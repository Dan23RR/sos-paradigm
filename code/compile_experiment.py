"""Theorem D (experimental) — can a small LLM COMPILE to the SOS substrate?

We define a minimal DSL for substrate programs, give the LLM a spec + 20 worked
examples, then present 50 unseen inputs and measure:

  (1) SYNTACTIC validity — does the LLM emit a parseable program?
  (2) SEMANTIC correctness — does executing it produce the right answer?

This is the LLM-as-compiler test.  If the LLM can address the substrate at
>= 80% correctness on held-out inputs, the paradigm is viable at 500M scale.

The task for this experiment is INTENTIONALLY simple: 'given n, compute n mod 7
using substrate ops'.  The point is NOT to beat a calculator, it is to see
whether the LLM can learn a TRANSLATION rule from a DSL spec.

DSL (Lisp-flavoured):
  (phase k d)    -> rotation phase 2*pi*k/d (returns a substrate constant)
  (shift k H)    -> compose the flow of H for time k
  (cleanup label)-> discretise to nearest labeled state
  (main n)       -> program entry point

Task:  input  n ∈ {0..999}
       output n mod 7

Correct program:  (main n) = (cleanup (shift n (phase 1 7)))

(Semantically: the substrate is a circle T¹ divided into 7 equal sectors.
 Shifting by n sectors from sector 0 lands in sector n mod 7.)
"""
import os, sys, time, re, json
os.environ['HF_HOME'] = '/sessions/gracious-beautiful-goldberg/.hf_cache'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Ground-truth DSL interpreter -----------------------------------------

def execute_program(program_text, n):
    """Parse and execute the minimal DSL. Returns int result in 0..6, or None
    if the program is syntactically invalid or semantically divergent."""
    try:
        # Normalise whitespace
        prog = program_text.strip()
        # Very light parser — look for the two idioms we expect.
        # Accept either (main n) = (cleanup (shift n (phase k d))) or just
        # an inline expression.
        # Extract (phase k d) and (shift <n-expr> ...).
        mph = re.search(r'\(\s*phase\s+(\d+)\s+(\d+)\s*\)', prog)
        msh = re.search(r'\(\s*shift\s+([A-Za-z0-9_]+)\s+', prog)
        mcl = re.search(r'\(\s*cleanup\b', prog)
        if not (mph and msh and mcl):
            return None
        k = int(mph.group(1))
        d = int(mph.group(2))
        shift_arg = msh.group(1)
        # shift_arg should be 'n' or a literal integer
        if shift_arg == 'n':
            total = n * k
        else:
            try:
                total = int(shift_arg) * k
            except ValueError:
                return None
        return total % d
    except Exception:
        return None

SPEC = """You are compiling symbolic tasks to a tiny DSL that runs on a phase-space substrate (a circle partitioned into d sectors).

DSL grammar:
  (phase k d)     — a phase constant equal to 2*pi*k/d on the circle T^1
  (shift n H)     — advance the current sector by n * (k/d of one turn), where H = (phase k d)
  (cleanup label) — snap to the nearest labeled sector and return its integer label in [0, d-1]
  (main n) = ...  — a top-level program with one integer input n

Semantics: the substrate starts at sector 0. (shift n H) moves you n*k sectors forward modulo d. (cleanup ...) returns the current sector index as an integer.

Your job: given a task description, output EXACTLY ONE program of the form
  (main n) = (cleanup (shift n (phase K D)))
with integer K, D appropriate to the task.  Put the program in a \\boxed{...}.
Do NOT print anything else inside the box.
"""

FEW_SHOTS = """
Example 1. Task: compute n mod 5.
Program: \\boxed{(main n) = (cleanup (shift n (phase 1 5)))}

Example 2. Task: compute n mod 10.
Program: \\boxed{(main n) = (cleanup (shift n (phase 1 10)))}

Example 3. Task: compute (2n) mod 9.
Program: \\boxed{(main n) = (cleanup (shift n (phase 2 9)))}

Example 4. Task: compute (3n) mod 11.
Program: \\boxed{(main n) = (cleanup (shift n (phase 3 11)))}

Example 5. Task: compute n mod 4.
Program: \\boxed{(main n) = (cleanup (shift n (phase 1 4)))}
"""

# ---- Test suite ------------------------------------------------------------

def make_tasks():
    """Generate 50 held-out tasks of the form '(a*n) mod d'."""
    rng = np.random.default_rng(1717)
    tasks = []
    for _ in range(50):
        d = int(rng.integers(3, 30))
        a = int(rng.integers(1, d))
        tasks.append({'a': a, 'd': d,
                      'desc': f'compute ({a}*n) mod {d}' if a > 1 else f'compute n mod {d}'})
    return tasks

def ground_truth(a, d, n):
    return (a * n) % d

# ---- LLM harness -----------------------------------------------------------

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

print(f"Loading {MODEL} ...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s")

def extract_boxed(text):
    # Greedy: take the LAST \boxed{...} in the response
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    return matches[-1] if matches else None

@torch.no_grad()
def llm_compile(task_desc):
    user = f"{SPEC}\n\nHere are examples:\n{FEW_SHOTS}\n\nNow the actual task.\nTask: {task_desc}.\nProgram:"
    messages = [{'role': 'user', 'content': user}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = model.generate(ids, max_new_tokens=120, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    prog = extract_boxed(text) or text.strip()
    return prog, text

# ---- Run -------------------------------------------------------------------

tasks = make_tasks()
print(f"\nRunning compile test on {len(tasks)} held-out tasks...")
print(f"(task format: compute (a*n) mod d, a,d random in reasonable ranges)\n")

rows = []
n_valid = 0
n_correct = 0
probe_ns = [0, 1, 2, 7, 13, 57, 101, 200, 500, 999]

for i, task in enumerate(tasks):
    t_start = time.time()
    prog, raw = llm_compile(task['desc'])
    dt = time.time() - t_start
    # Check validity on probe_ns
    valid = True
    correct_cnt = 0
    for n in probe_ns:
        pred = execute_program(prog, n)
        truth = ground_truth(task['a'], task['d'], n)
        if pred is None:
            valid = False
            break
        if pred == truth:
            correct_cnt += 1
    semantic_ok = valid and correct_cnt == len(probe_ns)
    n_valid += int(valid)
    n_correct += int(semantic_ok)
    marker = 'OK' if semantic_ok else ('VAL' if valid else 'INV')
    print(f"  t{i+1:2d} [{marker}] {task['desc']:35s}  -> {prog[:60]:60s}  ({correct_cnt}/{len(probe_ns)} probes, {dt:.1f}s)")
    rows.append({'i': i, 'task': task['desc'], 'a': task['a'], 'd': task['d'],
                 'prog': prog, 'valid': valid, 'semantic_ok': semantic_ok,
                 'probes_passed': correct_cnt, 'dt': dt, 'raw': raw[:400]})

print(f"\n=== SUMMARY ===")
print(f"  syntactic validity : {n_valid}/{len(tasks)}  ({100*n_valid/len(tasks):.1f}%)")
print(f"  semantic correctness: {n_correct}/{len(tasks)}  ({100*n_correct/len(tasks):.1f}%)")
print(f"  threshold for Theorem D viability: >=80% semantic correctness")
if n_correct/len(tasks) >= 0.8:
    print("  >>> Theorem D VIABLE at 500M-parameter scale. <<<")
else:
    print("  >>> Theorem D: unclear/negative at 500M scale. <<<")

np.savez('/sessions/gracious-beautiful-goldberg/sos_compile_results.npz',
         rows=np.array(rows, dtype=object),
         n_valid=n_valid, n_correct=n_correct, n_total=len(tasks))
print("\nSaved sos_compile_results.npz")
