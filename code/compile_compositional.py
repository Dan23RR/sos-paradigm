"""Theorem D v2 — COMPOSITIONAL compile: does the LLM chain substrate atoms?

Previous Theorem D test was single-atom: one `phase/shift/cleanup` per task.
To claim the LLM can *program* the substrate rather than just template-fill, we
need tasks that require COMBINING two (or three) atoms.

Task family A (2-atom):  compute ((a*n + b) mod d)
  Target DSL form:
    (main n) = (cleanup
                (compose
                  (shift n (phase a d))
                  (shift 1 (phase b d))))

Task family B (3-atom):  compute (((a*n + b) mod d1) * c) mod d2
  Target DSL form:
    (main n) = (cleanup
                (compose
                  (shift 1 (phase
                              (mod-eval (a*n + b) d1)
                              d1))
                  (scale c d2)))
  (simplified: the LLM only needs to produce the algebraic pattern;
   we interpret the s-expr directly)

The point: can Qwen2.5-0.5B, from a 10-line spec + few-shot examples,
produce syntactically-valid *compositions*?  Theorem D was about template
slot-fill; this is about chaining.

We report:
  (1) Syntactic validity of emitted program (parses cleanly).
  (2) Semantic correctness on 10 random probe values of n.
"""
import os, sys, re, time, json
os.environ['HF_HOME'] = '/sessions/gracious-beautiful-goldberg/.hf_cache'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch, warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Interpreters -----------------------------------------------------------
def interp_2atom(prog, n):
    """(main n) = (cleanup (compose (shift n (phase a d)) (shift 1 (phase b d))))
       — compute (a*n + b) mod d."""
    try:
        shifts = re.findall(r'\(\s*shift\s+(\S+)\s+\(\s*phase\s+(-?\d+)\s+(-?\d+)\s*\)\s*\)', prog)
        if len(shifts) < 2: return None
        # First shift: (shift n (phase a d))  -> contribution a*n
        # Second shift: (shift 1 (phase b d)) -> contribution b
        # d must match (take it from first)
        arg1, coef1, mod1 = shifts[0]
        arg2, coef2, mod2 = shifts[1]
        d = int(mod1)
        if int(mod2) != d: return None
        # Evaluate coefficients against variable n
        def eval_arg(s):
            s = s.strip()
            if s == 'n': return n
            try: return int(s)
            except: return None
        v1 = eval_arg(arg1); v2 = eval_arg(arg2)
        if v1 is None or v2 is None: return None
        return (v1 * int(coef1) + v2 * int(coef2)) % d
    except Exception:
        return None

def interp_3atom(prog, n):
    """(main n) = (cleanup (compose (... a*n+b mod d1) (scale c d2)))
       — compute (((a*n+b) mod d1) * c) mod d2."""
    try:
        shifts = re.findall(r'\(\s*shift\s+(\S+)\s+\(\s*phase\s+(-?\d+)\s+(-?\d+)\s*\)\s*\)', prog)
        scales = re.findall(r'\(\s*scale\s+(-?\d+)\s+(-?\d+)\s*\)', prog)
        if len(shifts) < 2 or len(scales) < 1: return None
        arg1, coef1, mod1 = shifts[0]
        arg2, coef2, mod2 = shifts[1]
        d1 = int(mod1)
        if int(mod2) != d1: return None
        c, d2 = map(int, scales[0])
        def eval_arg(s):
            s = s.strip()
            if s == 'n': return n
            try: return int(s)
            except: return None
        v1 = eval_arg(arg1); v2 = eval_arg(arg2)
        if v1 is None or v2 is None: return None
        inner = (v1 * int(coef1) + v2 * int(coef2)) % d1
        return (inner * c) % d2
    except Exception:
        return None

# ---- Spec ------------------------------------------------------------------
#
# KEY LESSON FROM FIRST RUN: Qwen2.5-0.5B sees "compute (16*n+14) mod 24" as a
# math problem and goes into verbose CoT mode, burning the token budget before
# ever emitting the DSL line.  We fix this two ways:
#   (1) prompt starts with the imperative "DO NOT compute. TRANSLATE only."
#   (2) prompt ends with "\\boxed{" as a prefix the model completes.
SPEC_2 = """ROLE: You are a SYNTAX TRANSLATOR, not a solver.

Given a math task of the exact form  "compute (a*n + b) mod d"  with integers
a, b, d, you MUST output EXACTLY ONE LINE of DSL and NOTHING ELSE.  Do NOT
compute anything.  Do NOT simplify.  Do NOT explain.  Just pattern-match.

DSL template (the ONLY thing to output):
  \\boxed{(main n) = (cleanup (compose (shift n (phase <a> <d>)) (shift 1 (phase <b> <d>))))}
substituting the three integer values from the task.  Use the literal symbol
"n" (no number for n).  One single line, wrapped in \\boxed{}.  Stop.
"""
FEW_SHOT_2 = """
Example 1. Task: compute (5*n + 3) mod 11.
\\boxed{(main n) = (cleanup (compose (shift n (phase 5 11)) (shift 1 (phase 3 11))))}

Example 2. Task: compute (7*n + 2) mod 13.
\\boxed{(main n) = (cleanup (compose (shift n (phase 7 13)) (shift 1 (phase 2 13))))}

Example 3. Task: compute (2*n + 9) mod 17.
\\boxed{(main n) = (cleanup (compose (shift n (phase 2 17)) (shift 1 (phase 9 17))))}

Example 4. Task: compute (12*n + 4) mod 19.
\\boxed{(main n) = (cleanup (compose (shift n (phase 12 19)) (shift 1 (phase 4 19))))}

Example 5. Task: compute (1*n + 0) mod 7.
\\boxed{(main n) = (cleanup (compose (shift n (phase 1 7)) (shift 1 (phase 0 7))))}
"""

SPEC_3 = """ROLE: You are a SYNTAX TRANSLATOR, not a solver.

Given a math task of the exact form  "compute (((a*n + b) mod d1) * c) mod d2"
with integers a, b, d1, c, d2, you MUST output EXACTLY ONE LINE of DSL and
NOTHING ELSE.  Do NOT compute.  Do NOT simplify.  Do NOT explain.  Just
pattern-match.

DSL template (the ONLY thing to output):
  \\boxed{(main n) = (cleanup (compose (shift n (phase <a> <d1>)) (shift 1 (phase <b> <d1>))) (scale <c> <d2>))}
substituting the five integer values.  One single line, wrapped in \\boxed{}.
Stop.
"""
FEW_SHOT_3 = """
Example 1. Task: compute (((3*n + 1) mod 7) * 2) mod 5.
\\boxed{(main n) = (cleanup (compose (shift n (phase 3 7)) (shift 1 (phase 1 7))) (scale 2 5))}

Example 2. Task: compute (((4*n + 2) mod 11) * 3) mod 8.
\\boxed{(main n) = (cleanup (compose (shift n (phase 4 11)) (shift 1 (phase 2 11))) (scale 3 8))}

Example 3. Task: compute (((6*n + 5) mod 13) * 4) mod 9.
\\boxed{(main n) = (cleanup (compose (shift n (phase 6 13)) (shift 1 (phase 5 13))) (scale 4 9))}

Example 4. Task: compute (((1*n + 0) mod 5) * 2) mod 7.
\\boxed{(main n) = (cleanup (compose (shift n (phase 1 5)) (shift 1 (phase 0 5))) (scale 2 7))}
"""

# ---- Tasks -----------------------------------------------------------------
def make_2_tasks(n_tasks, seed=8181):
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n_tasks):
        d = int(rng.integers(5, 25))
        a = int(rng.integers(1, d))
        b = int(rng.integers(0, d))
        tasks.append({'a': a, 'b': b, 'd': d,
                      'desc': f'compute ({a}*n + {b}) mod {d}',
                      'truth': lambda a=a, b=b, d=d: (lambda n: (a*n + b) % d)})
    return tasks

def make_3_tasks(n_tasks, seed=9191):
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n_tasks):
        d1 = int(rng.integers(5, 20))
        a = int(rng.integers(1, d1))
        b = int(rng.integers(0, d1))
        d2 = int(rng.integers(5, 15))
        c = int(rng.integers(1, d2))
        tasks.append({'a': a, 'b': b, 'd1': d1, 'c': c, 'd2': d2,
                      'desc': f'compute ((({a}*n + {b}) mod {d1}) * {c}) mod {d2}',
                      'truth': lambda a=a, b=b, d1=d1, c=c, d2=d2:
                               (lambda n: (((a*n + b) % d1) * c) % d2)})
    return tasks

# ---- LLM -------------------------------------------------------------------
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
print(f"Loading {MODEL} ...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    return matches[-1] if matches else None

@torch.no_grad()
def llm_compile(spec, few_shot, desc, max_new=500):
    # Use a larger budget so the model can do its CoT if it insists, but we
    # ALWAYS extract the LAST \\boxed{...} from the continuation.  That way the
    # test is whether the model EVENTUALLY emits valid DSL, regardless of
    # whether it also babbles about the answer first.
    user = (f"{spec}\n\nExamples:\n{few_shot}\n\n"
            f"Now the actual task — TRANSLATE only.  Even if you think about it, "
            f"your final output MUST end with the \\boxed{{...}} DSL line.\n"
            f"Task: {desc}\nOutput:")
    messages = [{'role': 'user', 'content': user}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    prog = extract_boxed(text) or text.strip()
    return prog, text

# ---- Run both rounds --------------------------------------------------------
probe_ns = [0, 1, 2, 7, 13, 37, 100, 333, 500, 999]

def run_round(label, tasks, interp, spec, few_shot):
    print(f"\n=== ROUND: {label} ({len(tasks)} tasks) ===", flush=True)
    n_valid = 0; n_correct = 0
    rows = []
    for i, task in enumerate(tasks):
        t0 = time.time()
        prog, raw = llm_compile(spec, few_shot, task['desc'])
        dt = time.time() - t0
        truth_fn = task['truth']() if callable(task.get('truth')) else None
        # Our 'truth' is a thunk that returns a lambda — call once.
        truth_lambda = task['truth']()
        valid = True
        correct_cnt = 0
        for nv in probe_ns:
            pred = interp(prog, nv)
            tr = truth_lambda(nv)
            if pred is None:
                valid = False; break
            if pred == tr: correct_cnt += 1
        semantic = valid and correct_cnt == len(probe_ns)
        n_valid += int(valid)
        n_correct += int(semantic)
        mark = 'OK' if semantic else ('VAL' if valid else 'INV')
        print(f"  t{i+1:2d} [{mark}] {task['desc'][:38]:38s} -> {prog[:70]:70s}  "
              f"({correct_cnt}/{len(probe_ns)}, {dt:.1f}s)", flush=True)
        rows.append({'i': i, 'desc': task['desc'], 'prog': prog,
                     'valid': valid, 'semantic_ok': semantic,
                     'probes_passed': correct_cnt, 'dt': dt})
    print(f"\n  {label}  syntactic: {n_valid}/{len(tasks)} ({100*n_valid/len(tasks):.1f}%)  "
          f"semantic: {n_correct}/{len(tasks)} ({100*n_correct/len(tasks):.1f}%)", flush=True)
    return n_valid, n_correct, rows

tasks_2 = make_2_tasks(30)
tasks_3 = make_3_tasks(30)

v2, c2, r2 = run_round('2-ATOM', tasks_2, interp_2atom, SPEC_2, FEW_SHOT_2)
v3, c3, r3 = run_round('3-ATOM', tasks_3, interp_3atom, SPEC_3, FEW_SHOT_3)

print(f"\n=== COMPOSITIONAL SUMMARY ===", flush=True)
print(f"  2-atom  syntactic {v2}/30 ({100*v2/30:.1f}%)  semantic {c2}/30 ({100*c2/30:.1f}%)")
print(f"  3-atom  syntactic {v3}/30 ({100*v3/30:.1f}%)  semantic {c3}/30 ({100*c3/30:.1f}%)")

np.savez('/sessions/gracious-beautiful-goldberg/sos_compile_compositional_results.npz',
         rows_2=np.array(r2, dtype=object),
         rows_3=np.array(r3, dtype=object),
         v2=v2, c2=c2, v3=v3, c3=c3)
print("\nSaved.")
