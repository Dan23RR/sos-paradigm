# Reproducibility code

A curated subset of the scripts referenced in the papers. Each script is runnable standalone on a CPU (some with a small GPU for speed).

## Requirements

Python 3.10+, `numpy`, `scipy`, `torch`. Optional: `matplotlib` for plots.

```
pip install numpy scipy torch matplotlib
```

## Scripts

### `theorem_a_fsa_embedding.py`
Numeric verification of Paper 1, Theorem A (FSA embedding into toroidal HRR). Scales to `m = 3000` automaton states and confirms the `d ≥ C · mk · log(m/delta)` capacity bound with tight constant `C ≈ 2`.

```
python code/theorem_a_fsa_embedding.py
```

### `compile_experiment.py`
The compile-learning experiment (Paper 1 Theorem D): a small LLM learns from a ~50-line DSL specification and ~20 worked examples to produce valid substrate programs for unseen inputs.

```
python code/compile_experiment.py
```

### `compile_compositional.py`
Chained-DSL compositional compile test. Validates that the compile-learning transfers to programs built by composing multiple primitives.

```
python code/compile_compositional.py
```

## Full reproducibility

The full experimental suite (~100 scripts, 97 unit tests covering Paper 1 theorems 1-13 and Paper 2 theorems 1-6, plus the RLVR-vs-SFT null-result rerun at Qwen2.5 0.5B / 1.5B) is available on request. Contact: `daniel.culotta@gmail.com`.
