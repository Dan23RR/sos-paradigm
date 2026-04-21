# SOS — Structural Separation & Substrate Scaling

> Can a language model truly *reason* with symbols — or is it 
> pattern-matching that happens to look like reasoning?

This research gives a formal answer. Standard transformers **cannot 
provably compose symbolic operations**: errors accumulate at every 
step, and the model cannot tell you why it's wrong. This isn't a 
training problem or a scale problem — it's structural.

The fix: a small external component called a **toroidal substrate** 
that sits beside the LLM, handles the symbolic computation exactly, 
and returns a verified result. Think of it as a calculator for 
logic — interpretable, provable, and sized by an explicit design rule.

Two papers. Independent research. April 2026.

*For the full technical treatment, see the Papers and Thesis sections below.*

---

Daniel Culotta · Independent research · 2026

---

## Papers

**Paper 1 — Structural Separation Theorems for Finite-Group Representations**
Zenodo DOI: [10.5281/zenodo.19642604](https://doi.org/10.5281/zenodo.19642604)
PDF: [`papers/paper1_structural_separation.pdf`](papers/paper1_structural_separation.pdf)

Key result (Thm 3'): no finite group G admits a non-trivial additive representation on R^d — a single no-go that subsumes the cyclic, abelian-product, and dihedral impossibility results previously known only case-by-case. Constructive positive results: toroidal T^k embeddings for abelian G, O(2)-matrix / Peter-Weyl embeddings for dihedral and non-abelian G. Capacity lower bound K(N, epsilon) >= (pi/epsilon)^N, validated numerically for N = 2..5. Complexity-class separation between additive, RoPE, MLP-residual, and torus-oracle representations with measured drift slopes 0.77 / 1.00 / 0.48 / 0.89.

**Paper 2 — Calibration Windows of Toroidal HRR Substrates**
PDF: [`papers/paper2_calibration_windows.pdf`](papers/paper2_calibration_windows.pdf)

Six contributions:
1. Three-regime scaling map for substrate composition at K/sqrt(D) ≈ 1/2.
2. Calibrated-regime margin formula m(K) ≈ 1 - C(V) sqrt(KV/D), with R^2 ≥ 0.97 at V = 16.
3. Operational design rule D* ≥ (C(V) / (1 - m*))^2 · KV, validated at K = 12 across nine substrate dimensions.
4. Peter-Weyl embedding for non-abelian D_n groups (Theorem 2 with decay O(sqrt(KV/d))).
5. LLM-substrate crossover L*(K, V, D) predicted in [3, 7] billion parameters for K = 10, V = 16, D = 512.
6. Controlled RLVR-with-substrate-reward vs SFT null result at matched compute (Qwen2.5 0.5B).

## Thesis

Standard transformer primitives cannot provably represent finite-group composition without error compounding linearly in depth. A toroidal HRR substrate can — provably and with explicit constants — provided D is sized per the calibration-window rule. The substrate sits beside the LLM as an interpretable external verifier: the LLM emits addresses (a compiled substrate program), the substrate computes, the verifier returns a typed result.

This framing is operational, not AGI-embryonic. The earlier AGI-embryo interpretation of the paradigm was retracted after five separate intrinsic-fitness signal failures in the `test5s-5w` pre-registered sequence — see the retractions ledger in Paper 2 §9.

## Relevance to AI safety

Substrate-as-verifier is an interpretable external alignment mechanism: not a learned reward model, hence not susceptible to reward-hacking in the usual RLHF sense. The RLVR-vs-SFT null result in Paper 2 identifies when substrate verifier reward cannot transmit usefully through the LLM policy (below-crossover regime L < L*). The calibration-window framework gives a per-task quantitative reliability predictor for substrate-augmented systems.

## Reproducibility

The papers are numerically reproducible from numpy + PyTorch + Qwen2.5 0.5B / 1.5B on a T4 Colab or RunPod A100. A subset of the core verification scripts is included:

- `code/theorem_a_fsa_embedding.py` — Paper 1 Theorem A numeric verification up to m = 3000 FSA states
- `code/compile_experiment.py` — compile-learning experiment (Theorem D)
- `code/compile_compositional.py` — chained DSL compositional compile

The full experimental suite (≈ 100 scripts, 97 unit tests) is available on request.

## Status

- Paper 1: shipped on Zenodo (April 2026), arXiv endorsement pending.
- Paper 2: draft complete (18pp), under internal review, arXiv-ready.
- Paper 3 (planned): hybrid architectures using m(K) routing criterion.

## Citation

```bibtex
@misc{culotta2026structural,
  author = {Culotta, Daniel},
  title  = {Structural Separation Theorems for Finite-Group Representations},
  year   = {2026},
  doi    = {10.5281/zenodo.19642604},
  url    = {https://doi.org/10.5281/zenodo.19642604}
}
```

## Contact

Daniel Culotta · `daniel.culotta@gmail.com` · [LinkedIn](https://www.linkedin.com/in/daniel-culotta-ml/)

## License

Code is released under Apache-2.0 (see `LICENSE`). Paper PDFs are distributed under CC-BY-4.0; please cite the Zenodo DOI.
