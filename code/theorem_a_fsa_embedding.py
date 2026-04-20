"""Memory-efficient m=3000 point for Theorem A sweep.

Strategy: use numpy memmaps for codebooks on disk, chunked memory build and
chunked cleanup argmax to keep RAM footprint bounded regardless of d.

Tests d candidates chosen from the empirical ratio d_min/d_pred ≈ 0.5 observed
at m ∈ {100, 300, 1000}.  For m=3000, d_pred = 264051, so we try
{60000, 90000, 120000, 150000}.
"""
import os, sys, time, math, json
os.environ['OPENBLAS_NUM_THREADS'] = '4'
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

WORK = '/sessions/gracious-beautiful-goldberg'
MMAP_TGT = WORK + '/S_tgt_mmap.npy'
MMAP_SRC = WORK + '/S_src_mmap.npy'


def gen_codebook_mmap(path, m, d, seed, dtype=np.complex64, chunk=200):
    """Build codebook as disk memmap in chunks to bound RAM."""
    if os.path.exists(path):
        os.remove(path)
    arr = np.memmap(path, dtype=dtype, mode='w+', shape=(m, d))
    # Seed per-row for reproducibility (doesn't matter here, just one pass)
    rng = np.random.default_rng(seed)
    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        phases = rng.random((end - start, d), dtype=np.float32)
        arr[start:end] = np.exp(1j * 2 * np.pi * phases).astype(dtype)
    arr.flush()
    return arr


def build_memory_chunked(delta, S_tgt, S_src, A, chunk=100):
    """M = Σ_{q,σ} S_tgt[δ(q,σ)] ⊛ conj(S_src[q]) ⊛ conj(A[σ])."""
    m, k = delta.shape
    d = S_tgt.shape[1]
    M = np.zeros(d, dtype=np.complex64)
    for sig in range(k):
        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            next_states = delta[start:end, sig]
            # Fetch rows from memmap into small RAM buffer
            T_tgt = np.asarray(S_tgt[next_states])        # (chunk, d)
            T_src = np.asarray(S_src[start:end])          # (chunk, d)
            T = T_tgt * np.conj(T_src)                    # bind
            M += (T * np.conj(A[sig])).sum(axis=0)
    return M


def cleanup_chunked(r, S_tgt, chunk=200):
    """Return argmax_q Re(conj(S_tgt[q]) · r), chunked over q."""
    m = S_tgt.shape[0]
    best_q = -1
    best_score = -np.inf
    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        T = np.asarray(S_tgt[start:end])
        scores = (np.conj(T) @ r).real
        local_best = int(np.argmax(scores))
        if scores[local_best] > best_score:
            best_score = scores[local_best]
            best_q = start + local_best
    return best_q


def walk_accuracy_chunked(delta, S_tgt, S_src, A, M, K, n_queries, seed):
    m, k = delta.shape
    rng = np.random.default_rng(seed + 2)
    correct = 0
    for i in range(n_queries):
        q = int(rng.integers(0, m))
        sigs = rng.integers(0, k, size=K, dtype=np.int32)
        q_truth = q
        for s in sigs:
            q_truth = int(delta[q_truth, s])
        q_hrr = q
        for s in sigs:
            r = M * np.asarray(S_src[q_hrr]) * A[int(s)]
            q_hrr = cleanup_chunked(r, S_tgt)
        if q_hrr == q_truth:
            correct += 1
    return correct / n_queries


def theoretical_dmin(m, k, delta_fail=0.05, C=4.0):
    return int(np.ceil(C * m * k * math.log(m / delta_fail)))


def test_point(m, k, d, K, n_queries, seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    delta = rng.integers(0, m, size=(m, k), dtype=np.int32)
    S_tgt = gen_codebook_mmap(MMAP_TGT, m, d, seed=seed + 1)
    S_src = gen_codebook_mmap(MMAP_SRC, m, d, seed=seed + 2)
    rng2 = np.random.default_rng(seed + 3)
    A = np.exp(1j * 2 * np.pi * rng2.random((k, d), dtype=np.float32)).astype(np.complex64)
    t_cb = time.time() - t0

    t0 = time.time()
    M = build_memory_chunked(delta, S_tgt, S_src, A)
    t_M = time.time() - t0

    t0 = time.time()
    acc = walk_accuracy_chunked(delta, S_tgt, S_src, A, M, K, n_queries, seed)
    t_walk = time.time() - t0

    # Clean up memmaps
    del S_tgt, S_src
    for p in (MMAP_TGT, MMAP_SRC):
        if os.path.exists(p):
            os.remove(p)

    print(f"    m={m:5d} k={k} d={d:7d} K={K:3d}  acc={acc*100:5.1f}%  "
          f"(cb {t_cb:.1f}s, M {t_M:.1f}s, walk {t_walk:.1f}s)", flush=True)
    return acc, {'t_cb': t_cb, 't_M': t_M, 't_walk': t_walk}


if __name__ == '__main__':
    m = 3000
    k = 2
    K = 1
    n_queries = 20  # fewer queries, each costs m × d float ops
    dp = theoretical_dmin(m, k)
    print(f"\n== m={m}  (d_pred = {dp}) memory-efficient run ==")
    # Start well below d_pred/2, then climb until hit 100%
    candidates = [60000, 100000, 140000]
    results = []
    dmin = None
    for d in candidates:
        acc, timings = test_point(m, k, d, K, n_queries, seed=3100)
        results.append({'m': m, 'k': k, 'd': d, 'K': K, 'acc': acc, **timings})
        if dmin is None and acc >= 0.95:
            dmin = d
        np.savez(WORK + '/sos_theorem_a_m3000_results.npz',
                 sweep=np.array(results, dtype=object),
                 dmin=dmin if dmin is not None else -1)
        if acc == 1.0:
            break

    print(f"\n  → m={m}  d_pred={dp}  d_min_emp={dmin}")
    if dmin:
        print(f"  ratio = {dmin/dp:.2f}")
    print("Done.")
