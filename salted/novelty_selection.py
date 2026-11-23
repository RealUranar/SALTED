"""Sparse (reference) environment selection by kernel novelty, the hybrid rule.

Kernel: k(a, b) = (x_a . x_b) ** zeta on the lambda = 0 descriptor rows as stored,
the same kernel rkhs_projector and the prediction use at lambda = 0. Kernels exist only
within a species. Novelty of atom a w.r.t. a reference set R of its species:

    nov(a) = 1 - k_aR K_RR^-1 k_Ra / k(a, a) = r(a, a) / k(a, a),

r the residual kernel, kept per species as an incremental pivoted Cholesky factor
L (N_s x m, one column per reference) and the residual diagonal d = diag(r).

Hybrid rule: the first round(f M) references (f = sparse_hybrid_fraction, rounded half up)
by totalnov, the rest by maxnov on the same Cholesky factors; f = 1 is totalnov alone,
f = 0 maxnov alone.
    totalnov next reference = candidate i maximising the novelty it removes,
             gain_i = sum_j r(j, i)^2 / (k_jj r(i, i)) over the valid atoms j of the species,
             i over ncand drawn candidates. The own term j = i, which is i's novelty, is
             taken exactly and the other terms are estimated from neval drawn atoms as
             (n_valid - 1) / |E - {i}| * sum over E - {i}, so species of different size are
             comparable. ncand = neval = 0 is the exact greedy (tests). Draws are fresh,
             without replacement, from one numpy Generator seeded by inp.salted.seed, made
             each time a species is evaluated.
    maxnov   next reference = highest d_i / k_ii.

Budget: one budget M over all species; each step picks the species whose best candidate
scores highest. Only the species that got the pick is re-evaluated; every other species
keeps its cached best candidate (for totalnov its cached subsample draw too), which stays
exact because a species' residual kernel changes only when it gains a reference. A species
stops being eligible once its largest normalised residual is below RTOL.

Atoms with k(a, a) <= 0 or a non-finite self-kernel are never candidates and are left
out of the totalnov evaluation sums; their count is printed.
"""

import numpy as np

NCAND = 256  # totalnov candidates per species and step
NEVAL = 4096  # totalnov evaluation atoms per species and step
RTOL = 1e-12  # relative residual below which an atom adds no new direction
_BLOCK = 32 << 20  # float64 entries per kernel block in the totalnov gain


class IncrementalCholesky:
    """Pivoted incremental Cholesky of the kernel (X X^T) ** zeta of one species."""

    def __init__(self, X, zeta, cap=64):
        X = self.X = np.asarray(X, dtype=np.float64)  # float32 rows would mix precisions with the float64 L
        self.zeta = zeta
        self.k = np.einsum("ij,ij->i", X, X) ** zeta
        self.valid = np.isfinite(self.k) & (self.k > 0)
        self.d = np.where(self.valid, self.k, 0.0)
        self.L = np.empty((len(X), max(1, cap)))
        self.m = 0
        self.picked = np.zeros(len(X), bool)

    def add(self, p):
        """Make atom p a reference; returns False when it adds no new direction."""
        self.picked[p] = True
        if not self.d[p] > RTOL * self.k[p]:
            return False
        if self.m == self.L.shape[1]:
            grown = np.empty((len(self.X), 2 * self.m))
            grown[:, : self.m] = self.L
            self.L = grown
        col = (self.X @ self.X[p]) ** self.zeta
        if self.m:
            col -= self.L[:, : self.m] @ self.L[p, : self.m]
        col /= np.sqrt(self.d[p])
        col[~self.valid] = 0.0
        if not np.all(np.isfinite(col)):
            raise ValueError(
                "Non-finite kernel column in novelty selection; a negative dot product "
                f"with non-integer zeta={self.zeta} gives NaN."
            )
        self.L[:, self.m] = col
        self.m += 1
        self.d -= col * col
        np.maximum(self.d, 0.0, out=self.d)  # rounding can push residuals slightly below 0
        self.d[p] = 0.0
        return True

    def eligible(self):
        return np.flatnonzero(self.valid & ~self.picked & (self.d > RTOL * self.k))

    def totalnov_gain(self, cand, evl):
        """Novelty removed from the atoms evl by adding each of cand, sum_j r(j,i)^2/(k_jj r(i,i))."""
        XE, LE, kE = self.X[evl], self.L[evl, : self.m], self.k[evl]
        nE = len(kE)
        out = np.empty(len(cand))
        step = max(1, _BLOCK // max(1, nE))
        for a in range(0, len(cand), step):
            c = cand[a : a + step]
            r = (XE @ self.X[c].T) ** self.zeta
            if self.m:
                r -= LE @ self.L[c, : self.m].T
            out[a : a + step] = np.einsum("ji,ji,j->i", r, r, 1.0 / kE) / self.d[c]
        return out

    def estimated_gain(self, cand, rng, neval):
        """totalnov gain of cand over all valid atoms, from neval drawn atoms (0 = exact)."""
        nvalid = int(self.valid.sum())
        if neval == 0 or neval >= nvalid:
            return self.totalnov_gain(cand, slice(None) if nvalid == len(self.k) else np.flatnonzero(self.valid))
        evl = rng.choice(np.flatnonzero(self.valid), neval, replace=False)
        in_e = np.zeros(len(self.k), bool)
        in_e[evl] = True
        in_e = in_e[cand]
        own = self.d[cand] / self.k[cand]
        # own term exact: its N_s/n_eval-scaled sampling noise would win the argmax and favour large species
        return own + (self.totalnov_gain(cand, evl) - in_e * own) * ((nvalid - 1) / np.maximum(neval - in_e, 1))

    def best(self, rule, rng, ncand, neval):
        """(score, atom) of the next reference by rule, or (-inf, -1) when none is eligible."""
        elig = self.eligible()
        if len(elig) == 0:
            return -np.inf, -1
        if rule == "maxnov":
            nov = self.d[elig] / self.k[elig]
            i = int(np.argmax(nov))
            return float(nov[i]), int(elig[i])
        cand = elig if ncand == 0 or ncand >= len(elig) else rng.choice(elig, ncand, replace=False)
        g = self.estimated_gain(cand, rng, neval)
        i = int(np.argmax(g))
        return float(g[i]), int(cand[i])

    def fallback(self):
        """Highest remaining normalised residual among unpicked atoms, else the first unpicked atom."""
        rest = np.flatnonzero(~self.picked)
        if len(rest) == 0:
            return -np.inf, -1
        v = rest[self.valid[rest]]
        if len(v) == 0:
            return -1.0, int(rest[0])
        nov = self.d[v] / self.k[v]
        i = int(np.argmax(nov))
        return float(nov[i]), int(v[i])


def residual_novelty(X_s, ref_idx, zeta, eval_idx=None):
    """Exact novelty r(a,a)/k(a,a) of atoms eval_idx (all rows when None) of one species
    w.r.t. the references ref_idx (rows of X_s). NaN for atoms with k(a,a) <= 0."""
    ref_idx = np.asarray(ref_idx, int)
    if eval_idx is None:
        rows, ref_loc = slice(None), ref_idx
        X = X_s
    else:
        rows, inv = np.unique(np.concatenate([ref_idx, np.asarray(eval_idx, int)]), return_inverse=True)
        X = X_s[rows]
        ref_loc = inv[: len(ref_idx)]
    ch = IncrementalCholesky(X, zeta, cap=len(ref_idx))
    for p in ref_loc:
        ch.add(p)
    nov = np.where(ch.valid, ch.d / np.where(ch.valid, ch.k, 1.0), np.nan)
    return nov if eval_idx is None else nov[inv[len(ref_idx):]]


def _check(ncand, neval, seed, f):
    if not isinstance(f, (int, float, np.floating)) or isinstance(f, bool) or not 0 <= f <= 1:
        raise ValueError(f"sparse_hybrid_fraction must be in [0, 1], got {f!r}")
    for name, v in (("ncand", ncand), ("neval", neval), ("salted.seed", seed)):
        if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 0:
            raise ValueError(f"{name} must be an int >= 0, got {v!r}")


def select_sparse(X, species, M, zeta=2.0, hybrid_fraction=0.6, ncand=NCAND, neval=NEVAL,
                  seed=42, nspecies=None, verbose=False):
    """Indices (rows of X, in pick order) of the M reference environments by the hybrid rule.

    X: (N, nfeat) lambda = 0 descriptor rows of the dense training atoms.
    species: (N,) int species index of each row.
    """
    _check(ncand, neval, seed, hybrid_fraction)
    species = np.asarray(species, int)
    if M > len(X):
        raise ValueError(f"Menv ({M}) cannot exceed the {len(X)} atomic environments.")

    nspecies = int(species.max()) + 1 if nspecies is None else nspecies
    members = [np.flatnonzero(species == s) for s in range(nspecies)]
    print(f"Sparse selection: hybrid Menv={M} zeta={zeta} hybrid_fraction={hybrid_fraction} "
          f"ncand={ncand} neval={neval} seed={seed}")

    rng = np.random.default_rng(seed)
    chols = []
    for s in range(nspecies):
        ch = IncrementalCholesky(np.ascontiguousarray(X[members[s]]), zeta)
        nbad = len(members[s]) - int(ch.valid.sum())
        if nbad:
            print(f"Species {s}: {nbad} of {len(members[s])} atoms excluded (k(a,a) <= 0 or non-finite)")
        chols.append(ch)

    n1 = int(np.floor(hybrid_fraction * M + 0.5))  # picks made by totalnov

    def rule(k):
        return "totalnov" if k < n1 else "maxnov"

    picks = []
    nfallback = np.zeros(nspecies, int)
    best = [ch.best(rule(0), rng, ncand, neval) for ch in chols]
    for step in range(M):
        if step == n1 and step > 0:
            # cached totalnov gains are not maxnov scores
            best = [ch.best("maxnov", rng, ncand, neval) for ch in chols]
        s = int(np.argmax([b[0] for b in best]))
        score, p = best[s]
        if p < 0:
            fb = [ch.fallback() for ch in chols]
            s = int(np.argmax([f[0] for f in fb]))
            score, p = fb[s]
            nfallback[s] += 1
        chols[s].add(p)
        picks.append(members[s][p])
        best[s] = chols[s].best(rule(step + 1), rng, ncand, neval)
        if verbose:
            print(f"Doing {step + 1} of {M}: species {s} score = {score}")

    idx = np.array(picks, int)
    final = np.bincount(species[idx], minlength=nspecies)
    print("References per species:", final.tolist())
    for s in np.flatnonzero(nfallback):
        print(f"WARNING: species {s}: {nfallback[s]} references picked with novelty below "
              f"{RTOL} (no novel atoms left)")
    for s in np.flatnonzero(final < 2):
        print(f"WARNING: species {s} has {final[s]} references (rkhs_projector needs a usable kernel)")
    return idx
