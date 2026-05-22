# Hard problem: joint multi-user adaptive control, and its decomposition

Phase 0-3 solved a *single* Charlie->user link with two scalars (mu, beta). That
problem is low-dimensional and nearly admits a closed-form rule, which weakens
the case for a learned controller. Here we state the **full multi-user,
multi-threat, time-varying problem** of [P3] (which we previously simplified
away) and decompose it into subproblems with clearly defined couplings. This is
the problem the controller actually targets.

Notation follows docs/formulation.md. Reduced per-link statistics (Eq. 16-22
there): for any link with normalized SNR `gamma = gamma0*mu` and DT coefficient
`beta`,
$$P_c(\gamma,\beta) = \mathbb E_h[Q(\beta+\gamma(1-h))],\quad
  P_e(\gamma,\beta) = \mathbb E_h[Q(\beta+\gamma(1+h))],$$
with sift `P_sift = P_c + P_e` and per-link `QBER = P_e/P_sift`.

---

## 1. System and threats (the [P3] setting we now model in full)

A GEO source **Charlie** distributes a common SIM/BPSK sequence (depth `mu`) via
two LEO relays to a server **Alice** and a cluster of `N` users
**B_1,...,B_N** on the ground. The secret key is established **per pair
(Alice, B_i)** (entanglement-based BBM92-style), not Charlie->user. Each node `u`
has its own geometry-driven state `s_u = (gamma0_u, sigma_{X,u}, w_{eq,u})`,
since users sit at different zeniths/distances over the pass.

Decision variables at time `t`:
- `mu` — modulation depth, **global** (one transmitter): couples every link and
  sets the URA security margin.
- `beta_A` — Alice's DT coefficient, **shared** by every pair (Alice is common).
- `{beta_i}_{i=1..N}` — per-user DT coefficients (heterogeneous).
- `chi in [0,1]` — exclusion ratio of the fading-randomness multiple access.

Threats:
- **URA**: an eavesdropper near each node at offset `d_E` decodes with the
  optimal threshold; security needs `P_e^E(u) > p_e^{min}` (=0.1).
- **BSA**: a beam-splitting attacker at a relay; detectable iff the sift-prob
  deviation exceeds a threshold, which requires `beta` large enough
  (a detectability margin `m_{BSA}(beta) >= m^{min}`).

---

## 2. Per-pair sift, exclusion, and key rate

**Pairwise raw sift** (both Alice and `B_i` decode; detections are conditionally
independent given Charlie's bit):
$$P^{AB_i} = P_{sift}^{A}\,P_{sift}^{i}. \tag{1}$$

**Pairwise QBER** (disagreement given both sifted):
$$\mathrm{QBER}^{AB_i} = \frac{P_c^A P_e^i + P_e^A P_c^i}{P_{sift}^A P_{sift}^i}. \tag{2}$$

**Exclusion term.** The proposed multiple access keeps only bits not shared with
other pairs. The m-fold intersection of pairs over a B-subset `S` (plus Alice)
equals `P_c^A \prod_{u in S} P_c^u` (from [P3] Eq. 22 with `P_C=1/2` and 0/1
symmetry). Inclusion-exclusion over all `S \subseteq \{j\neq i\}` collapses to a
**closed form** (this generalizes [P3] Eq. 24 to the asymmetric cluster):
$$P^{AB_i}_{excl} = P_c^A\,P_c^i\,\Big[\,1 - \prod_{j\neq i}\big(1 - P_c^{\,j}\big)\Big]. \tag{3}$$

**Excluded sift** and **sifted-key rate** ([P3] Eq. 19):
$$P^{AB_i}_{\chi} = P^{AB_i} - \chi\,P^{AB_i}_{excl},\qquad
  R_s^i = P^{AB_i}_{\chi}\,R_b. \tag{4}$$

**Per-pair secret fraction** (wiretap; Eve_1 near Alice, Eve_2 near `B_i`):
$$\sigma_i = \max\!\big(0,\; I(A;B_i) - \max\big(I(A;E_1),\,I(B_i;E_2)\big)\big),\quad
  I(A;B_i)=1-H_2(\mathrm{QBER}^{AB_i}),\ I(\cdot;E)=1-H_2(P_e^E). \tag{5}$$

**Total cluster key rate** (the objective):
$$R_f(\mathbf p; \mathbf s,t) = \sum_{i=1}^N R_s^i\,\sigma_i,
\qquad \mathbf p=(mu,\beta_A,\{\beta_i\},\chi). \tag{6}$$

---

## 3. The joint problem (P)

$$\textbf{(P)}\quad \max_{mu,\,\beta_A,\,\{\beta_i\},\,\chi}\ R_f
\quad\text{s.t.}\quad
\begin{cases}
\mathrm{QBER}^{AB_i} < q^{\max}, & i=1..N\\
P^{AB_i}_{\chi} > p_s^{\min}, & i=1..N\\
P_e^{E}(A), P_e^{E}(B_i) > p_e^{\min}, & \text{(URA)}\\
m_{BSA}(\beta_A), m_{BSA}(\beta_i) \ge m^{\min}, & \text{(BSA)}\\
mu\in(0,1),\ \beta\in[\beta_{lo},\beta_{hi}],\ \chi\in[0,1].
\end{cases}$$

**Why (P) is hard (and ML-worthy):**
- `mu` is **global** and appears in every `gamma_u = gamma0_u\,mu`, coupling all
  pairs and both threats nonlinearly.
- `chi` couples pairs through the product `\prod_{j\neq i}(1-P_c^j)` in (3): each
  pair's rate depends on **all** users' detection probabilities.
- `beta_A` is shared; `{beta_i}` are heterogeneous -> the action dimension grows
  with `N`.
- Constraints are tight and create a feasibility boundary (Phase 3 finding).
The map `s(t) -> p*(t)` is a coupled, variable-size, constrained mapping, not a
2-D lookup -> a learned (KAN/MLP) controller is justified.

---

## 4. Decomposition into subproblems

Exploit the variable roles: `(mu,chi)` are the **global/coupling** block;
`(beta_A,{beta_i})` are **link-local** given the block.

### Outer problem (O) -- 2-D global search
$$\textbf{(O)}\quad \max_{mu\in(0,1),\,\chi\in[0,1]}\ V(mu,\chi),$$
where `V(mu,chi)` is the optimal cluster rate returned by the inner problem.
`mu` is also screened by the URA/BSA constraints (they depend on `mu,beta`).
Two scalars -> grid or low-dim search.

### Inner problem (I) -- thresholds given (mu, chi)
$$\textbf{(I)}\quad V(mu,\chi)=\max_{\beta_A,\{\beta_i\}}\ \sum_i R_s^i\,\sigma_i
\ \text{ s.t. per-pair QBER/sift/URA/BSA.}$$
Couplings inside (I):
- through **`beta_A`** (shared, appears in every `P_c^A,P_e^A`), and
- through the **exclusion field** `phi_i = \prod_{j\neq i}(1-P_c^j)` in (3).

### Mean-field fixed point (M) -- decouples `{beta_i}`
Treat `phi = {phi_i}` as a field. Given `phi` and `beta_A`, each user's
subproblem **decouples**:
$$\textbf{(U_i)}\quad \beta_i^\star(phi,\beta_A)=\arg\max_{\beta_i}\ R_s^i\,\sigma_i
\ \text{ s.t. its own constraints (1-D).}$$
Then refresh the field from the new `{P_c^i}` and iterate to a fixed point:
$$phi_i^{(k+1)} = \prod_{j\neq i}\big(1 - P_c^{\,j}(\beta_j^{\star,(k)})\big).$$
`beta_A` is updated by a 1-D search holding `{beta_i}` (block coordinate). Under
the cluster-symmetric special case all `beta_i` collapse to one `beta_B` and (3)
reduces to `P_c^A P_c^B[1-(1-P_c^B)^{N-1}]` ([P3] Eq. 24), recovering the paper.

### Solver summary
```
solve (P):
  for (mu, chi) in outer grid:               # (O), 2-D
     init field phi, beta_A
     repeat until fixed point:               # (M)
        for each user i: beta_i = argmax_1D   # (U_i), decoupled
        update phi from {P_c^i}
        beta_A = argmax_1D given {beta_i}      # block coordinate
     V(mu,chi) = sum_i R_s^i sigma_i
  return argmax
```
This **oracle** generates labels `s(t) -> p*(t)` for the controller. The learned
controller predicts `(mu, beta_A, {beta_i}, chi)` directly from the cluster
state, amortizing the nested solve for onboard real-time use; KAN additionally
exposes closed-form rules for the link-local thresholds.

---

## 5. What this adds over [P3] and over Phase 0-3
- Full **multi-user coupling** (chi exclusion, heterogeneous users) -- previously
  dropped; restores and *generalizes* [P3] (asymmetric closed-form (3)).
- **BSA** detectability constraint alongside URA -- previously dropped.
- A **principled decomposition** (outer 2-D x mean-field inner) giving a tractable
  oracle and a clear ML target whose dimension scales with `N`.
- The controller now solves something a 2-D lookup cannot, justifying KAN/MLP.
