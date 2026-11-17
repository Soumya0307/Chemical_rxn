# Analytic vs. implemented parameters at the H$_3$ transition-state geometry

**Geometry.** Hamiltonian index 20 = NEB image 112 (the CI-NEB climbing image), taken verbatim from `run.log`:

```
H (-0.949629, -0.000254,  0.000916)
H ( 0.000733,  0.000001, -0.000004)   (Angstrom)
H ( 0.948896,  0.000253, -0.000911)
```

STO-3G, neutral doublet, $N_e=3$, no active-space truncation, $n_{\rm so}=6$ (6 qubits + 1 ancilla).

---

## 1. Spectral data required by the analytic formulas

All numbers below were recomputed independently (PySCF ROHF + OpenFermion JW + dense diagonalisation) at the geometry above, and cross-validated against the run log.

| Quantity | Value | Check against run |
|---|---|---|
| ROHF energy | $-1.531012867$ Ha | log: `converged SCF energy = -1.53101286734652` ✓ |
| $E_{\min}$ (full Fock) $=E_0^{(N=3)}$ | $-1.570989946$ Ha | log: `Fixed N=3 ground = -1.570989945643` ✓ |
| $E_{\max}$ (full Fock) | $+1.753548647$ Ha | — |
| $\|H\|$ (spectral norm) | **1.753549** Ha | — |
| $D=E_{\max}-E_{\min}$ | 3.324539 Ha | $2D=6.649077$ = log `wmax=6.64907719` ✓ |
| $N=3$ sector dimension | 20 | ✓ |
| $N=3$ distinct levels | $-1.570990,\ -1.248684,\ -0.912957,\ -0.842821,\ -0.493886,\ -0.457440,\ -0.246422,\ +0.117710,\ +0.353611$ | — |
| **Spectral gap $\Delta$** (to first *distinct* level) | **0.322306 Ha = 322.3 mHa** | same in the $(N_\alpha,N_\beta)=(2,1)$ sector |
| Ground-state degeneracy in $\mathcal H_{N_e=3}$ | **2-fold** ($S_z=\pm\tfrac12$) | — |
| Warm-start excess energy at this geometry | $6.2609\times10^{-4}$ Ha = 0.626 mHa | `energy_vs_w_applications.csv`, $n=0$ ✓ |
| Final energy error after $N_W=400$ | $5.3822\times10^{-4}$ Ha = 0.538 mHa | ✓ |

Two facts here matter for the comparison and should be stated in the chapter:

* **The gap at this geometry is 322 mHa, not milli-Hartree.** Every analytic parameter in Watts *et al.* scales as $1/\Delta$ (or $1/\Delta^2$, $1/\Delta^3$). The paper's own resource estimates are quoted for $\Delta\sim1$ mHa with $N_\omega=1000$. H$_3$/STO-3G at the climbing image is therefore **not** a small-gap instance, and the modest parameter values used are expected on the paper's own scaling — this is a consistency check, not evidence that the bounds are loose in the regime they were written for.
* **The $N_e=3$ ground state is exactly 2-fold degenerate** ($S_z=\pm1/2$), because the jump set conserves $\hat N$ but not $\hat S_z$. Theorem 2 / Lemma 4 of the paper assume a *nondegenerate* ground state. The formulas remain usable with $\Delta$ read as the gap to the first distinct eigenvalue (0.322306 Ha), which is what I use throughout, but the assumption mismatch is worth a sentence in the text.

---

## 2. Side-by-side comparison

$\varepsilon_{\rm chem}=1.6$ mHa throughout. Where the paper states a $\Theta(\cdot)$/$O(\cdot)$ scaling, the "analytic value" column evaluates the corresponding **explicit inequality proved in the appendix with the implicit constant set to 1** — this is stated per row, and is the only non-arbitrary way to turn the scaling into a number.

| # | Parameter | Analytic prescription (Watts *et al.*) | Analytic value at H$_3$ TS | Value used (matrix run) | Value used (circuit run) | Used / expected |
|---|---|---|---|---|---|---|
| 1 | Frequency period $\Omega$ | $\Omega\ge 4\|H\|$, Eq. (G8) | $\ge 7.0142$ | $\Omega=4D=13.2982$ | same | $1.90\times$ larger than needed |
| 2 | Filter bandwidth $L$ | chapter defines $L=2\|H\|$ | $3.5071$ | $L=2D=6.64908$ | same | see §4, flag (a) |
| 3 | Time spacing $\delta t=T_{\rm filter}$ | $\delta t=2\pi/\Omega$ | $\le 0.89578$ | $0.47249$ | same | $1.90\times$ finer |
| 4 | Time support $T_{N_\omega}$ | $T_{N_\omega}\ge 1/(\Delta\,\varepsilon_{\rm leak})$, Eq. (G9) | $1939$ a.u. at $\varepsilon_{\rm leak}=1.6\times10^{-3}$ | $(200{+}\tfrac12)\delta t=94.7$ a.u. | $47.5$ a.u. | $20\times$ / $41\times$ shorter |
| 5 | **Fourier cutoff $N_\omega$** | $N_\omega=\Theta\!\big(\tfrac{\|H\|}{\Delta}\tfrac{1}{\varepsilon_{\rm leak}}\big)$, Eq. (27) / §G | **4104** (run's $\delta t$) / **2164** (paper's minimal $\delta t$), at $\varepsilon_{\rm leak}=1.6\times10^{-3}$ | $k_{\max}=200$ (401 evolution calls); coefficients built with $N_\omega=500$ | $k_{\max}=400$ at $H_0$, $k_{\max}=100$ for $i>0$ | $20.5\times$ / $41\times$ fewer |
| 6 | Achieved leakage $\varepsilon_{\rm leak}$ | guaranteed $\le 1/(\Delta T_{N_\omega})$ | $3.28\times10^{-2}$ at $k_{\max}=200$ (guarantee) | **actual** $\sup_{\omega\ge\Delta}|\hat f|=6.65\times10^{-3}$; $|\hat f(+\Delta)|=4.22\times10^{-3}$ | $1.82\times10^{-2}$ / $8.96\times10^{-4}$ | envelope bound is $\sim5\times$ loose here |
| 7 | Window exponent $p$ | not prescribed (paper uses bare Dirichlet truncation) | — | Lanczos $p=1$ | $p=1$ | addition, not in paper |
| 8 | Dissipative step $\tau$ | no closed form; only $\mathcal E_\tau=e^{\tau\mathcal L_a}+O(\tau^2)$, Eq. (I9). Paper's own H$_4$ numerics use $\tau=0.01$ | $0.01$ (paper's demonstrated value) | $0.1$ | $0.1$ | $10\times$ larger — see §3 |
| 9 | **Cooling steps $N_W$ (=$t_i$)** | $t\ge\frac{2L_\mu}{p_{\min}}+\frac{8}{p_{\min}}\ln\frac1\varepsilon$, Eq. (B24) | **3548** with $L_\mu=1$, $p_{\min}=1.25\times10^{-2}$, $\varepsilon=5\times10^{-3}$; **4987** with the conservative extensive window $L_\mu=10$ | 400 | 500 ($H_0$), 200 ($i>0$) | $8.9\times$ / $17.7\times$ fewer |
| 10 | Layer radius $L_\mu$ | $L_\mu\le\lceil c_E N_o/\Delta E_{\rm layer}\rceil$, Eq. (B27); theorem uses $L_\mu=O(N_o)$ | $1$ from the *measured* warm-start excess (0.626 mHa) with $\Delta E_{\rm layer}=\Delta$; $10$ under the extensive-window assumption | not set (implicit) | not set | instance is $10\times$ better than the extensive assumption |
| 11 | Drift constant $p_{\min}$ | assumed $\Theta(1)$, Eq. (19) | **$1.251\times10^{-2}$** per $W$ application (computed, see §3) | — (empirically $1.57\times10^{-2}$) | — | assumption "$p_{\min}=\Theta(1)$" is off by $\sim80\times$; it is $\propto\tau$ |
| 12 | Jump set $|\mathcal A|$, $\mu_a$ | not prescribed; qDRIFT sampling $\mu_a$, Eq. (H1) | — | $N_A=2\binom62=30$, $\mu_a=1/30$ | same | as in paper's framework |
| 13 | Hamiltonian simulation | block-encoded QSP/QSVT | $N_{U_H}$ calls per $W$ | low-rank Trotter, $R=4$, $r_1=4$ | same | different primitive; not comparable term-by-term |

Supporting numbers for row 5/6 (actual truncated-and-windowed square-wave response at this geometry):

| $k_{\max}$ | $T_{N_\omega}$ (a.u.) | $\lvert\hat f(+\Delta)\rvert$ | $\sup_{\omega\ge\Delta}\lvert\hat f\rvert$ | bound $1/(\Delta T_{N_\omega})$ |
|---|---|---|---|---|
| 50 | 23.86 | $1.45\times10^{-2}$ | $3.27\times10^{-2}$ | $1.30\times10^{-1}$ |
| 100 | 47.48 | $1.66\times10^{-2}$ | $1.82\times10^{-2}$ | $6.53\times10^{-2}$ |
| 150 | 71.11 | $8.39\times10^{-3}$ | $1.09\times10^{-2}$ | $4.36\times10^{-2}$ |
| **200** | **94.73** | $4.22\times10^{-3}$ | $6.65\times10^{-3}$ | $3.28\times10^{-2}$ |
| 300 | 141.98 | $2.68\times10^{-4}$ | $3.20\times10^{-3}$ | $2.19\times10^{-2}$ |
| **400** | **189.23** | $5.22\times10^{-4}$ | $8.96\times10^{-4}$ | $1.64\times10^{-2}$ |
| 500 | 236.48 | $4.12\times10^{-5}$ | $4.99\times10^{-5}$ | $1.31\times10^{-2}$ |

---

## 3. Where the residual 0.538 mHa at the TS actually comes from

I built the ideal filtered-Lindbladian population kernel at this geometry,
$P_{j\to k}=\tau\,\frac{1}{N_A}\sum_a\lvert\hat f(E_k-E_j)\rvert^2\lvert\langle E_k|A_a|E_j\rangle\rvert^2$,
with the *same* filter used in the run ($T=0.47248551$, $k_{\max}=200$, Lanczos $p=1$, $N_\omega=500$), coarse-grained onto the 9 distinct $N=3$ levels.

| Diagnostic | Ideal-filter model | Observed in the run |
|---|---|---|
| Downhill probability from level 1 per $W$ application ($=p_{\min}$) | $1.251\times10^{-2}$ | $1.567\times10^{-2}$ (exponential fit to $E(n)-E_\infty$, first 60 steps) |
| Steady-state energy error at $k_{\max}=200$ | $2.27\times10^{-5}$ Ha = **0.023 mHa** | plateau at **0.538 mHa**, reached by $n\approx250$ |
| Steady state at $k_{\max}=100$ / $400$ | 0.231 / 0.0004 mHa | — |

Two conclusions, both testable:

1. **The mixing rate is predicted well** (1.25e-2 vs 1.57e-2, 25% agreement). So the *number* of cooling steps needed is governed by the filtered-jump rates as the paper's Markov-chain picture assumes — the theory is doing its job here.
2. **The error floor is not.** The observed plateau is $\sim23\times$ above the ideal-filter fixed point. Since the matrix run has no Trotter error and no jump-sampling error, and filter truncation alone predicts 0.023 mHa, the 0.538 mHa floor is most plausibly the $O(\tau^2)$ finite-step error of the $W(\sqrt\tau)$ construction at $\tau=0.1$ (row 8: the paper's own numerics use $\tau=0.01$). This also explains why the floor is essentially flat across indices 14–26 (0.54–0.61 mHa) — a $\tau$-artifact is geometry-insensitive, whereas a filter-truncation artifact would track $\Delta(s)$.
   **Recommended verification before this goes in the chapter:** rerun geometry index 20 only, with $\tau\in\{0.1,0.05,0.025,0.01\}$ at fixed $k_{\max}=200$, $N_W$ scaled as $1/\tau$. If the floor falls roughly as $\tau$, the attribution is confirmed. (The existing `tau_sweep` figure is at a different geometry and does not settle this.)

**A third point worth making explicitly in the chapter:** at the TS geometry the transported warm start already has an error of **0.626 mHa**, i.e. it is *inside* chemical accuracy before any cooling is applied. The 400 dissipative steps reduce the error by only $8.8\times10^{-5}$ Ha. So at this geometry the honest statement is not "400 steps suffice where the bound demands 3548" but "**0 steps suffice**, because continuation transported a state already within tolerance; 400 steps were spent recovering $0.09$ mHa of it." That is a much stronger statement of the warm-start mechanism than a step-count ratio.

---

## 4. Inconsistencies in the current draft that this comparison exposes

**(a) The $\|H\|$ / $L$ / $T_{\rm filter}$ sentence in §"Finite-filter parameters" is wrong for this system.** The draft says:

> "We take $L=2\|H\|$ … For the TS of hydrogen-exchange Hamiltonian, $\|H\|=1.405$, giving $L=2.81$ and $T_{\rm filter}=1.118$."

Three problems: (i) $\|H\|$ at image 112 is **1.753549**, not 1.405; (ii) the code did **not** use $L=2\|H\|$ — it used $L_i=2\big(E_{\max}(H_i)-E_{\min}(H_i)\big)$, which is what the §"Matrix run" paragraph correctly states and what reproduces every logged `wmax`; (iii) consequently the actual TS values are $L=6.64908$, $T_{\rm filter}=0.472486$, not $2.81$/$1.118$. The two definitions of $L$ in the chapter need to be reconciled, and the numerical example replaced.

**(b) $\Omega$ is 1.9$\times$ oversized, which is pure wasted cost.** The paper requires only $\Omega\ge4\|H\|=7.014$ to avoid aliasing; the run used $\Omega=4D=13.298$. Since $N_\omega\propto\Omega T_{N_\omega}$, switching to $L=2\|H\|$ would give the *same* time support with $\sim47\%$ fewer Fourier modes, i.e. $\sim47\%$ fewer real-time evolution calls per $W$. That is a free, defensible cost reduction to report.

**(c) $N_\omega=500$ vs $k_{\max}=200$ needs one clarifying sentence.** The log records `terms=1001` (i.e. $2\times500+1$ coefficients) while only 401 kicks are applied. As written, a reader cannot tell that $N_\omega=500$ enters only through the Lanczos window $\mathrm{sinc}(k/(N_\omega+1))$ while the retained mode set is $|k|\le200$. Also worth stating that only odd $k$ have nonzero $c_k$ ($c_0=1/2$, $c_k=i/\pi k$ for odd $k$), so the 401 retained modes carry 201 nonzero coefficients.

**(d) The chapter's error-floor claim.** The draft attributes the larger barrier-region errors to "the most demanding region of the present continuation run". Per §3 the floor there looks $\tau$-limited, not mixing-limited. If the $\tau$ sweep confirms this, the sentence should be rewritten — it changes the message from "the barrier is hard" to "our $\tau$ was too coarse and the barrier region is where that shows".

---

## 5. Ready-to-paste LaTeX table

```latex
\begin{table*}[t]
\caption{Analytic parameter prescriptions of Ref.~\cite{watts2026rapid} evaluated at the
H$_3$ climbing-image geometry (NEB image 112, Hamiltonian index 20; STO-3G, $N_e=3$,
$n_{\rm so}=6$), compared with the values used in this work. Spectral data at this
geometry: $\|H\|=1.7535$~Ha, $D=E_{\max}-E_{\min}=3.3245$~Ha,
$\Delta=0.3223$~Ha (gap to the first distinct level of the $N_e=3$ sector; the ground
state is twofold degenerate in $S_z$), $N_o=3$. Rows marked $^{\dagger}$ evaluate the
explicit appendix inequality with the implicit $\Theta(\cdot)$ constant set to unity.}
\label{tab:param_vs_theory}
\begin{ruledtabular}
\begin{tabular}{l l r r r}
Parameter & Analytic prescription & Analytic value & Used here & Ratio \\
\hline
Period $\Omega$                & $\Omega\ge4\|H\|$ [Eq.~(G8)]                                   & $\ge7.014$   & $13.298$ & $1.90\times$ \\
Spacing $T_{\rm filter}$       & $2\pi/\Omega$                                                  & $\le0.8958$  & $0.4725$ & $0.53\times$ \\
Time support $T_{N_\omega}^{\dagger}$ & $\ge1/(\Delta\varepsilon_{\rm leak})$ [Eq.~(G9)]         & $1939$       & $94.7$   & $0.049\times$ \\
Fourier cutoff $N_\omega^{\dagger}$   & $\Theta(\|H\|\Delta^{-1}\varepsilon_{\rm leak}^{-1})$ [Eq.~(27)] & $4104$ & $200$ & $0.049\times$ \\
Leakage $\varepsilon_{\rm leak}$ & $\le1/(\Delta T_{N_\omega})$                                  & $3.3\times10^{-2}$ & $6.7\times10^{-3}$ (actual) & $0.20\times$ \\
Step size $\tau$               & no closed form; $\mathcal{E}_\tau=e^{\tau\mathcal{L}}+O(\tau^2)$ & $0.01$ (Ref.) & $0.1$   & $10\times$ \\
Layer radius $L_\mu$           & $\lceil c_EN_o/\Delta E_{\rm layer}\rceil$ [Eq.~(B27)]          & $1$ (measured) & --      & -- \\
Drift $p_{\min}$               & assumed $\Theta(1)$ [Eq.~(19)]                                  & $1.25\times10^{-2}$ & $1.57\times10^{-2}$ (fit) & -- \\
Cooling steps $N_W^{\dagger}$  & $2L_\mu p_{\min}^{-1}+8p_{\min}^{-1}\ln\varepsilon^{-1}$ [Eq.~(B24)] & $3548$ & $400$ & $0.11\times$ \\
Jump set $N_A$                 & not prescribed                                                  & --           & $30$     & -- \\
\hline
Warm-start error at $s^\ast$   & --                                                             & --           & $0.626$~mHa & -- \\
Final error at $s^\ast$        & $\le\varepsilon_{\rm chem}=1.6$~mHa                            & --           & $0.538$~mHa & -- \\
Ideal-filter fixed point       & this work, §\ref{sec:observations}                             & $0.023$~mHa  & --       & -- \\
\end{tabular}
\end{ruledtabular}
\end{table*}
```

---

## 6. What I did *not* compute, and why

* **$N_H$ (number of geometries) vs $N_H=\Theta(c_EN_oC_{\rm DK}^2/\varepsilon_E)$, Eq. (B36).** This needs $C_{\rm DK}=\int\|\partial_sH\|/\Delta\,ds$, which is **gauge-dependent** — it must be evaluated in the same chained Procrustes-aligned orbital frame the run used. Reproducing that frame from the log alone is not reliable, and $c_E$ is an unspecified constant. Since you asked for the TS geometry only, I left it out; if you want it, the clean route is to dump $\|H_{i+1}-H_i\|$ and $\Delta_i$ directly from the run script rather than rebuilding.
* **A faithful re-implementation of $W(\sqrt\tau)$.** Eq. (I6) writes the kick angle as $\sqrt\tau$ while the chapter's §"jumps" text writes $\theta=\tau|c_k|/2$, and Eq. (I8) applies the forward and backward Strang halves at what appears to be full angle. I could not reproduce $\mathcal E_\tau=\mathbb 1+\tau\mathcal L+O(\tau^2)$ under either reading, so I did not use a rebuilt $W$ for any number above — every quantity in §3 comes from the ideal filtered-Lindbladian rates, which are unambiguous. **This notation inconsistency between Eq. (I6) and the $\theta=\tau|c_k|/2$ statement should be resolved in the text**; as written the circuit angle is ambiguous by a factor $\sqrt\tau/\tau$.
