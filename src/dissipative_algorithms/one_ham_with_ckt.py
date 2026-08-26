#!/usr/bin/env python3
"""
Circuit-construction core for the dissipative W(sqrt(tau)) workflow.

Stripped of all validation machinery. Removed relative to the original module:
dense-matrix construction and eigvalsh in build_molecule, the FCI call, the
density-matrix simulator loop, all explicit-matrix W construction
(A_matrix / build_W_matrix / run_W_chain_averaged_matrix), all fixed-sector
reference utilities (basis_fixed_N, basis_alpha_beta, lowest_spin_state_energy,
ground_energy_in_subspace, particle_number_distribution,
alpha_beta_distribution), hf_density_from_occupations, partial_trace_first_qubit,
pauli_sum_matrix, fast_qubit_op_to_paulisum, the unused A_ckt template,
WChainDiagnostics, parameter_sweep_subfigures, SquareWaveFilter.plot, and all
matplotlib/figure-saving.

Behavioural changes, both opt-out:
  * build_molecule(compute_dense=False) by default. The dense matrix and its
    eigendecomposition were O(8^n) time / O(16^n) memory and produced only the
    reference energy, which circuit construction never uses. The filter scale
    (E_max - E_min) now comes from two extremal Lanczos eigenvalues instead.
  * build_W_circuit accepts precomputed U_H_plus / U_H_minus, so the low-rank
    Trotter propagator is built once per geometry rather than once per W.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import cirq

from openfermion import (
    MolecularData,
    InteractionOperator,
    get_fermion_operator,
    get_sparse_operator,
)
from openfermion.transforms import jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.circuits import trotter
from openfermionpyscf import run_pyscf

from pyscf import gto, scf, mcscf, ao2mo

from .hamiltonians import HamiltonianData, HamiltonianBuilder

# =============================================================================
# Orbital gauge alignment
# =============================================================================

def procrustes_align_mo(mol_prev, C_prev, mo_occ_prev, mol_new, C_new, mo_occ_new):
    """
    Orbital-gauge alignment by blockwise orthogonal Procrustes (Proposition 1 /
    Appendix F), generalized to ROHF by aligning the doubly occupied, singly
    occupied, and virtual blocks separately (rotations within these blocks leave
    the mean-field state invariant).

    Solves  Q_b = argmax_Q Re Tr(Q^dagger M_b),  M_b = C_prev[:,b]^T S_x C_new[:,b],
    with S_x the AO cross-overlap between the two geometries, via SVD
    M_b = U S V^dagger, Q_b = U V^dagger.

    NOTE (verified for H3/STO-3G): in a minimal basis each block is 1x1, so this
    reduces to sign fixing and does NOT remove transport loss from large geometry
    steps; refine the path instead (max_geom_step).
    """
    S_x = gto.intor_cross("int1e_ovlp", mol_prev, mol_new)
    C = np.array(C_new, copy=True)
    for lo, hi in ((1.5, 2.5), (0.5, 1.5), (-0.5, 0.5)):
        bp = np.where((np.asarray(mo_occ_prev) > lo) & (np.asarray(mo_occ_prev) <= hi))[0]
        bn = np.where((np.asarray(mo_occ_new) > lo) & (np.asarray(mo_occ_new) <= hi))[0]
        if len(bn) == 0 or len(bp) != len(bn):
            continue
        M = C[:, bn].T @ S_x.T @ C_prev[:, bp]   # = (C_prev^T S_x C_new)^T
        U, _, Vt = np.linalg.svd(M.T)
        C[:, bn] = C[:, bn] @ (Vt.T @ U.T)
    return C


def hamiltonian_from_mo_coeff(mol, C):
    """
    Full-space molecular Hamiltonian in an EXPLICIT MO basis C (used by the
    Procrustes-aligned path). Verified to reproduce the openfermionpyscf
    pipeline Hamiltonian to ~1e-14 for canonical orbitals.
    """
    h1 = C.T @ scf.hf.get_hcore(mol) @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), C.shape[1])
    # OpenFermion integral ordering (same convention as openfermionpyscf):
    eri_of = np.asarray(eri.transpose(0, 2, 3, 1), order="C")
    one_body_spin, two_body_spin = spinorb_from_spatial(h1, eri_of)
    return InteractionOperator(float(mol.energy_nuc()), one_body_spin, 0.5 * two_body_spin)


# =============================================================================
# Spectral width without a dense matrix
# =============================================================================

def sparse_spectral_width(qubit_ham, n_system, tol=1e-4, maxiter=None):
    """
    Return E_max - E_min from two extremal Lanczos eigenvalues of the sparse
    Hamiltonian. No dense matrix is ever formed.

    The filter only needs this to set wmax = factor * width, so a loose
    tolerance is fine; tol=1e-4 keeps the iteration count small.
    """
    from scipy.sparse.linalg import eigsh, ArpackNoConvergence

    H_sparse = get_sparse_operator(qubit_ham, n_qubits=n_system)

    def extremal(which):
        try:
            return float(eigsh(H_sparse, k=1, which=which, tol=tol,
                               maxiter=maxiter, return_eigenvectors=False)[0])
        except ArpackNoConvergence as exc:
            if len(exc.eigenvalues):
                return float(exc.eigenvalues[0])
            raise

    lo = extremal("SA")
    hi = extremal("LA")
    return hi - lo



# =============================================================================
# Square-wave filter
# =============================================================================

class SquareWaveFilter:
    """
    Periodic square-wave cooling filter for the dissipative W(sqrt(tau)) block.

        fhat(omega) = sum_k c_k exp(i k T omega),      T = pi / L
        K_Nomega    = sum_k c_k A(-kT),
        A(-kT)      = exp(i H kT) A exp(-i H kT).

    Coefficients set the FSim/iSWAP kick angles and the ancilla phase:
        theta_k = sqrt(tau) * |c_k| / 2,     phi_k = arg(c_k).

    Convention:
        c_0 = A/2,
        c_k = sign * i A / (pi k)  for odd nonzero k,
        c_k = 0                    for even nonzero k.
    """

    def __init__(self):
        self.data = None

    def get_coefficients(self, L, K, tau, A=1.0, window=None, p=1,
                         drop_zeros=True, eps=1e-15, sign=+1):
        """
        Outputs used downstream:
            filter_terms : list(zip(ks, cvals)) -- consumed by build_W_circuit
            T_filter     : T = pi / L
            omega_filter : historical name; actually t_k = k * T_filter
            theta_filter, phi_filter : gate angles

        drop_zeros=False is REQUIRED for the paper circuit: build_W_circuit
        inserts exactly one exp(+iHT) per listed term, so dropping the even-k
        zero terms collapses every Delta-k=2 gap to a single T and the kicks
        land at the wrong times. Zero-coefficient kicks compile to identity but
        preserve the uniform time grid.
        """
        ks = np.arange(-K, K + 1, dtype=int)
        T_filter = np.pi / L
        omega = ks * T_filter                      # really t_k = k * T_filter

        cvals = np.zeros_like(omega, dtype=complex)
        cvals[ks == 0] = 0.5 * A
        odd_nonzero = (ks != 0) & (ks % 2 != 0)
        cvals[odd_nonzero] = sign * 1j * A / (np.pi * ks[odd_nonzero])

        if window:
            Kmax = (len(ks) - 1) // 2
            if window.lower().startswith("lanczos"):
                w = np.sinc(ks / (Kmax + 1)) ** p
            elif window.lower().startswith("fejer"):
                w = 1.0 - np.abs(ks) / (Kmax + 1)
                w[w < 0] = 0.0
            else:
                w = np.ones_like(ks, dtype=float)
            cvals = w * cvals
            cvals[ks == 0] = 0.5 * A           # keep c0 exact after windowing

        if drop_zeros:
            keep = np.abs(cvals) > eps
            ks, omega, cvals = ks[keep], omega[keep], cvals[keep]

        theta_filter = np.sqrt(tau) * np.abs(cvals) / 2.0
        phi_filter = np.array(
            [0.0 if k == 0 else np.angle(c) for k, c in zip(ks, cvals)], dtype=float
        )

        self.data = {
            "L": L, "K": K, "tau": tau, "A": A, "sign": sign,
            "T_filter": T_filter,
            "omega_filter": omega,
            "cvals": cvals,
            "ks_filter": ks,
            "filter_terms": list(zip(ks, cvals)),
            "theta_filter": theta_filter,
            "phi_filter": phi_filter,
            "filter_gate_params": list(zip(ks, cvals, theta_filter, phi_filter)),
        }
        return self.data


# =============================================================================
# Jump-pair normalization
# =============================================================================

def _normalize_jump_pair(pair):
    """
    Normalize a jump-pair spec to (legs, jump_type).

        (i, j)                          -> ([(i, j)], "Ax")
        (i, j, "Ax"/"Ay")               -> ([(i, j)], jump_type)
        (((i1,j1),(i2,j2),...), "Ax")   -> (list of legs, jump_type)

    Multi-leg pairs implement SPIN-ADAPTED jumps: the legs' JW hop operators
    commute (verified numerically), so exp(-i theta sigma x (A1+A2)) equals the
    product of the individual kicks applied back-to-back with the same
    (theta_k, phi_k). Used for singlet operators E_pq acting on the alpha and
    beta strands together, which conserve total S^2.
    """
    if len(pair) == 2 and all(isinstance(x, (int, np.integer)) for x in pair):
        return [(int(pair[0]), int(pair[1]))], "Ax"
    if len(pair) == 3 and all(isinstance(x, (int, np.integer)) for x in pair[:2]):
        return [(int(pair[0]), int(pair[1]))], str(pair[2])
    if len(pair) == 2:
        legs, jump_type = pair
        return [(int(a), int(b)) for a, b in legs], str(jump_type)
    raise ValueError(f"Unrecognized jump pair format: {pair!r}")


# =============================================================================
# JW-string-aware local kick
# =============================================================================
#
# For non-neighbouring JW modes the bare jump carries a parity string:
#     Ax_ij = 1/2 (X_i Z_{i+1}...Z_{j-1} X_j + Y_i Z_{i+1}...Z_{j-1} Y_j)
#     Ay_ij = 1/2 (Y_i Z_{i+1}...Z_{j-1} X_j - X_i Z_{i+1}...Z_{j-1} Y_j)
# so the string belongs in A_on_pair.

def _jw_string_system_qubits(system_qubits, i, j):
    """Return system qubits strictly between fermionic modes i and j."""
    lo, hi = sorted((i, j))
    return [system_qubits[m] for m in range(lo + 1, hi)]


def _pauli_string_rotation(pauli_ops, angle):
    """
    Return a Cirq operation implementing exp(-i * angle * P), P the product of
    pauli_ops. Cirq's PauliString power convention gives this up to a global
    phase, irrelevant for the induced channel.
    """
    ps = cirq.PauliString()
    for op in pauli_ops:
        ps *= op
    return ps ** (2.0 * angle / np.pi)


def _append_jw_string_hopping_evolution(circuit, system_qubits, i, j, theta, jump_type="Ax"):
    """
    Append exp(-i theta A_ij) for the JW-mapped bilinear jump. When i and j are
    neighbouring modes the string is empty and this reduces to the usual
    iSWAP-type evolution, up to a global phase.
    """
    jump_type = jump_type.lower()
    if jump_type not in ["ax", "ay"]:
        raise ValueError("jump_type must be either 'Ax' or 'Ay'.")

    qi = system_qubits[i]
    qj = system_qubits[j]
    z_ops = [cirq.Z(q) for q in _jw_string_system_qubits(system_qubits, i, j)]

    if jump_type == "ax":
        # exp[-i theta/2 X_i S X_j] exp[-i theta/2 Y_i S Y_j]; the two commute.
        circuit.append(_pauli_string_rotation([cirq.X(qi), *z_ops, cirq.X(qj)], +theta / 2.0))
        circuit.append(_pauli_string_rotation([cirq.Y(qi), *z_ops, cirq.Y(qj)], +theta / 2.0))
    else:  # "ay":  Ay = 1/2 (Y_i S X_j - X_i S Y_j)
        circuit.append(_pauli_string_rotation([cirq.Y(qi), *z_ops, cirq.X(qj)], +theta / 2.0))
        circuit.append(_pauli_string_rotation([cirq.X(qi), *z_ops, cirq.Y(qj)], -theta / 2.0))


def A_on_pair(ancilla, system_qubits, i, j, theta, phi, jump_type="Ax"):
    """
    Appendix-H local kick on the system pair (i, j):

        Rz(phi) H CZ(a,i) exp[-i theta A_ij] CZ(a,i) H Rz(-phi),

    with A_ij including the JW parity string Z_{min+1}...Z_{max-1}.
    """
    jump_type = jump_type.lower()
    if jump_type not in ["ax", "ay"]:
        raise ValueError("jump_type must be either 'Ax' or 'Ay'.")
    if i == j:
        raise ValueError("A_on_pair requires two distinct system modes i != j.")

    c = cirq.Circuit()
    c.append(cirq.rz(phi)(ancilla))
    c.append(cirq.H(ancilla))
    c.append(cirq.CZ(ancilla, system_qubits[i]))

    _append_jw_string_hopping_evolution(
        c, system_qubits=system_qubits, i=i, j=j, theta=theta, jump_type=jump_type
    )

    c.append(cirq.CZ(ancilla, system_qubits[i]))
    c.append(cirq.H(ancilla))
    c.append(cirq.rz(-phi)(ancilla))
    return c


# =============================================================================
# W(sqrt(tau)) circuit
# =============================================================================

def build_U_H_trotter(system_qubits, H_int, time_s, trotter_steps_H=1, final_rank=4):
    """
    Trotter circuit for exp(-i H time_s).

        positive time_s -> exp(-i H time_s)
        negative time_s -> exp(+i H |time_s|)

    So exp(+i H T_filter) is build_U_H_trotter(..., time_s=-T_filter).
    """
    custom_algorithm = trotter.LowRankTrotterAlgorithm(final_rank=final_rank)
    dt = time_s / trotter_steps_H

    circ = cirq.Circuit()
    for _ in range(trotter_steps_H):
        circ += cirq.Circuit(
            trotter.simulate_trotter(
                system_qubits, H_int, time=dt,
                omit_final_swaps=True, algorithm=custom_algorithm,
            ),
            strategy=cirq.InsertStrategy.EARLIEST,
        )
    bad = (cirq.PhasedISwapPowGate, cirq.ISwapPowGate, cirq.PauliStringPhasorGate)
    circ = cirq.expand_composite(
        circ, no_decomp=lambda op: not isinstance(op.gate, bad)
    ) 
    return circ


def build_W_circuit(
    ancilla, system_qubits, H_int, tau_s,
    filter_terms, pairs, weights,
    reps=None, trotter_steps_H=1, seed=None, T_filter=None,
    U_H_plus=None, U_H_minus=None, final_rank=4, chosen_pair=None,
):
    """
    Build W(sqrt(tau)) from complex square-wave filter coefficients.

    tau_s     : dissipative timestep; sets theta_k = sqrt(tau_s)|c_k|/2.
    T_filter  : Hamiltonian evolution spacing T in A(-kT) = e^{+iHkT} A e^{-iHkT}.
    reps      : if given, keep only modes with |k| <= reps.
    U_H_plus / U_H_minus :
        Optional precomputed propagators. Building them costs a low-rank ERI
        decomposition, so pass them in to avoid repeating that once per W.
        Omit them and they are built here (original behaviour).
    chosen_pair:
        Optional explicit jump pair. If provided, no RNG sampling is performed.
    """
    if T_filter is None:
        T_filter = tau_s
        print("Warning: T_filter not provided; using T_filter=tau_s for backward "
              "compatibility. For the paper circuit, pass T_filter explicitly.",
              flush=True)

    rng = np.random.default_rng(seed)
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    active_terms = (
        [(k, ck) for (k, ck) in filter_terms if abs(k) <= reps]
        if reps is not None else list(filter_terms)
    )

    # U_H_plus  = exp(+i H T_filter) = exp(-i H (-T_filter))
    # U_H_minus = exp(-i H T_filter)
    if U_H_plus is None:
        U_H_plus = build_U_H_trotter(
            system_qubits, H_int, time_s=-T_filter,
            trotter_steps_H=trotter_steps_H, final_rank=final_rank,
        )
    if U_H_minus is None:
        U_H_minus = cirq.Circuit(cirq.inverse(U_H_plus))

    circ = cirq.Circuit()

    def append_U_H_plus():
        for op in U_H_plus.all_operations():
            circ.append(op)

    def append_U_H_minus():
        for op in U_H_minus.all_operations():
            circ.append(op)

    # One jump operator is sampled for the whole W block unless the caller has
    # already sampled it. The explicit path lets the chain builder deduplicate
    # repeated W circuits.
    if chosen_pair is None:
        chosen_pair = pairs[rng.choice(len(pairs), p=w)]
    legs, jump_type = _normalize_jump_pair(chosen_pair)

    # Forward half:  product over k of  A_k(sqrt(tau)) exp(+i H T)
    for k, ck in active_terms:
        theta_k = np.sqrt(tau_s) * abs(ck) / 2.0
        phi_k = 0.0 if k == 0 else np.angle(ck)
        if theta_k != 0.0:
            for i, j in legs:
                circ += A_on_pair(ancilla, system_qubits, i, j, theta_k, phi_k,
                                  jump_type=jump_type)
        append_U_H_plus()

    # Backward half: reverse product over k of  exp(-i H T) A_k(sqrt(tau))
    for k, ck in reversed(active_terms):
        theta_k = np.sqrt(tau_s) * abs(ck) / 2.0
        phi_k = 0.0 if k == 0 else np.angle(ck)
        append_U_H_minus()
        if theta_k != 0.0:
            for i, j in reversed(legs):
                circ += A_on_pair(ancilla, system_qubits, i, j, theta_k, phi_k,
                                  jump_type=jump_type)

    return circ


# =============================================================================
# W-chain construction (no simulation)
# =============================================================================

def build_W_circuits_only(
    system_qubits: list[int],
    H_int: HamiltonianData,
    anc: int,
    tau: float,
    filter_terms: tuple[tuple[int , complex]],
    pairs: tuple[tuple[int, int]] | tuple[tuple[int, int, str]],
    weights: tuple[float],
    N_applications: int,
    reps_per_W: int | None = None,
    trotter_steps_H: int = 1,
    rng_seed: int = 123,
    T_filter: float = None,
    final_rank: int = 4,
    keep_circuits: str = "unique",
    ):
    """
    Construct the W(sqrt(tau)) circuits for one geometry. No density matrix, no
    simulator, no eigendecomposition, no energies.

    Returns (W_list, resources), where resources is one dict per application:
        {"index", "seed", "pair", "moments", "operations", "qubits", "seconds"}

    keep_circuits:
        "unique" -> return one circuit per distinct sampled jump pair (default)
        "all"    -> return the W sequence, reusing references for repeated pairs
        "first"  -> return only W_0; resources still cover all N_applications
        "none"   -> return []; resources still cover all N_applications

    Memory note: each W inlines the Trotter propagator 2*len(active_terms)
    times. With filter_K=200 and drop_zeros=False that is 802 copies per W.
    This builder samples the whole W sequence first, builds each distinct jump
    pair once, and then reuses the resource row for repeated applications.
    """
    if T_filter is None:
        raise ValueError(
            "T_filter must be provided. The paper circuit distinguishes the "
            "dissipative tau from the filter time T."
        )
    if keep_circuits not in ("unique", "all", "first", "none"):
        raise ValueError("keep_circuits must be 'unique', 'all', 'first', or 'none'.")

    n_qubits = 1 + len(system_qubits)

    t0 = time.perf_counter()
    U_H_plus = build_U_H_trotter(
        system_qubits, H_int, time_s=-T_filter,
        trotter_steps_H=trotter_steps_H, final_rank=final_rank,
    )
    U_H_minus = cirq.Circuit(cirq.inverse(U_H_plus))
    print(f"[trotter] propagator built once in {time.perf_counter()-t0:.1f} s "
          f"({len(U_H_plus)} moments); reused for all {N_applications} W blocks",
          flush=True)

    n_active = (
        len([1 for k, _ in filter_terms if abs(k) <= reps_per_W])
        if reps_per_W is not None else len(filter_terms)
    )
    print(f"[filter] {n_active} active terms -> {2*n_active} propagator "
          f"insertions per W", flush=True)

    w_norm = np.asarray(weights, dtype=float)
    w_norm = w_norm / w_norm.sum()

    sampled: List[dict] = []
    unique_pairs = {}
    for t in range(N_applications):
        seed = rng_seed + t
        chosen = pairs[np.random.default_rng(seed).choice(len(pairs), p=w_norm)]
        key = repr(chosen)
        sampled.append({"index": t, "seed": seed, "pair": chosen, "unique_key": key})
        if key not in unique_pairs:
            unique_pairs[key] = chosen

    print(f"[W sampling] {N_applications} applications -> "
          f"{len(unique_pairs)} unique sampled jump-pair circuits", flush=True)

    unique_circuits: dict = {}
    unique_resources: dict = {}

    for n_done, (key, chosen) in enumerate(unique_pairs.items(), start=1):
        ts = time.perf_counter()
        W = build_W_circuit(
            ancilla=anc, system_qubits=system_qubits, H_int=H_int, tau_s=tau,
            filter_terms=filter_terms, pairs=pairs, weights=weights,
            reps=reps_per_W, trotter_steps_H=trotter_steps_H,
            seed=None, T_filter=T_filter, chosen_pair=chosen,
            U_H_plus=U_H_plus, U_H_minus=U_H_minus, final_rank=final_rank,
        )

        n_moments = len(W)
        n_ops = sum(1 for _ in W.all_operations())
        dt = time.perf_counter() - ts

        unique_resources[key] = {
            "unique_key": key, "pair": chosen,
            "moments": n_moments, "operations": n_ops,
            "qubits": n_qubits, "build_seconds": dt,
        }
        if keep_circuits in ("unique", "all", "first"):
            unique_circuits[key] = W
        else:
            del W

        print(f"[unique W {n_done}/{len(unique_pairs)}] {n_moments} moments, "
              f"{n_ops} ops, {dt:.1f} s", flush=True)

    resources: List[dict] = []
    for item in sampled:
        ur = unique_resources[item["unique_key"]]
        resources.append({
            "index": item["index"], "seed": item["seed"], "pair": item["pair"],
            "unique_key": item["unique_key"],
            "moments": ur["moments"], "operations": ur["operations"],
            "qubits": ur["qubits"], "seconds": 0.0,
        })

    if keep_circuits == "unique":
        W_list = list(unique_circuits.values())
    elif keep_circuits == "all":
        W_list = [unique_circuits[item["unique_key"]] for item in sampled]
    elif keep_circuits == "first":
        W_list = [unique_circuits[sampled[0]["unique_key"]]] if sampled else []
    else:
        W_list = []

    total_moments = sum(r["moments"] for r in resources)
    total_ops = sum(r["operations"] for r in resources)
    width = resources[0]["qubits"] if resources else n_qubits
    print(f"[geometry total] width={width} "
          f"depth={total_moments} ops={total_ops}", flush=True)

    return W_list, resources
