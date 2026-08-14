#!/usr/bin/env python3
"""
Energy validation for the dissipative W(sqrt(tau)) circuit workflow.

WHY THIS EXISTS
---------------
`one_ham_with_ckt.build_W_circuits_only` constructs circuits and counts gates.
It never simulates anything, so a completed run proves only that circuits of a
certain size were built -- not that the dissipative dynamics cools the state.
This module closes that gap: it evolves a density matrix through the SAME W
blocks and reports Tr[H rho] against the exact ground energy in the fixed-N_e
sector.

It is additive. Nothing in one_ham_with_ckt.py is modified.

HOW IT AVOIDS SIMULATING 2e6 OPERATIONS
---------------------------------------
Each W block inlines the Trotter propagator 2*len(active_terms) times, giving
~2e6 operations. Simulating that op-by-op would take hours per block. Instead we
note that W acts on only (1 + n_system) qubits, so its unitary is a small dense
matrix, and it is a product of blocks that repeat:

    W = [backward half] [forward half]

    forward  half = product over k of   U_plus  * kick_k
    backward half = product over k of   kick_k' * U_minus

`U_plus` is built once per geometry and each distinct kick is built once, so the
whole W matrix costs ~1200 dense matmuls of size 2^(1+n_system). For n_system=8
that is seconds. We then take the two Kraus operators

    K_m = <m|_anc W |0>_anc ,      rho -> K_0 rho K_0^dag + K_1 rho K_1^dag

which is exactly "apply W to |0><0|_anc (x) rho, then trace out the ancilla".

FIDELITY TO THE CIRCUIT
-----------------------
`build_W_kraus` mirrors the loop structure of `one_ham_with_ckt.build_W_circuit`
term by term and calls the same `A_on_pair` and `build_U_H_trotter`. If you
change build_W_circuit, change this too. `verify_against_circuit` checks the two
agree on a small instance and should be run after any such change.

The sampled jump sequence is NOT re-derived here: it is read from the `resources`
rows returned by build_W_circuits_only, so the simulated chain uses exactly the
jump operators the counted circuits used.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import one_ham_with_ckt as linlin

try:
    import cirq
except ImportError as exc:  # pragma: no cover
    raise ImportError("linlin_energy_check requires cirq") from exc

from openfermion import get_sparse_operator


_I2 = np.eye(2, dtype=complex)


# =============================================================================
# Fixed-sector reference machinery (removed from one_ham_with_ckt; restored here)
# =============================================================================

def basis_fixed_N(n_system: int, N: int) -> np.ndarray:
    """Computational-basis indices with exactly N occupied spin orbitals.

    Convention matches OpenFermion's get_sparse_operator: spin orbital i is bit
    2^(n_system - 1 - i), so a plain popcount of the index gives the particle
    number regardless of ordering.
    """
    n_system = int(n_system)
    return np.array(
        [i for i in range(2 ** n_system) if int(i).bit_count() == int(N)],
        dtype=int,
    )


def basis_alpha_beta(n_system: int, Na: int, Nb: int) -> np.ndarray:
    """Indices with Na alpha (even spin orbitals) and Nb beta (odd) electrons."""
    n_system = int(n_system)
    out = []
    for idx in range(2 ** n_system):
        na = nb = 0
        for i in range(n_system):
            if idx >> (n_system - 1 - i) & 1:
                if i % 2 == 0:
                    na += 1
                else:
                    nb += 1
        if na == Na and nb == Nb:
            out.append(idx)
    return np.array(out, dtype=int)


def ground_energy_in_subspace(H, basis: np.ndarray) -> Tuple[float, np.ndarray]:
    """Lowest eigenvalue of H restricted to the given basis indices.

    Accepts a dense array or any scipy sparse matrix. Only the restricted block
    is densified, so this stays cheap even when the full matrix is large.
    """
    basis = np.asarray(basis, dtype=int)
    if basis.size == 0:
        raise ValueError("Empty subspace basis.")
    if hasattr(H, "tocsr"):
        sub = np.asarray(H.tocsr()[basis][:, basis].todense())
    else:
        sub = np.asarray(H)[np.ix_(basis, basis)]
    ev = np.linalg.eigvalsh(sub)
    return float(ev[0]), ev


def hf_density_msb(n_system: int, occ_spin_orbs: Sequence[int]) -> np.ndarray:
    """Pure computational-basis density matrix for the given occupations."""
    n_system = int(n_system)
    dim = 2 ** n_system
    idx = 0
    for i in occ_spin_orbs:
        i = int(i)
        if not (0 <= i < n_system):
            raise ValueError(f"Occupied spin orbital {i} outside [0,{n_system-1}].")
        idx |= 1 << (n_system - 1 - i)
    ket = np.zeros((dim, 1), dtype=complex)
    ket[idx, 0] = 1.0
    return ket @ ket.conj().T


def system_hamiltonian_sparse(qubit_ham, n_system: int):
    """Sparse system Hamiltonian in the same ordering as everything above."""
    return get_sparse_operator(qubit_ham, n_qubits=int(n_system))


def particle_number_expectation(rho: np.ndarray, n_system: int) -> float:
    diag = np.real(np.diag(np.asarray(rho, dtype=complex)))
    return float(sum(int(i).bit_count() * float(p) for i, p in enumerate(diag)))


# =============================================================================
# W as a pair of Kraus operators
# =============================================================================

def _sys_unitary(circ: "cirq.Circuit", system_qubits) -> np.ndarray:
    sq = list(system_qubits)
    return circ.unitary(qubit_order=sq, qubits_that_should_be_present=sq)


def _full_unitary(circ: "cirq.Circuit", ancilla, system_qubits) -> np.ndarray:
    order = [ancilla] + list(system_qubits)
    return circ.unitary(qubit_order=order, qubits_that_should_be_present=order)


def build_propagator_unitaries(
    system_qubits, H_int, T_filter: float,
    trotter_steps_H: int = 1, final_rank: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full-register unitaries for exp(+i H T_filter) and exp(-i H T_filter).

    Built from the identical `build_U_H_trotter` call used by build_W_circuit,
    including the same `final_rank`, so the low-rank truncation error present in
    the circuit is present here too. That is deliberate: this validates the
    circuit as built, not an idealised version of it.
    """
    U_plus_circ = linlin.build_U_H_trotter(
        system_qubits, H_int, time_s=-T_filter,
        trotter_steps_H=trotter_steps_H, final_rank=final_rank,
    )
    U_plus_sys = _sys_unitary(U_plus_circ, system_qubits)
    U_plus = np.kron(_I2, U_plus_sys)          # ancilla is the most significant qubit
    U_minus = U_plus.conj().T
    return U_plus, U_minus


def build_W_kraus(
    ancilla, system_qubits, tau_s: float, filter_terms, chosen_pair,
    U_plus: np.ndarray, U_minus: np.ndarray,
    reps: Optional[int] = None, return_unitary: bool = False,
):
    """Kraus operators (K0, K1) of one W(sqrt(tau)) block for a fixed jump pair.

    Mirrors one_ham_with_ckt.build_W_circuit exactly: same active-term filter,
    same theta_k = sqrt(tau)|c_k|/2 and phi_k = arg(c_k), same forward loop
    (kick then exp(+iHT)) and backward loop (exp(-iHT) then kick with the legs
    reversed), same skipping of theta_k == 0.
    """
    active_terms = (
        [(k, ck) for (k, ck) in filter_terms if abs(k) <= reps]
        if reps is not None else list(filter_terms)
    )
    legs, jump_type = linlin._normalize_jump_pair(chosen_pair)

    dim_sys = 2 ** len(list(system_qubits))
    dim = 2 * dim_sys

    # NOT cached. Every mode k has its own (theta_k, phi_k) and forward/backward
    # differ in leg order, so no two of the 2*(2K+1) kicks in a W block share a
    # key -- a cache would hit 0% of the time while holding one dim x dim matrix
    # each. At 9 qubits and K=200 that is ~20 GB of pure waste. Building the
    # unitary of a 16-operation circuit is milliseconds; just rebuild it.
    def kick(theta: float, phi: float, reverse: bool) -> np.ndarray:
        c = cirq.Circuit()
        for i, j in (reversed(legs) if reverse else legs):
            c += linlin.A_on_pair(
                ancilla, system_qubits, i, j, theta, phi, jump_type=jump_type
            )
        return _full_unitary(c, ancilla, system_qubits)

    U = np.eye(dim, dtype=complex)

    # Forward half: for each k, apply the kick then exp(+i H T).
    for k, ck in active_terms:
        theta = np.sqrt(tau_s) * abs(ck) / 2.0
        phi = 0.0 if k == 0 else float(np.angle(ck))
        if theta != 0.0:
            U = kick(theta, phi, reverse=False) @ U
        U = U_plus @ U

    # Backward half: reversed k, apply exp(-i H T) then the kick with legs reversed.
    for k, ck in reversed(active_terms):
        theta = np.sqrt(tau_s) * abs(ck) / 2.0
        phi = 0.0 if k == 0 else float(np.angle(ck))
        U = U_minus @ U
        if theta != 0.0:
            U = kick(theta, phi, reverse=True) @ U

    # rho -> Tr_anc[ W (|0><0| (x) rho) W^dag ] = sum_m K_m rho K_m^dag,
    # with the ancilla the most significant qubit and prepared in |0>.
    K0 = U[:dim_sys, :dim_sys].copy()
    K1 = U[dim_sys:, :dim_sys].copy()
    if return_unitary:
        return K0, K1, U
    return K0, K1


def kraus_completeness_error(K0: np.ndarray, K1: np.ndarray) -> float:
    """||K0^dag K0 + K1^dag K1 - I||_inf. Should be ~1e-13. Catches a broken W."""
    M = K0.conj().T @ K0 + K1.conj().T @ K1
    return float(np.max(np.abs(M - np.eye(M.shape[0]))))


# =============================================================================
# The chain
# =============================================================================

def run_W_chain_energies(
    ancilla, system_qubits, H_int, qubit_ham, n_system: int,
    tau: float, filter_terms, T_filter: float,
    pair_sequence: Sequence[Any],
    rho_init: Optional[np.ndarray] = None,
    occ_spin_orbs: Optional[Sequence[int]] = None,
    reps_per_W: Optional[int] = None,
    trotter_steps_H: int = 1, final_rank: int = 4,
    H_sparse=None, verbose: bool = True,
) -> Dict[str, Any]:
    """Evolve a density matrix through the sampled W sequence, tracking energy.

    `pair_sequence` must be the jump pair actually used at each application, in
    order -- take it from the `resources` rows of build_W_circuits_only so the
    simulated chain matches the counted circuits exactly.

    Returns the initial and FINAL energies only, the final density matrix (to be
    carried to the next geometry, which is what makes this a *continuation*),
    and diagnostics that fail loudly if the construction is wrong.
    """
    n_system = int(n_system)
    dim_sys = 2 ** n_system

    if H_sparse is None:
        H_sparse = system_hamiltonian_sparse(qubit_ham, n_system)

    if rho_init is None:
        if occ_spin_orbs is None:
            raise ValueError("Provide rho_init or occ_spin_orbs.")
        rho = hf_density_msb(n_system, occ_spin_orbs)
    else:
        rho = np.array(rho_init, dtype=complex, copy=True)

    def energy_of(r: np.ndarray) -> float:
        return float(np.real(np.trace(H_sparse @ r)))

    U_plus, U_minus = build_propagator_unitaries(
        system_qubits, H_int, T_filter,
        trotter_steps_H=trotter_steps_H, final_rank=final_rank,
    )

    # One Kraus pair per DISTINCT jump pair, reused across applications.
    kraus: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    completeness = 0.0
    unique = []
    for p in pair_sequence:
        if repr(p) not in kraus:
            unique.append(p)
            kraus[repr(p)] = None
    if verbose:
        print(f"[sim] {len(pair_sequence)} applications -> "
              f"{len(unique)} unique W channels to build", flush=True)

    for n, p in enumerate(unique, start=1):
        t0 = time.perf_counter()
        K0, K1 = build_W_kraus(
            ancilla, system_qubits, tau, filter_terms, p,
            U_plus, U_minus, reps=reps_per_W,
        )
        err = kraus_completeness_error(K0, K1)
        completeness = max(completeness, err)
        kraus[repr(p)] = (K0, K1)
        if verbose:
            print(f"[sim] unique W {n}/{len(unique)}  "
                  f"completeness={err:.2e}  {time.perf_counter()-t0:.1f} s", flush=True)

    if completeness > 1e-8:
        raise RuntimeError(
            f"Kraus completeness violated by {completeness:.2e}. The W block is "
            f"not a valid channel -- the construction is wrong, not the physics."
        )

    initial_energy = energy_of(rho)

    # The chain must run sequentially -- the ancilla is traced out between
    # applications, so W blocks cannot be composed across that boundary. Only
    # the endpoint energy is evaluated; the intermediate ones are not wanted.
    for p in pair_sequence:
        K0, K1 = kraus[repr(p)]
        rho = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

    final_energy = energy_of(rho)
    trace_drift = float(abs(np.real(np.trace(rho)) - 1.0))

    return {
        "n_applications": len(pair_sequence),
        "initial_energy": float(initial_energy),
        "final_energy": float(final_energy),
        "rho_final": rho,
        "trace_drift": trace_drift,
        "kraus_completeness_error": completeness,
        "final_particle_number": particle_number_expectation(rho, n_system),
        "n_unique_W": len(unique),
    }


def reference_energies(
    qubit_ham, n_system: int, n_electrons: Optional[int] = None,
    Na: Optional[int] = None, Nb: Optional[int] = None, H_sparse=None,
) -> Dict[str, Any]:
    """Exact ground energies in the sectors the dynamics can reach.

    The Type-II couplings conserve N_e, so the fixed-N_e value is the reference
    the dissipative evolution should converge to. If the jump set is not spin
    adapted it does NOT conserve N_alpha and N_beta separately, so the
    fixed-(Na,Nb) value is a diagnostic only.
    """
    n_system = int(n_system)
    if H_sparse is None:
        H_sparse = system_hamiltonian_sparse(qubit_ham, n_system)

    out: Dict[str, Any] = {"n_system": n_system}

    ev_full = np.linalg.eigvalsh(np.asarray(H_sparse.todense()))
    out["E_full_fock"] = float(ev_full[0])
    out["spectral_width"] = float(ev_full[-1] - ev_full[0])

    if n_electrons is not None:
        b = basis_fixed_N(n_system, n_electrons)
        e0, ev = ground_energy_in_subspace(H_sparse, b)
        out["E_fixed_N"] = e0
        out["sector_dimension"] = int(b.size)
        out["gap_fixed_N"] = float(ev[1] - ev[0]) if ev.size > 1 else float("nan")

    if Na is not None and Nb is not None:
        b = basis_alpha_beta(n_system, Na, Nb)
        if b.size:
            out["E_fixed_NaNb"] = ground_energy_in_subspace(H_sparse, b)[0]

    return out


# =============================================================================
# Compressed W circuit -- stores the repeated structure once
# =============================================================================
#
# build_W_circuit inlines the propagator by copying its operations:
#
#     for op in U_H_plus.all_operations():
#         circ.append(op)
#
# With 401 active terms that is 802 copies of a 2464-operation circuit, so one W
# holds ~2e6 operation objects and twelve of them exhaust the node.
#
# But the propagator is IDENTICAL every time. cirq can express that directly: a
# FrozenCircuit is stored once, and a CircuitOperation is a lightweight
# reference to it. So W becomes
#
#     802 references  +  402 small kicks (16 ops each)  ~=  7,200 objects
#
# instead of 1,982,560 -- and the single stored propagator is shared by every W
# at the geometry, so twelve W circuits cost ~89,000 objects rather than 24
# million. Pickle preserves that sharing, so the .pkl shrinks by the same factor.
#
# WHY THE PROPAGATORS ARE IDENTICAL, given that t_k = k * T_filter differs
# ------------------------------------------------------------------------
# The filtered jump needs exp(i H t_k) A exp(-i H t_k) for every mode k, and
# t_k = k * T_filter is a DIFFERENT time for every k. If each mode were built as
# its own block, no two would be alike and nothing could be shared.
#
# build_W_circuit does not do that. It walks k from -K to +K and inserts exactly
# ONE exp(+i H T_filter) per term, so the evolution accumulates as a ladder:
#
#     exp(i H t_k) = ( exp(i H T_filter) )^k
#
# The kick for mode k sits after k identical steps have accumulated, which is
# what puts it at time t_k. Every propagator block is therefore the same fixed
# duration T_filter, and the differing t_k comes from HOW MANY have been applied,
# not from any block being different.
#
# This is why SquareWaveFilter.get_coefficients must be called with
# drop_zeros=False, as its own docstring warns: the even-k terms have c_k = 0
# and contribute no kick, but they must still occupy a slot so the time grid
# stays uniform. Drop them and the kicks land at the wrong t_k.
#
# Evidence from the production log: 401 active terms -> 802 propagator
# insertions. Per-mode evolution times would need sum_k 2|k| = 80,400 steps,
# 100x more.
#
# The kicks are NOT shared: each carries its own theta_k, phi_k and is a
# genuinely different unitary. They are small, so inlining them costs nothing.
#
# The result is a real circuit. cirq simulators decompose CircuitOperation
# natively, cirq.unroll_circuit_op() expands it back to the flat form, and the
# unrolled circuit is operation-for-operation identical to build_W_circuit's.

def build_W_circuit_compressed(
    ancilla, system_qubits, H_int, tau_s: float, filter_terms, chosen_pair,
    T_filter: float, reps: Optional[int] = None, trotter_steps_H: int = 1,
    final_rank: int = 4,
    U_plus_frozen=None, U_minus_frozen=None,
):
    """W(sqrt(tau)) with the propagator stored once instead of 802 times.

    Mirrors one_ham_with_ckt.build_W_circuit exactly -- same active-term filter,
    same angles, same forward/backward ordering -- but references the propagator
    rather than copying it.

    Pass U_plus_frozen / U_minus_frozen (from freeze_propagators) so every W at a
    geometry shares one stored copy.
    """
    if U_plus_frozen is None or U_minus_frozen is None:
        U_plus_frozen, U_minus_frozen = freeze_propagators(
            system_qubits, H_int, T_filter,
            trotter_steps_H=trotter_steps_H, final_rank=final_rank,
        )

    active_terms = (
        [(k, ck) for (k, ck) in filter_terms if abs(k) <= reps]
        if reps is not None else list(filter_terms)
    )
    legs, jump_type = linlin._normalize_jump_pair(chosen_pair)

    ops: List[Any] = []

    def kick(theta, phi, reverse):
        out = []
        for i, j in (reversed(legs) if reverse else legs):
            out.extend(
                linlin.A_on_pair(
                    ancilla, system_qubits, i, j, theta, phi, jump_type=jump_type
                ).all_operations()
            )
        return out

    # Forward half: kick then exp(+i H T), for each k in order.
    for k, ck in active_terms:
        theta = np.sqrt(tau_s) * abs(ck) / 2.0
        phi = 0.0 if k == 0 else float(np.angle(ck))
        if theta != 0.0:
            ops.extend(kick(theta, phi, reverse=False))
        ops.append(cirq.CircuitOperation(U_plus_frozen))

    # Backward half: exp(-i H T) then kick with the legs reversed, k reversed.
    for k, ck in reversed(active_terms):
        theta = np.sqrt(tau_s) * abs(ck) / 2.0
        phi = 0.0 if k == 0 else float(np.angle(ck))
        ops.append(cirq.CircuitOperation(U_minus_frozen))
        if theta != 0.0:
            ops.extend(kick(theta, phi, reverse=True))

    return cirq.Circuit(ops)


def freeze_propagators(system_qubits, H_int, T_filter: float,
                       trotter_steps_H: int = 1, final_rank: int = 4):
    """Build exp(+-i H T_filter) once and freeze them for sharing across all W."""
    U_plus = linlin.build_U_H_trotter(
        system_qubits, H_int, time_s=-T_filter,
        trotter_steps_H=trotter_steps_H, final_rank=final_rank,
    )
    U_plus_frozen = cirq.FrozenCircuit(U_plus)
    U_minus_frozen = cirq.FrozenCircuit(cirq.inverse(U_plus))
    return U_plus_frozen, U_minus_frozen


def unroll_W(circ):
    """Expand the CircuitOperation references back to a flat circuit.

    Materialises every copy, so it costs exactly what build_W_circuit costs. Use
    it only to export QASM or to verify equivalence on a small cutoff.

    tags_to_check=None is REQUIRED: cirq.unroll_circuit_op defaults to unrolling
    only operations carrying the '<mapped_circuit_op>' tag, so without it the
    call silently returns the circuit unchanged.
    """
    return cirq.unroll_circuit_op(circ, deep=True, tags_to_check=None)


def stored_operation_count(circ) -> Dict[str, int]:
    """How many operation objects the circuit actually holds, versus how many it
    represents. The ratio is the storage saving."""
    stored = 0
    represented = 0
    seen_subcircuits = {}
    for op in circ.all_operations():
        if isinstance(op, cirq.CircuitOperation):
            sub = op.circuit
            key = id(sub)
            n_sub = sum(1 for _ in sub.all_operations())
            if key not in seen_subcircuits:
                seen_subcircuits[key] = n_sub
                stored += n_sub          # counted once, however often referenced
            stored += 1                  # the reference itself
            represented += n_sub * int(op.repetitions)
        else:
            stored += 1
            represented += 1
    return {
        "stored_objects": stored,
        "represented_operations": represented,
        "compression_ratio": (represented / stored) if stored else 1.0,
        "shared_subcircuits": len(seen_subcircuits),
    }


def verify_compressed_matches(
    ancilla, system_qubits, H_int, tau, filter_terms, chosen_pair, T_filter,
    reps=4, trotter_steps_H=1, final_rank=4,
) -> Dict[str, Any]:
    """Unroll the compressed W and compare it to build_W_circuit at a small
    mode cutoff. Run once after any change to either builder."""
    comp = build_W_circuit_compressed(
        ancilla, system_qubits, H_int, tau, filter_terms, chosen_pair, T_filter,
        reps=reps, trotter_steps_H=trotter_steps_H, final_rank=final_rank,
    )
    flat = unroll_W(comp)
    ref = linlin.build_W_circuit(
        ancilla=ancilla, system_qubits=system_qubits, H_int=H_int, tau_s=tau,
        filter_terms=filter_terms, pairs=[chosen_pair], weights=[1.0],
        reps=reps, trotter_steps_H=trotter_steps_H, seed=None,
        T_filter=T_filter, chosen_pair=chosen_pair, final_rank=final_rank,
    )
    flat_ops = list(flat.all_operations())
    ref_ops = list(ref.all_operations())
    return {
        "ops_unrolled": len(flat_ops),
        "ops_reference": len(ref_ops),
        "same_operation_sequence": flat_ops == ref_ops,
        **stored_operation_count(comp),
    }


# =============================================================================
# Analytic resource counting -- never materialises W
# =============================================================================
#
# build_W_circuits_only inlines the propagator 2*n_active times per W. At 8
# qubits that is ~2e6 cirq operations per W; at 20 qubits it is ~2e7, and
# holding even one of them exhausts memory. It is also unnecessary: W is a
# concatenation of two circuit templates repeated a known number of times, and
# cirq's default append strategy packs them without merging across the
# boundaries, so moments and operations are exactly additive.
#
#     moments = 2*(n_active * prop_moments + n_kick * kick_moments)
#     ops     = 2*(n_active * prop_ops     + n_kick * kick_ops)
#
# Verified against a CAS(4,4) run: 2*401*732 + 2*201*13 = 592290 moments and
# 2*401*2464 + 2*201*16 = 1982560 operations, both exact.
#
# n_kick < n_active because build_W_circuit skips terms with theta_k == 0, i.e.
# the even nonzero k of the square-wave filter.

def _is_non_clifford(op) -> bool:
    """True if the operation needs rotation synthesis in a Clifford+T basis.

    Uses cirq's stabilizer check where available; otherwise falls back to
    treating any gate whose exponent is not a multiple of 1/2 as a rotation.
    Errs toward counting, so the T estimate is an upper bound.
    """
    try:
        return not cirq.has_stabilizer_effect(op)
    except Exception:
        pass
    g = getattr(op, "gate", None)
    e = getattr(g, "exponent", None)
    if e is None:
        return False
    try:
        return not float(2.0 * float(e)).is_integer()
    except Exception:
        return True


def _circuit_stats(circ) -> Tuple[int, int, Dict[str, int], int]:
    """(moments, operations, gate histogram, non-Clifford rotation count)."""
    hist: Dict[str, int] = {}
    n_ops = 0
    n_rot = 0
    for op in circ.all_operations():
        n_ops += 1
        g = getattr(op, "gate", None)
        name = type(g).__name__ if g is not None else "unknown"
        hist[name] = hist.get(name, 0) + 1
        if _is_non_clifford(op):
            n_rot += 1
    return len(circ), n_ops, hist, n_rot


def t_count_from_rotations(n_rotations: int, eps_synth: float = 1e-10) -> int:
    """Ross-Selinger estimate: each arbitrary-angle rotation costs about
    3*log2(1/eps_synth) T gates. This is a synthesis estimate, not a
    transpilation -- no Clifford+T circuit is produced."""
    import math
    return int(round(n_rotations * 3.0 * math.log2(1.0 / float(eps_synth))))


def count_W_resources_analytic(
    ancilla, system_qubits, tau_s: float, filter_terms, chosen_pair,
    prop_stats: Tuple[int, int, Dict[str, int], int],
    reps: Optional[int] = None, n_qubits: Optional[int] = None,
    eps_synth: float = 1e-10, kick_stats_cache: Optional[dict] = None,
) -> Dict[str, Any]:
    """Exact resource counts for one W block, without building it.

    `prop_stats` is (_circuit_stats of the exp(+iHT) propagator), computed once
    per geometry. Kicks are measured per (pair, theta_k, phi_k): the jump pair
    sets the JW string length and hence how gates pack into moments, and the
    angles decide which gates are Clifford and hence the T cost.
    """
    active_terms = (
        [(k, ck) for (k, ck) in filter_terms if abs(k) <= reps]
        if reps is not None else list(filter_terms)
    )
    legs, jump_type = linlin._normalize_jump_pair(chosen_pair)

    n_active = len(active_terms)
    prop_m, prop_o, prop_h, prop_r = prop_stats

    # Each Fourier mode carries its OWN angles theta_k = sqrt(tau)|c_k|/2 and
    # phi_k = arg(c_k). Gate counts do not depend on them, but the T cost does:
    # for the square-wave filter c_k is purely imaginary, so phi_k = +-pi/2 and
    # rz(phi_k) is the Clifford S gate, costing zero T. Using a made-up angle
    # would count those as arbitrary rotations and inflate the T estimate.
    # So every distinct (theta_k, phi_k) is measured on its own small circuit.
    if kick_stats_cache is None:
        kick_stats_cache = {}

    def kick_stats(theta: float, phi: float):
        key = (float(theta), float(phi), tuple(map(tuple, legs)), jump_type)
        hit = kick_stats_cache.get(key)
        if hit is None:
            c = cirq.Circuit()
            for i, j in legs:
                c += linlin.A_on_pair(
                    ancilla, system_qubits, i, j, theta, phi, jump_type=jump_type
                )
            hit = _circuit_stats(c)
            kick_stats_cache[key] = hit
        return hit

    kick_m = kick_o = kick_r = 0
    kick_h: Dict[str, int] = {}
    n_kick = 0
    for k, ck in active_terms:
        theta = np.sqrt(tau_s) * abs(ck) / 2.0
        if theta == 0.0:
            continue                      # build_W_circuit skips these entirely
        n_kick += 1
        phi = 0.0 if k == 0 else float(np.angle(ck))
        m, o, h, r = kick_stats(theta, phi)
        kick_m += m
        kick_o += o
        kick_r += r
        for name, v in h.items():
            kick_h[name] = kick_h.get(name, 0) + v

    hist = {}
    for name in set(prop_h) | set(kick_h):
        hist[name] = 2 * (n_active * prop_h.get(name, 0) + kick_h.get(name, 0))

    n_rot = 2 * (n_active * prop_r + kick_r)

    return {
        "pair": chosen_pair,
        "moments": 2 * (n_active * prop_m + kick_m),
        "operations": 2 * (n_active * prop_o + kick_o),
        "rotations": n_rot,
        "t_count": t_count_from_rotations(n_rot, eps_synth),
        "t_synthesis_eps": float(eps_synth),
        "qubits": int(n_qubits if n_qubits is not None else 1 + len(list(system_qubits))),
        "gate_histogram": hist,
        "n_active_terms": n_active,
        "n_kick_terms": n_kick,
        "analytic": True,
    }


def analytic_resource_rows(
    ancilla, system_qubits, H_int, tau: float, filter_terms, T_filter: float,
    pair_sequence: Sequence[Any], reps_per_W: Optional[int] = None,
    trotter_steps_H: int = 1, final_rank: int = 4, eps_synth: float = 1e-10,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """One resource row per application, in constant memory.

    Drop-in replacement for the `resources` list returned by
    build_W_circuits_only. Builds the propagator once and one small kick circuit
    per distinct jump pair; W itself is never constructed.
    """
    t0 = time.perf_counter()
    U_plus = linlin.build_U_H_trotter(
        system_qubits, H_int, time_s=-T_filter,
        trotter_steps_H=trotter_steps_H, final_rank=final_rank,
    )
    prop_stats = _circuit_stats(U_plus)
    del U_plus
    if verbose:
        print(f"[analytic] propagator: {prop_stats[0]} moments, {prop_stats[1]} ops, "
              f"{prop_stats[3]} rotations ({time.perf_counter()-t0:.1f} s)", flush=True)

    n_qubits = 1 + len(list(system_qubits))
    cache: Dict[str, Dict[str, Any]] = {}
    kick_stats_cache: Dict[Any, Any] = {}
    rows: List[Dict[str, Any]] = []

    for t, p in enumerate(pair_sequence):
        key = repr(p)
        if key not in cache:
            cache[key] = count_W_resources_analytic(
                ancilla, system_qubits, tau, filter_terms, p,
                prop_stats, reps=reps_per_W, n_qubits=n_qubits,
                eps_synth=eps_synth, kick_stats_cache=kick_stats_cache,
            )
        r = dict(cache[key])
        r["index"] = t
        rows.append(r)

    if verbose and rows:
        print(f"[analytic] {len(rows)} applications, {len(cache)} unique pairs, "
              f"{rows[0]['moments']} moments, {rows[0]['operations']} ops, "
              f"{rows[0]['rotations']} rotations, {rows[0]['t_count']:,} T per W "
              f"(synthesis eps {eps_synth:g})", flush=True)
    return rows


def sequence_and_unique(pair_sequence: Sequence[Any]) -> Tuple[List[Any], List[int]]:
    """Split a sampled jump sequence into (unique pairs, index sequence).

    unique[i] is the i-th distinct pair in order of FIRST APPEARANCE, and
    sequence[t] is the index into `unique` used at application t. So

        unique_circuits[sequence[t]]  is the W applied at step t

    reconstructs the chain exactly. Unique circuits alone are not sufficient:
    W blocks do not commute, so total multiplicities do not determine the
    algorithm -- only the ordered sequence does.
    """
    order: Dict[str, int] = {}
    unique: List[Any] = []
    for p in pair_sequence:
        key = repr(p)
        if key not in order:
            order[key] = len(unique)
            unique.append(p)
    return unique, [order[repr(p)] for p in pair_sequence]


def build_W_circuits_compressed(
    ancilla, system_qubits, H_int, tau: float, filter_terms, T_filter: float,
    pair_sequence: Sequence[Any], reps_per_W: Optional[int] = None,
    trotter_steps_H: int = 1, final_rank: int = 4, eps_synth: float = 1e-10,
    keep_circuits: str = "unique", verbose: bool = True,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Drop-in replacement for one_ham_with_ckt.build_W_circuits_only that
    stores the repeated propagator once instead of 802 times.

    Returns (W_list, resources) with the same shapes. Resource rows come from
    the analytic counter, which reports the counts of the UNROLLED circuit --
    what you would run -- not the number of objects held in memory.
    """
    t0 = time.perf_counter()
    U_plus_frozen, U_minus_frozen = freeze_propagators(
        system_qubits, H_int, T_filter,
        trotter_steps_H=trotter_steps_H, final_rank=final_rank,
    )
    prop_stats = _circuit_stats(U_plus_frozen.unfreeze())
    if verbose:
        print(f"[compressed] propagator frozen once: {prop_stats[0]} moments, "
              f"{prop_stats[1]} ops ({time.perf_counter()-t0:.1f} s); "
              f"referenced, not copied", flush=True)

    n_qubits = 1 + len(list(system_qubits))
    kick_stats_cache: Dict[Any, Any] = {}
    circuits: Dict[str, Any] = {}
    res_cache: Dict[str, Dict[str, Any]] = {}

    # unique[i] <-> W_list[i] <-> sequence entries equal to i. This ordering is
    # the contract the replay file depends on; do not reorder W_list below.
    unique, sequence = sequence_and_unique(pair_sequence)

    for n, p in enumerate(unique, start=1):
        ts = time.perf_counter()
        W = build_W_circuit_compressed(
            ancilla, system_qubits, H_int, tau, filter_terms, p, T_filter,
            reps=reps_per_W, trotter_steps_H=trotter_steps_H,
            final_rank=final_rank,
            U_plus_frozen=U_plus_frozen, U_minus_frozen=U_minus_frozen,
        )
        res_cache[repr(p)] = count_W_resources_analytic(
            ancilla, system_qubits, tau, filter_terms, p, prop_stats,
            reps=reps_per_W, n_qubits=n_qubits, eps_synth=eps_synth,
            kick_stats_cache=kick_stats_cache,
        )
        if keep_circuits in ("unique", "all", "first"):
            circuits[repr(p)] = W
        if verbose:
            st = stored_operation_count(W)
            print(f"[compressed W {n}/{len(unique)}] holds "
                  f"{st['stored_objects']:,} objects, represents "
                  f"{st['represented_operations']:,} ops "
                  f"({st['compression_ratio']:.0f}x)  "
                  f"{time.perf_counter()-ts:.1f} s", flush=True)

    rows: List[Dict[str, Any]] = []
    for t, p in enumerate(pair_sequence):
        r = dict(res_cache[repr(p)])
        r["index"] = t
        rows.append(r)

    if keep_circuits == "unique":
        # Indexed to match `sequence`, not dict insertion order by accident.
        W_list = [circuits[repr(p)] for p in unique]
    elif keep_circuits == "all":
        W_list = [circuits[repr(p)] for p in pair_sequence]
    elif keep_circuits == "first":
        W_list = [circuits[repr(pair_sequence[0])]] if pair_sequence else []
    else:
        W_list = []
    return W_list, rows


def verify_analytic_counts(
    ancilla, system_qubits, H_int, tau, filter_terms, chosen_pair, T_filter,
    reps=4, trotter_steps_H=1, final_rank=4,
) -> Dict[str, Any]:
    """Build W directly at a small mode cutoff and compare with the analytic
    counts. Run once after any change to build_W_circuit."""
    U_plus = linlin.build_U_H_trotter(
        system_qubits, H_int, time_s=-T_filter,
        trotter_steps_H=trotter_steps_H, final_rank=final_rank,
    )
    prop_stats = _circuit_stats(U_plus)
    pred = count_W_resources_analytic(
        ancilla, system_qubits, tau, filter_terms, chosen_pair,
        prop_stats, reps=reps,
    )
    circ = linlin.build_W_circuit(
        ancilla=ancilla, system_qubits=system_qubits, H_int=H_int, tau_s=tau,
        filter_terms=filter_terms, pairs=[chosen_pair], weights=[1.0],
        reps=reps, trotter_steps_H=trotter_steps_H, seed=None,
        T_filter=T_filter, chosen_pair=chosen_pair, final_rank=final_rank,
    )
    true_m, true_o, _, true_r = _circuit_stats(circ)
    return {
        "moments_predicted": pred["moments"], "moments_actual": true_m,
        "operations_predicted": pred["operations"], "operations_actual": true_o,
        "rotations_predicted": pred["rotations"], "rotations_actual": true_r,
        "match": bool(pred["moments"] == true_m
                      and pred["operations"] == true_o
                      and pred["rotations"] == true_r),
    }


# =============================================================================
# Self-check: Kraus construction vs the actual circuit
# =============================================================================

def verify_against_circuit(
    ancilla, system_qubits, H_int, tau, filter_terms, chosen_pair,
    T_filter, reps=2, trotter_steps_H=1, final_rank=4,
) -> float:
    """Build W both ways on a tiny mode cutoff and compare. Run after any edit
    to build_W_circuit. Returns the max elementwise deviation."""
    U_plus, U_minus = build_propagator_unitaries(
        system_qubits, H_int, T_filter,
        trotter_steps_H=trotter_steps_H, final_rank=final_rank,
    )
    _, _, U_mat = build_W_kraus(
        ancilla, system_qubits, tau, filter_terms, chosen_pair,
        U_plus, U_minus, reps=reps, return_unitary=True,
    )
    circ = linlin.build_W_circuit(
        ancilla=ancilla, system_qubits=system_qubits, H_int=H_int, tau_s=tau,
        filter_terms=filter_terms, pairs=[chosen_pair], weights=[1.0],
        reps=reps, trotter_steps_H=trotter_steps_H, seed=None,
        T_filter=T_filter, chosen_pair=chosen_pair, final_rank=final_rank,
    )
    U_circ = _full_unitary(circ, ancilla, system_qubits)
    # A_on_pair uses PauliString powers, exact up to a global phase.
    ov = np.vdot(U_circ.reshape(-1), U_mat.reshape(-1))
    phase = ov / abs(ov) if abs(ov) > 1e-12 else 1.0
    return float(np.max(np.abs(U_circ - phase * U_mat)))


# =============================================================================
# Plot
# =============================================================================

def plot_energy_errors(rows: List[Dict[str, Any]], outpath: str,
                       chem_acc: float = 1.6e-3):
    """Two panels, one point per geometry: energies, and |E_final - E_0|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("final_energy") is not None]
    if not rows:
        print("[plot] no simulated geometries; nothing to plot", flush=True)
        return None

    idx = [r["hamiltonian_index"] for r in rows]
    e_fin = [r["final_energy"] for r in rows]
    e_ini = [r["initial_energy"] for r in rows]
    e_ref = [r["E_fixed_N"] for r in rows]
    e_fock = [r.get("E_full_fock") for r in rows]
    err = [abs(a - b) * 1e3 for a, b in zip(e_fin, e_ref)]   # mHa

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax[0].plot(idx, e_ini, "o--", ms=4, label="initial (before W chain)")
    ax[0].plot(idx, e_fin, "o-", ms=4, label="final (after W chain)")
    ax[0].plot(idx, e_ref, "s-", ms=3, label=r"exact fixed-$N_e$ ground")
    if all(v is not None for v in e_fock):
        ax[0].plot(idx, e_fock, "^:", ms=3, label="full Fock ground")
    ax[0].set_xlabel("Hamiltonian index / geometry index")
    ax[0].set_ylabel("Energy (Hartree)")
    ax[0].set_title("(a) Energy at each geometry")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].semilogy(idx, err, "o-", ms=4)
    ax[1].axhline(chem_acc * 1e3, ls=":", c="k",
                  label=f"chemical accuracy ({chem_acc*1e3:.1f} mHa)")
    ax[1].set_xlabel("Hamiltonian index / geometry index")
    ax[1].set_ylabel(r"$|E_{\mathrm{final}} - E_{0}^{(N_e)}|$ (mHa)")
    ax[1].set_title("(b) Final energy error")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    fig.savefig(str(outpath).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {outpath}", flush=True)
    return outpath