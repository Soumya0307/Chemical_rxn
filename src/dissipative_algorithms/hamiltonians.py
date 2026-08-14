'''
    Wrapper for Hamiltonian building
'''
import time
from dataclasses import dataclass

import numpy as np
import cirq
from openfermionpyscf import run_pyscf
from openfermion import (
    MolecularData,
    InteractionOperator,
    get_fermion_operator,
    get_sparse_operator,
)
from openfermion.transforms import jordan_wigner
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

@dataclass
class HamiltonianData:
    '''
        Data object returned by the HamiltonianBuilder
    '''
    interaction: object
    fermion: object
    qubit: object
    matrix: np.ndarray          # None unless compute_dense=True
    energy: float               # None unless compute_dense=True
    n_system: int
    ancilla: cirq.LineQubit
    system_qubits: list

    geometry: object = None
    input_pauli: object = None

    # Filled by the aligned (Procrustes) build path; propagates the orbital
    # gauge to the next geometry along the reaction path.
    scf_mol: object = None
    mo_coeff: object = None
    mo_occ: object = None

    # E_max - E_min. Set even when the dense matrix is skipped; this is the
    # only spectral quantity the filter construction actually needs.
    spectral_width: float = None


class HamiltonianBuilder:
    '''
        Namespace for Hamiltonian building methods
    '''

    @classmethod
    def build_molecule(
        cls,
        geometry: tuple[tuple[str, tuple[float, float, float]]],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        active_indices=None,
        ancilla_index=0,
        n_active_electrons=None,
        n_active_orbitals=None,
        use_casscf_orbitals=True,
        max_dense_qubits=12,
        use_procrustes_gauge=False,
        align_reference=None,
        fix_spin_ss=None,
        compute_dense=False,
        spectral_width_method="sparse",
    ):
        """
        Build a molecular Hamiltonian.

        Modes:
          1. No active space: standard OpenFermion + PySCF Hamiltonian.
          2. Active space, use_casscf_orbitals=False: CASCI-like with HF orbitals.
          3. Active space, use_casscf_orbitals=True: true CASSCF active space.
          0. use_procrustes_gauge=True: full space in an aligned orbital gauge.

        compute_dense=False (default) skips the dense matrix and eigvalsh.
        Set True only for validation runs; it is guarded by max_dense_qubits.

        spectral_width_method: "sparse" (extremal Lanczos, no dense matrix),
        "dense" (dense eigvalsh, matrix discarded unless compute_dense), or
        "none" (caller supplies the filter scale explicitly).
        """

        use_active_space = (
            n_active_electrons is not None and n_active_orbitals is not None
        )

        scf_mol = None
        mo_coeff_out = None
        mo_occ_out = None

        # ------------------------------------------------------------
        # Case 0: full-space Hamiltonian in a Procrustes-aligned gauge.
        # align_reference = dict(mol=..., mo_coeff=..., mo_occ=...) from the
        # previous geometry. First geometry: pass None.
        # ------------------------------------------------------------
        if use_procrustes_gauge:
            if use_active_space:
                raise NotImplementedError(
                    "use_procrustes_gauge is implemented for the full-space "
                    "(no active space) path only."
                )
            mol = gto.Mole()
            mol.atom = geometry
            mol.basis = basis
            mol.charge = charge
            mol.spin = multiplicity - 1
            mol.unit = "Angstrom"
            mol.build()

            mf = scf.RHF(mol) if multiplicity == 1 else scf.ROHF(mol)
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("SCF did not converge for this geometry.")

            C = mf.mo_coeff
            if align_reference is not None:
                C = procrustes_align_mo(
                    align_reference["mol"], align_reference["mo_coeff"],
                    align_reference["mo_occ"], mol, C, mf.mo_occ,
                )

            interaction_ham = hamiltonian_from_mo_coeff(mol, C)
            fermion_ham = get_fermion_operator(interaction_ham)
            n_system = 2 * mol.nao_nr()
            scf_mol = mol
            mo_coeff_out = C
            mo_occ_out = np.array(mf.mo_occ, copy=True)
            print("Using full-space Hamiltonian in Procrustes-aligned gauge"
                  + (" (reference: previous geometry)." if align_reference is not None
                     else " (first geometry: canonical orbitals)."), flush=True)

        # ------------------------------------------------------------
        # Case 1: true CASSCF active-space Hamiltonian
        # ------------------------------------------------------------
        elif use_active_space and use_casscf_orbitals:
            mol = gto.Mole()
            mol.atom = geometry
            mol.basis = basis
            mol.charge = charge
            mol.spin = multiplicity - 1
            mol.unit = "Angstrom"
            mol.build()

            mf = scf.RHF(mol) if multiplicity == 1 else scf.ROHF(mol)
            mf.kernel()

            mc = mcscf.CASSCF(mf, n_active_orbitals, n_active_electrons)
            if fix_spin_ss is not None:
                # Pin the CAS CI root to a spin sector: ss = S(S+1), e.g. 0 for a
                # singlet. Essential for singlet-O2 chemistry, where the lowest
                # Ms=0 CI root is otherwise the TRIPLET (verified for the O atom).
                from pyscf import fci as _fci
                mc.fcisolver = _fci.addons.fix_spin_(mc.fcisolver, ss=float(fix_spin_ss))
            mc.kernel()

            h1_active, ecore = mc.get_h1eff()
            h2_active = mc.get_h2eff()
            h2_active = ao2mo.restore(1, h2_active, n_active_orbitals)

            # BUGFIX: ao2mo.restore returns chemist-ordered integrals (ij|kl).
            # OpenFermion's spinorb_from_spatial/InteractionOperator expect
            # h[p,q,r,s] (as produced by openfermionpyscf via .transpose(0,2,3,1)).
            # Without this transpose the CAS Hamiltonian is wrong: CAS(3,3) for
            # H3/STO-3G gave E_min=-3.1428 instead of the exact FCI -1.5710.
            h2_active = np.asarray(h2_active.transpose(0, 2, 3, 1), order="C")

            one_body_spin, two_body_spin = spinorb_from_spatial(h1_active, h2_active)
            interaction_ham = InteractionOperator(ecore, one_body_spin, 0.5 * two_body_spin)
            fermion_ham = get_fermion_operator(interaction_ham)
            n_system = 2 * n_active_orbitals

            print("Using true CASSCF active-space Hamiltonian.", flush=True)
            print("CASSCF energy:", mc.e_tot, flush=True)
            print("Active electrons:", n_active_electrons, flush=True)
            print("Active spatial orbitals:", n_active_orbitals, flush=True)
            print("Active spin orbitals / qubits:", n_system, flush=True)

        # ------------------------------------------------------------
        # Case 2: OpenFermion Hamiltonian using HF orbitals
        # ------------------------------------------------------------
        else:
            molecule = MolecularData(geometry, basis, multiplicity, charge)
            # run_fci was validation-only; it is skipped unless explicitly wanted.
            molecule = run_pyscf(
                molecule,
                run_scf=True,
                run_fci=bool(compute_dense) and not use_active_space,
            )

            if use_active_space:
                n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
                occupied_indices = list(range(n_core_orbitals))
                if active_indices is None:
                    active_indices = list(
                        range(n_core_orbitals, n_core_orbitals + n_active_orbitals)
                    )
                interaction_ham = molecule.get_molecular_hamiltonian(
                    occupied_indices=occupied_indices,
                    active_indices=active_indices,
                )
                print("Using CASCI-like active-space Hamiltonian with HF orbitals.", flush=True)
                print("Frozen occupied orbitals:", occupied_indices, flush=True)
                print("Active spatial orbitals:", active_indices, flush=True)
                n_system = 2 * len(active_indices)
            else:
                interaction_ham = molecule.get_molecular_hamiltonian()
                n_system = 2 * molecule.n_orbitals

            fermion_ham = get_fermion_operator(interaction_ham)

        # ------------------------------------------------------------
        # Jordan-Wigner transform
        # ------------------------------------------------------------
        qubit_ham = jordan_wigner(fermion_ham)
        qubit_ham.compress()

        # ------------------------------------------------------------
        # Spectral data. Dense path is opt-in and size-guarded.
        # ------------------------------------------------------------
        H_mat = None
        E_corr = None
        spectral_width = None

        if compute_dense:
            if n_system > max_dense_qubits:
                raise ValueError(
                    f"compute_dense=True with n_system={n_system} exceeds "
                    f"max_dense_qubits={max_dense_qubits}. The dense matrix would "
                    f"be {2**n_system} x {2**n_system}."
                )
            t0 = time.perf_counter()
            H_mat = get_sparse_operator(qubit_ham, n_qubits=n_system).toarray()
            ev = np.linalg.eigvalsh(H_mat)
            E_corr = float(ev[0])
            spectral_width = float(ev[-1] - ev[0])
            print(f"[dense] matrix + eigvalsh in {time.perf_counter()-t0:.1f} s; "
                  f"E_min={E_corr:.10f}, width={spectral_width:.6f}", flush=True)

        elif spectral_width_method == "sparse":
            t0 = time.perf_counter()
            spectral_width = cls.sparse_spectral_width(qubit_ham, n_system)
            print(f"[spectral] width={spectral_width:.6f} via Lanczos in "
                  f"{time.perf_counter()-t0:.1f} s (no dense matrix)", flush=True)

        elif spectral_width_method == "dense":
            if n_system > max_dense_qubits:
                raise ValueError(
                    f"spectral_width_method='dense' with n_system={n_system} "
                    f"exceeds max_dense_qubits={max_dense_qubits}."
                )
            ev = np.linalg.eigvalsh(
                get_sparse_operator(qubit_ham, n_qubits=n_system).toarray()
            )
            spectral_width = float(ev[-1] - ev[0])

        elif spectral_width_method != "none":
            raise ValueError("spectral_width_method must be 'sparse', 'dense', or 'none'.")

        # ------------------------------------------------------------
        ancilla = cirq.LineQubit(ancilla_index)
        system_qubits = [
            cirq.LineQubit(i)
            for i in range(ancilla_index + 1, ancilla_index + 1 + n_system)
        ]

        return HamiltonianData(
            interaction=interaction_ham,
            fermion=fermion_ham,
            qubit=qubit_ham,
            matrix=H_mat,
            energy=E_corr,
            n_system=n_system,
            ancilla=ancilla,
            system_qubits=system_qubits,
            geometry=geometry,
            input_pauli=None,
            scf_mol=scf_mol,
            mo_coeff=mo_coeff_out,
            mo_occ=mo_occ_out,
            spectral_width=spectral_width,
        )

    @classmethod
    def sparse_spectral_width(cls, qubit_ham, n_system, tol=1e-4, maxiter=None):
        """
        Return E_max - E_min from two extremal Lanczos eigenvalues of the sparse
        Hamiltonian. No dense matrix is ever formed.

        The filter only needs this to set wmax = factor * width, so a loose
        tolerance is fine; tol=1e-4 keeps the iteration count small.
        """

        H_sparse = get_sparse_operator(qubit_ham, n_qubits=n_system)

        lo = cls.extremal(H_sparse, "SA", tol=tol, maxiter=maxiter)
        hi = cls.extremal(H_sparse, "LA", tol=tol, maxiter=maxiter)
        return hi - lo

    @classmethod
    def extremal(cls, H_sparse, which, tol=1e-4, maxiter=None):
        try:
           val = eigsh(
                    H_sparse,
                    k = 1,
                    which = which,
                    tol = tol,
                    maxiter = maxiter,
                    return_eigenvectors = False
                )[0]
        except ArpackNoConvergence as exc:
            if len(exc.eigenvalues):
                val = exc.eigenvalues[0]
        return float(val)

