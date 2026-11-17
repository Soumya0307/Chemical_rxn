"""XYZ -> PySCF orbitals -> electronic integrals -> a Cirq PauliSum.

All spin-orbital indices are interleaved alpha/beta and use big-endian matrix
ordering, as in Cirq and OpenFermion. Energies include the nuclear/core constant.
The optional active-space alignment rotates core, active and virtual blocks
separately; it never mixes an active orbital with a frozen orbital.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import cirq
from openfermion import InteractionOperator, QubitOperator, get_fermion_operator, jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial
from scipy.sparse.linalg import eigsh
from pyscf import gto, scf, mcscf, ao2mo, fci


def read_xyz(path):
    """Read every frame of a standard multi-frame XYZ trajectory (Angstrom)."""
    lines = Path(path).expanduser().read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i])
        if n < 1 or i + n + 2 > len(lines):
            raise ValueError(f"Incomplete XYZ frame at line {i + 1}.")
        frame = []
        for line in lines[i + 2:i + n + 2]:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ atom line: {line!r}")
            frame.append((parts[0], tuple(float(x) for x in parts[1:4])))
        if frames and [a for a, _ in frame] != [a for a, _ in frames[0]]:
            raise ValueError("Atom identities/order must remain fixed along the path.")
        frames.append(frame)
        i += n + 2
    if not frames:
        raise ValueError("The XYZ file contains no frames.")
    return frames


def to_cirq(operator, qubits):
    """OpenFermion QubitOperator -> Cirq PauliSum, preserving the scalar term."""
    paulis = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}
    return cirq.PauliSum.from_pauli_strings([
        cirq.PauliString({qubits[i]: paulis[p] for i, p in term}, coefficient=c)
        for term, c in operator.terms.items()
    ])


def to_openfermion(operator, qubits):
    positions = {q: i for i, q in enumerate(qubits)}
    result = QubitOperator()
    for term in cirq.PauliSum.wrap(operator):
        key = tuple(sorted((positions[q], str(p)) for q, p in term.items()))
        result += QubitOperator(key, term.coefficient)
    return result


def spectral_range(operator, qubits, *, method='sparse', max_qubits=14):
    """Return (lower, upper); 'bound' is a scalable, rigorous Pauli-norm bound."""
    terms = list(cirq.PauliSum.wrap(operator))
    constant = sum(float(t.coefficient.real) for t in terms if not t)
    if method == 'bound':
        radius = sum(abs(t.coefficient) for t in terms if t)
        return constant - radius, constant + radius
    if len(qubits) > max_qubits:
        raise ValueError("Spectral calculation exceeds max_spectral_qubits; use "
                         "spectral_method='bound' or an explicit filter_wmax.")
    from openfermion import get_sparse_operator
    sparse = get_sparse_operator(to_openfermion(operator, qubits), n_qubits=len(qubits))
    if method == 'dense' or sparse.shape[0] <= 4:
        ev = np.linalg.eigvalsh(sparse.toarray())
        return float(ev[0]), float(ev[-1])
    if method != 'sparse':
        raise ValueError("spectral method must be sparse, dense, or bound")
    # Fail on nonconvergence instead of accepting an unverified narrow bandwidth.
    lo = eigsh(sparse, k=1, which='SA', return_eigenvectors=False, tol=1e-10)[0]
    hi = eigsh(sparse, k=1, which='LA', return_eigenvectors=False, tol=1e-10)[0]
    return float(lo), float(hi)


def align_orbitals(mol, coefficients, blocks, reference):
    """Orthogonal Procrustes in corresponding orbital blocks; report overlaps."""
    if reference is None:
        return coefficients, {'first_geometry': True, 'min_block_overlap': 1.0}
    if tuple(map(tuple, blocks)) != tuple(map(tuple, reference['blocks'])):
        raise ValueError("Orbital block sizes changed: an explicit state-transport map is needed.")
    previous = reference['mo_coeff']
    if previous.shape[1] != coefficients.shape[1]:
        raise ValueError("The orbital dimension changed along the path.")
    overlap = gto.intor_cross('int1e_ovlp', reference['mol'], mol)
    aligned = coefficients.copy()
    singular_values = []
    for block in blocks:
        indices = np.asarray(block, dtype=int)
        if not len(indices):
            continue
        cross = previous[:, indices].T @ overlap @ aligned[:, indices]
        u, s, vt = np.linalg.svd(cross)
        aligned[:, indices] = aligned[:, indices] @ vt.T @ u.T
        singular_values.extend(s.tolist())
    return aligned, {'first_geometry': False,
                     'min_block_overlap': min(singular_values, default=1.0),
                     'block_singular_values': singular_values}


@dataclass
class HamiltonianData:
    interaction: object
    fermion: object
    qubit: cirq.PauliSum
    system_qubits: tuple
    ancilla: cirq.Qid
    n_electrons: int
    geometry: object
    reference: dict
    alignment: dict
    chemistry_seconds: float
    matrix: object = None
    energy: object = None
    spectral_width: object = None

    @property
    def n_system(self):
        return len(self.system_qubits)

    @property
    def scf_mol(self):
        return self.reference['mol']

    @property
    def mo_coeff(self):
        return self.reference['mo_coeff']

    @property
    def mo_occ(self):
        return self.reference['mo_occ']


def build_molecule(geometry, basis='sto-3g', multiplicity=1, charge=0,
                   active_indices=None, ancilla_index=0, n_active_electrons=None,
                   n_active_orbitals=None, use_casscf_orbitals=True,
                   max_dense_qubits=10, use_procrustes_gauge=False,
                   align_reference=None, fix_spin_ss=None, compute_dense=False,
                   spectral_width_method='none'):
    start = time.perf_counter()
    if (n_active_electrons is None) != (n_active_orbitals is None):
        raise ValueError("Supply both active electron and active orbital counts.")
    mol = gto.M(atom=geometry, basis=basis, charge=int(charge),
                spin=int(multiplicity)-1, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol) if multiplicity == 1 else scf.ROHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge.")
    coefficients = mf.mo_coeff
    active = n_active_orbitals is not None
    if active:
        ncas = int(n_active_orbitals)
        ne = (sum(n_active_electrons) if isinstance(n_active_electrons, (tuple,list))
              else int(n_active_electrons))
        if ne > mol.nelectron or (mol.nelectron - ne) % 2:
            raise ValueError("The active space must leave an integer number of doubly occupied core orbitals.")
        mc = (mcscf.CASSCF if use_casscf_orbitals else mcscf.CASCI)(mf, ncas, n_active_electrons)
        if fix_spin_ss is not None:
            mc.fcisolver = fci.addons.fix_spin_(mc.fcisolver, ss=float(fix_spin_ss))
        if active_indices is not None:
            if len(active_indices) != ncas or len(set(active_indices)) != ncas:
                raise ValueError("active_indices must contain n_active_orbitals distinct indices.")
            coefficients = mc.sort_mo(active_indices, mo_coeff=coefficients, base=0)
        mc.kernel(coefficients)
        if not getattr(mc, 'converged', True):
            raise RuntimeError("CASSCF did not converge.")
        coefficients = mc.mo_coeff
        nc = int(mc.ncore)
        blocks = [list(range(nc)), list(range(nc,nc+ncas)),
                  list(range(nc+ncas,coefficients.shape[1]))]
    else:
        if active_indices is not None:
            raise ValueError("active_indices requires an explicit active space.")
        ne = mol.nelectron
        blocks = [np.flatnonzero(mf.mo_occ == n).tolist() for n in (2,1,0)]
    alignment = {'enabled': bool(use_procrustes_gauge)}
    if use_procrustes_gauge:
        coefficients, info = align_orbitals(mol, coefficients, blocks, align_reference)
        alignment.update(info)
    if active:
        h1, constant = mc.get_h1eff(mo_coeff=coefficients)
        h2 = ao2mo.restore(1, mc.get_h2eff(mo_coeff=coefficients), ncas)
    else:
        h1 = coefficients.T @ mf.get_hcore() @ coefficients
        h2 = ao2mo.restore(1, ao2mo.kernel(mol, coefficients), coefficients.shape[1])
        constant = mol.energy_nuc()
    # PySCF chemist ordering -> OpenFermion ordering; required also for CASSCF.
    h2 = np.asarray(h2.transpose(0,2,3,1), order='C')
    one_body, two_body = spinorb_from_spatial(h1, h2)
    interaction = InteractionOperator(float(constant), one_body, .5*two_body)
    fermion = get_fermion_operator(interaction)
    of_qubit = jordan_wigner(fermion)
    of_qubit.compress()
    qubits = tuple(cirq.LineQubit.range(ancilla_index+1, ancilla_index+1+len(one_body)))
    result = HamiltonianData(interaction, fermion, to_cirq(of_qubit, qubits), qubits,
        cirq.LineQubit(ancilla_index), ne, geometry,
        {'mol':mol,'mo_coeff':coefficients,'mo_occ':mf.mo_occ.copy(),'blocks':blocks},
        alignment, time.perf_counter()-start)
    if compute_dense:
        if result.n_system > max_dense_qubits:
            raise ValueError("Dense Hamiltonian exceeds max_dense_qubits.")
        result.matrix = result.qubit.matrix(qubits=qubits)
        eigenvalues = np.linalg.eigvalsh(result.matrix)
        result.energy = float(eigenvalues[0])
        result.spectral_width = float(eigenvalues[-1]-eigenvalues[0])
    elif spectral_width_method != 'none':
        low, high = spectral_range(result.qubit, qubits, method=spectral_width_method)
        result.spectral_width = high-low
    return result


def from_xyz(path, frame=0, **parameters):
    return build_molecule(read_xyz(path)[frame], **parameters)


class HamiltonianBuilder:
    # The unchanged Rottnest executable expects this spelling.
    build_molecule = staticmethod(build_molecule)
