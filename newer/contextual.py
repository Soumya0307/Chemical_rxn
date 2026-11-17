"""Contextual-subspace approximation (thesis Ch. 4; Kirby et al., 2021).

Choose a noncontextual Pauli subset by a coefficient-ordered greedy search;
close it under inference within H. Its commuting generators G and mutually
anticommuting clique representatives C give E(q,r)=b(q)+sum_i a_i(q)r_i.
Enumerate q and minimize r analytically. Rotate sum r_i C_i to one Pauli,
then fix its eigenvalue and selected G eigenvalues. Release constraints until
the reference-energy tolerance is met. The subset/order is a heuristic, not
a claim of globally optimal reduction. All operators use the SAME isometry.

Operator projection is symbolic; automatic energy selection and state
projection are small-system diagnostics guarded by max_qubits. Large jobs can
specify retained_qubits explicitly; their energy tolerance is NOT certified.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import cirq
if __package__:
    from .filter import Jump, pauli_rotation
else:
    from filter import Jump, pauli_rotation


def _unit(p):
    return p.with_coefficient(1)


def _vector(p, qubits):
    return sum((int(p.get(q) in (cirq.X,cirq.Y)) << i) |
               (int(p.get(q) in (cirq.Z,cirq.Y)) << (i+len(qubits)))
               for i,q in enumerate(qubits))


def _independent(paulis, qubits):
    basis, result = {}, []
    for p in paulis:
        v = _vector(p,qubits)
        while v:
            top = v.bit_length()-1
            if top not in basis:
                basis[top] = v
                result.append(_unit(p))
                break
            v ^= basis[top]
    return result


def _cliques(terms):
    """Noncentral commutation must be an equivalence relation."""
    central = [p for p in terms if all(cirq.commutes(p,q) for q in terms)]
    other = [p for p in terms if p not in central]
    groups = []
    while other:
        first = other[0]
        group = [p for p in other if cirq.commutes(p,first)]
        if any(not cirq.commutes(p,q) for p,q in combinations(group,2)):
            return None
        rest = [p for p in other if p not in group]
        if any(cirq.commutes(p,q) for p in group for q in rest):
            return None
        groups.append(group)
        other = rest
    return central, groups


def _diagonalize(generators, qubits):
    """Clifford C with C G_j C† = signed Z on successive pivot qubits."""
    remaining = list(generators)
    operations, pivots, transformed = [], [], []
    for index in range(len(remaining)):
        p = remaining[index]
        # Eliminate earlier Z pivots using already diagonal generators.
        for old, pivot in zip(transformed,pivots):
            if p.get(pivot) == cirq.Z:
                p *= old
        support = [q for q in qubits if q in p and q not in pivots]
        if not support:
            raise ValueError('Dependent stabilizer generators.')
        pivot = support[0]
        ops = []
        for q in support:
            if p[q] == cirq.X:
                ops.append(cirq.H(q))
            elif p[q] == cirq.Y:
                ops.extend([cirq.S(q)**-1,cirq.H(q)])
        ops.extend(cirq.CNOT(q,pivot) for q in support[1:])
        operations.extend(ops)
        remaining = [r.after(ops) for r in remaining]
        transformed = [r.after(ops) for r in transformed]
        # Record the eliminated generator, while retaining the original G map.
        transformed.append(p.after(ops))
        pivots.append(pivot)
    return cirq.FrozenCircuit(operations), pivots


def _rotate_sum(operator, rotations):
    result = cirq.PauliSum.wrap(operator)
    for generator,angle in rotations:
        rotated = cirq.PauliSum()
        for term in result:
            if cirq.commutes(generator,term):
                rotated += term
            else:
                rotated += np.cos(2*angle)*term - 1j*np.sin(2*angle)*(generator*term)
        result = rotated
    return result


@dataclass
class Reduction:
    hamiltonian: cirq.PauliSum
    jumps: list
    qubits: tuple
    original_qubits: tuple
    rotations: list
    clifford: cirq.FrozenCircuit
    fixed: dict
    diagnostics: dict

    def project_operator(self, operator):
        transformed = _rotate_sum(operator,self.rotations)
        remaining = [q for q in self.original_qubits if q not in self.fixed]
        rename = dict(zip(remaining,self.qubits))
        result = cirq.PauliSum()
        for term in transformed:
            term = term.after(self.clifford)
            coefficient = term.coefficient
            paulis = {}
            for q,p in term.items():
                if q in self.fixed:
                    if p != cirq.Z:
                        coefficient = 0
                        break
                    coefficient *= self.fixed[q]
                else:
                    paulis[rename[q]] = p
            if abs(coefficient) > 1e-12:
                result += cirq.PauliString(paulis,coefficient=coefficient)
        return result

    def isometry(self, max_qubits=10):
        if len(self.original_qubits) > max_qubits:
            raise ValueError('Contextual isometry exceeds the dense qubit guard.')
        circuit = cirq.Circuit(pauli_rotation(g,a) for g,a in self.rotations)
        circuit += self.clifford
        u = circuit.unitary(qubit_order=self.original_qubits)
        indices = [i for i in range(2**len(self.original_qubits))
                   if all(1-2*((i >> (len(self.original_qubits)-1-j)) & 1) == self.fixed[q]
                          for j,q in enumerate(self.original_qubits) if q in self.fixed)]
        return u.conj().T[:,indices]


def identity_reduction(hamiltonian, jumps, qubits):
    return Reduction(cirq.PauliSum.wrap(hamiltonian),jumps,tuple(qubits),tuple(qubits),
                     [],cirq.FrozenCircuit(),{},dict(enabled=False,retained_qubits=len(qubits)))


def reduce_operators(hamiltonian, jumps, qubits, *, energy_tolerance=1.6e-3,
                     reference_energy=None, sector_isometry=None, retained_qubits=None,
                     max_qubits=10, max_assignment_generators=18, max_nc_terms=128):
    """Return H_CS, projected jumps and an explicit common operator/state map.

    sector_isometry selects the physical reference sector for energy tests;
    candidate states must belong to that sector, not merely have the right
    expected particle number. Unreachable sectors cause a constraint release.
    """
    qubits = tuple(qubits)
    n = len(qubits)
    if energy_tolerance < 0 or max_nc_terms < 1:
        raise ValueError('Invalid contextual tolerance/search size.')
    if retained_qubits is None and n > max_qubits:
        raise ValueError('Automatic contextual energy selection needs a small exact reference; '
                         'supply retained_qubits for an explicitly unvalidated larger reduction.')
    if retained_qubits is not None and not 0 <= retained_qubits <= n:
        raise ValueError('retained_qubits must lie between zero and the original qubit count.')
    terms = sorted([t for t in cirq.PauliSum.wrap(hamiltonian) if t],
                   key=lambda p:(-abs(p.coefficient),str(p)))
    selected = []
    for term in terms[:max_nc_terms]:
        candidate = selected+[_unit(term)]
        if _cliques(candidate) is not None:
            selected = candidate
    central,groups = _cliques(selected)
    representatives = [g[0] for g in groups]
    generators = _independent(central+[_unit(p*g[0]) for g in groups for p in g[1:]],qubits)
    if len(generators) > max_assignment_generators:
        raise ValueError('Noncontextual assignment search exceeds max_assignment_generators; '
                         'reduce max_nc_terms or increase the explicit search limit.')
    cg,pivots = _diagonalize(generators,qubits)
    reps_rotated = [p.after(cg) for p in representatives]
    # Inference closure: every H term in <G> or in a coset <G>C_i belongs to H_nc.
    def central_values(p, assignments):
        if any(q not in pivots or v != cirq.Z for q,v in p.items()):
            return None
        values = np.full(len(assignments),complex(p.coefficient))
        for q in p:
            values *= assignments[:,pivots.index(q)]
        return values.real
    assignments = np.array([[1-2*((i>>j)&1) for j in range(len(pivots))]
                            for i in range(2**len(pivots))],dtype=int).reshape(-1,len(pivots)) if pivots else np.empty((1,0),int)
    constant = np.zeros(len(assignments))
    beta = np.zeros((len(assignments),len(representatives)))
    closure_count = 0
    for term in cirq.PauliSum.wrap(hamiltonian):
        p = term.after(cg)
        values = central_values(p,assignments)
        if values is not None:
            constant += values
            closure_count += 1
            continue
        for i,c in enumerate(reps_rotated):
            values = central_values(p*c,assignments)
            if values is not None:
                beta[:,i] += values
                closure_count += 1
                break
    energies = constant-np.linalg.norm(beta,axis=1)
    best = int(np.argmin(energies))
    qvalues = assignments[best]
    # Desired original generator eigenvalues follow from its diagonal image.
    signs = [float(central_values(g.after(cg),qvalues[None,:])[0]) for g in generators]
    rotations = []
    if representatives:
        b = beta[best]
        r = -b/np.linalg.norm(b) if np.linalg.norm(b)>1e-14 else np.eye(1,len(b))[0]
        amplitude = float(r[0])
        for i in range(1,len(r)):
            angle = .5*np.arctan2(r[i],amplitude)
            rotations.append((1j*representatives[0]*representatives[i],float(angle)))
            amplitude = float(np.hypot(amplitude,r[i]))
        if len(r) == 1 and r[0] < 0:
            signs += [-1.]
        else:
            signs += [1.]
        constraints = generators+[representatives[0]]
    else:
        r = np.array([])
        constraints = generators
    diagnostics = dict(enabled=True,method='greedy noncontextual cliques with inference closure',
        noncontextual_terms=closure_count,commuting_generators=len(generators),
        anticommuting_representatives=len(representatives),noncontextual_energy=float(energies[best]),
        r=r.tolist(),assignment=signs,energy_tolerance=energy_tolerance,candidates=[])
    full_matrix = None
    if n <= max_qubits:
        full_matrix = hamiltonian.matrix(qubits=qubits)
        if sector_isometry is None:
            sector_isometry = np.eye(2**n)
        if reference_energy is None:
            reference_energy = float(np.linalg.eigvalsh(sector_isometry.conj().T@full_matrix@sector_isometry)[0])
    maximum_fixed = len(constraints)
    if retained_qubits is not None:
        maximum_fixed = n-retained_qubits
        if maximum_fixed > len(constraints):
            raise ValueError('Requested more fixed qubits than the noncontextual constraints provide.')
    for count in range(maximum_fixed,-1,-1):
        chosen,chosen_signs = constraints[:count],signs[:count]
        cliff,piv = _diagonalize(chosen,qubits)
        # Solve the diagonal parity equations for the fixed pivot bits.
        values = {}
        for g,sign in zip(chosen,chosen_signs):
            image = g.after(cliff)
            unknown = [q for q in image if q not in values]
            if len(unknown) != 1:
                raise RuntimeError('Unexpected stabilizer elimination order.')
            values[unknown[0]] = int(round(sign/float(image.coefficient.real)*
                                      np.prod([values[q] for q in image if q in values])))
        output_qubits = tuple(cirq.LineQubit.range(qubits[0].x,qubits[0].x+n-count)) if qubits else ()
        reduction = Reduction(None,[],output_qubits,qubits,rotations,cliff,values,diagnostics)
        reduction.hamiltonian = reduction.project_operator(hamiltonian)
        reduction.jumps = [Jump(j.name,reduction.project_operator(j.operator),j.pair) for j in jumps]
        error,energy = None,None
        if full_matrix is not None:
            v = reduction.isometry(max_qubits)
            # Exact intersection with the requested physical sector.
            gram = v.conj().T@sector_isometry@sector_isometry.conj().T@v
            ev,vec = np.linalg.eigh(gram)
            basis = vec[:,ev>1-1e-8]
            if basis.shape[1]:
                reduced = reduction.hamiltonian.matrix(qubits=output_qubits)
                energy = float(np.linalg.eigvalsh(basis.conj().T@reduced@basis)[0])
                error = abs(energy-reference_energy)
        diagnostics['candidates'].append(dict(retained_qubits=n-count,energy=energy,error=error))
        if retained_qubits is not None or (error is not None and error<=energy_tolerance) or count == 0:
            diagnostics.update(retained_qubits=n-count,energy_error=error,
                energy_validated=error is not None and error<=energy_tolerance,
                reference_energy=reference_energy)
            # With no constraints, return the original representation exactly.
            if count == 0:
                reduction = identity_reduction(hamiltonian,jumps,qubits)
                reduction.diagnostics = diagnostics
            return reduction
