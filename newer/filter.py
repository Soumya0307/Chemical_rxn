"""Square-wave Fourier filter and Hermitian Jordan–Wigner couplings, in Cirq.

Indices are zero-based spin orbitals, interleaved alpha/beta. Ax = a†i aj +
a†j ai; Ay = i(a†i aj - a†j ai). A whole W uses ONE sampled coupling.
Keep the zero Fourier coefficients: the W circuit uses a uniform time grid.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import cirq


class SquareWaveFilter:
    def get_coefficients(self, L, K, tau, A=1., window='lanczos', p=1,
                         drop_zeros=False, eps=1e-15, sign=+1):
        if not np.isfinite(L) or L <= 0 or K < 0 or int(K) != K or tau < 0:
            raise ValueError('Require finite L > 0, integer K >= 0 and tau >= 0.')
        if sign not in (-1, 1) or A < 0 or p < 0:
            raise ValueError('Require sign = ±1, A >= 0 and p >= 0.')
        if drop_zeros:
            raise ValueError('Use drop_zeros=False to preserve the W time grid.')
        ks = np.arange(-int(K), int(K)+1)
        c = np.zeros(len(ks), complex)
        c[ks == 0] = A/2
        odd = ks % 2 != 0
        c[odd] = sign*1j*A/(np.pi*ks[odd])
        if window == 'lanczos':
            c *= np.sinc(ks/(K+1))**p
        elif window == 'fejer':
            c *= 1-abs(ks)/(K+1)
        elif window not in (None, 'none'):
            raise ValueError('window must be lanczos, fejer or none.')
        self.data = dict(L=float(L), K=int(K), tau=float(tau), A=A, sign=sign,
            window=window, p=p, T_filter=np.pi/L, cvals=c, ks_filter=ks,
            filter_terms=list(zip(ks.tolist(), c.tolist())),
            omega_filter=ks*np.pi/L, theta_filter=np.sqrt(tau)*abs(c)/2,
            phi_filter=np.angle(c))
        return self.data


def normalize_pair(pair):
    if len(pair) in (2, 3) and all(isinstance(x, (int, np.integer)) for x in pair[:2]):
        legs, kind = [(int(pair[0]), int(pair[1]))], 'Ax' if len(pair) == 2 else pair[2]
    elif len(pair) == 2:
        legs, kind = [tuple(x) for x in pair[0]], pair[1]
    else:
        raise ValueError(f'Invalid jump specification: {pair!r}')
    if kind not in ('Ax', 'Ay') or not legs:
        raise ValueError('Choose Ax or Ay and at least one pair.')
    return legs, kind


@dataclass
class Jump:
    name: str
    operator: cirq.PauliSum
    pair: object = None


def jw_jump(qubits, pair):
    legs, kind = normalize_pair(pair)
    result = cirq.PauliSum()
    for i, j in legs:
        if i == j or min(i,j) < 0 or max(i,j) >= len(qubits):
            raise ValueError(f'Invalid spin-orbital pair {(i,j)} for {len(qubits)} modes.')
        parity = {qubits[k]:cirq.Z for k in range(min(i,j)+1, max(i,j))}
        if kind == 'Ax':
            result += cirq.PauliString({**parity,qubits[i]:cirq.X,qubits[j]:cirq.X}, coefficient=.5)
            result += cirq.PauliString({**parity,qubits[i]:cirq.Y,qubits[j]:cirq.Y}, coefficient=.5)
        else:
            result += cirq.PauliString({**parity,qubits[i]:cirq.Y,qubits[j]:cirq.X}, coefficient=.5)
            result -= cirq.PauliString({**parity,qubits[i]:cirq.X,qubits[j]:cirq.Y}, coefficient=.5)
    return Jump(f'{kind}:{legs}', result, pair)


def make_jumps(qubits, pairs=None, *, conserve_spin=False, spin_adapted=False,
               max_jump_support=None):
    """A support cap changes the coupling family, and therefore the dynamics."""
    if pairs is None:
        if spin_adapted:
            if len(qubits) % 2:
                raise ValueError('Spin-adapted couplings require alpha/beta pairs.')
            pairs = [(((2*i,2*j),(2*i+1,2*j+1)),kind)
                     for i,j in combinations(range(len(qubits)//2),2) for kind in ('Ax','Ay')]
        else:
            pairs = [(i,j,kind) for i,j in combinations(range(len(qubits)),2)
                     if not conserve_spin or i % 2 == j % 2 for kind in ('Ax','Ay')]
    jumps = [jw_jump(qubits,p) for p in pairs]
    if max_jump_support is not None:
        jumps = [j for j in jumps if len(set().union(*(set(p.qubits) for p in j.operator)))
                 <= max_jump_support]
    if not jumps:
        raise ValueError('The selected coupling family is empty.')
    return jumps


def pauli_rotation(term, angle):
    """Exactly exp(-i angle*term), INCLUDING its scalar/global phase."""
    coefficient = complex(term.coefficient)
    if abs(coefficient.imag) > 1e-10:
        raise ValueError('Evolution requires a Hermitian Pauli sum.')
    angle *= coefficient.real
    if not term:
        return cirq.global_phase_operation(np.exp(-1j*angle))
    return cirq.PauliStringPhasor(term.with_coefficient(1),
        exponent_neg=angle/np.pi, exponent_pos=-angle/np.pi)


def kick(ancilla, operator, theta, phi, *, steps=1, legacy_phase=True):
    """Ancilla-dilated kick. JW terms commute; reduced jumps may require splitting.

    With legacy_phase=True the chronological Rz(phi), H, controlled-A, H,
    Rz(-phi) matches the existing W convention. False gives the direct
    [[0,c* A],[c A,0]] dilation. Noncommuting sums use symmetric second order.
    """
    if steps < 1 or int(steps) != steps:
        raise ValueError('kick steps must be a positive integer.')
    terms = list(cirq.PauliSum.wrap(operator))
    commuting = all(cirq.commutes(a,b) for a,b in combinations(terms,2))
    phase = phi if legacy_phase else -phi
    circuit = cirq.Circuit(cirq.rz(phase)(ancilla), cirq.H(ancilla))
    if commuting:
        circuit.append(pauli_rotation(cirq.Z(ancilla)*t,theta) for t in terms)
    else:
        step = cirq.FrozenCircuit(pauli_rotation(cirq.Z(ancilla)*t,theta/(2*steps))
                                 for t in terms+terms[::-1])
        circuit.append(cirq.CircuitOperation(step).repeat(int(steps)))
    circuit.append([cirq.H(ancilla), cirq.rz(-phase)(ancilla)])
    return circuit.freeze()
