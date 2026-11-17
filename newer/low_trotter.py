"""Hamiltonian simulation: old low-rank, Pauli Trotter, and low-energy tuning.

The low-energy method in thesis Ch. 4 does NOT project away high-energy states.
It chooses a product-formula repetition count using ||(U_r-U_exact) P_A||.
P_A enlarges the initial energy window to accommodate ALL selected jumps.
This finite-instance check is exponential classical preprocessing; it is not a
scalable proof of accuracy over a whole dissipative trajectory. Operators are
unchanged. Contextual reduction, if requested, happens first in contextual.py.
"""
from dataclasses import dataclass
import time
import numpy as np
import scipy.linalg
import cirq
from openfermion.circuits import trotter
if __package__:
    from .filter import pauli_rotation
else:
    from filter import pauli_rotation


def pauli_formula(hamiltonian, time_s, steps=1, order=2):
    if steps < 1 or int(steps) != steps or order not in (1,2):
        raise ValueError('Use positive integer steps and order 1 or 2.')
    terms = list(cirq.PauliSum.wrap(hamiltonian))
    factors = terms if order == 1 else terms+terms[::-1]
    step = cirq.FrozenCircuit(pauli_rotation(p,time_s/(steps*order)) for p in factors)
    return cirq.FrozenCircuit(cirq.CircuitOperation(step).repeat(int(steps)))


def circuit_matrix(circuit, qubits):
    """Evaluate repeated blocks by matrix powers, without flattening the circuit."""
    memo = {}
    def evaluate(block):
        if block in memo:
            return memo[block]
        result = np.eye(2**len(qubits),dtype=complex)
        for op in block.all_operations():
            if isinstance(op,cirq.CircuitOperation) and not op.qubit_map and not op.param_resolver:
                matrix = np.linalg.matrix_power(evaluate(op.circuit),int(op.repetitions))
            else:
                matrix = cirq.Circuit(op).unitary(qubit_order=qubits)
            result = matrix@result
        memo[block] = result
        return result
    return evaluate(cirq.FrozenCircuit(circuit))


def _window_count(eigenvalues, count):
    count = max(1,min(int(count),len(eigenvalues)))
    while count < len(eigenvalues) and abs(eigenvalues[count]-eigenvalues[count-1]) < 1e-9:
        count += 1
    return count


def calibrate_low_energy(hamiltonian, jumps, qubits, rho, time_s, *,
                         sector_isometry=None, epsilon_state=.1, epsilon_jump=.1,
                         tolerance=1.6e-3, candidates=(1,2,4,8,16,32,64,128),
                         order=2, max_qubits=10):
    if len(qubits)>max_qubits:
        raise ValueError('Low-energy calibration exceeds max_dense_qubits. Use explicit '
                         'Pauli/low-rank steps for a larger circuit build; calibration is not scalable.')
    if not 0 <= epsilon_state < 1 or epsilon_jump < 0 or tolerance <= 0:
        raise ValueError('Invalid low-energy tolerances.')
    h = hamiltonian.matrix(qubits=qubits)
    sector = np.eye(len(h)) if sector_isometry is None else sector_isometry
    if np.linalg.norm(h@sector-sector@(sector.conj().T@h@sector))>1e-8:
        raise ValueError('The supplied sector is not invariant under H.')
    ev,u = np.linalg.eigh(sector.conj().T@h@sector)
    v = sector@u
    populations = np.real(np.diag(v.conj().T@rho@v))
    outside_sector = max(0.,1-float(populations.sum()))
    if outside_sector > epsilon_state**2+1e-9:
        raise ValueError('The input state has too much weight outside the selected energy sector.')
    count = _window_count(ev,np.searchsorted(np.cumsum(populations),1-epsilon_state**2)+1)
    initial = v[:,:count]
    jump_matrices = [j.operator.matrix(qubits=qubits) for j in jumps]
    enlarged_count = count
    while True:
        enlarged = v[:,:enlarged_count]
        leakage = max((np.linalg.norm(a@initial-enlarged@(enlarged.conj().T@a@initial),2)
                       for a in jump_matrices),default=0.)
        if leakage <= epsilon_jump+1e-12:
            break
        if enlarged_count == len(ev):
            raise ValueError('Jumps leave the supplied symmetry sector; enlarge the sector.')
        enlarged_count = _window_count(ev,enlarged_count+1)
    exact = scipy.linalg.expm(-1j*time_s*h)
    spaces = {'full':np.eye(len(h)), 'sector':v, 'initial_window':initial,
              'jump_window':enlarged}
    rows, minima = [], {name:None for name in spaces}
    selected_circuit = None
    for steps in sorted(set(int(x) for x in candidates)):
        circuit = pauli_formula(hamiltonian,time_s,steps,order)
        difference = circuit_matrix(circuit,qubits)-exact
        errors = {name:float(np.linalg.norm(difference@space,2)) for name,space in spaces.items()}
        rows.append(dict(steps=steps,**errors))
        for name,error in errors.items():
            if error <= tolerance and minima[name] is None:
                minima[name] = steps
        if selected_circuit is None and errors['jump_window'] <= tolerance:
            selected_circuit = circuit
        if all(value is not None for value in minima.values()):
            break
    if selected_circuit is None:
        raise ValueError('No candidate Trotter count meets the jump-window error tolerance; '
                         'increase low_energy_candidates.')
    terms = [p for p in hamiltonian if p]
    # Shift each Pauli term by |coefficient| I: h_X >= 0, ||h_X||=2|coefficient|.
    local_g = max((sum(2*abs(p.coefficient) for p in terms if q in p) for q in qubits),default=0.)
    diagnostics = dict(method='finite-instance restricted product-formula error',
        initial_window_size=count,jump_window_size=enlarged_count,
        initial_cutoff=float(ev[count-1]),jump_cutoff=float(ev[enlarged_count-1]),
        initial_width=float(ev[count-1]-ev[0]),jump_width=float(ev[enlarged_count-1]-ev[0]),
        input_leakage_amplitude=float(np.sqrt(max(0.,1-populations[:count].sum()))),
        jump_leakage=leakage,epsilon_state=epsilon_state,epsilon_jump=epsilon_jump,
        tolerance=tolerance,selected_steps=minima['jump_window'],minimum_tested_steps=minima,
        error_scan=rows,psd_shift=float(sum(abs(p.coefficient) for p in terms)),
        g=float(local_g),max_pauli_support=max((len(p) for p in terms),default=0),
        certification='one Hamiltonian interval on the recorded input window; not the whole W chain')
    return selected_circuit,enlarged,diagnostics


@dataclass
class Propagator:
    circuit: cirq.FrozenCircuit
    qubits: tuple
    diagnostics: dict
    window: object = None
    _matrix: object = None

    def matrix(self, max_qubits=10):
        if len(self.qubits)>max_qubits:
            raise ValueError('Propagator matrix exceeds max_dense_qubits.')
        if self._matrix is None:
            self._matrix = circuit_matrix(self.circuit,self.qubits)
        return self._matrix


def build_propagator(hamiltonian, qubits, time_s, *, method='low_rank', interaction=None,
                     steps=4, final_rank=4, order=2, jumps=None, rho=None,
                     max_qubits=10, **low_energy_options):
    """Return exp(-i H time_s); its inverse is used for the other W half."""
    start = time.perf_counter()
    diagnostics = dict(method=method,time=float(time_s),steps=steps,final_rank=final_rank)
    window = None
    if method == 'low_rank':
        if interaction is None:
            raise ValueError('Low-rank simulation needs the original two-body InteractionOperator. '
                             'After contextual reduction use pauli or low_energy.')
        if steps < 1 or int(steps) != steps:
            raise ValueError('steps must be a positive integer.')
        algorithm = trotter.LowRankTrotterAlgorithm(final_rank=final_rank)
        # Let OpenFermion track all intermediate permutations and restore the output order.
        circuit = cirq.FrozenCircuit(trotter.simulate_trotter(qubits,interaction,time=time_s,
            n_steps=int(steps),algorithm=algorithm,omit_final_swaps=False))
    elif method == 'pauli':
        circuit = pauli_formula(hamiltonian,time_s,steps,order)
    elif method == 'low_energy':
        if rho is None or jumps is None:
            raise ValueError('Low-energy tuning needs the actual input state and coupling family.')
        circuit,window,info = calibrate_low_energy(hamiltonian,jumps,qubits,rho,time_s,
                              order=order,max_qubits=max_qubits,**low_energy_options)
        diagnostics.update(info)
        diagnostics['steps'] = info['selected_steps']
    elif method == 'exact_matrix':
        if len(qubits)>max_qubits:
            raise ValueError('Exact matrix propagation exceeds max_dense_qubits.')
        matrix = scipy.linalg.expm(-1j*time_s*hamiltonian.matrix(qubits=qubits))
        circuit = cirq.FrozenCircuit(cirq.MatrixGate(matrix).on(*qubits)) if qubits else cirq.FrozenCircuit(
            cirq.global_phase_operation(matrix[0,0]))
        diagnostics['resource_counts_meaningful'] = False
    else:
        raise ValueError('method must be low_rank, pauli, low_energy or exact_matrix.')
    diagnostics['build_seconds'] = time.perf_counter()-start
    return Propagator(circuit,tuple(qubits),diagnostics,window)
