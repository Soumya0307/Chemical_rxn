"""One Hamiltonian: build W once per coupling, then reuse its circuit/channel.

averaged_matrix applies the weighted average of the finite W channels;
sampled_circuit samples ONE coupling per W and uses cached Cirq-equivalent
Kraus matrices; direct_circuit runs those same choices through Cirq with an
ancilla reset. Sampling refers to the coupling choice, not to energy shots.
All density-matrix modes are exponential classical benchmarks. build-only
circuits and logical resources do not require density matrices.

The default 'centered' convention implements the explicit conjugated Fourier
kicks in the thesis. 'legacy' reproduces the old repo's shifted W frame.
Low-rank, filter and sampling choices must match before comparing results.
"""
from dataclasses import dataclass, field
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import hashlib
import importlib
import json
import os
import sys
import time
import types
import numpy as np
import cirq
if __package__:
    from . import Hamiltonian
    from .filter import SquareWaveFilter, kick, make_jumps
    from .low_trotter import build_propagator, circuit_matrix
else:
    import Hamiltonian
    from filter import SquareWaveFilter, kick, make_jumps
    from low_trotter import build_propagator, circuit_matrix


@dataclass
class WFactory:
    qubits: tuple
    ancilla: cirq.Qid
    jumps: list
    propagator: object                 # exp(+i H T), i.e. time_s=-T
    filter_terms: list
    tau: float = .1
    cutoff: object = None
    convention: str = 'centered'
    kick_steps: int = 1
    max_qubits: int = 10
    circuits: dict = field(default_factory=dict)
    channels: dict = field(default_factory=dict)
    timings: dict = field(default_factory=lambda:dict(circuit_build_seconds=0.,kraus_build_seconds=0.))

    def __post_init__(self):
        self.qubits = tuple(self.qubits)
        if self.tau < 0 or self.convention not in ('centered','legacy'):
            raise ValueError('Require tau >= 0 and centered or legacy convention.')
        terms = [(int(k),complex(c)) for k,c in self.filter_terms
                 if self.cutoff is None or abs(k)<=self.cutoff]
        ks = [k for k,c in terms]
        if not ks or ks != list(range(ks[0],ks[-1]+1)) or ks[0] != -ks[-1]:
            raise ValueError('W needs the complete symmetric Fourier grid, including zero terms.')
        self.terms = terms
        self._kick_cache = {}
        self._prop_operations = {}
        self._kick_operations = {}

    def _kick(self, index, coefficient):
        key = index,coefficient
        if key not in self._kick_cache:
            self._kick_cache[key] = kick(self.ancilla,self.jumps[index].operator,
                np.sqrt(self.tau)*abs(coefficient)/2,np.angle(coefficient),
                steps=self.kick_steps,legacy_phase=self.convention=='legacy')
        return self._kick_cache[key]

    def _schedule(self):
        """Chronological (propagator power, coefficient) instructions."""
        if self.convention == 'legacy':
            for k,c in self.terms:
                if c: yield 'kick',c
                yield 'prop',1
            for k,c in self.terms[::-1]:
                yield 'prop',-1
                if c: yield 'kick',c
        else:
            modes = self.terms+self.terms[::-1]
            previous = modes[0][0]
            yield 'prop',-previous
            for k,c in modes:
                if previous != k: yield 'prop',previous-k
                if c: yield 'kick',c
                previous = k
            yield 'prop',previous

    def circuit(self, index):
        if index not in self.circuits:
            start = time.perf_counter()
            operations = []
            for kind,value in self._schedule():
                if kind == 'prop' and value:
                    if value not in self._prop_operations:
                        self._prop_operations[value] = cirq.CircuitOperation(self.propagator.circuit).repeat(value)
                    operations.append(self._prop_operations[value])
                elif kind == 'kick':
                    key = index,value
                    if key not in self._kick_operations:
                        self._kick_operations[key] = cirq.CircuitOperation(self._kick(index,value))
                    operations.append(self._kick_operations[key])
            self.circuits[index] = cirq.FrozenCircuit(operations)
            self.timings['circuit_build_seconds'] += time.perf_counter()-start
        return self.circuits[index]

    def kraus(self, index):
        if len(self.qubits)>self.max_qubits:
            raise ValueError('Kraus simulation exceeds max_dense_qubits; use simulate=False.')
        if index not in self.channels:
            start = time.perf_counter()
            self.circuit(index)
            d = 2**len(self.qubits)
            u = self.propagator.matrix(self.max_qubits)
            powers, kicks = {1:u,-1:u.conj().T,0:np.eye(d)},{}
            # Only the ancilla-|0> input columns are needed for the channel.
            blocks = np.vstack([np.eye(d,dtype=complex),np.zeros((d,d),complex)])
            operator = self.jumps[index].operator
            terms = list(operator)
            commuting = all(cirq.commutes(a,b) for i,a in enumerate(terms) for b in terms[i+1:])
            if commuting:
                eigenvalues,eigenvectors = np.linalg.eigh(operator.matrix(qubits=self.qubits))
            for kind,value in self._schedule():
                if kind == 'prop':
                    if value not in powers:
                        powers[value] = np.linalg.matrix_power(u,int(value))
                    blocks = np.vstack([powers[value]@blocks[:d],powers[value]@blocks[d:]])
                else:
                    if value not in kicks:
                        if commuting:
                            theta = np.sqrt(self.tau)*abs(value)/2
                            phi = np.angle(value)*(1 if self.convention=='legacy' else -1)
                            cosine = (eigenvectors*np.cos(theta*eigenvalues))@eigenvectors.conj().T
                            sine = (eigenvectors*(-1j*np.sin(theta*eigenvalues)))@eigenvectors.conj().T
                            kicks[value] = np.block([[cosine,np.exp(1j*phi)*sine],
                                                     [np.exp(-1j*phi)*sine,cosine]])
                        else:
                            kicks[value] = circuit_matrix(self._kick(index,value),
                                                          (self.ancilla,)+self.qubits)
                    blocks = kicks[value]@blocks
            k0,k1 = blocks[:d],blocks[d:]
            defect = np.linalg.norm(k0.conj().T@k0+k1.conj().T@k1-np.eye(d),2)
            if defect>1e-8:
                raise RuntimeError(f'W channel failed trace-preservation check: {defect:.3g}')
            self.channels[index] = k0,k1
            self.timings['kraus_build_seconds'] += time.perf_counter()-start
        return self.channels[index]


def sample_sequence(n_jumps, applications, seed=123, weights=None):
    if applications < 0 or int(applications) != applications:
        raise ValueError('N_applications must be a nonnegative integer.')
    weights = np.ones(n_jumps) if weights is None else np.asarray(weights,float)
    if weights.shape != (n_jumps,) or np.any(weights<0) or not np.all(np.isfinite(weights)) or weights.sum()<=0:
        raise ValueError('Supply one finite nonnegative weight per jump with positive sum.')
    weights = weights/weights.sum()
    # Retain the old seed+step convention for reproducible compatibility runs.
    sequence = [int(np.random.default_rng(seed+i).choice(n_jumps,p=weights)) for i in range(applications)]
    return sequence,weights


def energy_measurement_circuits(hamiltonian, qubits):
    """Independent Pauli readout circuits (basis change followed by measurement)."""
    measurements = []
    for index,term in enumerate(hamiltonian):
        if not term: continue
        support = [q for q in qubits if q in term]
        rotations = []
        for q in support:
            if term[q] == cirq.X: rotations.append(cirq.H(q))
            elif term[q] == cirq.Y: rotations.extend([cirq.S(q)**-1,cirq.H(q)])
        basis = cirq.FrozenCircuit(rotations)
        measurement = cirq.FrozenCircuit(basis,cirq.measure(*support,key=f'energy_{index}'))
        measurements.append((float(term.coefficient.real),support,basis,measurement))
    return measurements


def measure_energy(hamiltonian, qubits, rho, shots=0, seed=0):
    """Independent Pauli measurements on copies; never collapse the carried state.

    shots is the number PER nonconstant Pauli term. The reported standard
    error assumes independent measurement batches. shots=0 gives exact energy.
    """
    if not shots:
        return float(np.trace(hamiltonian.matrix(qubits=qubits)@rho).real),0.
    if shots < 2 or int(shots) != shots:
        raise ValueError('Use at least two energy shots per Pauli term, or zero for exact energy.')
    rng = np.random.RandomState(seed)
    energy = sum(float(term.coefficient.real) for term in hamiltonian if not term)
    variance = 0.
    simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)
    for coefficient,support,basis,measurement in energy_measurement_circuits(hamiltonian,qubits):
        rotated = simulator.simulate(basis,initial_state=rho,qubit_order=qubits).final_density_matrix
        samples = cirq.sample_density_matrix(rotated,indices=[qubits.index(q) for q in support],
                                             repetitions=int(shots),seed=rng)
        estimate = float(np.mean(1-2*(samples.sum(axis=1)%2).astype(float)))
        energy += coefficient*estimate
        variance += coefficient**2*(1-estimate**2)/(shots-1)
    return energy,float(np.sqrt(max(0.,variance)))


def run_chain(factory, hamiltonian, initial_rho, *, applications=10,
              mode='sampled_circuit', seed=123, weights=None, energy_stride=1,
              energy_shots=0, sequence=None):
    if len(factory.qubits)>factory.max_qubits:
        raise ValueError('Density-matrix simulation exceeds max_dense_qubits.')
    if mode not in ('sampled_circuit','direct_circuit','averaged_matrix'):
        raise ValueError('Unknown evolution mode.')
    choices,probabilities = sample_sequence(len(factory.jumps),applications,seed,weights)
    if sequence is not None:
        if len(sequence)!=applications or any(i<0 or i>=len(factory.jumps) for i in sequence):
            raise ValueError('Invalid explicit jump sequence.')
        choices = list(sequence)
    rho = np.asarray(initial_rho,dtype=complex).copy()
    d = 2**len(factory.qubits)
    if rho.shape!=(d,d) or abs(np.trace(rho)-1)>1e-8 or np.linalg.norm(rho-rho.conj().T)>1e-8:
        raise ValueError('initial_rho must be a normalized Hermitian density matrix.')
    if np.linalg.eigvalsh(rho)[0]<-1e-8:
        raise ValueError('initial_rho must be positive semidefinite.')
    used = range(len(factory.jumps)) if mode=='averaged_matrix' else sorted(set(choices))
    for i in used:
        factory.circuit(i)
        if mode != 'direct_circuit': factory.kraus(i)
    simulator = cirq.DensityMatrixSimulator(dtype=np.complex128,seed=seed)
    direct = {}
    if mode == 'direct_circuit':
        # Cirq 1.7's density simulator has a buffer bug on nested global-phase
        # operations. Elementary unitary gates have the identical channel.
        for i in used:
            direct[i] = cirq.Circuit(op for op in cirq.decompose(factory.circuit(i)) if op.qubits)
            direct[i].append(cirq.reset(factory.ancilla))
    rows,apply_seconds,energy_seconds = [],0.,0.
    energy_matrix = hamiltonian.matrix(qubits=factory.qubits) if not energy_shots else None
    def record(step):
        nonlocal energy_seconds
        start = time.perf_counter()
        if energy_matrix is not None:
            energy,error = float(np.trace(energy_matrix@rho).real),0.
        else:
            energy,error = measure_energy(hamiltonian,factory.qubits,rho,energy_shots,seed+step)
        row = dict(step=step,energy=energy,energy_standard_error=error,
                   trace=float(np.trace(rho).real))
        if factory.propagator.window is not None:
            v = factory.propagator.window
            row['low_energy_leakage_probability'] = max(0.,1-float(np.trace(v.conj().T@rho@v).real))
        rows.append(row)
        energy_seconds += time.perf_counter()-start
    record(0)
    for step,index in enumerate(choices,1):
        start = time.perf_counter()
        if mode == 'direct_circuit':
            circuit = direct[index]
            total = np.zeros((2*d,2*d),complex)
            total[:d,:d] = rho
            output = simulator.simulate(circuit,initial_state=total,
                                        qubit_order=(factory.ancilla,)+factory.qubits)
            rho = output.final_density_matrix[:d,:d]
        elif mode == 'sampled_circuit':
            rho = sum(k@rho@k.conj().T for k in factory.kraus(index))
        else:
            rho = sum(p*sum(k@rho@k.conj().T for k in factory.kraus(i))
                      for i,p in enumerate(probabilities) if p)
        apply_seconds += time.perf_counter()-start
        if step == applications or (energy_stride and step%energy_stride==0): record(step)
    return dict(rho=rho,energies=rows,sequence=choices,probabilities=probabilities,
        circuits={i:factory.circuit(i) for i in used},mode=mode,
        timings={**factory.timings,'channel_apply_seconds':apply_seconds,'energy_seconds':energy_seconds},
        diagnostics=dict(trace=float(np.trace(rho).real),min_eigenvalue=float(np.linalg.eigvalsh(rho)[0]),
            max_hermiticity_error=float(np.max(abs(rho-rho.conj().T)))))


def pyliqtr_resources(circuits, *, circuit_precision=1e-3):
    """pyLIQTR counts on unique leaves, multiplied by their actual occurrences.

    Accepts either a sequence of block references or {FrozenCircuit: count}.
    The latter needs no list of W applications and no full reaction circuit.
    One precision budget is shared by ALL rotations in the supplied sequence.
    Includes unitary W only; reset, preparation and measurement are reported
    separately by the driver. Does not mistake a cached block for one gate.
    """
    if not 0<circuit_precision<1:
        raise ValueError('circuit_precision must lie between zero and one.')
    # Compatibility aliases for this installation of Qualtran/pyLIQTR.
    if not hasattr(np,'complex_'): np.complex_ = np.complex128
    from pyLIQTR.utils.resource_analysis import estimate_resources, get_T_counts_from_rotations
    memo,leaf_memo = {},{}
    def count(circuit):
        frozen = circuit if isinstance(circuit,cirq.FrozenCircuit) else cirq.FrozenCircuit(circuit)
        if frozen in memo: return memo[frozen]
        total = Counter(T=0,Clifford=0,Rotations=0)
        for op in frozen.all_operations():
            if isinstance(op,cirq.CircuitOperation):
                # Qubit renaming does not change a gate count. Resolve angles
                # on the UNIQUE child, never mapped_circuit() on its repeats.
                if cirq.is_parameterized(op.repetitions) or op.repeat_until is not None:
                    raise ValueError('Logical counting requires a fixed integer repetition count.')
                child_block = (cirq.resolve_parameters(op.circuit,op.param_resolver)
                               if op.param_resolver else op.circuit)
                child = count(child_block)
                multiplier = abs(int(op.repetitions))
                for key in total: total[key] += multiplier*child[key]
            elif op.qubits:
                if cirq.is_measurement(op) or isinstance(op.gate,cirq.ResetChannel):
                    raise ValueError('Pass unitary circuits; report measurements/resets separately.')
                if isinstance(op.gate,cirq.MatrixGate):
                    raise ValueError('Dense benchmark MatrixGates are not a scalable logical resource model.')
                if op not in leaf_memo:
                    leaf_memo[op] = estimate_resources(cirq.Circuit(op),profile=True)
                result = leaf_memo[op]
                for key in total: total[key] += result[key]
        memo[frozen] = total
        return total
    # Combine selected jump multiplicities BEFORE visiting any W circuit.
    multiplicities = Counter()
    entries = circuits.items() if isinstance(circuits,Mapping) else ((c,1) for c in circuits)
    for circuit,multiplicity in entries:
        if int(multiplicity)!=multiplicity or multiplicity<0:
            raise ValueError('Block multiplicities must be nonnegative integers.')
        if multiplicity:
            frozen = circuit if isinstance(circuit,cirq.FrozenCircuit) else cirq.FrozenCircuit(circuit)
            multiplicities[frozen] += int(multiplicity)
    total = Counter(T=0,Clifford=0,Rotations=0)
    qubits = set()
    for circuit,multiplicity in multiplicities.items():
        qubits.update(circuit.all_qubits())
        for key,value in count(circuit).items(): total[key] += multiplicity*value
    synthesized = get_T_counts_from_rotations(total['Rotations'],circuit_precision=circuit_precision) if total['Rotations'] else 0
    return dict(LogicalQubits=len(qubits),T=int(total['T']+synthesized),
        Clifford=int(total['Clifford']+(2*synthesized+1 if synthesized else 0)),
        Rotations=int(total['Rotations']),native_T=int(total['T']),
        circuit_precision=circuit_precision,
        per_rotation_precision=circuit_precision/total['Rotations'] if total['Rotations'] else None,
        unique_counted_blocks=len(memo),scope='unitary W sequence',
        unique_profiled_operations=len(leaf_memo),unique_W_blocks=len(multiplicities),
        W_applications=sum(multiplicities.values()),
        T_method='pyLIQTR synthesis estimate; not an explicitly synthesized circuit')


def selected_block_counts(factory, sequence):
    """One W reference per selected jump, with its sampled multiplicity."""
    blocks = Counter()
    for index,multiplicity in Counter(sequence).items():
        blocks[factory.circuit(index)] += multiplicity
    return blocks


def build_W_circuits_only(system_qubits,H_int,anc,tau,filter_terms,pairs,weights,
                         N_applications,reps_per_W=None,trotter_steps_H=1,rng_seed=123,
                         T_filter=None,final_rank=4,keep_circuits='unique'):
    """Compatibility entry point for the unchanged owner-supplied executable."""
    if T_filter is None: raise ValueError('T_filter is required.')
    from openfermion import jordan_wigner,get_fermion_operator
    h = Hamiltonian.to_cirq(jordan_wigner(get_fermion_operator(H_int)),system_qubits)
    p = build_propagator(h,system_qubits,-T_filter,interaction=H_int,
                        steps=trotter_steps_H,final_rank=final_rank)
    factory = WFactory(tuple(system_qubits),anc,make_jumps(system_qubits,pairs),p,
                       filter_terms,tau,reps_per_W,convention='legacy')
    sequence,_ = sample_sequence(len(pairs),N_applications,rng_seed,weights)
    unique = list(dict.fromkeys(sequence))
    for i in unique: factory.circuit(i)
    if keep_circuits == 'all': circuits = [factory.circuit(i) for i in sequence]
    elif keep_circuits == 'unique': circuits = [factory.circuit(i) for i in unique]
    elif keep_circuits == 'first': circuits = [factory.circuit(sequence[0])] if sequence else []
    elif keep_circuits == 'none': circuits = []
    else: raise ValueError('Invalid keep_circuits option.')
    return circuits,[dict(index=s,seed=rng_seed+s,pair=pairs[i],qubits=len(system_qubits)+1,
                          compressed=True) for s,i in enumerate(sequence)]


def register_legacy_imports():
    """Explicitly connect the untouched executables to these six new modules."""
    existing = sys.modules.get('dissipative_algorithms')
    if existing is not None and not getattr(existing,'_newer_compatibility',False):
        raise RuntimeError('An older dissipative_algorithms package is already loaded; use a fresh process.')
    package = existing or types.ModuleType('dissipative_algorithms')
    package.__path__ = [str(Path(__file__).parent)]
    package._newer_compatibility = True
    package.hamiltonians = Hamiltonian
    package.one_ham_with_ckt = sys.modules[__name__]
    sys.modules['dissipative_algorithms'] = package
    sys.modules['dissipative_algorithms.hamiltonians'] = Hamiltonian
    sys.modules['dissipative_algorithms.one_ham_with_ckt'] = sys.modules[__name__]
    return importlib.import_module('dissipative_algorithms.executables.simple_molecules')


# Rottnest adapter. Kept here so the owner-supplied executables stay byte-identical.
class CachedBlock(cirq.Gate):
    def __init__(self,circuit,key):
        self.circuit = cirq.FrozenCircuit(circuit)
        self.qubits = tuple(sorted(circuit.all_qubits()))
        self.key = key
    def _num_qubits_(self): return len(self.qubits)
    def _decompose_(self,qubits):
        return self.circuit.unfreeze().transform_qubits(dict(zip(self.qubits,qubits))).all_operations()
    def _rottnest_hash(self,operation): return bytes.fromhex(operation.gate.key)


def _native_circuit(circuit):
    """Rewrite leaf gates to the gate set understood by this Rottnest parser."""
    def supported(op):
        gate = op.gate
        if isinstance(gate,(cirq.CZPowGate,cirq.CXPowGate,cirq.HPowGate)):
            return float(gate.exponent)%2==1 and gate.global_shift==0
        return isinstance(gate,(cirq.XPowGate,cirq.YPowGate,cirq.ZPowGate))
    result = []
    for op in cirq.decompose(circuit,keep=supported,on_stuck_raise=None):
        if not op.qubits: continue
        if cirq.is_measurement(op) or isinstance(op.gate,cirq.ResetChannel):
            raise ValueError('Rottnest adapter accepts unitary W blocks; reset/readout require a hardware model.')
        if supported(op): result.append(op);continue
        expanded = cirq.optimize_for_target_gateset(cirq.Circuit(op),gateset=cirq.CZTargetGateset())
        for sub in expanded.all_operations():
            if not sub.qubits: continue
            if supported(sub): result.append(sub)
            elif len(sub.qubits)==1:
                result.extend((p**t).on(sub.qubits[0]) for p,t in
                              cirq.single_qubit_matrix_to_pauli_rotations(cirq.unitary(sub),atol=1e-10))
            else: raise ValueError(f'Unsupported Rottnest operation: {sub}')
    return cirq.Circuit(result)


def _resource_manifest(circuits):
    records,seen,memo,child_keys = [],set(),{},{}
    def leaf(circuit):
        payload = cirq.to_json(_native_circuit(circuit))
        key = hashlib.sha256(('newer-leaf-v1'+payload).encode()).hexdigest()
        if key not in seen:
            records.append(dict(key=key,json=payload));seen.add(key)
        return key
    def register(circuit):
        frozen = circuit if isinstance(circuit,cirq.FrozenCircuit) else cirq.FrozenCircuit(circuit)
        if frozen in memo: return memo[frozen]
        children,pending = [],[]
        def flush():
            if pending: children.append([leaf(cirq.Circuit(pending)),1]);pending.clear()
        for op in frozen.all_operations():
            if isinstance(op,cirq.CircuitOperation):
                flush()
                if op.qubit_map or op.param_resolver:
                    children.append([register(op.mapped_circuit(deep=False)),1])
                else:
                    # The same propagator/inverse appears thousands of times.
                    # Register each unique signed child once, keeping repeat
                    # counts as integers instead of rebuilding its operations.
                    key = op.circuit,op.repetitions<0
                    if key not in child_keys:
                        child = op.circuit if op.repetitions>=0 else cirq.inverse(op.circuit)
                        child_keys[key] = register(child)
                    children.append([child_keys[key],abs(int(op.repetitions))])
            else: pending.append(op)
        flush()
        key = hashlib.sha256(('newer-group-v1'+json.dumps(children)).encode()).hexdigest()
        if key not in seen:
            records.append(dict(key=key,children=children));seen.add(key)
        memo[frozen] = key
        return key
    sequence = [register(c) for c in circuits]
    return dict(blocks=records,sequence=sequence)


def __getattr__(name):
    if name != 'rottnest_executables': raise AttributeError(name)
    if not hasattr(np,'complex_'): np.complex_ = np.complex128
    if not hasattr(cirq.ops.op_tree,'OpTree'): cirq.ops.op_tree.OpTree = cirq.OP_TREE
    from rottnest.executables.t_rz_executable import T_RZ_RottnestExecutable
    from rottnest.input_parsers.pyliqtr_parser import update_tracking_targets
    from rottnest.monkey_patchers.decomposition_targets import add_cirq_hash,load_hash_patcher
    add_cirq_hash(CachedBlock,lambda _,op:bytes.fromhex(op.gate.key))
    load_hash_patcher();update_tracking_targets({CachedBlock})
    class ReactionExecutable(T_RZ_RottnestExecutable):
        @staticmethod
        def get_name(): return 'Newer reaction circuit blocks'
        @staticmethod
        def get_parameters(): return {'manifest':(str,'')}
        def _generate_circuit(self):
            manifest = json.loads(Path(self.manifest).read_text())
            blocks = {}
            for item in manifest['blocks']:
                if 'json' in item: circuit = cirq.read_json(json_text=item['json'])
                else:
                    circuit = cirq.Circuit(blocks[key].on(*blocks[key].qubits)
                        for key,count in item['children'] for _ in range(count))
                blocks[item['key']] = CachedBlock(circuit,item['key'])
            return cirq.Circuit(blocks[key].on(*blocks[key].qubits) for key in manifest['sequence'])
    ReactionExecutable.__qualname__ = 'ReactionExecutable'
    globals()['ReactionExecutable'] = ReactionExecutable
    globals()['rottnest_executables'] = [ReactionExecutable]
    return globals()['rottnest_executables']


def _poll_rottnest(pool):
    """Keep one status request outstanding while draining cached-block reports."""
    from rottnest.process_pool import commands
    from rottnest.process_pool.ipc_manager import IPCManager
    pool.manager_priority_task_queue.put((commands.POLL,))
    while True:
        status = pool.ipc.get_item(commands.POLL,blocking=True)
        if status is not IPCManager.NOT_FOUND:
            return status
        # Rottnest fetches at most five messages per call. Sending another POLL
        # before its reply can fill both queues on large cached hierarchies.
        pool.flush_results_cache()


def _rottnest_worker(job_path):
    """One isolated Rottnest pool; used by physical_resources below."""
    from rottnest.plugins import executables,architectures
    from rottnest.procedures import preprocess_and_execute
    from rottnest.process_pool import get_pool
    from rottnest.compute_units.layout_proxy import LayoutProxy
    from rottnest.error_budgets import get_error_budget
    from t_scheduler.region_builder.json_to_region import example
    job = json.loads(Path(job_path).read_text())
    executables.load_modules_from_strings('one_ham')
    executables.set_current_executable('Newer reaction circuit blocks')
    executables.set_executable_params(**(executables.get_executable_params()|{'manifest':job['manifest']}))
    architectures.set_current_architecture(job['architecture'])
    layout = job['layout'] or example
    LayoutProxy.add_layout_with_id(0,layout)
    pool = get_pool()
    pool.poll = lambda: _poll_rottnest(pool)
    start = time.perf_counter()
    procedure = preprocess_and_execute.PreprocAndExecuteProcedure(reporting=False)
    try:
        procedure.execute()
        while not procedure.complete(): procedure.poll()
        pre = procedure.preprocessor
        hardware = procedure.pool_procedure.get_results.get_results()
        if hasattr(hardware,'to_runchart'): hardware = hardware.to_runchart()
        result = dict(architecture=job['architecture'],rz_count=pre.get_rz_count(),
            t_count=pre.get_t_count(),t_infidelity=pre.get_t_infidelity(),hardware=hardware,
            seconds=time.perf_counter()-start,
            assumptions=dict(layout=layout,physical_error_rate=get_error_budget().get_p_physical(),
                target_error=get_error_budget().get_target_error(),
                rz_error_budget=get_error_budget().get_rz_precision_budget()))
        print('NEWER_ROTTNEST_RESULT '+json.dumps(result,default=str),flush=True)
    finally: get_pool().terminate()


def physical_resources(circuits, *, architecture='Four Stage Superconducting',layout=None,
                       timeout=1800,verbose=False):
    """Rottnest hardware schedule for the SAME W sequence used by the simulation.

    Temporary worker interchange files are removed. Returned hardware values
    retain Rottnest's units (tocks and region space-time volume); no undocumented
    conversion to seconds or physical qubits is made. Resets/readout/preparation
    are excluded, so this is the unitary portion of the physical estimate.
    """
    import signal
    import subprocess
    import tempfile
    manifest = _resource_manifest(circuits)
    if not manifest['sequence']: raise ValueError('No circuits to estimate.')
    with tempfile.TemporaryDirectory(prefix='newer_rottnest_') as folder:
        folder = Path(folder)
        path = folder/'blocks.json';path.write_text(json.dumps(manifest))
        job = folder/'job.json';job.write_text(json.dumps(dict(manifest=str(path),architecture=architecture,layout=layout)))
        # Spawned legacy Qualtran workers import these aliases before our module.
        (folder/'sitecustomize.py').write_text("import numpy as np\nif not hasattr(np,'complex_'): np.complex_=np.complex128\nimport cirq\nif not hasattr(cirq.ops.op_tree,'OpTree'): cirq.ops.op_tree.OpTree=cirq.OP_TREE\n")
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(folder),str(Path(__file__).parent),env.get('PYTHONPATH','')])
        sdk = '/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk'
        if Path(sdk).exists(): env.setdefault('SDKROOT',sdk)
        code = ('import sys,os,traceback; import one_ham; '
                'one_ham._rottnest_worker(sys.argv[1]); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)')
        with tempfile.TemporaryFile(mode='w+') as log:
            process = subprocess.Popen([sys.executable,'-c',code,str(job)],env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out = False
            try: process.wait(timeout=timeout)
            except subprocess.TimeoutExpired: timed_out = True
            finally:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            log.seek(0);output = log.read()
        if verbose: print(output)
        if timed_out: raise TimeoutError(f'Rottnest exceeded {timeout}s.\n'+output[-6000:])
        for line in output.splitlines():
            if line.startswith('NEWER_ROTTNEST_RESULT '):
                result = json.loads(line.split(' ',1)[1])
                result.update(W_applications=len(circuits),unique_cached_blocks=len(manifest['blocks']),
                    scope='unitary W sequence; excludes state preparation, reset and measurement',
                    circuit_manifest_sha256=hashlib.sha256(json.dumps(manifest).encode()).hexdigest())
                return result
        raise RuntimeError(f'Rottnest failed with exit code {process.returncode}.\n'+output[-6000:])
