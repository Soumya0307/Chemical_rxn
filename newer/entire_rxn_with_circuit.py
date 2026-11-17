#!/usr/bin/env python3
"""Reaction driver shared by both notebooks and HPC jobs.

Edit USER_CONFIG, or pass --config settings.json plus command-line overrides.
run_reaction(config) returns tables, circuits, states and diagnostics in memory.
plot_results(result) returns the same figures used by the notebooks. Nothing
is saved unless save_outputs is per_geometry or at_end. Use simulate=False
for large circuit/resource jobs: it deliberately does not invent energy data.

Dependencies: NumPy, SciPy, Cirq, OpenFermion, PySCF, matplotlib; pyLIQTR and
Rottnest for resources. Relative input paths are resolved against this file.
"""
from pathlib import Path
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
import pickle
import tempfile
import time
import numpy as np
import cirq
if __package__:
    from . import Hamiltonian, contextual, low_trotter, one_ham
    from .filter import SquareWaveFilter, make_jumps
else:
    import Hamiltonian, contextual, low_trotter, one_ham
    from filter import SquareWaveFilter, make_jumps


USER_CONFIG = {
    # Chemistry and path. Interleaved spin orbitals: alpha=0,2,...; beta=1,3,...
    'xyz_trajectory':'H3_rxn_trj.xyz', 'geometry_start':92, 'geometry_stop':133,
    'geometry_stride':1, 'include_final_geometry':True,
    'basis':'sto-3g', 'charge':0, 'multiplicity':2,
    'active_indices':None, 'n_active_electrons':None, 'n_active_orbitals':None,
    'use_casscf_orbitals':True, 'fix_spin_ss':None, 'ancilla_index':0,
    'align_orbitals':True, 'carry_state':True, 'max_geom_step':None,
    'displacement_min_bohr':None, 'displacement_max_bohr':None,
    'occ_spin_orbs':None, 'initial_state':'HF', 'reference_sector':'N',
    'conserve_spin':False, 'spin_adapted_jumps':False,
    'jump_pairs':None, 'jump_weights':None, 'max_jump_support':None,
    # Filter: L = factor*||H|| is the thesis choice; spectral_width is legacy.
    'filter_K':500, 'filter_window':'lanczos', 'filter_window_power':1,
    'filter_wmax_factor':2., 'filter_scale':'norm', 'filter_wmax':None,
    'spectral_method':'sparse', 'max_spectral_qubits':14,
    'reps_per_W':200, 'tau_W':.1, 'N_applications':400,
    'first_N_applications':None, 'w_convention':'centered', 'kick_steps':1,
    # Alternatives: low_rank (original), pauli, low_energy, exact_matrix.
    # Eight steps resolve the longer norm-based interval in the verified H3 run.
    # Set four with filter_scale='spectral_width' for the historical settings.
    'hamiltonian_simulation':'low_rank', 'trotter_steps_H':8, 'final_rank':4,
    'pauli_order':2, 'low_energy_epsilon_state':.1, 'low_energy_epsilon_jump':.1,
    'low_energy_tolerance':1.6e-3, 'low_energy_candidates':[1,2,4,8,16,32,64,128],
    'contextual':False, 'contextual_energy_tolerance':1.6e-3,
    'contextual_retained_qubits':None, 'contextual_max_nc_terms':128,
    # sampled_circuit caches exact channel blocks; direct_circuit checks Cirq.
    'evolution_mode':'sampled_circuit', 'rng_seed':123, 'simulate':True,
    'max_dense_qubits':10, 'energy_stride':10, 'energy_shots':0,
    'logical_resources':False, 'logical_circuit_precision':1e-3,
    'physical_resources':False, 'physical_timeout':1800,
    'physical_architecture':'Four Stage Superconducting', 'physical_layout':None,
    'save_outputs':'none', 'outdir':os.environ.get('TARGET_DIR','rxn_outputs'),
    'show_plots':False,
}


def small_config():
    """Fast two-frame H3 example for the logical notebook's pipeline checks.

    Small Fourier/update counts are a smoke test, not converged chemistry.
    Increase these in USER_CONFIG for production or a convergence study.
    """
    return dict(USER_CONFIG,geometry_start=111,geometry_stop=113,filter_K=2,
        reps_per_W=2,N_applications=4,trotter_steps_H=2,energy_stride=1,
        save_outputs='none',logical_resources=False)


def convergence_config():
    """Verified H3 frame-111 run from HF; exact original-H energy is the target.

    Uses the centered filter and the original rank-four low-rank option.
    Eight Hamiltonian-simulation steps resolve the longer norm-based filter
    interval better than four. No ground-state initialization or projection.
    This is an empirical one-geometry preset, not a guarantee for other inputs.
    """
    return dict(USER_CONFIG,geometry_start=111,geometry_stop=112,
        filter_K=500,reps_per_W=200,N_applications=1500,tau_W=.1,
        hamiltonian_simulation='low_rank',trotter_steps_H=8,final_rank=4,
        w_convention='centered',filter_scale='norm',initial_state='HF',
        evolution_mode='sampled_circuit',contextual=False,energy_stride=1,
        rng_seed=123,energy_shots=0,save_outputs='none')


def sector_basis(n_qubits, n_electrons, multiplicity, sector='N'):
    if sector == 'Fock': return np.eye(2**n_qubits)
    nalpha = (n_electrons+multiplicity-1)//2
    nbeta = n_electrons-nalpha
    indices = []
    for index in range(2**n_qubits):
        occupied = [i for i in range(n_qubits) if index & (1<<(n_qubits-1-i))]
        if len(occupied) != n_electrons: continue
        if sector == 'NaNb' and sum(i%2==0 for i in occupied)!=nalpha: continue
        indices.append(index)
    if not indices: raise ValueError('The requested particle/spin sector is empty.')
    basis = np.eye(2**n_qubits)[:,indices]
    if sector == 'singlet':
        from openfermion import s_squared_operator, jordan_wigner, get_sparse_operator
        s2 = get_sparse_operator(jordan_wigner(s_squared_operator(n_qubits//2)),n_qubits=n_qubits)
        ev,v = np.linalg.eigh(basis.conj().T@s2@basis)
        basis = basis@v[:,abs(ev)<1e-8]
        if not basis.shape[1]: raise ValueError('No singlets in the selected electron sector.')
    elif sector not in ('N','NaNb'):
        raise ValueError('reference_sector must be N, NaNb, singlet or Fock.')
    return basis


def _frames(config):
    path = Path(config['xyz_trajectory']).expanduser()
    if not path.is_absolute(): path = Path(__file__).parent/path
    frames = Hamiltonian.read_xyz(path)
    stride = config['geometry_stride']
    if stride<1: raise ValueError('geometry_stride must be positive.')
    start = config['geometry_start']
    stop = len(frames) if config['geometry_stop'] is None else config['geometry_stop']
    if not 0<=start<stop<=len(frames): raise ValueError('Invalid geometry slice.')
    indices = list(range(start,stop,stride))
    if config['include_final_geometry'] and indices[-1]!=stop-1: indices.append(stop-1)
    coords = np.array([[xyz for atom,xyz in g] for g in frames])
    displacement = np.r_[0,np.cumsum(np.linalg.norm(np.diff(coords,axis=0).reshape(len(frames)-1,-1),axis=1))]/.529177210903
    selected = []
    for i in indices:
        lo,hi = config['displacement_min_bohr'],config['displacement_max_bohr']
        if lo is not None and displacement[i]<lo: continue
        if hi is not None and displacement[i]>hi: continue
        if selected and config['max_geom_step'] is not None:
            limit = config['max_geom_step']
            if limit<=0: raise ValueError('max_geom_step must be positive.')
            previous = selected[-1][1]
            a = np.array([x for _,x in previous]); b = coords[i]
            subdivisions = int(np.ceil(np.max(np.linalg.norm(b-a,axis=1))/limit))
            previous_index = selected[-1][0]
            for j in range(1,subdivisions):
                t = j/subdivisions
                geometry = [(atom,tuple(x)) for (atom,_),x in zip(frames[i],a+t*(b-a))]
                selected.append((previous_index+t*(i-previous_index),geometry))
        selected.append((i,frames[i]))
    if not selected: raise ValueError('No geometries remain after selection.')
    return path,selected


def run_reaction(config=None):
    supplied = {} if config is None else dict(config)
    unknown = set(supplied)-set(USER_CONFIG)
    if unknown: raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    c = dict(USER_CONFIG,**supplied)
    if c['save_outputs'] not in ('none','per_geometry','at_end'):
        raise ValueError('save_outputs must be none, per_geometry or at_end.')
    if c['contextual'] and c['hamiltonian_simulation']=='low_rank':
        raise ValueError('After contextual projection use pauli or low_energy; '
                         'a generic reduced Pauli sum is not an electronic two-body Hamiltonian.')
    if not c['simulate'] and c['hamiltonian_simulation']=='low_energy':
        raise ValueError('Automatic low-energy calibration requires a state and exact small-system reference.')
    if c['carry_state'] and not c['align_orbitals']:
        raise ValueError('Carrying states along this driver requires aligned orbital coordinates.')
    path,frames = _frames(c)
    result = dict(config=c,summary=[],energy_rows=[],geometries=[],logical=[],logical_total=None,physical=[],
        metadata=dict(input_path=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            data_source_commit='2a200b7e3f899ce8d1ad642748139f88807d28fd',
            energy_units='Hartree',save_outputs=c['save_outputs'],complete=False,
            resource_scope='unitary W sequence; preparation, ancilla reset and energy measurement excluded'))
    reference,full_rho,cumulative = None,None,0
    try:
        for geometry_index,(frame_index,geometry) in enumerate(frames):
            start = time.perf_counter()
            chemistry_keys = ['basis','multiplicity','charge','active_indices','ancilla_index',
                'n_active_electrons','n_active_orbitals','use_casscf_orbitals','fix_spin_ss']
            data = Hamiltonian.build_molecule(geometry,**{k:c[k] for k in chemistry_keys},
                use_procrustes_gauge=c['align_orbitals'],align_reference=reference)
            reference = data.reference
            jumps = make_jumps(data.system_qubits,c['jump_pairs'],conserve_spin=c['conserve_spin'],
                               spin_adapted=c['spin_adapted_jumps'],max_jump_support=c['max_jump_support'])
            basis,target,h = None,None,None
            if c['simulate']:
                if data.n_system>c['max_dense_qubits']:
                    raise ValueError(f'{data.n_system} system qubits exceeds max_dense_qubits. '
                        'Use simulate=False for circuits/resources; classical energy plots need a state simulator.')
                h = data.qubit.matrix(qubits=data.system_qubits)
                basis = sector_basis(data.n_system,data.n_electrons,c['multiplicity'],c['reference_sector'])
                energies,vectors = np.linalg.eigh(basis.conj().T@h@basis)
                target = float(energies[0])
                if full_rho is None or not c['carry_state']:
                    if c['initial_state']=='ground':
                        state = basis@vectors[:,0]
                    elif c['initial_state']=='HF':
                        na = (data.n_electrons+c['multiplicity']-1)//2
                        nb = data.n_electrons-na
                        occupied = c['occ_spin_orbs']
                        if occupied is None: occupied = list(range(0,2*na,2))+list(range(1,2*nb,2))
                        if len(set(occupied))!=data.n_electrons or any(i<0 or i>=data.n_system for i in occupied):
                            raise ValueError('HF occupations do not match the active electron count.')
                        index = sum(1<<(data.n_system-1-i) for i in occupied)
                        state = np.eye(1,2**data.n_system,index).ravel().astype(complex)
                    else: raise ValueError('initial_state must be HF or ground.')
                    full_rho = np.outer(state,state.conj())
                if full_rho.shape!=h.shape: raise ValueError('Active-space dimension changed along the path.')
            reduction = contextual.identity_reduction(data.qubit,jumps,data.system_qubits)
            if c['contextual']:
                reduction = contextual.reduce_operators(data.qubit,jumps,data.system_qubits,
                    energy_tolerance=c['contextual_energy_tolerance'],reference_energy=target,
                    sector_isometry=basis,retained_qubits=c['contextual_retained_qubits'],
                    max_qubits=c['max_dense_qubits'],max_nc_terms=c['contextual_max_nc_terms'])
            rho,v,projected_sector,projection_weight = None,None,None,1.
            if c['simulate']:
                v = reduction.isometry(c['max_dense_qubits'])
                rho = v.conj().T@full_rho@v
                projection_weight = float(np.trace(rho).real)
                if projection_weight<1e-12:
                    raise ValueError('The contextual subspace is orthogonal to the carried state; '
                                     'retain more contextual qubits or disable reduction.')
                rho /= projection_weight
                gram = v.conj().T@basis@basis.conj().T@v
                ev,vec = np.linalg.eigh(gram)
                projected_sector = vec[:,ev>1-1e-8]
            effective,qubits = reduction.hamiltonian,reduction.qubits
            lo,hi = Hamiltonian.spectral_range(effective,qubits,method=c['spectral_method'],
                                               max_qubits=c['max_spectral_qubits'])
            if c['filter_scale'] not in ('norm','spectral_width'):
                raise ValueError('filter_scale must be norm or spectral_width.')
            scale = max(abs(lo),abs(hi)) if c['filter_scale']=='norm' else hi-lo
            wmax = c['filter_wmax'] if c['filter_wmax'] is not None else c['filter_wmax_factor']*scale
            if wmax+1e-10<hi-lo:
                raise ValueError('Filter L is narrower than the verified Bohr-frequency range.')
            coefficients = SquareWaveFilter().get_coefficients(wmax,c['filter_K'],c['tau_W'],
                window=c['filter_window'],p=c['filter_window_power'])
            options = dict(sector_isometry=projected_sector,epsilon_state=c['low_energy_epsilon_state'],
                epsilon_jump=c['low_energy_epsilon_jump'],tolerance=c['low_energy_tolerance'],
                candidates=c['low_energy_candidates'])
            propagator = low_trotter.build_propagator(effective,qubits,-coefficients['T_filter'],
                method=c['hamiltonian_simulation'],interaction=data.interaction if not c['contextual'] else None,
                steps=c['trotter_steps_H'],final_rank=c['final_rank'],order=c['pauli_order'],
                jumps=reduction.jumps,rho=rho,max_qubits=c['max_dense_qubits'],**options)
            factory = one_ham.WFactory(qubits,data.ancilla,reduction.jumps,propagator,
                coefficients['filter_terms'],c['tau_W'],c['reps_per_W'],c['w_convention'],
                c['kick_steps'],c['max_dense_qubits'])
            applications = c['first_N_applications'] if geometry_index==0 and c['first_N_applications'] is not None else c['N_applications']
            seed = c['rng_seed']+geometry_index
            if c['simulate']:
                chain = one_ham.run_chain(factory,effective,rho,applications=applications,
                    mode=c['evolution_mode'],seed=seed,weights=c['jump_weights'],
                    energy_stride=c['energy_stride'],energy_shots=c['energy_shots'])
                full_rho = v@chain['rho']@v.conj().T
                initial_energy = float(np.trace(effective.matrix(qubits=qubits)@rho).real)
                final_energy = float(np.trace(h@full_rho).real)
                overlap = float(np.trace(rho@chain['rho']).real)
                sector_weight = float(np.trace(basis.conj().T@full_rho@basis).real)
                for row in chain['energies']:
                    result['energy_rows'].append(dict(row,hamiltonian_index=frame_index,
                        cumulative_W_applications=cumulative+row['step'],target_energy=target))
            else:
                sequence,probabilities = one_ham.sample_sequence(len(jumps),applications,seed,c['jump_weights'])
                chain = dict(sequence=sequence,probabilities=probabilities,
                    circuits={i:factory.circuit(i) for i in set(sequence)},timings=factory.timings,
                    rho=None,diagnostics={})
                initial_energy=final_energy=overlap=sector_weight=None
            if c['logical_resources']:
                blocks = one_ham.selected_block_counts(factory,chain['sequence'])
                logical = one_ham.pyliqtr_resources(blocks,circuit_precision=c['logical_circuit_precision'])
                result['logical'].append(dict(hamiltonian_index=frame_index,precision_scope='this geometry',**logical))
            if c['physical_resources']:
                # Hardware scheduling needs order; these are references to
                # cached W blocks, never a flattened reaction circuit.
                sequence_circuits = [factory.circuit(i) for i in chain['sequence']]
                physical = one_ham.physical_resources(sequence_circuits,
                    architecture=c['physical_architecture'],layout=c['physical_layout'],
                    timeout=c['physical_timeout'])
                result['physical'].append(dict(hamiltonian_index=frame_index,**physical))
            cumulative += applications
            summary = dict(hamiltonian_index=frame_index,geometry_index=geometry_index,
                initial_energy=initial_energy,final_energy=final_energy,target_energy=target,
                final_energy_error=None if target is None else final_energy-target,
                initial_final_overlap=overlap,sector_weight=sector_weight,
                original_qubits=data.n_system,retained_qubits=len(qubits),
                contextual_projection_weight=projection_weight,W_applications=applications,
                unique_W_blocks=len(factory.circuits),jump_family_size=len(jumps),
                filter_L=wmax,T_filter=coefficients['T_filter'],
                trotter_steps=propagator.diagnostics['steps'],chemistry_seconds=data.chemistry_seconds,
                propagator_build_seconds=propagator.diagnostics['build_seconds'],
                **chain['timings'],total_seconds=time.perf_counter()-start)
            result['summary'].append(summary)
            result['geometries'].append(dict(frame_index=frame_index,hamiltonian=data,reduction=reduction,
                factory=factory,chain=chain,initial_rho=rho,filter=coefficients,
                measurement_circuits=one_ham.energy_measurement_circuits(effective,qubits) if c['energy_shots'] else [],
                diagnostics=dict(alignment=data.alignment,contextual=reduction.diagnostics,
                                 propagation=propagator.diagnostics,channel=chain['diagnostics'])))
            print(f'Frame {frame_index}: {len(qubits)} qubits, {applications} W, '
                  f'{len(factory.circuits)} cached blocks, energy={final_energy}, '
                  f'{summary["total_seconds"]:.2f} s',flush=True)
            if c['save_outputs']=='per_geometry': save_results(result)
        if c['logical_resources']:
            # A separate total uses ONE synthesis-error budget for the path;
            # summing separately budgeted per-geometry T estimates is incorrect.
            blocks = Counter()
            for g in result['geometries']:
                blocks.update(one_ham.selected_block_counts(g['factory'],g['chain']['sequence']))
            result['logical_total'] = one_ham.pyliqtr_resources(
                blocks,circuit_precision=c['logical_circuit_precision'])
        result['metadata']['complete'] = True
    except Exception as error:
        result['metadata']['error'] = f'{type(error).__name__}: {error}'
        if c['save_outputs']!='none': save_results(result)
        raise
    if c['save_outputs']!='none': save_results(result)
    return result


def plot_results(result, show=True):
    """The four original plots plus timings and enabled reduction diagnostics."""
    import matplotlib.pyplot as plt
    rows = result['summary']; figures = {}
    if not rows: return figures
    xs = [r['hamiltonian_index'] for r in rows]
    def figure(name,ylabel,title):
        fig,ax = plt.subplots(figsize=(8,4))
        ax.set(xlabel='Hamiltonian / original geometry index',ylabel=ylabel,title=title)
        ax.grid(alpha=.25); figures[name] = fig
        return fig,ax
    if result['config']['simulate']:
        fig,ax = figure('energy_vs_hamiltonian_index','Energy (Hartree)','Energy along the reaction path')
        for key,label,marker in [('initial_energy','Before W','v'),('final_energy','After W','o'),('target_energy','Reference ground','x')]:
            ax.plot(xs,[r[key] for r in rows],marker=marker,label=label)
        ax.legend();fig.tight_layout()
        fig,ax = figure('energy_error_vs_hamiltonian_index','Absolute energy error (mHa)','Final energy error')
        ax.semilogy(xs,[max(abs(r['final_energy_error'])*1000,1e-8) for r in rows],'-o')
        ax.axhline(1.6,ls=':',color='black',label='Chemical accuracy (1.6 mHa)');ax.legend();fig.tight_layout()
        fig,ax = figure('overlap_vs_hamiltonian_index',r'Re Tr($\rho_{in}\rho_{out}$)','Initial/final density overlap')
        ax.plot(xs,[r['initial_final_overlap'] for r in rows],'-o');fig.tight_layout()
        fig,ax = figure('energy_vs_w_applications','Energy (Hartree)','Energy during the W chain')
        ax.set_xlabel('Cumulative W applications')
        for x in xs:
            selected = [r for r in result['energy_rows'] if r['hamiltonian_index']==x]
            positions = [r['cumulative_W_applications'] for r in selected]
            ax.errorbar(positions,[r['energy'] for r in selected],
                        yerr=[r['energy_standard_error'] for r in selected],marker='.',label=f'H {x}')
            ax.hlines(selected[0]['target_energy'],min(positions),max(positions),linestyle=':',alpha=.6)
        ax.legend(fontsize=7,ncol=min(4,len(xs)));fig.tight_layout()
    fig,ax = figure('timing_vs_hamiltonian_index','Wall time (s)','Measured classical run time')
    for key in ['chemistry_seconds','propagator_build_seconds','kraus_build_seconds','channel_apply_seconds']:
        ax.plot(xs,[r.get(key,0.) for r in rows],'-o',label=key.replace('_seconds','').replace('_',' '))
    ax.legend();fig.tight_layout()
    if result['config']['contextual']:
        fig,ax = figure('contextual_reduction','System qubits','Selected contextual subspaces')
        ax.plot(xs,[r['original_qubits'] for r in rows],label='Original')
        ax.plot(xs,[r['retained_qubits'] for r in rows],'-o',label='Retained');ax.legend();fig.tight_layout()
    if result['config']['hamiltonian_simulation']=='low_energy':
        fig,ax = figure('low_energy_trotter','Restricted operator error','Low-energy product-formula calibration')
        ax.set_xlabel('Trotter repetitions')
        for g in result['geometries']:
            scan = g['diagnostics']['propagation']['error_scan']
            for name in ['full','sector','initial_window','jump_window']:
                ax.loglog([r['steps'] for r in scan],[max(r[name],1e-16) for r in scan],
                          marker='.',label=f'H {g["frame_index"]}: {name}')
        ax.axhline(result['config']['low_energy_tolerance'],ls=':',color='black')
        ax.legend(fontsize=7);fig.tight_layout()
    if result['logical']:
        fig,ax = figure('logical_resources','Estimated T count','pyLIQTR: all W applications')
        ax.plot(xs,[r['T'] for r in result['logical']],'-o');fig.tight_layout()
    if result.get('physical'):
        fig,ax = figure('physical_resources','Runtime (Rottnest tocks)','Rottnest: unitary W sequence')
        ax.plot(xs,[r['hardware']['tocks']['total'] for r in result['physical']],'-o');fig.tight_layout()
    if show: plt.show()
    return figures


def save_results(result):
    """Explicit opt-in only; write atomic tables/checkpoints and familiar PNGs."""
    import matplotlib.pyplot as plt
    root = Path(result['config']['outdir']);root.mkdir(parents=True,exist_ok=True)
    def atomic(name,writer,mode='w'):
        with tempfile.NamedTemporaryFile(mode=mode,dir=root,delete=False) as f:
            writer(f);temporary = f.name
        os.replace(temporary,root/name)
    for name,rows in [('energy_vs_hamiltonian_index.csv',result['summary']),
                      ('energy_vs_w_applications.csv',result['energy_rows']),
                      ('overlap_vs_hamiltonian_index.csv',result['summary'])]:
        def write(f,rows=rows):
            if rows:
                keys = list(dict.fromkeys(k for row in rows for k in row))
                w = csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
        atomic(name,write)
    meta = dict(result['metadata'],config=result['config'],logical=result['logical'],
                logical_total=result['logical_total'],physical=result['physical'],
                diagnostics=[g['diagnostics'] for g in result['geometries']])
    atomic('run_metadata.json',lambda f:json.dump(meta,f,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
    atomic('all_W_circuits_by_geometry.pkl',lambda f:pickle.dump(
        [[g['factory'].circuit(i) for i in g['chain']['sequence']] for g in result['geometries']],f),'wb')
    atomic('final_rhos_by_geometry.pkl',lambda f:pickle.dump([g['chain']['rho'] for g in result['geometries']],f),'wb')
    for name,fig in plot_results(result,show=False).items():
        fig.savefig(root/(name+'.png'),dpi=160);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',help='JSON object overriding USER_CONFIG')
    parser.add_argument('--small-example',action='store_true')
    parser.add_argument('--convergence-example',action='store_true')
    for key,default in USER_CONFIG.items():
        option = '--'+key.replace('_','-')
        if isinstance(default,bool): parser.add_argument(option,dest=key,action=argparse.BooleanOptionalAction,default=None)
        else: parser.add_argument(option,dest=key,default=None)
    args = vars(parser.parse_args())
    small,convergence = args.pop('small_example'),args.pop('convergence_example')
    if small and convergence: parser.error('Choose either --small-example or --convergence-example.')
    config = convergence_config() if convergence else small_config() if small else dict(USER_CONFIG)
    file = args.pop('config')
    if file: config.update(json.loads(Path(file).read_text()))
    for key,value in args.items():
        if value is not None:
            if isinstance(value,str):
                try: value = json.loads(value)
                except json.JSONDecodeError: pass
            config[key] = value
    result = run_reaction(config)
    print(json.dumps({key:result[key] for key in ['summary','logical','logical_total','physical']},indent=2))
    if config['show_plots']: plot_results(result)


if __name__=='__main__': main()
