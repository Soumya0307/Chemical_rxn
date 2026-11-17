from dissipative_algorithms import hamiltonians
from dissipative_algorithms import one_ham_with_ckt

def build_circuit(mol):
    '''
        TODO: Parameterise
    '''
    hamiltonian = hamiltonians.HamiltonianBuilder.build_molecule(
        mol,
        basis="sto-3g",
        multiplicity=2,
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
    )

    # TODO: More parameterisation
    filter_terms = one_ham_with_ckt.SquareWaveFilter().get_coefficients(
            L = 2,
            K = 100,
            tau = 0.1,
            A = 1.0,
            window = "lanczos",
            p = 1,
            drop_zeros = False,
            eps = 1e-15,
            sign = +1
    )

    # TODO: Parameterise
    pairs = [(i, j) 
             for orbs in ([0, 1, 2], [3, 4, 5])
             for n, i in enumerate(orbs) 
             for j in orbs[n + 1:]
            ]
    weights = [1.0] * len(pairs)

    w_list, resources = one_ham_with_ckt.build_W_circuits_only(
        system_qubits=hamiltonian.system_qubits,
        H_int=hamiltonian.interaction,
        anc=hamiltonian.ancilla,
        tau=filter_terms["tau"],
        filter_terms=filter_terms["filter_terms"],
        T_filter=filter_terms["T_filter"],
        pairs=pairs,
        weights=weights,
        N_applications=100,
        reps_per_W=100,
        trotter_steps_H=1,
        rng_seed=123,
        final_rank=4,
        keep_circuits="all",
    )
    return w_list 
