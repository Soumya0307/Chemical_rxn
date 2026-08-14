#!/usr/bin/env python3
"""
LinLin reaction-path circuit builder, WITH per-geometry energy validation.

This is your existing driver with one addition: after the W circuits are built
at each geometry, the same W blocks are evolved as a channel on a density matrix
and the final energy is compared with the exact ground energy in the fixed-N_e
sector. One number per geometry, plotted at the end.

The circuit-construction path is unchanged. Pass --no-simulate to get exactly
the old behaviour.

WHAT THE ENERGY CHECK ACTUALLY TESTS
    - that W is a valid channel (Kraus completeness, trace preservation)
    - that the sampled jump operators conserve particle number
    - that the dissipative chain lowers the energy toward E_0 in the fixed-N_e
      sector, i.e. that the protocol cools rather than merely producing circuits

CONTINUATION
    rho at the end of geometry i is carried to geometry i+1 as the warm start.
    Only geometry 0 starts from the reference determinant. This is the actual
    continuation algorithm; without the carry-forward each geometry would be an
    independent cooling run.

    NOTE: the carry-forward is only meaningful if the orbital gauge is aligned
    along the path. With --no-align-orbitals the Hamiltonians at successive
    geometries are expressed in unrelated CASSCF orbital frames and the
    transported rho is not comparable between them. Use --align-orbitals for
    any multi-geometry run.
"""

from __future__ import annotations
import argparse, inspect, json, os, pickle, sys, tempfile, time, traceback
from dataclasses import dataclass, asdict, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import one_ham_with_ckt as linlin
import linlin_energy_check as echk          # <-- added

Geometry = List[Tuple[str, Tuple[float, float, float]]]

USER_CONFIG = {
    "xyz_trajectory": None,
    "geometry_start": -1, "geometry_stop": None, "geometry_stride": 1,
    "include_final_geometry": True,

    "basis": "def2-SVP", "charge": 0, "multiplicity": 1,
    "active_indices": None, "ancilla_index": 0,
    "n_active_electrons": 4, "n_active_orbitals": 4,
    "use_casscf_orbitals": True, "fix_spin_ss": 0,
    "max_geom_step": None, "align_orbitals": False,

    "n_electrons": 4, "occ_spin_orbs": None,
    "spin_adapted_jumps": True, "conserve_spin": False,

    "tau_filter": 0.1, "filter_K": 200, "filter_window": "lanczos",
    "filter_wmax_factor": 2.0, "filter_wmax": None,   # None -> diagonalise
    "use_global_filter": False,
    "tau_W": 0.1, "N_applications": 200, "reps_per_W": 200,
    "trotter_steps_H": 4, "rng_seed": 123, "keep_circuits": "unique",

    # ---- added: energy validation --------------------------------------
    "compress_circuits": False,   # store the repeated propagator once (CircuitOperation)
    "analytic_resources": False,  # count W without building it (constant memory)
    "t_synthesis_eps": 1e-10,     # Ross-Selinger precision for the T estimate
    "simulate": True,          # run the density-matrix check per geometry
    "sim_max_qubits": 10,      # refuse to build 2^(1+n) x 2^(1+n) beyond this
    "sim_carry_rho": True,     # continuation: carry rho to the next geometry
    "plot": True,              # write energy_error_by_geometry.png/.pdf
    "chem_acc": 1.6e-3,        # Hartree, for the plot reference line
    # --------------------------------------------------------------------

    "outdir": os.environ.get("TARGET_DIR", "rxn_linlin_circuits"),
}

# ---------------------------------------------------------------- atomic I/O
def _json_default(x):
    if isinstance(x, np.ndarray):          return x.tolist()
    if isinstance(x, np.integer):          return int(x)
    if isinstance(x, np.floating):         return float(x)
    if isinstance(x, (np.complexfloating, complex)):
        return {"real": float(np.real(x)), "imag": float(np.imag(x))}
    return str(x)

def _atomic(write_fn, path: Path, mode="wb", **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode, delete=False, dir=str(path.parent),
                                     suffix=".tmp", **kw) as tmp:
        write_fn(tmp); tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)

def atomic_pickle(obj, path: Path):
    _atomic(lambda f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL), path)

def atomic_json(obj, path: Path):
    _atomic(lambda f: json.dump(obj, f, indent=2, default=_json_default),
            path, mode="w", encoding="utf-8")

# ------------------------------------------------------------------ geometry
def resolve_xyz_trajectory(spec: Optional[str]) -> Path:
    cands: List[Path] = []
    if spec:
        raw = Path(spec).expanduser()
        if raw.exists():
            return raw.resolve()
        cands += [Path(x) for x in sorted(glob(str(raw)))]
    seen = set(cands)
    for pat in ("*_trj.xyz", "*_TRJ.xyz"):
        for m in sorted(glob(pat)):
            p = Path(m)
            if p not in seen and p.is_file():
                seen.add(p); cands.append(p)
    cands = [p for p in cands if p.is_file()]
    if len(cands) == 1:
        return cands[0].resolve()
    if not cands:
        raise FileNotFoundError("No '*_trj.xyz' here; pass --xyz-trajectory.")
    raise FileExistsError("Multiple candidates: " + ", ".join(map(str, cands)))

def load_xyz_trajectory(path: Path) -> List[Geometry]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    geoms, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1; continue
        try:
            natoms = int(s)
        except ValueError as exc:
            raise ValueError(f"Expected atom count at line {i+1}: {s!r}") from exc
        i += 2
        g: Geometry = []
        for _ in range(natoms):
            parts = lines[i].split()
            if len(parts) < 4:
                raise ValueError(f"Malformed atom line {i+1}: {lines[i]!r}")
            g.append((parts[0], tuple(float(v) for v in parts[1:4]))); i += 1
        geoms.append(g)
    if not geoms:
        raise ValueError(f"No frames in {path}")
    return geoms

def refine_geometry_path(geoms, idxs, max_step):
    if max_step is None:
        return list(geoms), list(idxs), [False] * len(geoms)
    out_g, out_i, out_f = [geoms[0]], [int(idxs[0])], [False]
    for a in range(1, len(geoms)):
        ga = np.array([xyz for _, xyz in geoms[a-1]], float)
        gb = np.array([xyz for _, xyz in geoms[a]], float)
        syms = [s for s, _ in geoms[a-1]]
        nsub = max(1, int(np.ceil(float(np.abs(gb-ga).max()) / float(max_step))))
        for s in range(1, nsub):
            gi = ga + (gb - ga) * (s / nsub)
            out_g.append([(syms[k], tuple(gi[k])) for k in range(len(syms))])
            out_i.append(int(idxs[a-1])); out_f.append(True)
        out_g.append(geoms[a]); out_i.append(int(idxs[a])); out_f.append(False)
    return out_g, out_i, out_f

# --------------------------------------------------------------------- parse
def parse_int_list(s):
    if s is None: return None
    s = str(s).strip()
    return None if (not s or s.lower() == "none") else [int(x) for x in s.split(",") if x.strip()]

def parse_opt_int(s):
    return None if s is None or str(s).lower() in {"none", ""} else int(s)

def parse_opt_float(s):
    return None if s is None or str(s).lower() in {"none", ""} else float(s)

# ---------------------------------------------------------------- jump pairs
def make_pairs_and_weights(n_system, conserve_spin=False, spin_adapted=False):
    n_system = int(n_system); pairs: List[Any] = []
    if spin_adapted:
        for p in range(n_system // 2):
            for q in range(p + 1, n_system // 2):
                legs = ((2*p, 2*q), (2*p+1, 2*q+1))
                pairs += [(legs, "Ax"), (legs, "Ay")]
    else:
        for i in range(n_system):
            for j in range(i + 1, n_system):
                if conserve_spin and (i % 2 != j % 2): continue
                pairs += [(i, j, "Ax"), (i, j, "Ay")]
    if not pairs:
        raise ValueError(f"No jump pairs (n_system={n_system}).")
    return pairs, [1.0 / len(pairs)] * len(pairs)

# -------------------------------------------------------------------- filter
def compute_filter(spectral_width, tau_filter, filter_K, window, wmax_factor, wmax_override):
    if wmax_override is not None:
        wmax, width = float(wmax_override), None
    else:
        if spectral_width is None:
            raise ValueError(
                "No spectral width is available. Pass --filter-wmax, or let "
                "HamiltonianBuilder compute Ham.spectral_width."
            )
        width = float(spectral_width)
        wmax = wmax_factor * max(width, 1e-12)
    data = linlin.SquareWaveFilter().get_coefficients(
        L=wmax, K=filter_K, tau=tau_filter, A=1.0, window=window,
        drop_zeros=False,   # keep even-k zeros: preserves the uniform time grid
        sign=+1,
    )
    data.update(wmax=wmax, spectral_width=width, tau_filter=tau_filter,
                filter_K=filter_K, filter_window=window,
                filter_wmax_factor=wmax_factor)
    return data

# --------------------------------------------------------- circuit resources
def circuit_resources(circ) -> Dict[str, Any]:
    try:
        ops = list(circ.all_operations())
        hist: Dict[str, int] = {}
        for op in ops:
            g = getattr(op, "gate", None)
            hist[type(g).__name__ if g is not None else "unknown"] = \
                hist.get(type(g).__name__ if g is not None else "unknown", 0) + 1
        return {"n_qubits": len(circ.all_qubits()), "depth_moments": len(circ),
                "n_operations": len(ops), "gate_histogram": hist}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

def aggregate(res_list):
    tot = {"n_qubits": 0, "depth_moments": 0, "n_operations": 0,
           "n_rotations": 0, "t_count": 0, "gate_histogram": {}}
    for r in res_list:
        if "error" in r: continue
        n_qubits = r.get("n_qubits", r.get("qubits", 0))
        depth = r.get("depth_moments", r.get("moments", 0))
        n_ops = r.get("n_operations", r.get("operations", 0))
        tot["n_qubits"] = max(tot["n_qubits"], n_qubits)   # width = max
        tot["depth_moments"] += depth                      # depth accumulates
        tot["n_operations"] += n_ops
        tot["n_rotations"] += r.get("rotations", 0)
        tot["t_count"] += r.get("t_count", 0)
        for k, v in r.get("gate_histogram", {}).items():
            tot["gate_histogram"][k] = tot["gate_histogram"].get(k, 0) + v
    return tot

# ----------------------------------------------------------- circuit builder
_BUILDER_NAMES = (
    "build_W_circuits_only",
    "build_W_circuits",
    "build_W_chain_circuits",
    "build_W_circuit",
    "make_W_circuit",
    "build_W",
    "W_circuit",
)

def _resolve_builder():
    for name in _BUILDER_NAMES:
        fn = getattr(linlin, name, None)
        if callable(fn):
            return name, fn, False
    fn = getattr(linlin, "run_W_chain_circuit", None)
    if callable(fn):
        return "run_W_chain_circuit", fn, True
    raise RuntimeError(
        "No circuit builder found in one_ham_with_ckt. Looked for: "
        + ", ".join(_BUILDER_NAMES) + ", run_W_chain_circuit."
    )

def build_circuits(**kw):
    """Return (W circuits, resource rows if supplied by the builder)."""
    name, fn, is_chain = _resolve_builder()
    accepted = set(inspect.signature(fn).parameters)
    call = {k: v for k, v in kw.items() if k in accepted}
    missing = [k for k in kw if k not in accepted]
    if missing:
        print(f"[builder] {name} ignores: {', '.join(sorted(missing))}", flush=True)
    out = fn(**call)
    if is_chain:
        print("[builder] WARNING: fell back to run_W_chain_circuit, which also "
              "simulates. Expose a single-W builder to skip that.", flush=True)
        out = out[2] if isinstance(out, tuple) and len(out) >= 3 else out
        return (out if isinstance(out, list) else [out]), None
    if isinstance(out, tuple):
        circuits = out[0]
        resources = out[1] if len(out) > 1 else None
        return circuits if isinstance(circuits, list) else [circuits], resources
    return out if isinstance(out, list) else [out], None

# ------------------------------------------------- jump sequence for the sim
def pair_sequence_from_resources(resources, pairs, weights, N_applications, rng_seed):
    """The jump pair used at each application, in order.

    Preferred source is the builder's own resource rows, so the simulated chain
    uses exactly the operators the counted circuits used. Falls back to
    reproducing the builder's sampling rule if the rows are unavailable.
    """
    if resources:
        rows = [r for r in resources if "pair" in r and "index" in r]
        if len(rows) == N_applications:
            rows.sort(key=lambda r: r["index"])
            return [r["pair"] for r in rows]
        print("[sim] builder resources lack a usable pair sequence; "
              "reproducing the sampling rule instead", flush=True)
    w = np.asarray(weights, dtype=float); w = w / w.sum()
    return [pairs[np.random.default_rng(rng_seed + t).choice(len(pairs), p=w)]
            for t in range(N_applications)]

# --------------------------------------------------------------------- record
@dataclass
class GeometryRecord:
    hamiltonian_index: int
    original_geometry_index: int
    n_system: int
    n_W_applications: int
    n_W_circuits: int
    n_filter_terms: int
    wmax: float
    T_filter: float
    n_jump_pairs: int
    keep_circuits: str
    resources_total: Dict[str, Any]

    # ---- added: energy validation (None when --no-simulate) ----
    initial_energy: Optional[float] = None
    final_energy: Optional[float] = None
    E_fixed_N: Optional[float] = None
    E_full_fock: Optional[float] = None
    energy_error: Optional[float] = None
    energy_error_mHa: Optional[float] = None
    gap_fixed_N: Optional[float] = None
    sector_dimension: Optional[int] = None
    spectral_width: Optional[float] = None
    final_particle_number: Optional[float] = None
    trace_drift: Optional[float] = None
    kraus_completeness_error: Optional[float] = None
    sim_seconds: Optional[float] = None
    sim_skipped_reason: Optional[str] = None

# ----------------------------------------------------------------------- run
def run(args):
    outdir = Path(args.outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    n_electrons = parse_opt_int(args.n_electrons)
    user_occ = parse_int_list(args.occ_spin_orbs)
    if user_occ is None and n_electrons is None:
        raise ValueError("Pass --occ-spin-orbs or --n-electrons.")

    xyz = resolve_xyz_trajectory(args.xyz_trajectory)
    all_geoms = load_xyz_trajectory(xyz)

    start = int(args.geometry_start)
    if start < 0: start += len(all_geoms)
    start = max(0, start)
    stop = len(all_geoms) if args.geometry_stop is None else min(len(all_geoms), int(args.geometry_stop))
    if stop <= start:
        raise ValueError(f"Bad range: start={start} stop={stop} total={len(all_geoms)}")
    stride = max(1, int(args.geometry_stride))
    idxs = list(range(start, stop, stride))
    if args.include_final_geometry and (stop - 1) not in idxs: idxs.append(stop - 1)
    idxs = sorted(set(idxs))
    geoms = [all_geoms[i] for i in idxs]

    max_step = parse_opt_float(args.max_geom_step)
    geoms, idxs, interp = refine_geometry_path(geoms, idxs, max_step)
    if sum(interp):
        print(f"[path-refine] +{sum(interp)} frames -> {len(geoms)} total", flush=True)

    if len(geoms) == 1:
        print(f"[warn] ONE geometry selected (original index {idxs[0]} of "
              f"{len(all_geoms)}). This is a single-point resource estimate, "
              f"not a continuation run.", flush=True)
    if len(geoms) > 1 and not bool(args.align_orbitals):
        print("[warn] multi-geometry run with --no-align-orbitals: the orbital "
              "gauge is not propagated, so the transported rho is not "
              "comparable between geometries.", flush=True)

    meta: Dict[str, Any] = {
        "linlin_path": str(Path(linlin.__file__).resolve()),
        "geometry_source": str(xyz), "n_input_geometries": len(all_geoms),
        "n_used_geometries": len(geoms), "selected_original_indices": idxs,
        "interpolated_flags": interp, "args": vars(args), "status": "running",
    }
    atomic_json(meta, outdir / "run_metadata.json")

    circuits_by_geometry: List[List[Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    replay_by_geometry: List[Dict[str, Any]] = []   # ordered W sequence per geometry
    align_prev = None
    filter_cache = None
    rho_carry = None                                   # <-- continuation state
    run_t0 = time.time()

    try:
        for h_idx, (orig, geom) in enumerate(zip(idxs, geoms)):
            t0 = time.time()
            print(f"\n===== geometry {h_idx}/{len(geoms)-1} | original {orig} =====", flush=True)

            Ham = linlin.HamiltonianBuilder().build_molecule(
                geometry=geom, basis=args.basis, multiplicity=args.multiplicity,
                charge=args.charge, active_indices=parse_int_list(args.active_indices),
                ancilla_index=args.ancilla_index,
                n_active_electrons=parse_opt_int(args.n_active_electrons),
                n_active_orbitals=parse_opt_int(args.n_active_orbitals),
                use_casscf_orbitals=bool(args.use_casscf_orbitals),
                use_procrustes_gauge=bool(args.align_orbitals),
                align_reference=align_prev,
                fix_spin_ss=parse_opt_float(args.fix_spin_ss),
            )
            if bool(args.align_orbitals):
                align_prev = {"mol": Ham.scf_mol, "mo_coeff": Ham.mo_coeff, "mo_occ": Ham.mo_occ}

            n_system = int(Ham.n_system)
            occ = user_occ if user_occ is not None else list(range(n_electrons))
            if max(occ, default=-1) >= n_system:
                raise ValueError(f"occ_spin_orbs={occ} vs n_system={n_system}")

            if args.use_global_filter and filter_cache is not None:
                fdata = filter_cache
            else:
                fdata = compute_filter(
                    getattr(Ham, "spectral_width", None), args.tau_filter, args.filter_K,
                    args.filter_window, args.filter_wmax_factor,
                    parse_opt_float(args.filter_wmax),
                )
                if args.use_global_filter: filter_cache = fdata

            pairs, weights = make_pairs_and_weights(
                n_system, bool(args.conserve_spin), bool(args.spin_adapted_jumps))

            print(f"n_system={n_system}  wmax={fdata['wmax']:.6f}  "
                  f"T_filter={fdata['T_filter']:.6f}  terms={len(fdata['filter_terms'])}  "
                  f"pairs={len(pairs)}", flush=True)

            if bool(args.analytic_resources):
                # W is the propagator inlined 2*n_active times plus the kicks,
                # and cirq packs those templates without merging across the
                # boundaries, so the counts are exactly additive. Counting them
                # that way is constant in memory; building W is not. Required
                # beyond ~12 qubits, where a single W exhausts the node.
                W_circuits = []
                builder_resources = echk.analytic_resource_rows(
                    ancilla=Ham.ancilla, system_qubits=Ham.system_qubits,
                    H_int=Ham.interaction, tau=args.tau_W,
                    filter_terms=fdata["filter_terms"], T_filter=fdata["T_filter"],
                    pair_sequence=pair_sequence_from_resources(
                        None, pairs, weights,
                        int(args.N_applications), int(args.rng_seed) + h_idx,
                    ),
                    reps_per_W=args.reps_per_W,
                    trotter_steps_H=args.trotter_steps_H,
                    eps_synth=float(args.t_synthesis_eps),
                )
            elif bool(args.compress_circuits):
                # Real circuits, but the propagator is stored once as a
                # FrozenCircuit and referenced 2*n_active times instead of being
                # copied. Unroll with echk.unroll_W() to recover the flat form.
                W_circuits, builder_resources = echk.build_W_circuits_compressed(
                    ancilla=Ham.ancilla, system_qubits=Ham.system_qubits,
                    H_int=Ham.interaction, tau=args.tau_W,
                    filter_terms=fdata["filter_terms"], T_filter=fdata["T_filter"],
                    pair_sequence=pair_sequence_from_resources(
                        None, pairs, weights,
                        int(args.N_applications), int(args.rng_seed) + h_idx,
                    ),
                    reps_per_W=args.reps_per_W,
                    trotter_steps_H=args.trotter_steps_H,
                    eps_synth=float(args.t_synthesis_eps),
                    keep_circuits=args.keep_circuits,
                )
            else:
                W_circuits, builder_resources = build_circuits(
                    system_qubits=Ham.system_qubits, H_int=Ham.interaction, anc=Ham.ancilla,
                    tau=args.tau_W, filter_terms=fdata["filter_terms"],
                    pairs=pairs, weights=weights,
                    N_applications=args.N_applications, reps_per_W=args.reps_per_W,
                    occ_spin_orbs=occ, trotter_steps_H=args.trotter_steps_H,
                    rng_seed=args.rng_seed + h_idx, T_filter=fdata["T_filter"],
                    keep_circuits=args.keep_circuits,
                )

            res = builder_resources if builder_resources is not None else [
                circuit_resources(c) for c in W_circuits
            ]
            total = aggregate(res)

            # The ordered replay sequence. Unique circuits alone are NOT enough
            # to reconstruct the chain: W blocks do not commute, so total
            # multiplicities do not determine the algorithm. Store which unique
            # block is applied at each of the N_applications steps.
            seq_pairs = pair_sequence_from_resources(
                builder_resources, pairs, weights,
                int(args.N_applications), int(args.rng_seed) + h_idx,
            )
            uniq_pairs, seq_idx = echk.sequence_and_unique(seq_pairs)
            if W_circuits and args.keep_circuits == "unique" \
                    and len(W_circuits) != len(uniq_pairs):
                raise RuntimeError(
                    f"replay index mismatch: {len(W_circuits)} stored circuits "
                    f"vs {len(uniq_pairs)} unique pairs. The pickle and the "
                    f"sequence file would not line up."
                )
            replay_by_geometry.append({
                "hamiltonian_index": h_idx,
                "original_geometry_index": int(orig),
                "n_applications": int(args.N_applications),
                "unique_pairs": [repr(p) for p in uniq_pairs],
                "sequence": seq_idx,
            })
            print(f"[replay] {len(seq_idx)} applications over "
                  f"{len(uniq_pairs)} unique blocks; order recorded", flush=True)

            # ---------------- energy validation ---------------------------
            sim: Dict[str, Any] = {}
            if not bool(args.simulate):
                sim["sim_skipped_reason"] = "--no-simulate"
            elif n_system > int(args.sim_max_qubits):
                sim["sim_skipped_reason"] = (
                    f"n_system={n_system} > sim_max_qubits={args.sim_max_qubits}"
                )
                print(f"[sim] skipped: {sim['sim_skipped_reason']}", flush=True)
            else:
                ts = time.time()
                H_sparse = echk.system_hamiltonian_sparse(Ham.qubit, n_system)

                Na = (n_electrons + (int(args.multiplicity) - 1)) // 2
                Nb = n_electrons - Na
                refs = echk.reference_energies(
                    Ham.qubit, n_system, n_electrons=n_electrons,
                    Na=Na, Nb=Nb, H_sparse=H_sparse,
                )
                print(f"[ref] E_fixed_N({n_electrons}) = {refs['E_fixed_N']:.10f} Ha  "
                      f"(sector dim {refs['sector_dimension']}, "
                      f"gap {refs['gap_fixed_N']:.6f})", flush=True)

                seq = pair_sequence_from_resources(
                    builder_resources, pairs, weights,
                    int(args.N_applications), int(args.rng_seed) + h_idx,
                )

                chain = echk.run_W_chain_energies(
                    ancilla=Ham.ancilla, system_qubits=Ham.system_qubits,
                    H_int=Ham.interaction, qubit_ham=Ham.qubit, n_system=n_system,
                    tau=args.tau_W, filter_terms=fdata["filter_terms"],
                    T_filter=fdata["T_filter"], pair_sequence=seq,
                    rho_init=(rho_carry if bool(args.sim_carry_rho) else None),
                    occ_spin_orbs=occ, reps_per_W=args.reps_per_W,
                    trotter_steps_H=args.trotter_steps_H, H_sparse=H_sparse,
                )

                if bool(args.sim_carry_rho):
                    rho_carry = chain["rho_final"]

                err = abs(chain["final_energy"] - refs["E_fixed_N"])
                sim.update(
                    initial_energy=chain["initial_energy"],
                    final_energy=chain["final_energy"],
                    E_fixed_N=refs["E_fixed_N"],
                    E_full_fock=refs.get("E_full_fock"),
                    energy_error=err, energy_error_mHa=err * 1e3,
                    gap_fixed_N=refs.get("gap_fixed_N"),
                    sector_dimension=refs.get("sector_dimension"),
                    spectral_width=refs.get("spectral_width"),
                    final_particle_number=chain["final_particle_number"],
                    trace_drift=chain["trace_drift"],
                    kraus_completeness_error=chain["kraus_completeness_error"],
                    sim_seconds=time.time() - ts,
                )

                flag = "OK " if err <= float(args.chem_acc) else "OVER"
                print(f"[energy] initial {chain['initial_energy']:.10f} -> "
                      f"final {chain['final_energy']:.10f} Ha", flush=True)
                print(f"[energy] error {err*1e3:.4f} mHa  [{flag} vs "
                      f"{float(args.chem_acc)*1e3:.1f} mHa]   "
                      f"N_e = {chain['final_particle_number']:.6f}   "
                      f"trace drift {chain['trace_drift']:.1e}", flush=True)
            # --------------------------------------------------------------

            circuits_by_geometry.append(W_circuits)
            summary_rows.append(asdict(GeometryRecord(
                hamiltonian_index=h_idx, original_geometry_index=int(orig),
                n_system=n_system, n_W_applications=int(args.N_applications),
                n_W_circuits=len(W_circuits),
                n_filter_terms=len(fdata["filter_terms"]),
                wmax=float(fdata["wmax"]), T_filter=float(fdata["T_filter"]),
                n_jump_pairs=len(pairs), keep_circuits=str(args.keep_circuits),
                resources_total=total, **sim,
            )))

            # Pickling accumulates every circuit from every geometry and is the
            # dominant memory cost once n_system grows. Skip it when there is
            # nothing to store.
            if any(circuits_by_geometry):
                atomic_pickle(circuits_by_geometry, outdir / "unique_W_circuits.pkl")
            atomic_json(summary_rows, outdir / "circuit_resources.json")
            atomic_json(replay_by_geometry, outdir / "W_sequence_by_geometry.json")
            meta["last_completed_hamiltonian_index"] = h_idx
            atomic_json(meta, outdir / "run_metadata.json")

            # replot after every geometry so a killed job still leaves a figure
            if bool(args.plot) and any(
                r.get("final_energy") is not None for r in summary_rows
            ):
                echk.plot_energy_errors(
                    summary_rows, str(outdir / "energy_error_by_geometry.png"),
                    chem_acc=float(args.chem_acc),
                )

            dt, el = time.time() - t0, time.time() - run_t0
            left = len(geoms) - (h_idx + 1)
            print(f"[timing] geometry {h_idx}: {dt/60:.1f} min | elapsed {el/3600:.2f} h "
                  f"| ETA {(el/(h_idx+1))*left/3600:.2f} h", flush=True)
            print(f"[resources] qubits={total['n_qubits']} depth={total['depth_moments']} "
                  f"ops={total['n_operations']} rotations={total['n_rotations']} "
                  f"T={total['t_count']:,}", flush=True)

        meta["status"] = "completed"
        atomic_json(meta, outdir / "run_metadata.json")

        sims = [r for r in summary_rows if r.get("final_energy") is not None]
        if sims:
            e = np.array([r["energy_error_mHa"] for r in sims])
            print(f"\n[summary] {len(sims)} simulated geometries | "
                  f"mean {e.mean():.4f} mHa | min {e.min():.4f} | max {e.max():.4f} | "
                  f"std {e.std():.4f} | over threshold: "
                  f"{int((e > float(args.chem_acc)*1e3).sum())}", flush=True)
        print(f"\nDone -> {outdir}", flush=True)

    except Exception as exc:
        meta.update(status="failed_after_partial_save", error_type=type(exc).__name__,
                    error_message=str(exc), traceback=traceback.format_exc())
        atomic_pickle(circuits_by_geometry, outdir / "unique_W_circuits.pkl")
        atomic_json(summary_rows, outdir / "circuit_resources.json")
        atomic_json(replay_by_geometry, outdir / "W_sequence_by_geometry.json")
        atomic_json(meta, outdir / "run_metadata.json")
        (outdir / "ERROR.txt").write_text(meta["traceback"], encoding="utf-8")
        print(meta["traceback"], file=sys.stderr, flush=True)
        raise


def build_arg_parser():
    p = argparse.ArgumentParser(description="LinLin reaction-path circuit builder.")
    C = USER_CONFIG
    p.add_argument("--xyz-trajectory", default=C["xyz_trajectory"])
    p.add_argument("--geometry-start", type=int, default=C["geometry_start"])
    p.add_argument("--geometry-stop", type=int, default=C["geometry_stop"])
    p.add_argument("--geometry-stride", type=int, default=C["geometry_stride"])
    p.add_argument("--include-final-geometry", action=argparse.BooleanOptionalAction,
                   default=C["include_final_geometry"])
    p.add_argument("--basis", default=C["basis"])
    p.add_argument("--multiplicity", type=int, default=C["multiplicity"])
    p.add_argument("--charge", type=int, default=C["charge"])
    p.add_argument("--active-indices", default=C["active_indices"])
    p.add_argument("--ancilla-index", type=int, default=C["ancilla_index"])
    p.add_argument("--n-active-electrons", default=C["n_active_electrons"])
    p.add_argument("--n-active-orbitals", default=C["n_active_orbitals"])
    p.add_argument("--use-casscf-orbitals", action=argparse.BooleanOptionalAction,
                   default=C["use_casscf_orbitals"])
    p.add_argument("--fix-spin-ss", default=C["fix_spin_ss"])
    p.add_argument("--max-geom-step", default=C["max_geom_step"])
    p.add_argument("--align-orbitals", action=argparse.BooleanOptionalAction,
                   default=C["align_orbitals"])
    p.add_argument("--n-electrons", default=C["n_electrons"])
    p.add_argument("--occ-spin-orbs", default=C["occ_spin_orbs"])
    p.add_argument("--spin-adapted-jumps", action=argparse.BooleanOptionalAction,
                   default=C["spin_adapted_jumps"])
    p.add_argument("--conserve-spin", action=argparse.BooleanOptionalAction,
                   default=C["conserve_spin"])
    p.add_argument("--tau-filter", type=float, default=C["tau_filter"])
    p.add_argument("--filter-K", type=int, default=C["filter_K"])
    p.add_argument("--filter-window", default=C["filter_window"])
    p.add_argument("--filter-wmax-factor", type=float, default=C["filter_wmax_factor"])
    p.add_argument("--filter-wmax", default=C["filter_wmax"],
                   help="Supply the filter scale directly and skip diagonalisation.")
    p.add_argument("--use-global-filter", action=argparse.BooleanOptionalAction,
                   default=C["use_global_filter"])
    p.add_argument("--tau-W", type=float, default=C["tau_W"])
    p.add_argument("--N-applications", type=int, default=C["N_applications"])
    p.add_argument("--reps-per-W", type=int, default=C["reps_per_W"],
                   help="Fourier mode cutoff: keeps |k| <= reps_per_W.")
    p.add_argument("--trotter-steps-H", type=int, default=C["trotter_steps_H"])
    p.add_argument("--rng-seed", type=int, default=C["rng_seed"])
    p.add_argument("--keep-circuits", choices=("unique", "all", "first", "none"),
                   default=C["keep_circuits"])
    # ---- added ----
    p.add_argument("--compress-circuits", action=argparse.BooleanOptionalAction,
                   default=C["compress_circuits"],
                   help="Build real W circuits but store the repeated propagator "
                        "once via cirq.CircuitOperation instead of copying it "
                        "802 times. ~270x fewer objects and a ~270x smaller "
                        "pickle. Ignored if --analytic-resources is set.")
    p.add_argument("--analytic-resources", action=argparse.BooleanOptionalAction,
                   default=C["analytic_resources"],
                   help="Count W analytically instead of building it. Constant "
                        "memory in filter_K; also fills the gate histogram. "
                        "No circuits are produced, so the pickle is skipped.")
    p.add_argument("--t-synthesis-eps", type=float, default=C["t_synthesis_eps"],
                   help="Rotation-synthesis precision for the T-count estimate "
                        "(Ross-Selinger, 3*log2(1/eps) T per rotation).")
    p.add_argument("--simulate", action=argparse.BooleanOptionalAction,
                   default=C["simulate"],
                   help="Evolve rho through the same W blocks and report the "
                        "final energy error per geometry.")
    p.add_argument("--sim-max-qubits", type=int, default=C["sim_max_qubits"])
    p.add_argument("--sim-carry-rho", action=argparse.BooleanOptionalAction,
                   default=C["sim_carry_rho"],
                   help="Carry rho to the next geometry (the continuation). "
                        "Disable to cool each geometry from the reference "
                        "determinant independently.")
    p.add_argument("--plot", action=argparse.BooleanOptionalAction,
                   default=C["plot"],
                   help="Write energy_error_by_geometry.png/.pdf. Numbers still "
                        "go to circuit_resources.json when disabled.")
    p.add_argument("--chem-acc", type=float, default=C["chem_acc"])
    # ---------------
    p.add_argument("--outdir", default=C["outdir"])
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
