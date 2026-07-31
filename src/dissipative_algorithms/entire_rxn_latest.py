#!/usr/bin/env python3
"""
Clean LinLin reaction-path driver extracted from The_entire_rxn.ipynb.

This script deliberately excludes the notebook's example/demo cells and QPE section.
It keeps only the reaction-path W-chain workflow:

1. Load the LinLin definitions from a flexible base name/path.
2. Read the molecular geometry sequence from a NEB/IRC multi-frame ``*.trj.xyz`` file.
3. Build molecular Hamiltonians for those geometries.
4. Carry the output density matrix from geometry g into geometry g+1.
5. Save the initial state at the final geometry.
6. Save energy data versus Hamiltonian index and versus cumulative W applications.
7. Save every generated W circuit for every geometry to a pickle file.
8. Save overlap(initial rho, final rho) per geometry and plot it versus Hamiltonian index.

There is intentionally no built-in/generated H3 path and no H3-specific default spin/electron sector.
The geometry path must come from your NEB/IRC trajectory file, and the chemical charge,
multiplicity, and initial occupied spin orbitals/electron count must be supplied explicitly.

The script saves partial results after each geometry, so completed outputs survive if a later
geometry fails.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import pickle
import runpy
import sys
import tempfile
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Use a non-interactive backend for HPC/headless runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import one_ham as linlin


Geometry = List[Tuple[str, Tuple[float, float, float]]]


# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------
# Put run/science parameters here. The shell script should only handle scratch,
# conda, threads, logging, and copy-back. Command-line arguments can still
# override these values when needed.
USER_CONFIG = {
    # Input files expected in the same directory as this script when run in scratch.
    "xyz_trajectory": "H3_rxn_trj.xyz",

    # Geometry slicing. None means use the end of the trajectory.
    "geometry_start": 0,
    "geometry_stop": None,
    "geometry_stride": 1,
    "include_final_geometry": True,

    # Chemistry / Hamiltonian settings.
    "basis": "sto-3g",
    "charge": 0,
    "multiplicity": 2,
    "active_indices": None,
    "ancilla_index": 0,

    # Neutral H3 doublet CASSCF active space:
    # total active electrons = 3, with Na=2, Nb=1
    "n_active_electrons": "2,1",
    "n_active_orbitals": 3,
    "use_casscf_orbitals": True,

    # Orbital transport / alignment.
    "use_projected_initial_guess": True,
    "use_procrustes_alignment": True,

    # Initial state / sector settings for neutral H3 doublet.
    "n_electrons": 3,
    "Na": 2,
    "Nb": 1,

    # Interleaved spin-orbital ordering:
    # spatial 0 alpha, spatial 0 beta, spatial 1 alpha, spatial 1 beta, ...
    # For H3 doublet: alpha0, beta0, alpha1
    "occ_spin_orbs": "0,1,2",

    # If your ordering is alpha-block then beta-block:
    # alpha0, alpha1, alpha2, beta0, beta1, beta2
    # use:
    # "occ_spin_orbs": "0,1,3",

    # Filter and W-chain settings.
    "tau_filter": 0.1,
    "filter_K": 500,
    "filter_window": "lanczos",
    "filter_wmax_factor": 2.0,
    "use_global_filter": False,

    "tau_W": 1.0,
    "N_applications": 500,
    "reps_per_W": 50,
    "trotter_steps_H": 1,
    "rng_seed": 123,

    # Output directory.
    "outdir": os.environ.get("TARGET_DIR", "rxn_linlin_outputs"),
}


# -----------------------------------------------------------------------------
# Robust saving utilities
# -----------------------------------------------------------------------------

def atomic_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
        pickle.dump(obj, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp", encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, default=_json_default)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
        np.save(tmp, arr)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def write_csv_rows(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Preserve first-row order, then append any keys that appear later.
    fieldnames: List[str] = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp", newline="", encoding="utf-8") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _json_default(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.complexfloating, complex)):
        return {"real": float(np.real(x)), "imag": float(np.imag(x))}
    return str(x)


# -----------------------------------------------------------------------------
# Geometry utilities
# -----------------------------------------------------------------------------

def resolve_xyz_trajectory(path_or_pattern: Optional[str]) -> Path:
    """
    Resolve the NEB/IRC trajectory file.

    If --xyz-trajectory is supplied, it may be an explicit path or a glob pattern.
    If it is omitted, the script looks for exactly one trajectory-like XYZ file in
    the current directory. It never generates a toy geometry path.
    """
    candidates: List[Path] = []

    if path_or_pattern:
        raw = Path(path_or_pattern).expanduser()
        # Explicit existing path.
        if raw.exists():
            return raw.resolve()
        # Glob pattern, e.g. "*trj.xyz".
        matches = [Path(x) for x in sorted(glob(str(raw)))]
        candidates.extend(matches)
    else:
        patterns = ["H3_rxn_trj.xyz", "IRC_trj.xyz", "*trj.xyz", "*TRJ.xyz", "*.trj.xyz"]
        seen = set()
        for pattern in patterns:
            for match in sorted(glob(pattern)):
                p = Path(match)
                if p not in seen and p.is_file():
                    seen.add(p)
                    candidates.append(p)

    candidates = [p for p in candidates if p.is_file()]
    if len(candidates) == 1:
        return candidates[0].resolve()

    if not candidates:
        raise FileNotFoundError(
            "No NEB/IRC trajectory file found. Pass --xyz-trajectory MEP_trj.xyz "
            "or --xyz-trajectory '*trj.xyz'. This script no longer generates an H3 path."
        )

    raise FileExistsError(
        "Multiple trajectory candidates matched; pass one explicitly with --xyz-trajectory. Candidates: "
        + ", ".join(str(p) for p in candidates)
    )


def load_xyz_trajectory(path: Path) -> List[Geometry]:
    """
    Load a multi-frame XYZ trajectory, including ORCA-style MEP_trj.xyz files.

    Each frame is:
        natoms
        comment
        Atom x y z
        ...
    """
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    geometries: List[Geometry] = []
    i = 0
    while i < len(text):
        line = text[i].strip()
        if not line:
            i += 1
            continue
        try:
            natoms = int(line)
        except ValueError as exc:
            raise ValueError(f"Expected atom count at line {i + 1} of {path}, got: {line!r}") from exc
        if i + 1 + natoms >= len(text) + 1:
            raise ValueError(f"Incomplete XYZ frame starting at line {i + 1} of {path}")
        i += 2  # skip natoms and comment
        geom: Geometry = []
        for _ in range(natoms):
            parts = text[i].split()
            if len(parts) < 4:
                raise ValueError(f"Malformed XYZ atom line {i + 1}: {text[i]!r}")
            atom = parts[0]
            xyz = tuple(float(v) for v in parts[1:4])
            geom.append((atom, xyz))
            i += 1
        geometries.append(geom)
    if not geometries:
        raise ValueError(f"No XYZ frames found in {path}")
    return geometries


# -----------------------------------------------------------------------------
# Physics/workflow helpers
# -----------------------------------------------------------------------------

def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if not s or s.lower() == "none":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_optional_int(s: Optional[str]) -> Optional[int]:
    if s is None or str(s).lower() in {"none", ""}:
        return None
    return int(s)
    
def parse_active_electrons(x: Any) -> Optional[Any]:
    """
    Parse n_active_electrons.

    Allowed forms:
        None
        3
        "3"
        (2, 1)
        [2, 1]
        "2,1"
        "(2, 1)"

    For H3 doublet CASSCF, this should return (2, 1).
    """
    if x is None:
        return None

    if isinstance(x, tuple):
        return tuple(int(v) for v in x)

    if isinstance(x, list):
        return tuple(int(v) for v in x)

    if isinstance(x, int):
        return int(x)

    s = str(x).strip()
    if not s or s.lower() == "none":
        return None

    if "," in s:
        s_clean = s.strip("()[]")
        vals = [int(v.strip()) for v in s_clean.split(",") if v.strip()]
        return tuple(vals)

    return int(s)


def make_pairs_and_weights(n_system: int, occ_spin_orbs: Sequence[int]) -> Tuple[List[Tuple[int, int, str]], List[float]]:
    """
    Automatic jump list matching the notebook comment:
    jumps from initially occupied to initially unoccupied spin-orbitals, with Ax and Ay channels.
    """
    occ = sorted(set(int(i) for i in occ_spin_orbs))
    unocc = [j for j in range(n_system) if j not in occ]
    pairs = [(i, j, axis) for i in occ for j in unocc for axis in ("Ax", "Ay")]
    if not pairs:
        raise ValueError(
            f"No jump pairs were generated. n_system={n_system}, occ_spin_orbs={list(occ_spin_orbs)}"
        )
    weights = [1.0 / len(pairs)] * len(pairs)
    return pairs, weights


def compute_filter_for_hamiltonian(
    H_mat: np.ndarray,
    SquareWaveFilter: Any,
    tau_filter: float,
    filter_K: int,
    filter_window: str,
    filter_wmax_factor: float,
) -> Dict[str, Any]:
    wmax = filter_wmax_factor * float(np.linalg.norm(H_mat, ord=2))
    sq_filter = SquareWaveFilter()
    data = sq_filter.get_coefficients(
        L=wmax,
        K=filter_K,
        tau=tau_filter,
        A=1.0,
        window=filter_window,
        drop_zeros=True,
        sign=+1,
    )
    data["wmax"] = wmax
    data["tau_filter"] = tau_filter
    data["filter_K"] = filter_K
    data["filter_window"] = filter_window
    data["filter_wmax_factor"] = filter_wmax_factor
    return data


def density_overlap(rho_a: Optional[np.ndarray], rho_b: Optional[np.ndarray]) -> float:
    """
    Overlap-like diagnostic between two density matrices: Re Tr(rho_a rho_b).
    For pure states this equals |<psi|phi>|^2.
    """
    if rho_a is None or rho_b is None:
        return float("nan")
    a = np.asarray(rho_a)
    b = np.asarray(rho_b)
    if a.shape != b.shape:
        return float("nan")
    return float(np.real(np.trace(a @ b)))


def call_basis_alpha_beta(basis_alpha_beta: Any, n_system: int, Na: int, Nb: int) -> Any:
    try:
        return basis_alpha_beta(n_system, Na=Na, Nb=Nb)
    except TypeError:
        return basis_alpha_beta(n_system, Na, Nb)


def maybe_subspace_energy(
    H_mat: np.ndarray,
    n_system: int,
    n_electrons: Optional[int],
    Na: Optional[int],
    Nb: Optional[int],
    basis_fixed_N: Optional[Any],
    basis_alpha_beta: Optional[Any],
    ground_energy_in_subspace: Optional[Any],
) -> Tuple[Optional[float], Optional[float]]:
    if ground_energy_in_subspace is None:
        return None, None

    E_N = None
    E_NaNb = None

    if n_electrons is not None and basis_fixed_N is not None:
        basis_N = basis_fixed_N(n_system, n_electrons)
        E_N, _ = ground_energy_in_subspace(H_mat, basis_N)
        E_N = float(E_N)

    if Na is not None and Nb is not None and basis_alpha_beta is not None:
        basis_NaNb = call_basis_alpha_beta(basis_alpha_beta, n_system, Na, Nb)
        E_NaNb, _ = ground_energy_in_subspace(H_mat, basis_NaNb)
        E_NaNb = float(E_NaNb)

    return E_N, E_NaNb


@dataclass
class GeometryRecord:
    hamiltonian_index: int
    original_geometry_index: int
    E_corr: float
    E_fixed_N: Optional[float]
    E_fixed_NaNb: Optional[float]
    initial_final_overlap: float
    n_system: int
    n_W_circuits: int
    n_energy_points: int
    final_energy: float
    wmax: float
    T_filter: float
    n_filter_terms: int


# -----------------------------------------------------------------------------
# Plotting and incremental output
# -----------------------------------------------------------------------------

def plot_energy_vs_w(energy_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not energy_rows:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    by_geom: Dict[int, List[Dict[str, Any]]] = {}
    for row in energy_rows:
        by_geom.setdefault(int(row["hamiltonian_index"]), []).append(row)

    for g_idx, rows in by_geom.items():
        xs = [r["cumulative_W_applications"] for r in rows]
        ys = [r["energy"] for r in rows]
        ax.plot(xs, ys, lw=1.5, label=f"H {g_idx}")
        if rows[0].get("E_corr") is not None:
            ax.hlines(rows[0]["E_corr"], xs[0], xs[-1], linestyles=":", lw=1.0)
        if g_idx > 0:
            ax.axvline(xs[0], lw=0.8, ls="--", alpha=0.5)

    ax.set_xlabel("Cumulative W applications")
    ax.set_ylabel(r"$\langle H_g \rangle$")
    ax.set_title("LinLin W-chain energy along reaction path")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_energy_vs_hamiltonian(summary_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not summary_rows:
        return
    xs = [row["hamiltonian_index"] for row in summary_rows]
    ys_final = [row["final_energy"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys_final, marker="o", lw=1.5, label="Final energy after W-chain")

    if any(row.get("E_corr") is not None for row in summary_rows):
        ys_corr = [row.get("E_corr", np.nan) for row in summary_rows]
        ax.plot(xs, ys_corr, marker="x", lw=1.0, label="Full Fock-space ground")

    if any(row.get("E_fixed_NaNb") is not None for row in summary_rows):
        ys_sector = [row.get("E_fixed_NaNb", np.nan) for row in summary_rows]
        ax.plot(xs, ys_sector, marker="s", lw=1.0, label="Fixed Na/Nb ground")

    ax.set_xlabel("Hamiltonian index / geometry index")
    ax.set_ylabel("Energy")
    ax.set_title("Energy versus number of Hamiltonians")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overlap(summary_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not summary_rows:
        return
    xs = [row["hamiltonian_index"] for row in summary_rows]
    ys = [row["initial_final_overlap"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", lw=1.5)
    ax.set_xlabel("Hamiltonian index / geometry index")
    ax.set_ylabel(r"Re Tr($\rho_{\rm in}\rho_{\rm out}$)")
    ax.set_title("Overlap between initial and final state at each geometry")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_partial_outputs(
    outdir: Path,
    circuits_by_geometry: List[List[Any]],
    rho_by_geometry: List[np.ndarray],
    summary_rows: List[Dict[str, Any]],
    energy_rows: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    resources_by_geometry: Optional[List[List[Any]]] = None,
    orbital_alignment_by_geometry: Optional[List[Any]] = None,
) -> None:
    atomic_pickle(circuits_by_geometry, outdir / "all_W_circuits_by_geometry.pkl")
    atomic_pickle(rho_by_geometry, outdir / "final_rhos_by_geometry.pkl")

    if resources_by_geometry is not None:
        atomic_pickle(resources_by_geometry, outdir / "all_resources_by_geometry.pkl")

    if orbital_alignment_by_geometry is not None:
        atomic_pickle(orbital_alignment_by_geometry, outdir / "all_orbital_alignment.pkl")
        atomic_json(orbital_alignment_by_geometry, outdir / "all_orbital_alignment.json")

    write_csv_rows(summary_rows, outdir / "energy_vs_hamiltonian_index.csv")
    write_csv_rows(energy_rows, outdir / "energy_vs_w_applications.csv")
    write_csv_rows(summary_rows, outdir / "overlap_vs_hamiltonian_index.csv")
    atomic_json(metadata, outdir / "run_metadata.json")

    plot_energy_vs_hamiltonian(summary_rows, outdir / "energy_vs_hamiltonian_index.png")
    plot_energy_vs_w(energy_rows, outdir / "energy_vs_w_applications.png")
    plot_overlap(summary_rows, outdir / "overlap_vs_hamiltonian_index.png")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.charge is None:
        raise ValueError("Pass --charge explicitly. No molecule-specific charge default is assumed.")
    if args.multiplicity is None:
        raise ValueError("Pass --multiplicity explicitly. No H3/default spin multiplicity is assumed.")
    if args.occ_spin_orbs is None and parse_optional_int(args.n_electrons) is None:
        raise ValueError(
            "Pass either --occ-spin-orbs, e.g. 0,1,2,3, or --n-electrons. "
            "The script will not assume an H3 electron sector."
        )

    HamiltonianBuilder = linlin.HamiltonianBuilder
    SquareWaveFilter = linlin.SquareWaveFilter
    run_W_chain_circuit = linlin.run_W_chain_circuit

    hf_density_from_occupations = getattr(linlin, "hf_density_from_occupations", None)
    basis_fixed_N = getattr(linlin, "basis_fixed_N", None)
    basis_alpha_beta = getattr(linlin, "basis_alpha_beta", None)
    ground_energy_in_subspace = getattr(linlin, "ground_energy_in_subspace", None)
    estimate_resources = getattr(linlin, "estimate_resources", None)

    linlin_path = Path(linlin.__file__).resolve()

    xyz_path = resolve_xyz_trajectory(args.xyz_trajectory)
    geometries_all = load_xyz_trajectory(xyz_path)
    source = str(xyz_path)

    start_idx = max(0, int(args.geometry_start))
    stop_idx = len(geometries_all) if args.geometry_stop is None else min(len(geometries_all), int(args.geometry_stop))
    if stop_idx <= start_idx:
        raise ValueError(
            f"Invalid geometry range: start={start_idx}, stop={stop_idx}, total={len(geometries_all)}"
        )

    stride = max(1, int(args.geometry_stride))
    selected_indices = list(range(start_idx, stop_idx, stride))
    if args.include_final_geometry and (stop_idx - 1) not in selected_indices:
        selected_indices.append(stop_idx - 1)
    selected_indices = sorted(set(selected_indices))
    geometries = [geometries_all[i] for i in selected_indices]

    print(f"Loaded {len(geometries)} geometries.", flush=True)
    print("First geometry:", flush=True)
    print(geometries[0], flush=True)
    print("Last geometry:", flush=True)
    print(geometries[-1], flush=True)

    selector_info = {
        "selector": "generic_xyz_range_stride",
        "n_original": len(geometries_all),
        "n_selected": len(geometries),
        "geometry_start": start_idx,
        "geometry_stop_exclusive": stop_idx,
        "geometry_stride": stride,
        "include_final_geometry": bool(args.include_final_geometry),
        "selected_indices": selected_indices,
    }

    if not geometries:
        raise ValueError("No geometries selected from the trajectory.")

    n_active_electrons = parse_active_electrons(args.n_active_electrons)
    n_active_orbitals = parse_optional_int(args.n_active_orbitals)

    n_electrons = parse_optional_int(args.n_electrons)
    Na = parse_optional_int(args.Na)
    Nb = parse_optional_int(args.Nb)
    user_occ = parse_int_list(args.occ_spin_orbs)

    metadata: Dict[str, Any] = {
        "linlin_path": str(linlin_path),
        "geometry_source": source,
        "n_input_geometries": len(geometries_all),
        "n_used_geometries": len(geometries),
        "selected_original_indices": selected_indices,
        "selector_info": selector_info,
        "args": vars(args),
        "n_active_electrons_parsed": n_active_electrons,
        "n_active_orbitals_parsed": n_active_orbitals,
        "status": "running",
    }
    atomic_json(metadata, outdir / "run_metadata.json")

    prev_mol = None
    prev_mo_coeff = None
    rho_current: Optional[np.ndarray] = None

    circuits_by_geometry: List[List[Any]] = []
    resources_by_geometry: List[List[Any]] = []
    rho_by_geometry: List[np.ndarray] = []
    summary_rows: List[Dict[str, Any]] = []
    energy_rows: List[Dict[str, Any]] = []
    orbital_alignment_by_geometry: List[Any] = []

    filter_cache: Optional[Dict[str, Any]] = None
    cumulative_W = 0

    try:
        for h_idx, (original_idx, geometry) in enumerate(zip(selected_indices, geometries)):
            print(
                f"\n===== Hamiltonian/geometry {h_idx}/{len(geometries)-1} | original index {original_idx} =====",
                flush=True,
            )
            print(geometry, flush=True)

            Ham = HamiltonianBuilder().build_molecule(
                geometry=geometry,
                basis=args.basis,
                multiplicity=args.multiplicity,
                charge=args.charge,
                active_indices=parse_int_list(args.active_indices),
                ancilla_index=args.ancilla_index,

                # Essential for entering the CASSCF branch.
                n_active_electrons=n_active_electrons,
                n_active_orbitals=n_active_orbitals,
                use_casscf_orbitals=bool(args.use_casscf_orbitals),

                # Orbital transport options.
                prev_mol=prev_mol,
                prev_mo_coeff=prev_mo_coeff,
                use_projected_initial_guess=bool(args.use_projected_initial_guess),
                use_procrustes_alignment=bool(args.use_procrustes_alignment),
            )

            interaction_ham = Ham.interaction
            fermion_ham = Ham.fermion
            qubit_ham = Ham.qubit
            H_mat = Ham.matrix
            E_corr = float(Ham.energy)
            n_system = int(Ham.n_system)
            ancilla = Ham.ancilla
            system_qubits = Ham.system_qubits

            if user_occ is not None:
                occ_spin_orbs = user_occ
            elif n_electrons is not None:
                occ_spin_orbs = list(range(n_electrons))
            else:
                raise ValueError(
                    "Could not infer the initial occupied spin orbitals. "
                    "Pass --occ-spin-orbs 0,1,2 or pass --n-electrons."
                )

            if max(occ_spin_orbs, default=-1) >= n_system:
                raise ValueError(
                    f"occ_spin_orbs={occ_spin_orbs} is incompatible with n_system={n_system}."
                )

            # Initialize the quantum state only once.
            # Afterwards rho_current is transported through the W-chain.
            if rho_current is None:
                if hf_density_from_occupations is None:
                    raise KeyError(
                        "`hf_density_from_occupations` was not found in LinLin. "
                        "Needed to initialize the first density matrix."
                    )
                rho_in = np.asarray(
                    hf_density_from_occupations(
                        len(system_qubits),
                        occ_spin_orbs,
                    )
                )
            else:
                rho_in = np.asarray(rho_current)

            # Save the initial state for the final geometry immediately before running it.
            if h_idx == len(geometries) - 1:
                atomic_npy(rho_in, outdir / "initial_state_at_final_geometry.npy")
                atomic_pickle(
                    {
                        "rho_initial_final_geometry": rho_in,
                        "hamiltonian_index": h_idx,
                        "original_geometry_index": original_idx,
                        "geometry": geometry,
                    },
                    outdir / "initial_state_at_final_geometry.pkl",
                )
                print(
                    f"Saved initial state at final geometry -> {outdir / 'initial_state_at_final_geometry.npy'}",
                    flush=True,
                )

            if args.use_global_filter and filter_cache is not None:
                filter_data = filter_cache
            else:
                filter_data = compute_filter_for_hamiltonian(
                    H_mat=H_mat,
                    SquareWaveFilter=SquareWaveFilter,
                    tau_filter=args.tau_filter,
                    filter_K=args.filter_K,
                    filter_window=args.filter_window,
                    filter_wmax_factor=args.filter_wmax_factor,
                )
                if args.use_global_filter and filter_cache is None:
                    filter_cache = filter_data

            filter_terms = filter_data["filter_terms"]
            T_filter = filter_data["T_filter"]

            pairs, weights = make_pairs_and_weights(
                n_system=n_system,
                occ_spin_orbs=occ_spin_orbs,
            )

            E_N, E_NaNb = maybe_subspace_energy(
                H_mat=H_mat,
                n_system=n_system,
                n_electrons=n_electrons,
                Na=Na,
                Nb=Nb,
                basis_fixed_N=basis_fixed_N,
                basis_alpha_beta=basis_alpha_beta,
                ground_energy_in_subspace=ground_energy_in_subspace,
            )

            print(f"E_corr = {E_corr:.10f} Hartree", flush=True)
            if E_N is not None:
                print(f"Fixed N={n_electrons} ground = {E_N:.10f} Hartree", flush=True)
            if E_NaNb is not None:
                print(f"Fixed Na={Na}, Nb={Nb} ground = {E_NaNb:.10f} Hartree", flush=True)

            alignment_info = getattr(Ham, "alignment_info", None)
            if alignment_info is not None:
                print(
                    "Orbital alignment active min/max cosine = "
                    f"{alignment_info['active_min_cosine']:.8f}/"
                    f"{alignment_info['active_max_cosine']:.8f}",
                    flush=True,
                )
            else:
                print("Orbital alignment: first geometry, no previous frame.", flush=True)

            print(
                f"Filter: wmax={filter_data['wmax']:.8f}, "
                f"T_filter={T_filter:.8f}, terms={len(filter_terms)}",
                flush=True,
            )
            print(
                f"Generated {len(pairs)} jump pairs from occ_spin_orbs={occ_spin_orbs}",
                flush=True,
            )

            steps, energies, W_circuits, rho_current = run_W_chain_circuit(
                system_qubits=system_qubits,
                H_int=interaction_ham,
                anc=ancilla,
                tau=args.tau_W,
                filter_terms=filter_terms,
                pairs=pairs,
                weights=weights,
                N_applications=args.N_applications,
                reps_per_W=args.reps_per_W,
                occ_spin_orbs=occ_spin_orbs,
                trotter_steps_H=args.trotter_steps_H,
                rng_seed=args.rng_seed + h_idx,
                T_filter=T_filter,
                H_mat_sys=H_mat,
                initial_rho_sys=rho_in,
            )

            # Store the orbital frame actually used to build this Hamiltonian.
            # The next geometry aligns to this frame.
            prev_mol = getattr(Ham, "mol", None)
            prev_mo_coeff = getattr(Ham, "mo_coeff", None)

            geometry_resources: List[Any] = []
            if estimate_resources is not None:
                for c_idx, W_circuit in enumerate(W_circuits):
                    resources = estimate_resources(W_circuit)
                    geometry_resources.append(resources)
            else:
                print(
                    "Warning: estimate_resources was not found in LinLin; skipping resource estimates.",
                    flush=True,
                )

            rho_out = np.asarray(rho_current)
            overlap = density_overlap(rho_in, rho_out)
            final_energy = float(energies[-1]) if len(energies) else float("nan")

            circuits_by_geometry.append(W_circuits)
            resources_by_geometry.append(geometry_resources)
            rho_by_geometry.append(rho_out)
            orbital_alignment_by_geometry.append(alignment_info)

            record = GeometryRecord(
                hamiltonian_index=h_idx,
                original_geometry_index=int(original_idx),
                E_corr=E_corr,
                E_fixed_N=E_N,
                E_fixed_NaNb=E_NaNb,
                initial_final_overlap=overlap,
                n_system=n_system,
                n_W_circuits=len(W_circuits),
                n_energy_points=len(energies),
                final_energy=final_energy,
                wmax=float(filter_data["wmax"]),
                T_filter=float(T_filter),
                n_filter_terms=len(filter_terms),
            )
            summary_rows.append(asdict(record))

            for local_step, energy in zip(steps, energies):
                energy_rows.append(
                    {
                        "hamiltonian_index": h_idx,
                        "original_geometry_index": int(original_idx),
                        "local_W_applications": int(local_step),
                        "cumulative_W_applications": int(cumulative_W + local_step),
                        "energy": float(energy),
                        "E_corr": E_corr,
                        "E_fixed_N": E_N,
                        "E_fixed_NaNb": E_NaNb,
                    }
                )

            if len(steps):
                cumulative_W += int(steps[-1])

            metadata["last_completed_hamiltonian_index"] = h_idx
            metadata["status"] = "running_partial_saved"

            save_partial_outputs(
                outdir=outdir,
                circuits_by_geometry=circuits_by_geometry,
                rho_by_geometry=rho_by_geometry,
                summary_rows=summary_rows,
                energy_rows=energy_rows,
                metadata=metadata,
                resources_by_geometry=resources_by_geometry,
                orbital_alignment_by_geometry=orbital_alignment_by_geometry,
            )

            print(f"Saved partial outputs after geometry {h_idx} -> {outdir}", flush=True)

        metadata["status"] = "completed"

        save_partial_outputs(
            outdir=outdir,
            circuits_by_geometry=circuits_by_geometry,
            rho_by_geometry=rho_by_geometry,
            summary_rows=summary_rows,
            energy_rows=energy_rows,
            metadata=metadata,
            resources_by_geometry=resources_by_geometry,
            orbital_alignment_by_geometry=orbital_alignment_by_geometry,
        )

        print(f"\nCompleted. Outputs saved in: {outdir}", flush=True)

    except Exception as exc:
        metadata["status"] = "failed_after_partial_save"
        metadata["error_type"] = type(exc).__name__
        metadata["error_message"] = str(exc)
        metadata["traceback"] = traceback.format_exc()

        save_partial_outputs(
            outdir=outdir,
            circuits_by_geometry=circuits_by_geometry,
            rho_by_geometry=rho_by_geometry,
            summary_rows=summary_rows,
            energy_rows=energy_rows,
            metadata=metadata,
            resources_by_geometry=resources_by_geometry,
            orbital_alignment_by_geometry=orbital_alignment_by_geometry,
        )

        (outdir / "ERROR.txt").write_text(metadata["traceback"], encoding="utf-8")

        print("\nERROR occurred. Partial outputs were saved before exiting.", file=sys.stderr, flush=True)
        print(metadata["traceback"], file=sys.stderr, flush=True)

        raise


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean LinLin reaction-path W-chain driver.")

    # Geometry source: NEB/IRC trajectory only; no generated toy path.
    p.add_argument(
        "--xyz-trajectory",
        default=USER_CONFIG["xyz_trajectory"],
        help=(
            "ORCA/XYZ multi-frame trajectory, e.g. MEP_trj.xyz. "
            "May also be a glob like '*trj.xyz'. If omitted, the script searches "
            "the current directory and requires exactly one trajectory-like XYZ file."
        ),
    )
    p.add_argument(
        "--geometry-start",
        type=int,
        default=USER_CONFIG["geometry_start"],
        help="First trajectory frame index to use, 0-based.",
    )
    p.add_argument(
        "--geometry-stop",
        type=int,
        default=USER_CONFIG["geometry_stop"],
        help="Exclusive stop frame index. Default: end of trajectory.",
    )
    p.add_argument(
        "--geometry-stride",
        type=int,
        default=USER_CONFIG["geometry_stride"],
        help="Use every stride-th frame from the trajectory.",
    )
    p.add_argument(
        "--include-final-geometry",
        action=argparse.BooleanOptionalAction,
        default=USER_CONFIG["include_final_geometry"],
        help="Always include the final selected frame even when stride skips it.",
    )

    # Chemistry settings.
    p.add_argument("--basis", default=USER_CONFIG["basis"])
    p.add_argument(
        "--multiplicity",
        type=int,
        default=USER_CONFIG["multiplicity"],
        help="Spin multiplicity. For neutral H3 doublet, use 2.",
    )
    p.add_argument(
        "--charge",
        type=int,
        default=USER_CONFIG["charge"],
        help="Molecular charge. For neutral H3, use 0.",
    )
    p.add_argument(
        "--active-indices",
        default=USER_CONFIG["active_indices"],
        help="Comma-separated active orbital indices, or None.",
    )
    p.add_argument(
        "--ancilla-index",
        type=int,
        default=USER_CONFIG["ancilla_index"],
    )

    # CASSCF active-space settings.
    # Important: n_active_electrons may be a tuple string such as "2,1".
    p.add_argument(
        "--n-active-electrons",
        default=USER_CONFIG["n_active_electrons"],
        help=(
            "Active electron count. Use an integer string like '3' for spin-summed "
            "or a tuple/comma string like '2,1' for alpha/beta. "
            "For neutral H3 doublet, use '2,1'."
        ),
    )
    p.add_argument(
        "--n-active-orbitals",
        default=USER_CONFIG["n_active_orbitals"],
        help="Number of active spatial orbitals. For neutral H3/STO-3G, use 3.",
    )
    p.add_argument(
        "--use-casscf-orbitals",
        action=argparse.BooleanOptionalAction,
        default=USER_CONFIG["use_casscf_orbitals"],
        help="Use CASSCF orbitals instead of plain HF orbitals.",
    )

    # Orbital transport / alignment settings.
    p.add_argument(
        "--use-projected-initial-guess",
        action=argparse.BooleanOptionalAction,
        default=USER_CONFIG["use_projected_initial_guess"],
        help="Use previous-frame orbitals projected into the current AO basis as the CASSCF initial guess.",
    )
    p.add_argument(
        "--use-procrustes-alignment",
        action=argparse.BooleanOptionalAction,
        default=USER_CONFIG["use_procrustes_alignment"],
        help="Apply Procrustes alignment between consecutive active orbital frames.",
    )

    # Sector / initial state.
    p.add_argument(
        "--n-electrons",
        default=USER_CONFIG["n_electrons"],
        help="Total electron number for fixed-N diagnostics and default HF occupations.",
    )
    p.add_argument(
        "--Na",
        default=USER_CONFIG["Na"],
        help="Alpha electron count for fixed Na/Nb sector diagnostics. For neutral H3 doublet, use 2.",
    )
    p.add_argument(
        "--Nb",
        default=USER_CONFIG["Nb"],
        help="Beta electron count for fixed Na/Nb sector diagnostics. For neutral H3 doublet, use 1.",
    )
    p.add_argument(
        "--occ-spin-orbs",
        default=USER_CONFIG["occ_spin_orbs"],
        help=(
            "Comma-separated occupied spin orbitals for the initial HF density. "
            "For interleaved H3 doublet ordering, use 0,1,2. "
            "For alpha-block/beta-block ordering, use 0,1,3."
        ),
    )

    # Filter and W-chain settings.
    p.add_argument(
        "--tau-filter",
        type=float,
        default=USER_CONFIG["tau_filter"],
        help="Tau used in SquareWaveFilter coefficients.",
    )
    p.add_argument("--filter-K", type=int, default=USER_CONFIG["filter_K"])
    p.add_argument("--filter-window", default=USER_CONFIG["filter_window"])
    p.add_argument(
        "--filter-wmax-factor",
        type=float,
        default=USER_CONFIG["filter_wmax_factor"],
    )
    p.add_argument(
        "--use-global-filter",
        action=argparse.BooleanOptionalAction,
        default=USER_CONFIG["use_global_filter"],
        help="Use first geometry's filter for all geometries. Default recomputes per geometry.",
    )
    p.add_argument(
        "--tau-W",
        type=float,
        default=USER_CONFIG["tau_W"],
        help="tau argument passed to run_W_chain_circuit.",
    )
    p.add_argument("--N-applications", type=int, default=USER_CONFIG["N_applications"])
    p.add_argument("--reps-per-W", type=int, default=USER_CONFIG["reps_per_W"])
    p.add_argument("--trotter-steps-H", type=int, default=USER_CONFIG["trotter_steps_H"])
    p.add_argument("--rng-seed", type=int, default=USER_CONFIG["rng_seed"])

    # Output.
    p.add_argument(
        "--outdir",
        default=USER_CONFIG["outdir"],
        help="Output directory.",
    )

    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
