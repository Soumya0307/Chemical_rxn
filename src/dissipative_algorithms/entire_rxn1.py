#!/usr/bin/env python3
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
import time 
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cirq  
import one_ham as linlin
Geometry = List[Tuple[str, Tuple[float, float, float]]]

USER_CONFIG = {
    "xyz_trajectory": None,
    "geometry_start": 0,
    "geometry_stop": None,
    "geometry_stride": 1,
    "include_final_geometry": True,
    "basis": "STO-3G",
    "charge": 0,
    "multiplicity": 2,         
    "active_indices": None,
    "ancilla_index": 0,
    "n_active_electrons": None,
    "n_active_orbitals": None,     
    "use_casscf_orbitals": True,
    "fix_spin_ss": None,
    "max_geom_step": None,
    "align_orbitals": True,
    "n_electrons": 3,
    "Na": 2,
    "Nb": 1,
    "occ_spin_orbs": None,
    "spin_adapted_jumps": False,
    # Filter and W-chain settings.
    "tau_filter": 1,
    "filter_K": 500,
    "filter_window": "lanczos",
    "filter_wmax_factor": 2.0,
    "use_global_filter": False,
    "tau_W": 0.1,
    "N_applications": 400,  
    "reps_per_W": 200,
    "trotter_steps_H": 4,
    "rng_seed": 123,
    "evolution_mode": "averaged_matrix",  # sampled_circuit or averaged_matrix
    "energy_stride": 0,
    "save_outputs": "per_geometry",
    "reference_sector": "N",       
    "conserve_spin": False,                
    "displacement_min_bohr": 9.0,
    "displacement_max_bohr": 12.0,
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
    patterns_to_search: List[str] = []
    if path_or_pattern:
        raw = Path(path_or_pattern).expanduser()
        # Explicit existing path.
        if raw.exists():
            return raw.resolve()
        # Glob pattern, e.g. "*_trj.xyz".
        matches = [Path(x) for x in sorted(glob(str(raw)))]
        candidates.extend(matches)
    # Generic convention: the trajectory file name ends with "_trj.xyz".
    patterns_to_search = ["*_trj.xyz", "*_TRJ.xyz"]
    seen = set(candidates)
    for pattern in patterns_to_search:
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
            "No trajectory file matching '*_trj.xyz' found in this folder. "
            "Place exactly one <name>_trj.xyz here or pass --xyz-trajectory explicitly."
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
ANGSTROM_TO_BOHR = 1.8897261254578281
def cumulative_displacements_bohr(
    geometries: Sequence[Geometry],
) -> np.ndarray:
    """
    Cumulative Cartesian reaction-path displacement in Bohr.
    Assumes the XYZ coordinates are in Angstrom.
    """
    displacements_bohr = [0.0]
    for geometry_previous, geometry_current in zip(
        geometries[:-1],
        geometries[1:],
    ):
        symbols_previous = [
            atom for atom, _ in geometry_previous
        ]
        symbols_current = [
            atom for atom, _ in geometry_current
        ]
        if symbols_previous != symbols_current:
            raise ValueError(
                "Atom identities or ordering changed between XYZ frames."
            )
        coordinates_previous = np.asarray(
            [xyz for _, xyz in geometry_previous],
            dtype=float,
        )
        coordinates_current = np.asarray(
            [xyz for _, xyz in geometry_current],
            dtype=float,
        )
        step_angstrom = np.linalg.norm(
            (coordinates_current - coordinates_previous).ravel()
        )
        step_bohr = step_angstrom * ANGSTROM_TO_BOHR
        displacements_bohr.append(
            displacements_bohr[-1] + step_bohr
        )
    return np.asarray(displacements_bohr, dtype=float)
def refine_geometry_path(
    geometries: List[Geometry],
    original_indices: List[int],
    max_step: Optional[float],
) -> Tuple[List[Geometry], List[int], List[bool]]:
    """
    Insert linearly interpolated frames wherever the maximum atomic displacement
    between consecutive geometries exceeds max_step (Angstrom).
    Returns (geometries, original_indices, is_interpolated). Interpolated frames
    inherit the original index of the segment start.
    """
    if max_step is None:
        return list(geometries), list(original_indices), [False] * len(geometries)
    out_g: List[Geometry] = [geometries[0]]
    out_i: List[int] = [int(original_indices[0])]
    out_f: List[bool] = [False]
    for a in range(1, len(geometries)):
        ga = np.array([xyz for _, xyz in geometries[a - 1]], dtype=float)
        gb = np.array([xyz for _, xyz in geometries[a]], dtype=float)
        symbols = [s for s, _ in geometries[a - 1]]
        disp = float(np.abs(gb - ga).max())
        nsub = max(1, int(np.ceil(disp / float(max_step))))
        for s in range(1, nsub):
            gi = ga + (gb - ga) * (s / nsub)
            out_g.append([(symbols[k], tuple(gi[k])) for k in range(len(symbols))])
            out_i.append(int(original_indices[a - 1]))
            out_f.append(True)
        out_g.append(geometries[a])
        out_i.append(int(original_indices[a]))
        out_f.append(False)
    return out_g, out_i, out_f
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
def make_pairs_and_weights(
    n_system: int,
    occ_spin_orbs: Sequence[int],
    conserve_spin: bool = False,
    spin_adapted: bool = False,
) -> Tuple[List[Any], List[float]]:
    """
    Generate number-conserving one-body jump operators.
    spin_adapted=False: all unordered spin-orbital pairs (i,j) with Ax/Ay
    hops. Conserves N (and Sz if conserve_spin=True), but NOT total S^2.
    spin_adapted=True: singlet excitation operators per SPATIAL pair (p,q),
    E_pq + E_qp and i(E_pq - E_qp), realized as two commuting JW hops on the
    alpha strand (2p,2q) and beta strand (2p+1,2q+1) applied in the same kick.
    These conserve N, Sz AND S^2: dynamics started from a closed-shell
    determinant remains in the singlet subspace, so the fixed point is the
    lowest SINGLET even when a triplet lies below it (singlet-O2 chemistry).
    """
    n_system = int(n_system)
    pairs: List[Any] = []
    if spin_adapted:
        n_orb = n_system // 2
        for p in range(n_orb):
            for q in range(p + 1, n_orb):
                legs = ((2 * p, 2 * q), (2 * p + 1, 2 * q + 1))
                pairs.append((legs, "Ax"))
                pairs.append((legs, "Ay"))
    else:
        for i in range(n_system):
            for j in range(i + 1, n_system):
                if conserve_spin and (i % 2 != j % 2):
                    continue
                pairs.append((i, j, "Ax"))
                pairs.append((i, j, "Ay"))
    if not pairs:
        raise ValueError(
            f"No jump pairs were generated. n_system={n_system}, "
            f"conserve_spin={conserve_spin}, spin_adapted={spin_adapted}."
        )
    weights = [1.0 / len(pairs)] * len(pairs)
    return pairs, weights
def choose_target_energy(
    E_corr: float,
    E_N: Optional[float],
    E_NaNb: Optional[float],
    reference_sector: str,
    conserve_spin: bool,
    E_singlet: Optional[float] = None,
) -> Tuple[float, str]:
    """Choose the reference line used in plots and error columns."""
    mode = str(reference_sector or "auto").lower()
    if mode == "singlet":
        if E_singlet is None:
            raise ValueError("reference_sector='singlet' requested, but the lowest-singlet energy is unavailable.")
        return float(E_singlet), "Lowest singlet"
    if mode == "fock":
        return float(E_corr), "Full Fock-space ground"
    if mode == "n":
        if E_N is None:
            raise ValueError("reference_sector='N' requested, but E_fixed_N is unavailable.")
        return float(E_N), "Fixed-N ground"
    if mode == "nanb":
        if E_NaNb is None:
            raise ValueError("reference_sector='NaNb' requested, but E_fixed_NaNb is unavailable.")
        return float(E_NaNb), "Fixed Na/Nb ground"
    if mode != "auto":
        raise ValueError("reference_sector must be one of: auto, N, NaNb, Fock, singlet.")
    if conserve_spin and E_NaNb is not None:
        return float(E_NaNb), "Fixed Na/Nb ground"
    if E_N is not None:
        return float(E_N), "Fixed-N ground"
    return float(E_corr), "Full Fock-space ground"
def compute_filter_for_hamiltonian(
    H_mat: np.ndarray,
    SquareWaveFilter: Any,
    tau_filter: float,
    filter_K: int,
    filter_window: str,
    filter_wmax_factor: float,
) -> Dict[str, Any]:
    evals = np.linalg.eigvalsh(np.asarray(H_mat, dtype=complex))
    spectral_width = float(np.real(evals[-1] - evals[0]))
    wmax = filter_wmax_factor * max(spectral_width, 1e-12)
    sq_filter = SquareWaveFilter()
    data = sq_filter.get_coefficients(
        L=wmax,
        K=filter_K,
        tau=tau_filter,
        A=1.0,
        window=filter_window,
        drop_zeros=False,
        sign=+1,
    )
    data["wmax"] = wmax
    data["tau_filter"] = tau_filter
    data["filter_K"] = filter_K
    data["filter_window"] = filter_window
    data["filter_wmax_factor"] = filter_wmax_factor
    data["spectral_width"] = spectral_width
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
    E_singlet: Optional[float]
    initial_final_overlap: float
    n_system: int
    n_W_circuits: int
    n_energy_points: int
    final_energy: float
    target_energy: float
    target_label: str
    final_energy_error: float
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
        ax.plot(xs, ys, lw=0.5, marker="o", ms=3, label=f"H {g_idx}")
        target = rows[0].get("target_energy")
        if target is not None:
            ax.hlines(target, xs[0], xs[-1], linestyles=":", lw=1.0)
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
    if any(row.get("initial_energy") is not None for row in summary_rows):
        ys_init = [row.get("initial_energy", np.nan) for row in summary_rows]
        ax.plot(xs, ys_init, marker="v", lw=1.0, alpha=0.7, label="Initial energy (before W-chain)")
    if any(row.get("target_energy") is not None for row in summary_rows):
        ys_target = [row.get("target_energy", np.nan) for row in summary_rows]
        label = summary_rows[0].get("target_label", "Reference ground")
        ax.plot(xs, ys_target, marker="x", lw=1.2, label=label)
    if any(row.get("E_corr") is not None for row in summary_rows):
        ys_corr = [row.get("E_corr", np.nan) for row in summary_rows]
        ax.plot(xs, ys_corr, marker=".", lw=0.8, alpha=0.5, label="Full Fock-space ground")
    if any(row.get("E_fixed_NaNb") is not None for row in summary_rows):
        ys_sector = [row.get("E_fixed_NaNb", np.nan) for row in summary_rows]
        ax.plot(xs, ys_sector, marker="s", lw=0.8, alpha=0.7, label="Fixed Na/Nb ground")
    ax.set_xlabel("Hamiltonian index / geometry index")
    ax.set_ylabel("Energy")
    ax.set_title("Energy versus number of Hamiltonians")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
def plot_energy_error_vs_hamiltonian(summary_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    """|final_energy - target| per Hamiltonian, log scale, with the chemical-accuracy line."""
    if not summary_rows:
        return
    xs = [row["hamiltonian_index"] for row in summary_rows]
    errs = [max(abs(float(row["final_energy_error"])) * 1000.0, 1e-6) for row in summary_rows]  # mHa
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(xs, errs, marker="o", lw=1.5)
    ax.axhline(1.6, color="k", ls=":", lw=1.2, label="chemical accuracy (1.6 mHa)")
    ax.set_xlabel("Hamiltonian index / geometry index")
    ax.set_ylabel(r"$|E_{\rm final} - E_0^{(g)}|$ (mHa)")
    ax.set_title("Final energy error versus Hamiltonian index")
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
) -> None:
    atomic_pickle(circuits_by_geometry, outdir / "all_W_circuits_by_geometry.pkl")
    atomic_pickle(rho_by_geometry, outdir / "final_rhos_by_geometry.pkl")
    write_csv_rows(summary_rows, outdir / "energy_vs_hamiltonian_index.csv")
    write_csv_rows(energy_rows, outdir / "energy_vs_w_applications.csv")
    write_csv_rows(summary_rows, outdir / "overlap_vs_hamiltonian_index.csv")
    atomic_json(metadata, outdir / "run_metadata.json")
    plot_energy_vs_hamiltonian(summary_rows, outdir / "energy_vs_hamiltonian_index.png")
    plot_energy_error_vs_hamiltonian(summary_rows, outdir / "energy_error_vs_hamiltonian_index.png")
    plot_energy_vs_w(energy_rows, outdir / "energy_vs_w_applications.png")
    plot_overlap(summary_rows, outdir / "overlap_vs_hamiltonian_index.png")
# -----------------------------------------------------------------------------
# --- SPEED --- Cached sampled-circuit runner (local; does NOT touch one_ham)
# -----------------------------------------------------------------------------
def run_W_chain_circuit_cached(
    system_qubits,
    H_int,
    anc,
    tau,
    filter_terms,
    pairs,
    weights,
    N_applications,
    reps_per_W=None,
    occ_spin_orbs=None,
    trotter_steps_H=1,
    rng_seed=123,
    T_filter=None,
    H_mat_sys=None,
    initial_rho_sys=None,
    energy_stride=0,
):
    if T_filter is None:
        raise ValueError(
            "T_filter must be provided. "
            "The paper circuit distinguishes dissipative tau from filter time T."
        )
    if H_mat_sys is None:
        raise ValueError(
            "H_mat_sys must be provided explicitly to avoid using a stale global H_sys_mat."
        )
    qubits_all = [anc] + list(system_qubits)
    n_sys = len(system_qubits)
    d_sys = 2 ** n_sys
    if H_mat_sys.shape != (d_sys, d_sys):
        raise ValueError(
            f"H_mat_sys has shape {H_mat_sys.shape}, but system_qubits has "
            f"{n_sys} qubits, so expected {(d_sys, d_sys)}. "
            "You are mixing Hamiltonian/model variables."
        )
    if initial_rho_sys is None:
        rho_sys = linlin.hf_density_from_occupations(
            n_sys,
            occ_spin_orbs or [0, 1],
        )
    else:
        rho_sys = np.asarray(initial_rho_sys, dtype=complex)
    if rho_sys.shape != H_mat_sys.shape:
        raise ValueError(
            f"rho_sys has shape {rho_sys.shape}, but H_mat_sys has shape {H_mat_sys.shape}."
        )
    initial_energy = float(np.real(np.trace(rho_sys @ H_mat_sys)))
    steps = [0]
    energies = [initial_energy]
    print(f"initial circuit energy = {initial_energy:.10f}", flush=True)
    min_eig_in = np.linalg.eigvalsh(0.5 * (rho_sys + rho_sys.conj().T)).min()
    if min_eig_in < -1e-5:
        raise ValueError(
            f"initial rho_sys is non-physical. Minimum eigenvalue = {min_eig_in}"
        )
    energy_stride = int(energy_stride or 0)
    w_norm = np.array(weights, dtype=float)
    w_norm = w_norm / w_norm.sum()
    channel_cache = {} 
    W_ckt_list = []
    run_t0 = time.perf_counter()
    print(
        f"[chain] starting {N_applications} W applications on {n_sys}+1 qubits; "
        f"{len(pairs)} possible jump pairs; energies recorded at "
        + (f"every {energy_stride} steps + endpoints" if energy_stride > 0 else "endpoints only"),
        flush=True,
    )
    for t in range(N_applications):
        step_t0 = time.perf_counter()
        rng = np.random.default_rng(rng_seed + t)
        pair_idx = int(rng.choice(len(pairs), p=w_norm))
        chosen_pair = pairs[pair_idx]
        cache_state = "HIT"
        if pair_idx not in channel_cache:
            cache_state = "MISS"
            print(
                f"[W {t + 1}/{N_applications}] pair {chosen_pair}: cache MISS -> "
                f"building circuit (distinct pair #{len(channel_cache) + 1})...",
                flush=True,
            )
            build_t0 = time.perf_counter()
            W_circ = linlin.build_W_circuit(
                ancilla=anc,
                system_qubits=system_qubits,
                H_int=H_int,
                tau_s=tau,
                filter_terms=filter_terms,
                pairs=[chosen_pair],
                weights=[1.0],
                reps=reps_per_W,
                trotter_steps_H=trotter_steps_H,
                seed=0,
                T_filter=T_filter,
            )
            W_circ = cirq.Circuit(cirq.I.on_each(*qubits_all)) + W_circ
            print(
                f"[W {t + 1}/{N_applications}] circuit built: "
                f"{len(W_circ)} moments, "
                f"{sum(1 for _ in W_circ.all_operations())} operations, "
                f"time = {time.perf_counter() - build_t0:.1f} s; "
                f"computing unitary (one-off slow part for this pair)...",
                flush=True,
            )
            uni_t0 = time.perf_counter()
            U_full = W_circ.unitary(qubit_order=qubits_all, dtype=np.complex128)
            M0 = np.array(U_full[0:d_sys, 0:d_sys], copy=True)
            M1 = np.array(U_full[d_sys:2 * d_sys, 0:d_sys], copy=True)
            del U_full 
            channel_cache[pair_idx] = (W_circ, M0, M1)
            print(
                f"[W {t + 1}/{N_applications}] unitary + Kraus blocks cached in "
                f"{time.perf_counter() - uni_t0:.1f} s",
                flush=True,
            )
        W_circ, M0, M1 = channel_cache[pair_idx]
        W_ckt_list.append(W_circ)
        rho_sys = M0 @ rho_sys @ M0.conj().T + M1 @ rho_sys @ M1.conj().T
        rho_sys = 0.5 * (rho_sys + rho_sys.conj().T)
        rho_sys = rho_sys / np.trace(rho_sys)
        energy_text = "skipped"
        if energy_stride > 0 and (t + 1) % energy_stride == 0 and (t + 1) < N_applications:
            e_now = float(np.real(np.trace(rho_sys @ H_mat_sys)))
            steps.append(t + 1)
            energies.append(e_now)
            energy_text = f"{e_now:.10f}"
        step_dt = time.perf_counter() - step_t0
        elapsed = time.perf_counter() - run_t0
        eta = elapsed / (t + 1) * (N_applications - t - 1)
        print(
            f"[W {t + 1}/{N_applications}] done (cache {cache_state}, "
            f"pair {chosen_pair}): step = {step_dt:.2f} s; "
            f"elapsed = {elapsed / 60.0:.1f} min; est. remaining = {eta / 60.0:.1f} min; "
            f"{len(channel_cache)} distinct pairs cached; energy = {energy_text}",
            flush=True,
        )
    # ------------------------------------------------------------
    # Endpoint: one physicality check + the final energy.
    # ------------------------------------------------------------
    min_eig_out = np.linalg.eigvalsh(rho_sys).min()
    if min_eig_out < -1e-8:
        raise ValueError(
            f"channel chain produced non-physical rho_sys. "
            f"Minimum eigenvalue = {min_eig_out}."
        )
    final_energy = float(np.real(np.trace(rho_sys @ H_mat_sys)))
    steps.append(N_applications)
    energies.append(final_energy)
    print(
        f"[chain] COMPLETED {N_applications} W applications: "
        f"initial energy = {initial_energy:.10f}, final energy = {final_energy:.10f}, "
        f"reduction = {initial_energy - final_energy:.10f}; "
        f"{len(channel_cache)} distinct circuits built for {N_applications} steps; "
        f"total time = {(time.perf_counter() - run_t0) / 60.0:.1f} min",
        flush=True,
    )
    return steps, energies, W_ckt_list, rho_sys
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
    run_W_chain_averaged_matrix = getattr(linlin, "run_W_chain_averaged_matrix", None)
    hf_density_from_occupations = getattr(linlin, "hf_density_from_occupations", None)
    basis_fixed_N = getattr(linlin, "basis_fixed_N", None)
    basis_alpha_beta = getattr(linlin, "basis_alpha_beta", None)
    ground_energy_in_subspace = getattr(linlin, "ground_energy_in_subspace", None)
    linlin_path = Path(linlin.__file__).resolve()
    xyz_path = resolve_xyz_trajectory(args.xyz_trajectory)
    source = str(xyz_path)
    geometries_all = load_xyz_trajectory(xyz_path)
    displacements_bohr = cumulative_displacements_bohr(
        geometries_all
    )
    x_min = float(args.displacement_min_bohr)
    x_max = float(args.displacement_max_bohr)

    selected_indices = [
        i
        for i, x_i in enumerate(displacements_bohr)
        if x_min < x_i < x_max
    ]

    if not selected_indices:
        raise ValueError(
            f"No geometries lie in the displacement range "
            f"{x_min} < x < {x_max} Bohr."
        )

    geometries = [geometries_all[i] for i in selected_indices]
    print(
    f"Selected {len(geometries)} geometries out of "
    f"{len(geometries_all)} total geometries.",
    flush=True,
    )

    print(
        f"Selected original geometry indices: {selected_indices}",
        flush=True,
    )

    print(
        f"Displacement window: {x_min} < x < {x_max} Bohr",
        flush=True,
    )
    max_geom_step = None if args.max_geom_step in (None, "none", "None") else float(args.max_geom_step)
    geometries, selected_indices, interpolated_flags = refine_geometry_path(
        geometries, selected_indices, max_geom_step
    )
    n_interpolated = int(sum(interpolated_flags))
    if n_interpolated:
        print(f"[path-refine] inserted {n_interpolated} interpolated frame(s) "
              f"(max_geom_step={max_geom_step} A); {len(geometries)} geometries total.", flush=True)
    selector_info = {
    "selector": "displacement_window_bohr",
    "n_original": len(geometries_all),
    "n_selected": len(geometries),
    "displacement_min_bohr": x_min,
    "displacement_max_bohr": x_max,
    "selected_indices": selected_indices,
    "selected_displacements_bohr": [
        float(displacements_bohr[i])
        for i in selected_indices
        if i < len(displacements_bohr)
    ],
    "max_geom_step": max_geom_step,
    "n_interpolated_frames": n_interpolated,
    "interpolated_flags": interpolated_flags,
    "align_orbitals": bool(args.align_orbitals),
    }
    if not geometries:
        raise ValueError("No geometries selected from the trajectory.")
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
        "status": "running",
    }
    atomic_json(metadata, outdir / "run_metadata.json")
    rho_current: Optional[np.ndarray] = None
    align_prev: Optional[Dict[str, Any]] = None
    circuits_by_geometry: List[List[Any]] = []
    rho_by_geometry: List[np.ndarray] = []
    summary_rows: List[Dict[str, Any]] = []
    energy_rows: List[Dict[str, Any]] = []
    filter_cache: Optional[Dict[str, Any]] = None
    cumulative_W = 0
    run_t0 = time.time() 
    try:
        for h_idx, (original_idx, geometry) in enumerate(zip(selected_indices, geometries)):
            t_geom = time.time() 
            print(f"\n===== Hamiltonian/geometry {h_idx}/{len(geometries)-1} | original index {original_idx} =====", flush=True)
            print(geometry, flush=True)
            print(f"[alive] building Hamiltonian for geometry {h_idx} (SCF/CASSCF + JW + dense matrix)...", flush=True) 
            Ham = HamiltonianBuilder().build_molecule(
                geometry=geometry,
                basis=args.basis,
                multiplicity=args.multiplicity,
                charge=args.charge,
                active_indices=parse_int_list(args.active_indices),
                ancilla_index=args.ancilla_index,
                n_active_electrons=parse_optional_int(args.n_active_electrons),
                n_active_orbitals=parse_optional_int(args.n_active_orbitals),
                use_casscf_orbitals=bool(args.use_casscf_orbitals),
                use_procrustes_gauge=bool(args.align_orbitals),
                align_reference=align_prev,
                fix_spin_ss=None if str(args.fix_spin_ss).lower() in ("none", "") else float(args.fix_spin_ss),
            )
            print(f"[alive] Hamiltonian ready for geometry {h_idx} "
                  f"({time.time() - t_geom:.1f} s so far this geometry)", flush=True) 
            if bool(args.align_orbitals):
                align_prev = {
                    "mol": Ham.scf_mol,
                    "mo_coeff": Ham.mo_coeff,
                    "mo_occ": Ham.mo_occ,
                }
            interaction_ham = Ham.interaction
            H_mat = Ham.matrix
            if int(Ham.n_system) > 12:
                print(f"WARNING: n_system={Ham.n_system} spin orbitals -> dense "
                      f"{2**int(Ham.n_system)}-dim matrices; this workflow is exact-diagonalization "
                      "based and will be very slow/memory-heavy. Use an active space "
                      "(n_active_electrons/n_active_orbitals) to reduce the size.", flush=True)
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
            if rho_current is None:
                if hf_density_from_occupations is None:
                    raise KeyError(
                        "`hf_density_from_occupations` was not found in LinLin. "
                        "Needed to explicitly save the initial density matrix."
                    )
                rho_in = np.asarray(hf_density_from_occupations(len(system_qubits), occ_spin_orbs))
            else:
                rho_in = np.asarray(rho_current)
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
                print(f"Saved initial state at final geometry -> {outdir / 'initial_state_at_final_geometry.npy'}", flush=True)
            if args.use_global_filter and filter_cache is not None:
                filter_data = filter_cache
            else:
                print(f"[alive] computing square-wave filter for geometry {h_idx} "
                      f"(full eigvalsh of the {H_mat.shape[0]}-dim Hamiltonian)...", flush=True)  
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
                conserve_spin=bool(args.conserve_spin),
                spin_adapted=bool(args.spin_adapted_jumps),
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
            E_singlet = None
            need_singlet = (str(args.reference_sector).lower() == "singlet") or bool(args.spin_adapted_jumps)
            if need_singlet and n_electrons is not None and n_electrons % 2 == 0:
                lowest_spin_state_energy = getattr(linlin, "lowest_spin_state_energy", None)
                if lowest_spin_state_energy is not None:
                    E_singlet = float(lowest_spin_state_energy(H_mat, n_system, n_electrons, ss_target=0.0))
            target_energy, target_label = choose_target_energy(
                E_corr=E_corr,
                E_N=E_N,
                E_NaNb=E_NaNb,
                reference_sector=args.reference_sector,
                conserve_spin=bool(args.conserve_spin),
                E_singlet=E_singlet,
            )
            print(f"E_corr/full Fock-space ground = {E_corr:.12f}", flush=True)
            if E_N is not None:
                print(f"Fixed N={n_electrons} ground = {E_N:.12f}", flush=True)
            if E_NaNb is not None:
                print(f"Fixed Na={Na}, Nb={Nb} ground = {E_NaNb:.12f}", flush=True)
            if E_singlet is not None:
                print(f"Lowest singlet = {E_singlet:.12f}", flush=True)
            print(f"Target reference = {target_label}: {target_energy:.12f}", flush=True)
            print(f"Filter: wmax={filter_data['wmax']:.8f}, T_filter={T_filter:.8f}, terms={len(filter_terms)}", flush=True)
            print(f"Generated {len(pairs)} jump pairs from occ_spin_orbs={occ_spin_orbs}; conserve_spin={bool(args.conserve_spin)}", flush=True)
            if args.evolution_mode == "sampled_circuit":
                steps, energies, W_circuits, rho_current = run_W_chain_circuit_cached(
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
                    energy_stride=int(args.energy_stride),
                )
            elif args.evolution_mode == "averaged_matrix":
                if run_W_chain_averaged_matrix is None:
                    raise RuntimeError("one_ham.py does not provide run_W_chain_averaged_matrix.")
                steps, energies, W_circuits, rho_current = run_W_chain_averaged_matrix(
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
            else:
                raise ValueError("evolution_mode must be sampled_circuit or averaged_matrix.")
            rho_out = np.asarray(rho_current)
            overlap = density_overlap(rho_in, rho_out)
            final_energy = float(energies[-1]) if len(energies) else float("nan")
            initial_energy = float(energies[0]) if len(energies) else float("nan")  # --- SPEED ---
            circuits_by_geometry.append(W_circuits)
            rho_by_geometry.append(rho_out)
            record = GeometryRecord(
                hamiltonian_index=h_idx,
                original_geometry_index=int(original_idx),
                E_corr=E_corr,
                E_fixed_N=E_N,
                E_fixed_NaNb=E_NaNb,
                E_singlet=E_singlet,
                initial_final_overlap=overlap,
                n_system=n_system,
                n_W_circuits=len(W_circuits),
                n_energy_points=len(energies),
                final_energy=final_energy,
                target_energy=float(target_energy),
                target_label=target_label,
                final_energy_error=float(final_energy - target_energy),
                wmax=float(filter_data["wmax"]),
                T_filter=float(T_filter),
                n_filter_terms=len(filter_terms),
            )
            summary_row = asdict(record)
            summary_row["initial_energy"] = initial_energy
            summary_row["energy_reduction"] = float(initial_energy - final_energy)
            summary_rows.append(summary_row)
            for local_step, energy in zip(steps, energies):
                energy_rows.append({
                    "hamiltonian_index": h_idx,
                    "original_geometry_index": int(original_idx),
                    "local_W_applications": int(local_step),
                    "cumulative_W_applications": int(cumulative_W + local_step),
                    "energy": float(energy),
                    "E_corr": E_corr,
                    "E_fixed_N": E_N,
                    "E_fixed_NaNb": E_NaNb,
                    "E_singlet": E_singlet,
                    "target_energy": target_energy,
                    "target_label": target_label,
                    "energy_error": float(energy - target_energy),
                })
            if len(steps):
                cumulative_W += int(steps[-1])
            metadata["last_completed_hamiltonian_index"] = h_idx
            if str(args.save_outputs) == "per_geometry":
                metadata["status"] = "running_partial_saved"
                save_partial_outputs(outdir, circuits_by_geometry, rho_by_geometry, summary_rows, energy_rows, metadata)
                print(f"Saved partial outputs after geometry {h_idx} -> {outdir}", flush=True)
            else:
                metadata["status"] = "running_outputs_deferred"
                atomic_json(metadata, outdir / "run_metadata.json")
                print(f"Completed geometry {h_idx} (outputs deferred to end of run).", flush=True)
            geom_dt = time.time() - t_geom
            elapsed = time.time() - run_t0
            n_done = h_idx + 1
            n_left = len(geometries) - n_done
            eta = (elapsed / n_done) * n_left
            print(
                f"[timing] geometry {h_idx} finished in {geom_dt / 3600.0:.2f} h "
                f"({geom_dt / 60.0:.1f} min); total elapsed {elapsed / 3600.0:.2f} h; "
                f"est. remaining for {n_left} geometrie(s): {eta / 3600.0:.2f} h",
                flush=True,
            )
        metadata["status"] = "completed"
        save_partial_outputs(outdir, circuits_by_geometry, rho_by_geometry, summary_rows, energy_rows, metadata)
        print(f"\nCompleted. Outputs saved in: {outdir}", flush=True)
    except Exception as exc:
        metadata["status"] = "failed_after_partial_save"
        metadata["error_type"] = type(exc).__name__
        metadata["error_message"] = str(exc)
        metadata["traceback"] = traceback.format_exc()
        save_partial_outputs(outdir, circuits_by_geometry, rho_by_geometry, summary_rows, energy_rows, metadata)
        (outdir / "ERROR.txt").write_text(metadata["traceback"], encoding="utf-8")
        print("\nERROR occurred. Partial outputs were saved before exiting.", file=sys.stderr, flush=True)
        print(metadata["traceback"], file=sys.stderr, flush=True)
        raise
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean LinLin reaction-path W-chain driver.")
    p.add_argument(
        "--xyz-trajectory",
        default=USER_CONFIG["xyz_trajectory"],
        help=(
            "ORCA/XYZ multi-frame trajectory, e.g. MEP_trj.xyz. "
            "May also be a glob like '*trj.xyz'. If omitted, the script searches "
            "the current directory and requires exactly one trajectory-like XYZ file."
        ),
    )
    p.add_argument("--geometry-start", type=int, default=USER_CONFIG["geometry_start"], help="First trajectory frame index to use, 0-based.")
    p.add_argument("--geometry-stop", type=int, default=USER_CONFIG["geometry_stop"], help="Exclusive stop frame index. Default: end of trajectory.")
    p.add_argument("--geometry-stride", type=int, default=USER_CONFIG["geometry_stride"], help="Use every stride-th frame from the trajectory.")
    p.add_argument(
        "--include-final-geometry",
        action=argparse.BooleanOptionalAction,
        default=USER_CONFIG["include_final_geometry"],
        help="Always include the final selected frame even when stride skips it.",
    )
    p.add_argument("--basis", default=USER_CONFIG["basis"])
    p.add_argument("--multiplicity", type=int, default=USER_CONFIG["multiplicity"], help="Spin multiplicity. Required; no H3/default spin is assumed.")
    p.add_argument("--charge", type=int, default=USER_CONFIG["charge"], help="Molecular charge. Required; no molecule-specific charge is assumed.")
    p.add_argument("--active-indices", default=USER_CONFIG["active_indices"], help="Comma-separated active indices, or None.")
    p.add_argument("--ancilla-index", type=int, default=USER_CONFIG["ancilla_index"])
    p.add_argument("--n-active-electrons", default=USER_CONFIG["n_active_electrons"], help="Integer or None.")
    p.add_argument("--n-active-orbitals", default=USER_CONFIG["n_active_orbitals"], help="Integer or None.")
    p.add_argument("--use-casscf-orbitals", action=argparse.BooleanOptionalAction, default=USER_CONFIG["use_casscf_orbitals"])
    p.add_argument("--max-geom-step", default=USER_CONFIG["max_geom_step"], help="Max atomic displacement (Angstrom) between consecutive geometries before interpolated frames are inserted. 'none' disables path refinement.")
    p.add_argument("--align-orbitals", action=argparse.BooleanOptionalAction, default=USER_CONFIG["align_orbitals"], help="Procrustes orbital-gauge alignment along the path (Prop. 1). Full-space (no active space) runs only.")
    p.add_argument("--n-electrons", default=USER_CONFIG["n_electrons"], help="Electron number for sector diagnostics and default HF occupations. Required unless --occ-spin-orbs is supplied.")
    p.add_argument("--Na", default=USER_CONFIG["Na"], help="Alpha electron count for Na/Nb subspace diagnostics. Optional.")
    p.add_argument("--Nb", default=USER_CONFIG["Nb"], help="Beta electron count for Na/Nb subspace diagnostics. Optional.")
    p.add_argument("--occ-spin-orbs", default=USER_CONFIG["occ_spin_orbs"], help="Comma-separated occupied spin orbitals for the initial HF density. Default: range(n_electrons).")
    p.add_argument("--tau-filter", type=float, default=USER_CONFIG["tau_filter"], help="Tau used in SquareWaveFilter coefficients.")
    p.add_argument("--filter-K", type=int, default=USER_CONFIG["filter_K"])
    p.add_argument("--filter-window", default=USER_CONFIG["filter_window"])
    p.add_argument("--filter-wmax-factor", type=float, default=USER_CONFIG["filter_wmax_factor"])
    p.add_argument("--use-global-filter", action=argparse.BooleanOptionalAction, default=USER_CONFIG["use_global_filter"], help="Use first geometry's filter for all geometries. Default recomputes per geometry.")
    p.add_argument("--tau-W", type=float, default=USER_CONFIG["tau_W"], help="tau argument passed to the W chain.")
    p.add_argument("--N-applications", type=int, default=USER_CONFIG["N_applications"])
    p.add_argument("--reps-per-W", type=int, default=USER_CONFIG["reps_per_W"])
    p.add_argument("--trotter-steps-H", type=int, default=USER_CONFIG["trotter_steps_H"])
    p.add_argument("--rng-seed", type=int, default=USER_CONFIG["rng_seed"])
    p.add_argument("--evolution-mode", choices=["sampled_circuit", "averaged_matrix"], default=USER_CONFIG["evolution_mode"], help="sampled_circuit is one stochastic branch; averaged_matrix is the deterministic Lindblad-average channel.")
    p.add_argument("--energy-stride", type=int, default=USER_CONFIG["energy_stride"], help="sampled_circuit only: 0 = record energy at endpoints of each geometry only; m > 0 = also record every m steps.")
    p.add_argument("--save-outputs", choices=["per_geometry", "at_end"], default=USER_CONFIG["save_outputs"], help="Write CSVs/pickles/plots after every geometry (crash-safe) or once at the end (faster for sampled_circuit).")
    p.add_argument("--reference-sector", choices=["auto", "N", "NaNb", "Fock", "singlet"], default=USER_CONFIG["reference_sector"], help="Which ground energy to use as the convergence target in plots. 'singlet' = lowest S^2~0 state (use with --spin-adapted-jumps).")
    p.add_argument("--spin-adapted-jumps", action=argparse.BooleanOptionalAction, default=USER_CONFIG["spin_adapted_jumps"], help="Use S^2-conserving singlet excitation jump operators E_pq (alpha+beta strands per kick). Keeps closed-shell dynamics in the singlet subspace.")
    p.add_argument("--fix-spin-ss", default=USER_CONFIG["fix_spin_ss"], help="Pin the CASSCF CI root to S(S+1)=ss (0=singlet). 'none' disables.")
    p.add_argument("--conserve-spin", action=argparse.BooleanOptionalAction, default=USER_CONFIG["conserve_spin"], help="Restrict jumps to same spin-orbital parity. Use with --reference-sector NaNb.")
    p.add_argument("--displacement-min-bohr", type=float, default=USER_CONFIG["displacement_min_bohr"])
    p.add_argument("--displacement-max-bohr", type=float, default=USER_CONFIG["displacement_max_bohr"])

    p.add_argument("--outdir", default=USER_CONFIG["outdir"], help="Output directory.")
    return p
if __name__ == "__main__":
    run(build_arg_parser().parse_args())