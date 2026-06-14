# Chemical_rxn

This repository contains folders for different chemical reactions. Each folder includes the required input files and geometries for generating a sequence of structures between the reactant and transition state (TS).

## Folder Contents

Each reaction folder contains: 

### `*_MEP_trj.xyz` (Most important for further resource estimation)

This file contains the minimum energy path (MEP) trajectory.

It can be used for:

1. Visualizing the reaction animation.
2. Resuming the reaction in the forward direction.
3. Providing the full sequence of geometries as input for the dissipative algorithm.

### `*.interp` 

This file is used as input for `ORCA-NEB.ipynb` to plot the MEP.

To use it:

1. Place the `*.interp` file and `ORCA-NEB.ipynb` in the same directory.
2. Run the notebook to generate the MEP plot.

## Notes

- Each folder corresponds to a separate reaction.
- Reactant and TS geometries are provided to construct the reaction pathway.
- The `*_MEP_trj.xyz` file includes all intermediate geometries along the path.
