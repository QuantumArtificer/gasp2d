# orbital-spectrum

**orbital-spectrum** is a lightweight, efficient toolkit for analyzing the orbital (angular harmonic) decomposition of discrete (numerical) and continuous (analytic) functions.

The initial focus is on 2D fields, with a 3D extension on the way. The library performs an angular spectral decomposition, separating a field into angular harmonics and their associated radial components. This lets you:

- Inspect the angular character and radial extent of a function visually.
- Perform robust symmetry analysis in condensed matter and materials science contexts.
- Use the radial components directly in calculations (e.g. electrostatics) to reduce the number of variables.

Built-in tooling allows you to **prune angular-momentum channels** using a physically meaningful fractional power criterion on the radial components. This helps you minimize computational load while retaining the dominant physics in interaction or energy calculations.

Developed by **Alex Santacruz** (PhD student, 2DQMAT Research @ IF-UNAM).
