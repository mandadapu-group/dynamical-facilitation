# Kinteic Monte Carlo - Facilitation
Modified kinetic Monte Carlo (kMC) code for running simulation of elastically interacting excitations. 

## How to Use
To compile the program, you need a C compiler, such as `gcc`, `clang`, etc. Use the following command:
```bash
gcc kmcres.c gsd.c -o kmcres.exe -lm
```
### Running the Program
The executable requires 5 command-line arguments:
```bash
kmcres.exe (L) (kappa) (betaJ) (filename) (gsd_mode)
```
- `L`: System size
- `kappa`: Interaction strength parameter
- `betaJ`: Onverse temperature parameter
- `filename`: Base name for output files
- `gsd_mode`: Set to `on` or `off`. If `on`, the program generates a GSD file for trajectory data.

### Output Files

Note that this program will generates 4 files.
1. `filename.log`:

Contains:
- Number of timesteps
- Simulation time
- Bond-order correlation
- Self-intermediate scattering function
- Mean-squared displacement
- Persistence function
- Number of excitation
- Boolean flag indicating whether the threshold has been reached.

2. `filename.bin`
Checkpoint file used for restarting simulations.

3. `filename.fin`
An empty file created when the simulation ends normally. Useful as a completion flag.

4. `filename.gsd` (only if `gsd_mode` is `on`)
Stores trajectory data in GSD format.
- The particles/diameter data contains persistence variable of each site, 0 or 1.
- The particles/typid data contains 0 for type and 1 for type B, where type B corresponds to lattice sites with excitations.

## Notes
- Ensure the `gsd.c` source file is included and properly linked.
- The program assumes a lattice-based system in 2 dimension with kinetic constraints.
- GSD files can be visualized using Ovito or analyzed with Python libraries such as `gsd` and `hoomd`.
