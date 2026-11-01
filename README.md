## SeismoSim

SeismoSim is a Python based structural dynamics simulator that models how buildings respond to earthquake ground motion using numerical methods. The project focuses on single degree of freedom (SDOF) systems and lets you see how mass, stiffness and damping affect a structure's dynamic response.

## my comments

so i initially started with eulers method, it was what i knew before actually going in coding more deeply - after researching i got to know that there are more methods to do a lower error for approximation. thats how i learned RK4. i am thinking of improving this model to MDOF, i am AMAZED by initial value problems <3

## Features
- Free vibration of an SDOF system (any damping level)
- Response to a ground acceleration record (time history of relative displacement, relative velocity and absolute acceleration)
- Elastic response spectra (Sd, pseudo-velocity, pseudo-acceleration, absolute acceleration)
- Ground motion loading from text/CSV files, plus synthetic signals (harmonic, sine pulse)
- Closed-form solutions and a convergence study that validate the RK4 solver
- Plots with matplotlib

## Installation
Python 3.10 or newer is recommended.
- pip install -r requirements.txt

## How to use
 python main.py
Select one of the 3 modes
- (1) Free vibration: enter mass (kg), stiffness (N/m), damping (Ns/m), then initial conditions and duration.
- (2) Earthquake response: enter the system, then the path of a ground motion record.
- (3) Response spectrum: enter a record, a damping ratio and a period range.

Then, run the RK4 
- python converge.py

## Ground motion file format
- Two columns: time, acceleration. The time step is read from the file and the time column must be strictly increasing and uniformly spaced.
- One column: acceleration only. The time step is asked for.
- Separators: spaces, commas or semicolons. Lines that are not numeric (headers, comments) are skipped.
- Units: acceleration in g or m/s2. Internally everything is m/s².
- Files with NaN or infinite values, inconsistent column counts, or more than two columns are rejected.
- Multi-value-per-row formats such as PEER NGA .AT2 are not supported yet.

## Structure
- model.py, SDOF system class (frequency, period, damping ratio)
- simulation.py, RK4 free vibration and base excitation solvers
- ground_motion.py, Record loading and synthetic ground motions
- spectrum.py, Elastic response spectrum
- analytical.py, Closed form solutions used for validation 
- visualization.py, plotting functions (fav part)
- converge.py, RK4 convergence analysis
- main.py, command line menu
- test.py, unit and integration tests

## Method
The equation of motion m u'' + c u' + k u = -m a_g(t) is integrated with a fourth order RK. The ground acceleration is linearly interpolated between samples. For stiff (short period) systems, each sampling interval is split into substeps so that wn * h <= 0.1, which keeps RK4 stable and accurate

## Tests
- python -m unittest -v

The suite checks the solver against closed form solutions and an independent Newmark beta implementation, verifies linearity, superposition, energy conservation and the observed convergence order (about 4), and exercises file loading and the menu with mocked input.

## Limitations
- SDOF, linear elastic systems only
- Uniformly sampled records on
