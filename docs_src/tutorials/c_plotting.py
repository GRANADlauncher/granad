# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: base
# ---

# # Plotting
#
# We introduce built-in plotting functions. 
#
# A typical simulation requires visualizing:

# 1. geometry
# 2. simulation results
# 3. space-dependent arrays (such as eigenstates, charges, ...)

# All public plotting functions are associated with a list of orbitals.

### Geometry
# 
# Geometries can be visualized in 2D or 3D.

# +
from granad import MaterialCatalog, Rectangle

flake = MaterialCatalog.get("graphene").cut_flake( Rectangle(10, 10) )
flake.show_2d()
# -

# If we have a stack

# +
flake_shifted = MaterialCatalog.get("graphene").cut_flake( Rectangle(10, 10) )
flake_shifted.shift_by_vector( [0,0,1] )
stack = flake + flake_shifted
stack.show_3d()
# -

### Simulation output

# There is a dedicated function for showing time-domain simulation results. Let's illustrate this by tracking the current considering the previously defined flake under x-polarized Gaussian pulsed illumination, which in GRANAD's units is equivalent to the velocity operator given by the commutator of the system hamiltonian $H$ and the polarization operator $P$

# $$\bf{v} = \frac{-i}{\hbar} [H, \bf{P}]$$

# The time-dependent induced current (density) is given by the expectation value of the current density operator $j$, which is related to the velocity operator as $j = ev$. In GRANAD's units, setting the elementary charge $e = 1$, the velocity operator is thus identical to the current density operator, with which we can immediately evalute the induced currents due to the external field.

# +
from granad import Pulse
pulse = Pulse(
    amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
)
result = flake.master_equation(
    illumination=pulse,
    expectation_values = [flake.velocity_operator],
    end_time=40,
    relaxation_rate=1/10,
    grid=100, # save every 100-th time step
)
plot_labels = ['j_x', 'j_y', 'j_z'] 
flake.show_res( result, plot_labels = plot_labels )
# -

# A detailed description of parameters to the Gaussian pulse is given in the electric fields tutorial.

# We can also Fourier transform the result to obtain the frequency domain result by specifying the omega limits

# +
flake.show_res( result, omega_max = 5, omega_min = 0, plot_labels = plot_labels )
# -

# Note that the sparsity of the spectral grid ows to the sparse sampling in time domain which we have chosen for demonstrational purposes. A real simulation requires more careful refinement of the temporal grid. A finer temporal grid or longer integration time generally results in higher quality spectra.

### Space-dependent arrays

# The functions show_2d and show_3d are a bit more versatile than initially indicated.

# If we supply a "display" argument, we can plot an arbitrary array on the grid spanned by the orbitals and filter it by orbital tags. Let's demonstrate this by visualizing the lowest energy single-particle state of the flake

# +
flake.show_2d( display = flake.eigenvectors[:, 0] )
# -

# If you want a normalized plot of the absolute values, do

# +
flake.show_2d( display = flake.eigenvectors[:, 0], scale = True )
# -

# Additionally, if you supply the keyword argument name = "MY-PLOT-NAME.pdf" to any plotting function, the plot will not be displayed, but instead saved to disk in the directory you invoked Python.

# There exists a convenience function to compute and visualize the induced field at an arbitrary time step. 
# To achieve this, we first run a full simulation of the system using the `flake.master_equation` function
# and extract the density matrix at the last time step.

# +
result = flake.master_equation(
    relaxation_rate = 1/10,
    illumination = pulse,
    end_time = 40,
    density_matrix = ["full"], # this argument may be removed
     )
density_matrix = result.output[-1]
# -

# Next, we define the extent of a 2D grid. This grid represents the spatial domain where the induced field 
# will be evaluated. The grid points are created to encompass the range of particle positions in the x and y 
# dimensions, extended slightly to include a buffer region.

# +
import jax.numpy as jnp
xmin, xmax = flake.positions[:, 0].min(), flake.positions[:, 0].max()  # Compute the minimum and maximum x-coordinates 
                                                                       # of the particle positions.
x_grid = jnp.linspace(xmin - 1, xmax + 1, 40)  # Create a 1D array of 40 evenly spaced points in the x-direction, 
                                               # extending slightly beyond the range of the particles.

ymin, ymax = flake.positions[:, 1].min(), flake.positions[:, 1].max()  # Compute the minimum and maximum y-coordinates 
                                                                       # of the particle positions.
y_grid = jnp.linspace(ymin - 1, ymax + 1, 40)  # Create a 1D array of 40 evenly spaced points in the y-direction, 
                                               # extending slightly beyond the range of the particles.
# -

# Finally, we use the `show_induced_field` function to compute and visualize the induced field on the 
# defined 2D grid. This function calculates the field based on the density matrix, evaluates it on the grid, 
# and plots the normalized logarithmic magnitude of the induced field.

# +
flake.show_induced_field(
    x = x_grid,              # The x-coordinates of the grid points where the field will be evaluated.
    y = y_grid,              # The y-coordinates of the grid points where the field will be evaluated.
    component = 0,           # Specifies which component (direction) of the field to visualize. 
                             # Here, the x-component (0) is selected.
    z = jnp.array([0]),      # The z-coordinate of the plane in which the field is evaluated. 
                             # Here, it is set to 0 to visualize the field in the xy-plane.
    density_matrix = density_matrix[-1]  # The final density matrix from the simulation is passed as an argument, 
                                         # ensuring that the field reflects the system's state at the final time step.
)
# -
