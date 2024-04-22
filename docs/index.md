## Welcome to GRANAD

GRANAD (GRAphene Nanoflakes with ADatoms) is a tight-binding simulation utility geared towards the exploration of systems at the intersection of solid state physics and quantum optics.

### Features

- Computes optical and electronic properties 
- Full access to time-dependent density matrices via master equation
- Relies on [JAX](https://jax.readthedocs.io/en/latest/) for performance and differentiability

### Installation

If you want to just install the package, run

```bash
pip install git+https://github.com/GRANADlauncher/granad.git
```

The documentations is built automatically by executing

```bash
git clone https://github.com/GRANADlauncher/granad.git
cd granad
bash install.sh
```

### Quickstart

Set up the simulation


```python
import jax.numpy as jnp
from granad import Material2D, Triangle

# get material
graphene = Material2D.get( "graphene" )

# cut a 15 Angström wide triangle from the lattice (can also be an arbitrary polygon)
flake = graphene.cut_orbitals( Triangle(15)  ) 

# frequencies
omegas = jnp.linspace( 0, 5, 50 )

# compute optical properties in the RPA with GPU-acceleration
polarizability = flake.get_polarizability_rpa(
    omegas,
    relaxation_rate = 1/10,
    polarization = 0,
    hungry = True )
absorption = polarizability.imag * 4 * jnp.pi * omegas
```

Plot the results


```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plt.plot(omegas, absorption / jnp.max(absorption), linewidth=2)
plt.xlabel(r'$\hbar\omega$', fontsize=20)
plt.ylabel(r'$\sigma(\omega)$', fontsize=25)
plt.title('Absorption Spectrum as a Function of Photon Energy', fontsize=15)
plt.grid(True)
```


    
![png](index_files/index_8_0.png)
    

