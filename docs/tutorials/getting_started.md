## Getting started

We introduce the basics of GRANAD and do a quick simulation.


## Orbitals

At its core, GRANAD is all about orbitals. Let's create one at the origin and inspect it.



```python
from granad import Orbital

my_first_orbital = Orbital(
    position = (0, 0, 0),
    tag = "a tag contains arbitrary information",
)
print(my_first_orbital)
```

    {'position': Array([0., 0., 0.], dtype=float64),
     'layer_index': None,
     'tag': 'a tag contains arbitrary information',
     'energy_level': None,
     'angular_momentum': None,
     'angular_momentum_z': None,
     'spin': None,
     'atom_name': None,
     'group_id': 1}


The group_id, unsurprisingly, groups orbitals. For example: if you create a two-level adatom, you need two orbitals that share the same group_id. In the same way, all orbitals in a graphene sheet share the same group_id.


```python

my_second_orbital = Orbital(
    position = (1, 1, 1),
    tag = "a new, shifted orbital",
)
print(my_second_orbital)

```

    {'position': Array([1., 1., 1.], dtype=float64),
     'layer_index': None,
     'tag': 'a new, shifted orbital',
     'energy_level': None,
     'angular_momentum': None,
     'angular_momentum_z': None,
     'spin': None,
     'atom_name': None,
     'group_id': 2}


This is all there is to know about orbitals!


## Materials

Materials are stuff you can cut orbitals from. We will see below how this works.


```python
from granad import Material2D
Material2D.available()
```

    Available materials:
    graphene


Let's pick a material and move on.


```python
graphene = Material2D.get("graphene")
```

## OrbitalLists

OrbitalLists are the last class you need to know. Unsurprisingly, an OrbitalList is a list of orbitals. You can create one yourself from the two orbitals you created above


```python
from granad import OrbitalList

my_first_orbital_list = OrbitalList([my_first_orbital, my_second_orbital])
print(my_first_orbital_list)
```

    List with 2 orbitals, 2 electrons.
    [0] electrons excited from [0] to [0].
    group id 1 : 1 orbitals
    group id 2 : 1 orbitals


Alternatively, you get orbital lists if you cut a flake from a material. You do this by specifying the shape of the flake.
You can specify any shape you want, but this is covered in a separate tutorial.
For now, we will use a built-in shape: an equilateral triangle with a side length of 15 Angström.


```python
from granad import Triangle
import jax.numpy as jnp
triangle = Triangle(15, armchair = True) + jnp.array([10,10])
```

Now, our shape is ready and we can start cutting. To make sure that we are satisfied with what we get, we plot the flake. By default, GRANAD cuts any "dangling" atoms.


```python
my_first_flake = graphene.cut_orbitals(triangle, plot = True)
print(my_first_flake)
```


    
![png](getting_started_files/getting_started_15_0.png)
    


    List with 36 orbitals, 36 electrons.
    [0] electrons excited from [0] to [0].
    group id 3 : 36 orbitals


For more information on cutting, including different edge types and how to keep dangling atoms, have a look at the corresponding tutorial.

## A first simulation

To get a feeling of the setup, we first inspect the energies of the flake


```python
my_first_flake.show_energies()
```


    
![png](getting_started_files/getting_started_19_0.png)
    


Let's say we want to compute the absorption spectrum of the flake. One way of doing this is based on integrating a modified von-Neumann equation in time-domain. You can do this in two steps: 

1. Excite the flake with an electric field.
2. Compute its dipole moment $p(\omega)$ from the expectation value of its dipole operator to obtain the absorption spectrum from $Im[p(\omega)]$.

We first do step 1. To make sure that we get a meaningful spectrum, we must pick an electric field with many frequencies, i.e. a narrow pulse in time-domain


```python
from granad import Pulse

my_first_illumination = Pulse(
    amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
)  # TODO: units
```

For step 2, we have to think of a few parameters:

1. TD simulation duration: we go from 0 to 40 in 1e5 steps.
2. Relaxation rate: this is $r$ in the dissipation term $D[\rho] = r \cdot(\rho - \rho_0)$.
3. Frequency domain limits: we choose the interval [0, 16].
4. Density matrix sampling rate: producing 1e5 density matrices can quickly tire our RAM. so we only save every 100th density matrix, such that we get 1000 density matrices.

A simulation is just passing all of these parameters to the corresponding method of our flake.


```python
omegas, dipole_omega, pulse_omega = (
    my_first_flake.get_expectation_value_frequency_domain(
        operator=my_first_flake.dipole_operator,  # the dipole moment is the expectation value of the dipole operator
        end_time=40,
        steps_time=1e5,
        relaxation_rate=1 / 10,
        illumination=my_first_illumination,
        omega_min=0,
        omega_max=10,
        skip=100,
    )
)
```

    /home/david/miniconda3/lib/python3.9/site-packages/equinox/_jit.py:49: UserWarning: Complex dtype support is work in progress, please read https://github.com/patrick-kidger/diffrax/pull/197 and proceed carefully.
      out = fun(*args, **kwargs)


We see that three variables are returned: the omega axis we have specified, the dipole moment and the pulse in freqeuency domain. There is no way to control the number of points in the omega axis, because
it is the result of a Fourier transform.

We now plot the dipole moment and the pulse in frequency domain.


```python
import matplotlib.pyplot as plt

plt.plot(omegas, dipole_omega)
plt.plot(omegas, pulse_omega, "--")
```

    /home/david/.local/lib/python3.9/site-packages/matplotlib/cbook/__init__.py:1289: ComplexWarning: Casting complex values to real discards the imaginary part
      return np.asarray(x, float)
    /home/david/.local/lib/python3.9/site-packages/matplotlib/cbook/__init__.py:1289: ComplexWarning: Casting complex values to real discards the imaginary part
      return np.asarray(x, float)





    [<matplotlib.lines.Line2D at 0x68ff4f943100>,
     <matplotlib.lines.Line2D at 0x68ff4f943220>,
     <matplotlib.lines.Line2D at 0x68ff4f943280>]




    
![png](getting_started_files/getting_started_28_2.png)
    

