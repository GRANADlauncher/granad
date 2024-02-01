.. highlight:: console
.. highlight:: python

========== 
Quickstart
==========

GRANAD offers

#. Calculations of independent-particle (IP) quantities based on parametric tight-binding (TB) Hamiltonians.
#. Many-body calculations of e.g. absorption spectra based on the time-domain formalism presented in https://pubs.acs.org/doi/abs/10.1021/nn204780e.

The following is an illustration of GRANAD's core capabilities.

A typical simulation
====================
A typical GRANAD simulation consists of two steps:

1. Describing a stacked nanomaterial with the help of the class :class:`granad.StackBuilder`.
2. Extracting physical quantities from the nanomaterial with the help of the class :class:`granad.Stack`.


.. note:: GRANAD is built on top of `JAX <https://jax.readthedocs.io/en/latest/>`_.
	  For this reason, JAX's equivalent to numpy should be used.
	  In most cases, this amounts to a simple replacement of :code:`numpy` with :code:`jax.numpy`,
	  as in :code:`numpy.array([1,2,3])` and :code:`jax.numpy.array([1,2,3])`.

	  
In the following, we detail both steps to visualize the indepdendent-particle
energy landscape and first excited state of a triangular graphene flake.

Building the Stack
------------------

The first simulation step mainly operates on Python primitives and is designed
to offer an easy interactive way to specify a nanomaterial.

We start by initializing a :class:`granad.StackBuilder` object:

.. literalinclude:: ../examples/stack_building.py
   :language: python
   :lines: 6
	 
We want to model

#. a *triangular* flake of 7.4 Å (0.74 nm)
#. with an *armchair* edge type, which is 
#. made from graphene, i.e. a *honeycomb* lattice with a *lattice constant* of 2.46 Å.

GRANAD directly understands the triangle requirement. To get a triangular flake, we use the class :class:`granad.Triangle` like so

.. literalinclude:: ../examples/stack_building.py
   :language: python
   :lines: 9

.. note:: There are more shape options in GRANAD, e.g. :class:`granad.Rectangle`. They
	  are used completly analogously.
   
GRANAD lets us put all these requirements together to specify a flake cut from a larger lattice via its class :class:`granad.Lattice` as follows

.. literalinclude:: ../examples/stack_building.py
   :language: python
   :lines: 10-15

Now that we have specified the lattice, we add it to the StackBuilder.

.. literalinclude:: ../examples/stack_building.py
   :language: python
   :lines: 16

GRANAD considers this as the addition of a single orbital named :code:`"pz"` located at exactly the positions we want (we consider spin-traced dynamics). We can quickly visualize the stack with the method :py:meth:`granad.StackBuilder.show3D`.

.. literalinclude:: ../examples/stack_building.py
   :language: python
   :lines: 18

.. plot:: ../examples/stack_building.py

Now that we have built the stack, we can proceed to specify its TB-parameters.

Defining parameters
-------------------

GRANAD knows two different couplings, which can be flexibly specified:

1. TB-hoppings are specified using :py:meth:`granad.StackBuilder.set_hopping`
2. Coulomb interaction matrix elements are specified using :py:meth:`granad.StackBuilder.set_coulomb`.

The two methods work exactly analogously and have the same signature. They accept a class describing the coupling between the orbitals.

.. _coupling classes:

Coupling Class Types   
^^^^^^^^^^^^^^^^^^^^

1. :class:`granad.DistanceCoupling`: coupling as a distance-dependent function.
2. :class:`granad.LatticeCoupling`: coupling in a regular structure (like a graphene flake) via a list of neighbor couplings.
3. :class:`granad.LatticeSpotCoupling`: coupling between a single spot (hosting, e.g. an isolated atom) and a regular structure.
4. :class:`granad.SpotCoupling`: couplings in an isolated spot.

Returning to our stack, we are just considering a single graphene flake and want to use the a nearest-neighbor hopping rate of :code:`-2.66` with :code:`0` onsite energies. Since we have a regular structure of many orbitals, we need the second class. Its constructor expects:

* The ids of the two orbitals we want to couple.
* The lattice we have just specified.
* A list of couplings of the form :code:`[onsite, nearest_neighbor]` (actually, this list can have arbitrarily many entries for arbitrarily many neighbor-couplings).

In our case, this translates to

.. literalinclude:: ../examples/energies_layer.py
   :language: python
   :lines: 18-21

If we wanted to have a next-to-nearest-neighbor coupling of :code:`-1`, we would have to change the list to :code:`[0, -2.66, -1]`.

Now, we want to set the coulomb matrix energies.
We want to set the neighbor couplings to :code:`[16.522, 8.64, 5.333]` and apply the classical
law :math:`c(r) = 14.399 / r` for any pz-Orbital that is further away.
GRANAD offers a way to achieve this by supplying this classical function an optional parameter to :class:`granad.LatticeCoupling` as follows:

.. note:: Due to internal reasons, any coupling function *must return a complex number*.

.. literalinclude:: ../examples/energies_layer.py
   :language: python
   :lines: 23-30
	  

We have now specified all couplings and can proceed to extract physical quantities.

Extract physical quantities
---------------------------
Since the stack is now fully built and we want to start doing numerics,
we have no need for the :code:`StackBuilder` anymore.
The central numerical class in GRANAD is :class:`granad.Stack`.
A :code:`Stack` object is just a container. It holds arrays that correspond to the numerical description of the nanomaterial we have just specified.

.. note:: The state of a stack corresponds to its 1RDM density matrix and is thus a complex array.

We obtain it using the method :py:meth:`granad.StackBuilder.get_stack`.
This method allows us to set the initial IP state of the stack. By default, it
produces the ground state, but we could also decide to use, e.g. the first excited state.
Additionally, we can control the number of doping electrons in the stack,
add a diagonal term to the Hamiltonian corresponding to the influence
of a homogeneous electric field or set some internal numerical precision.
For now, we don't need to touch on these options, but instead initialize the
stack in the ground state, without any additional terms. This means we just call the method without any arguments and store the resulting stack

.. literalinclude:: ../examples/energies_layer.py
   :language: python
   :lines: 33

	   
.. note:: Internally, a stack is just used to store data and has no mutable state. So, any changes to coupling parameters must happen at the level of the :code:`StackBuilder`, not the :code:`Stack`!


Visualizing IP physical quantities is now easy:

.. literalinclude:: ../examples/energies_layer.py
   :language: python
   :lines: 35-39

.. plot:: ../examples/energies_layer.py
	  

Dynamic quantities
==================

We now turn to time-domain simulations. We want to calculate the absorption spectrum
of the triangular flake we just built. The absorption cross section is calculated as follows

#. We illuminate the flake with an electric field pulse for a specific time and record the site occupations :math:`\rho_{ii}`, where :math:`\rho` is the real-space density matrix.
#. The induced dipole moments are calculated at every orbital position :math:`R_i` according to :math:`p_i \sim R_i \rho_{ii}`.
#. We Fourier transform the dipole moments to obtain the dynamic polarizability as :math:`\alpha(\omega) = p(\omega) / E(\omega)`.
#. Finally, we compute the absorption cross section as :math:`\sigma(\omega) = -\omega \text{Im}[\alpha]`.
   

Modelling an Electric Field Pulse
---------------------------------

First, we need to define the illumination. GRANAD offers multiple functions for modelling electric fields. All of them are assumed to propagate in z-direction. The one for our purpose is called :py:func:`granad.electric_field_pulse`. As for all electric field functions, it takes in the properties of the electric field and returns a function. This function takes in a point in time and returns the corresponding electric field. We will see in detail how this works below.

In general, for using electric field functions, we need to know the properties of the field, i.e. 

#. The amplitudes of its vectors components :math:`(E_x, E_y, E_z)`.
#. Its frequency :math:`f`.
#. The location where we need it to be evaluated. In our case, we want to know the field at the position of the atoms in the stack. Fortunately, the class :class:`granad.Stack` has a corresponding attribute.

Additionally, we are considering a pulse, so we also need to know the spectral
location of its peak and its full width at half maximum (fwhm).

Putting all of this together, we can obtain an electric field as follows:
   
.. literalinclude:: ../examples/time_domain_simulation_loss.py
   :language: python
   :lines: 57-59

Absorption spectra
------------------

We now get to the actual simulation. First, we need to do a necessary import

.. literalinclude:: ../examples/time_domain_simulation_loss.py
   :language: python
   :lines: 4


This just imports JAX's version of numpy. You can mostly use the same functions here as you would in regular numpy, just replacing :code:`np`, which :code:`jnp`.

For example, this is how we define the simulation duration (GRANAD's units
are explained in :doc:`units`.)

.. literalinclude:: ../examples/time_domain_simulation_loss.py
   :language: python
   :lines: 62

The actual simulation is performed using the function :py:func:`granad.evolution`. We must supply this function with the following arguments

#. The stack object we just built.
#. The simulation time.
#. The electric field we just defined.
#. A damping function describing losses in our structure.
   
This function returns the updated stack and an array of density matrices in real space. The dimension of this array is :math:`N \times N \times T`, where :math:`N` is the number of orbitals and :math:`T` is the number of timesteps in the simulation. This array can get very large, quickly exhausting our computational resources.

One strategy to overcome this problem is to chunk the simulation duration into little bits, run them one by one and save all intermediary results to disk. For the present example, we chose an easier route.

For the calculation of the absorption spectrum, we only need the diagonal elements of the density matrix. Conveniently, the function :py:func:`granad.evolution` takes an optional extra-argument, which is a postprocessing function. This function must return an array and is applied to all :math:`T` density matrices we get from the simulation.

Only the diagonal elements are needed, so our postprocessing function should be :code:`jnp.diag`, resulting in an array of shape :math:`N \times T`, drastically reducing memory consumption.

For the damping function, we just chose :py:func:`granad.evolution` with a damping rate of 0.1. Ultimately, this results in 

.. literalinclude:: ../examples/time_domain_simulation_loss.py
   :language: python
   :lines: 64-70

After letting the simulation run for a while, we can calculate the induced dipole moment in real space

.. literalinclude:: ../examples/time_domain_simulation_loss.py
   :language: python
   :lines: 73

We then perform a Fourier transform (note that you have to supply a custom implementation of this function as of now)	   

.. literalinclude:: ../examples/time_domain_simulation_loss.py
   :language: python
   :lines: 76

After Fourier transforming the x-component of the electric field as well, we can
compute the absorption cross section, and plot the result

.. literalinclude:: ../examples/time_domain_simulation_loss.py
   :language: python
   :lines: 85-93

This results in	   

.. plot:: ../examples/time_domain_simulation_loss.py


.. note:: The output of time domain simulations can be used to calculate a variety of other quantities, such as the EPI, induced fields, and more; all of them are described in the reference.
	  

Modelling Adatoms
=================

We already know how to handle flake nanomaterials cut from a larger lattice. Now, we need to proceed how to model individual adatoms. In contrast to graphene, where we needed to model a lattice, we now need to model a single spot. This is done with the help of :class:`granad.Spot`.

First, we again initialize a StackBuilder.

.. literalinclude:: ../examples/energies_adatom.py
   :language: python
   :lines: 3

Then, we define the spot we want to use. This is just a point in 3D-space. For simplicity, we use the coordinate origin

.. literalinclude:: ../examples/energies_adatom.py
   :language: python
   :lines: 5

We then want to model our adatom as a two-level system (TLS) at this point, containing
a low-energy orbital, called :code:`"A"` and high-energy orbital, called :code:`"B"`.
So, we need to add two orbitals at this spot. However, we want only the :code:`"A"` to be occupied, so we explicitly set the occupation of :code:`"B"` to :code:`0`.

.. literalinclude:: ../examples/energies_adatom.py
   :language: python
   :lines: 6-7


Now, we need the couplings. Remembering the list under `coupling classes`_, we use the
class :class:`granad.SpotCoupling` to specify how the two orbitals in the isolated spot couple to each other.
This happens very analogously to the lattice coupling. The only difference is that, since we consider a single point in space, the coupling is described by a single number and not a list.

We want the two adatom levels to couple as follows:

#. :code:`"A"` should have an energy of :code:`0`.
#. :code:`"B"` should have an energy of :code:`2`.
#. :code:`"A"` and :code:`"B"` should have a coupling energy of :code:`1`.
#. :code:`"A"` should have an onsite coulomb repulsion of :code:`1`.
#. :code:`"B"` should have an onsite coulomb repulsion of :code:`1`.
#. :code:`"A"` and :code:`"B"` should have an onsite coulomb repulsion of :code:`1`.

The corresponding use of the :code:`SpotCoupling` class to achieve this is as follows:   
   
.. literalinclude:: ../examples/energies_adatom.py
   :language: python
   :lines: 9-17

We can now again obtain the stack, and plot the energies.

.. plot:: ../examples/energies_adatom.py


Combining adatoms and layers
============================

.. DANGER:: Currently, only a single two-level adatom and a single graphene layer are implemented.


So far, we know how to handle lattice-based and isolated systems. We can, however, also couple both of them. In this example, we want to add a TLS in the top-position of a pz-Orbital in a benzene ring. The initial setup of the benzene ring is familiar: we need a :py:meth:`granad.StackBuilder` object and capture the geometry of the benzene ring with a :py:meth:`granad.Lattice` object.

.. literalinclude:: ../examples/energies_layer_adatom.py
   :language: python
   :lines: 6-16

.. note::
   Using geometry types such as :class:`granad.Triangle`, the largest possible shape is chosen that matches the size requirement. 

Inspecting with the method :py:meth:`granad.StackBuilder.show3D`, we
see that each individual orbital is annotated with an index. We pick the 0-th pz-Orbital to place the TLS on top of.

.. plot:: ../examples/single_benzene_ring.py

Then, we use the method :py:meth:`granad.StackBuilder.get_positions`: It returns all positions in the stack as an array. The array is of dimension :math:`N \times 3`, where the first index corresponds to the index shown by :py:meth:`granad.StackBuilder.show3D`. We now place the TLS with a low-energy orbital called :code:`"A"` and a high energy orbital called :code:`"B"` in the top-position in a distance in z-direction of 1 Å of the 0-th pz-Orbital. Here, we must not forget to set the (non-interacting) occupation of the higher level to 0.

.. literalinclude:: ../examples/energies_layer_adatom.py
   :language: python
   :lines: 18-22

Now, we need to specify the couplings. For the isolated flake and TLS, this proceeds exactly analogously to the previous sections and is omitted here for brevity. The new problem, however, is to couple the two components. Looking at the possible `coupling classes`_ again, we see that we need to couple a lattice system with a single spot and thus need :class:`granad.LatticeSpotCoupling`.
2
We want to couple the :code:`"A"` orbital of the TLS to the benzene ring as follows:

#. The hopping between :code:`"A"` and :code:`"pz"` should be :code:`1` for nearest neighbours, :code:`0.5` for next-to-nearest neighbors and :code:`0` beyond.
#. The coulomb interaction between :code:`"A"` and :code:`"pz"` should be :code:`2` for nearest neighbours and the classical relation :math:`c(r) = \frac{1}{r}` beyond. For sake of simplicity, we assume identical coupling strengths for :code:`"B"`.

The way to do this with :class:`granad.LatticeSpotCoupling` is thus 

.. literalinclude:: ../examples/energies_layer_adatom.py
   :language: python
   :lines: 50-60

.. DANGER:: The constructor arguments to :class:`granad.LatticeSpotCoupling` make an explicit distinction between the id of the lattice orbital and the id of the isolated orbital. Make sure not to mix up the two.
	  
Of course, we now have to specify the coupling between :code:`"pz"` and :code:`"B"`. GRANAD will complain
if any coupling combination is not supplied. For sake of simplicity, we chose this coupling exactly as in the lower-energy case.

.. literalinclude:: ../examples/energies_layer_adatom.py
   :language: python
   :lines: 62-72

Physical quantities can be visualized as usual.

.. literalinclude:: ../examples/energies_layer_adatom.py
   :language: python
   :lines: 74-77

.. plot:: ../examples/energies_layer_adatom.py
	     
	  
Differentiability
=================
.. DANGER:: Currently, this feature is implemented, but not tested.

GRANAD is built on `JAX <https://jax.readthedocs.io/en/latest/>`_. One motivation behind expressing GRANAD's computations as pure function composition is thus given by the possibility to leverage JAX's automatic differentiation capabilities, e.g. for parametric optimization.

This can serve, e.g. as a simplified way to determine the adsorption position of an adatom.

.. literalinclude:: ../examples/optimization.py

More Examples
=============

More examples can be found in the examples folder of the offical release.
