# CD-regular-networks

This repository contains the code to generate, post-process and visualize the results presented in Lars Talsma’s Master thesis named “Continuous Distribution of Entanglement in Quantum  Networks with Regular Topologies.” 

## Quantum network topology
We investigate the performance of a protocol that continuously distributes entangled states between quantum nodes in networks with a physical node degree $`k_{\mathrm{p}}`$. This means that each quantum node shares physical channels with $`k_{\mathrm{p}}`$ other nodes, such that a chain, honeycomb lattice, square grid and triangular lattice structure emerges for `$k_{\mathrm{p}}=2,3,4,6`$, respectively. 

## Quantum network model and dynamics
The number of entangled states between nodes, so-called entangled links, evolves as quantum nodes create new links through (1) entanglement generation and (2) entanglement swaps, and (3) remove low-fidelity links. The nodes probabilistically attempt swaps and can maximise the performance metrics by varying this probability. In particular, we assume that the quantum networks are homogeneous, such that:
- all nodes generate entangled links with the same fidelity $`F_{\mathrm{gen}}`$ and with the same probability of success $`p`$;
- all nodes successfully attempt entanglement swaps with the same probability $`p_{\mathrm{s}}`$;
- all nodes have an infinite number of memories;
- all entangled links decohere according to the same coherence time $`T_{2}`$; and
- all nodes discard entangled links with an age equal to the cutoff time $`t_{\mathrm{cut}}`$ or have been involved in more than $`M`$ swaps.

## Entanglement distribution protocol
The network discretises time, and all quantum nodes implement a continuous distribution (CD) protocol simultaneously during each time step, as prescribed by:
1. Cutoff time. Discard entangled links with ages equal to the cutoff time $`t_{\mathrm{cut}}`$.
2. Entanglement generation. Attempt to generate shared Werner states (entangled links) with physical neighbours and succeed with a probability $`p`$.
3. Entanglement swapping. Attempt to swap two entangled links with a probability $`q`$ and succeed with a probability $p_{\mathrm{s}}`$. A quantum node randomly chooses the first entangled link from its memory. The node chooses the second link randomly from the set of links stored in a differently-oriented qubit. If the swap succeeds, the two initial links transform into a new link; if it fails, the nodes discard the two initial links. Nodes do not know what swaps the other nodes implement.
4. Maximum swap distance. Discard entangled links that have been involved in more than $`M`$ entanglement swaps. The quantum nodes communicate their results to conclude which entangled links they have swapped.


## Performance characterisation
We use two metrics to measure the performance of such CD protocol explicitly considering the time-dependent fidelity of entangled links: the virtual node degree which denotes the number of entangled links stored by a certain node at a specific time, and the virtual neighbourhood size which denotes the number of nodes a certain node shares entangled states with a at a specific time. 

## Files
The network topologies with $`k_{\mathrm{p}}=2,3,4,6`$ (emerging structures corresponding to a `chain`, `honeycomb lattice`, `square grid` and `triangular lattice`) as discussed in the thesis for `infinite` and `finite` networks (with boundaries; only for the `chain` and `square grid`) each have their own folder containing: 
1. A `skeleton` Python file containing the code to simulate the quantum network running the CD protocol over time;
2. A simulation Jupyter notebook (`sim_x_y`, with `x` either `fin`ite or `inf`inite and `y` the topology: `chain`, `honeycomb`, `sq` or `tri`) containing the code to initiate and save (multi-core) simulations of the networks that sweep over several system parameters;
3. The corresponding Python code to run the parameter sweeps, which includes post-processing of the simulations (`sweep_z`, where `z` is the network parameter that we sweep over: `prob_swap` ($`q`$), `prob_succ_gen` ($`p`$), `prob_succ_swap` ($`p_{\mathrm{s}}`$), `time_cutoff` ($`t_{\mathrm{cut}}`$) or `max_link_dist` ($`M`$); a combination of parameters is also possible).

Finally, the `figures` folder contains the Jupyter notebooks to generate the figures in the thesis. The data is saved in the Jupyter notebooks as NumPy (.np) `results` files per network topology (infinite/finite + chain/honeycomb/square/triangular) and per sweep parameter (e.g., FOLDER_LOCATION/Infinite networks/Infinite honeycomb/sweep_prob_swap_max_link_dist/results), which can, subsequently, be used by the figure Jupyter notebooks. There is figure.mplstyle file that handles homogenisation of the figure parameters (font size, tick size etc.). 

The data present in the thesis can be retrieved by running the parameter sweeps with the corresponding random number generator seed (0). 

For any questions, you can contact me at larstalsma@gmail.com

