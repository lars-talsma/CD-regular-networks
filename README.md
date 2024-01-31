# CD-regular-networks

This repository is dedicated to investigating the continuous distribution (CD) of entanglement in quantum networks with a regular topology and is divided into three folders:

#### `/simulation`
This folder contains the `skeleton_<topology>.py` files that are required to run simulations of quantum networks operating a CD protocol, for networks with regular topologies (`chain`, `honeycomb`, `sq`, `tri`), both without (`inf`) and with (`fin`) boundaries (`fin` only for `chain` and `sq`).

#### `/manuscript`
This folder contains the code to run, post-process and plot the results presented in the [paper "Continuously Distributing Entanglement in Quantum Networks with Regular Topologies]()" by Lars Talsma, Alvaro G Iñesta and Stephanie Wehner. In particular, it contains a Jupyter Notebook tutorial on how to run network simulations (using the `skeleton_<topology>.py` files), and two Jupyter Notebooks that contain the code to generate the plots in the manuscript for both `inf`inite and `fin`ite networks (as well as explanations and additional analysis). Furthermore, there is a `figure.MPLSTYLE` file to standardize figure appearance. Then there are three subfolders: `/data_generation` contains the `data_gen_<topology>_<parameter>.py` files that have been used to generate the data presented in the manuscript (and used in the Jupyter Notebooks; this folder contains a copy of the `skeleton_<topology>.py` files discussed above for easy usage; see below for more information on the simulation `<parameter>`s); the generated results are stored in the `/data` folder as `data_<topology>_<parameter>.npy` NumPy files (also stored in [this data repository]()); lastly, `/figures` contains the figures produced in the two plot Jupyter Notebooks. The `data_gen` files have been included for reproducibility and as a reference to the values used in the simulations. To retrieve the results quicker, the data generation files use the `multiprocessing` package, and in turn, call a `data_gen_<topology>_<parameter>_run_sim.py` to run a simulation for a fixed set of simulation parameters; see the tutorial for more information.

#### `/thesis`
This folder contains the code to run, post-process and plot the results presented in Lars Talsma’s [Master thesis “Continuous Distribution of Entanglement in Quantum Networks with Regular Topologies”](https://repository.tudelft.nl/islandora/object/uuid%3A7a891675-6bb9-4353-b510-e1e01d945023). In particular, it contains subfolders for all the network topologies (`infinite` for the `chain`, `honeycomb lattice`, `square-lattice` and `triangular-lattice`,  and `finite` for the `chain` and `square-lattice`), which subsequently contain
1. A `skeleton` Python file containing the code to simulate the quantum network running the CD protocol over time (a copy of the above-disucced files in the `simulation` folder -- for ease of use);
2. A simulation Jupyter notebook (`sim_x_y`, with `x` either `fin`ite or `inf`inite and `y` the topology: `chain`, `honeycomb`, `sq` or `tri`) containing the code to initiate and save (multi-core) simulations of the networks that sweep over several system parameters;
3. The corresponding Python code to run the parameter sweeps, which includes post-processing of the simulations (`sweep_z`, where `z` is the network parameter that we sweep over: `prob_swap` ($q$), `prob_succ_gen` ($p_{\mathrm{gen}}$), `prob_succ_swap` ($p_{\mathrm{swap}}$), `time_cutoff` ($t_{\mathrm{cut}}$) or `max_link_dist` ($M$); a combination of parameters is also possible).

Finally, the `figures` folder contains the Jupyter notebooks to generate the figures in the thesis. The data is saved in the Jupyter notebooks as NumPy (.npy) `results` files per network topology (infinite/finite + chain/honeycomb/square/triangular) and per sweep parameter (e.g., FOLDER_LOCATION/Infinite networks/Infinite honeycomb/sweep_prob_swap_max_link_dist/results), which can, subsequently, be used by the figure Jupyter notebooks. There is figure.MPLSTYLE file that handles homogenization of the figure parameters (font size, tick size etc.). 

We now shortly elaborate on what networks we investigate, how entanglement can be distributed in such networks and how we characterize the performance of entanglement distribution; for more details, we refer to the manuscript and thesis above.

## Quantum network topology
We investigate the performance of a protocol that continuously distributes entangled states between quantum nodes in networks with a physical node degree $d$. This means that each quantum node shares physical channels with $d$ other nodes, such that a chain, honeycomb lattice, square grid and triangular lattice structure emerges for $d=2,3,4,6$, respectively. (The thesis naming convention for the physical node degree is $k_{\mathrm{p}}$.)

## Quantum network model and dynamics
The number of entangled states between nodes, so-called entangled links, evolves as quantum nodes create new links through (1) entanglement generation and (2) entanglement swaps, and (3) remove low-fidelity links. The nodes probabilistically attempt swaps and can maximize the performance metrics by varying this probability. In particular, we assume that the quantum networks are homogeneous, such that:
- all nodes generate entangled links with the same fidelity $F_{\mathrm{new}}$ (thesis: $F_{\mathrm{gen}}$) and with the same probability of success $p_{\mathrm{gen}}$ (thesis: $p$);
- all nodes successfully implement entanglement swaps with the same probability $p_{\mathrm{swap}}$ (thesis: $p_{\mathrm{s}}$);
- all entangled links decohere according to the same coherence time $T$ (thesis: $T_{2}$);
- all nodes discard entangled links with an age equal to the cutoff time $t_{\mathrm{cut}}$ or have been involved in more than $M$ swaps. Note that the definition of the maximum swap distance $M$ in the manuscript differs slightly compared to that in the simulation code and the thesis: in the manuscript, nodes discard links that are the fusion of more than $M$ short-distance links (generated between physical neighbors), while in the code/thesis, nodes discard links that have been involved in more than $M$ swaps; i.e., $M$(manuscript) = $M$(code/thesis)$-1$; lastly,
- all nodes have a "large enough" number of memories to store,  i.e., nodes can store all generated entangled links until they discard them when the links age to the cutoff time;

## Entanglement distribution protocol
The network discretizes time, and all quantum nodes implement a continuous distribution (CD) protocol simultaneously during each time step, as prescribed by:
1. Cutoff time. Discard entangled links with ages equal to the cutoff time $t_{\mathrm{cut}}$.
2. Entanglement generation. Attempt to generate shared Werner states (entangled links) with physical neighbors and succeed with a probability $p_{\mathrm{gen}}$.
3. Entanglement swapping. Attempt to swap two entangled links with a probability $q$ and succeed with a probability $p_{\mathrm{swap}}$. A quantum node randomly chooses the first entangled link from its memory. The node chooses the second link randomly from the set of links stored in a differently-oriented qubit. If the swap succeeds, the two initial links transform into a new link; if it fails, the nodes discard the two initial links. Nodes do not know what swaps the other nodes implement.
4. Maximum swap distance. Discard entangled links are the fusion of more than $M$ short-distance links (see note on $M$ above). The quantum nodes communicate their results to conclude which entangled links they have swapped.


## Performance characterization
We use two metrics to measure the performance of such CD protocol explicitly considering the time-dependent fidelity of entangled links: the virtual node degree, which denotes the number of entangled links stored by a certain node at a specific time, and the virtual neighborhood size, which denotes the number of nodes a certain node shares entangled states at a specific time. 

For any questions, you can contact me at larstalsma@gmail.com

