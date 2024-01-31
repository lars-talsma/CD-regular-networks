"""
Skeleton of the simulations of a quantum network with an infinite square
lattice topology performing entanglement generation, entanglement swaps
and cutoffs.
"""

import numpy as np
from itertools import product


def simulation_infinite_chain(prob_succ_gen, prob_succ_swap, prob_swap,
                              time_cutoff, qubits_per_node, time,
                              max_swap_dist, number_of_nodes, test):
    """
    Perform a discrete time simulation with a number of 'time'
    timesteps of a continuous distribution of entanglement protocol on
    an infinite square grid of nodes (which is represented by a finite
    square lattice containing 'number_of_nodes' × 'number_of_nodes'
    nodes with periodic boundary conditions) where the nodes have a
    'qubits_per_node'-sized qubit register. Entanglement between nodes
    is generated successfully with a probability 'prob_succ_gen',
    entanglement swaps happen with a probability 'prob_swap' and succeed
    with probability 'prob_succ_swap'. Entangled pairs are cutoff after
    living 'time_cutoff' and the maximum number of swaps allowed to
    generate a virtual link is 'max_swap_dist'. Return the ages of the
    virtual links (which are characterized by the horizontal and vertical
    indices of the two nodes it connects and the qubit index storing the
    link) in an adjacency array 'virtual_link_adj_array', the figures of
    merit, 'virtual_node_degree' and 'virtual_neighbourhood_size', and
    the number of swaps that have been attempted 'swap_count' and how
    many that have failed during swapping 'swap_fail_count'. Run tests
    on the metrics to check if they are correctly bounded. Return
    indices of nodes containing virtual links if 'test' is True.

    Parameters
    ----------
    prob_succ_gen : float
        The probability of successful entanglement generation
    prob_succ_swap : float
        The probability of a successful swap operation
    prob_swap : float
        The probability of performing a swap
    time_cutoff : int
        The cutoff time
    qubits_per_node : integer
        The number of qubits per node
    time : int
        The number of time steps that will be used in the simulation
    max_swap_dist : int
        The maximum number of swaps allowed to generate a virtual link
    number_of_nodes : int
        The number of nodes in the finite chain of nodes
    test: Bool
        If test is True, virtual_link_adj_array is printed each time
        step to check its evolution, else nothing happens

    Returns
    -------
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
    virtual_node_degree : numpy.ndarray
        number_of_nodes^2 × time-dimensional array storing the virtual
        node degree of each node in the network during all time steps
    virtual_neighbourhood_size : numpy.ndarray
        number_of_nodes^2 × time-dimensional array storing the virtual
        neighbourhood size of each node during all time steps
    swap_count : int
        The number of swaps that have been attempted during simulation
    swap_fail_count : int
        The number of swaps that have failed during the simulation
    """
    # To denote no links exist, the adjacency array on virtual links,
    # contains infinities, and the arrays on the swap distance and qubit
    # orientation zero entries.
    virtual_link_adj_array = np.empty((number_of_nodes, 1,
                                       number_of_nodes, 1,
                                       qubits_per_node))
    virtual_link_adj_array[:] = np.inf

    swap_dist_adj_array = np.zeros((number_of_nodes, 1,
                                    number_of_nodes, 1,
                                    qubits_per_node))
    qubit_orient_adj_array = np.zeros((number_of_nodes, 1,
                                       number_of_nodes, 1,
                                       qubits_per_node))

    virtual_node_degree = np.empty((number_of_nodes, 1, time))
    virtual_neighbourhood_size = np.empty((number_of_nodes, 1,
                                           time))

    swap_count = 0
    swap_fail_count = 0

    for t in range(time):
        (virtual_link_adj_array_post_cd, swap_count, swap_fail_count,
         swap_dist_adj_array, qubit_orient_adj_array) = (
            continuous_distribution_protocol(
                 prob_succ_gen, prob_succ_swap, prob_swap, time_cutoff,
                 max_swap_dist, virtual_link_adj_array, swap_count,
                 swap_fail_count, swap_dist_adj_array,
                 qubit_orient_adj_array))

        virtual_node_degree[:, :, t], virtual_neighbourhood_size[:, :, t] = (
            virtual_metrics(virtual_link_adj_array_post_cd))

        # Test figures of merit to see if they are correctly bounded.
        np.testing.assert_array_less(
            virtual_node_degree[:, :, t],
            2*min([qubits_per_node, time_cutoff]) + 1)
        np.testing.assert_array_less(
            virtual_neighbourhood_size[:, :, t],
            2*min([qubits_per_node, max_swap_dist + 1, time_cutoff])+1)
        np.testing.assert_array_less(
            virtual_neighbourhood_size[:, :, t],
            virtual_node_degree[:, :, t] + 1)

        if test:
            print(np.transpose(np.nonzero(np.isfinite(
                virtual_link_adj_array_post_cd))))

        virtual_link_adj_array = virtual_link_adj_array_post_cd + 1

    return (virtual_link_adj_array, virtual_node_degree,
            virtual_neighbourhood_size, swap_count, swap_fail_count)


def continuous_distribution_protocol(prob_succ_gen, prob_succ_swap, prob_swap,
                                     time_cutoff, max_swap_dist,
                                     virtual_link_adj_array, swap_count,
                                     swap_fail_count, swap_dist_adj_array,
                                     qubit_orient_adj_array):
    """
    Perform one round of a continuous distribution protocol for an
    infinite square lattice of 'number_of_nodes × number_of_nodes' nodes
    with each a qubit register containing 'qubits_per_node', where (1)
    entanglement is generated with a probability 'prob_succ_gen', (2)
    entanglement swaps are attempted with probability 'prob_swap' and
    succeed with 'prob_succ_swap' and (3) virtual links older than
    time_cutoff are removed. Virtual links can be at most of distance
    'max_swap_dist'. Store the ages of the virtual links (which are
    characterized by the horizontal and vertical indices of the two
    nodes it connects and the qubit index storing the link) in an
    adjacency array, 'virtual_link_adj_array', the number of swaps
    required to generate the link, 'swap_dist_adj_array'-1, and the
    orientation of the qubits storing a link, 'qubit_orient_adj_array'.
    Record the number of swaps attempted, 'swap_count', and failed swaps,
    'swap_fail_count'.

    Parameters
    ----------
    prob_succ_gen : float
        The probability of successful entanglement generation
    prob_succ_swap : float
        The probability of a successful swap operation
    prob_swap : float
        The probability of performing a swap
    time_cutoff : int
        The cutoff time
    max_swap_dist : int
        The maximum number of swaps allowed to generate a virtual link
    number_of_nodes : integer
        The number of nodes in the repeater chain
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages at
        the start of a round of the CD protocol
    swap_count : int
        The number of swaps that have been attempted up to the start of
        a round of the CD protocol
    swap_fail_count : int
        The number of swaps that have failed up to the start of a round
        of the CD protocol
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links at CD protocol start
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links at CD protocol start,
        specifically of the node defined by the first two node indices.
        0 denotes an unused qubit, 1 a top-oriented qubit, 2 right, 3
        bottom, and 4 left.

    Returns
    -------
    virtual_link_adj_array_post_cutoff: numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
        after a round of the CD protocol
    swap_count : int
        The number of swaps that have been attempted during simulation
        after a round of the CD protocol
    swap_fail_count : int
        The number of swaps that have failed during simulation due to
        the newly generated link being longer than 'max_swap_dist'
        after a round of the CD protocol
    swap_dist_adj_array_post_cutoff : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links after the CD protocol
    qubit_orient_adj_array_post_cutoff : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links after the CD protocol,
        specifically of the node defined by the first two node indices.
        0 denotes an unused qubit, 1 a top-oriented qubit, 2 right, 3
        bottom, and 4 left.
    """
    (virtual_link_adj_array_post_cutoff, swap_dist_adj_array_post_cutoff,
     qubit_orient_adj_array_post_cutoff) = cutoff_application(
        time_cutoff, virtual_link_adj_array, swap_dist_adj_array,
        qubit_orient_adj_array)

    (virtual_link_adj_array_post_gen, swap_dist_adj_array_post_gen,
     qubit_orient_adj_array_post_gen) = entanglement_generation(
        prob_succ_gen, virtual_link_adj_array_post_cutoff,
        swap_dist_adj_array_post_cutoff, qubit_orient_adj_array_post_cutoff)

    (virtual_link_adj_array_post_swap, swap_count, swap_fail_count,
     swap_dist_adj_array_post_swap, qubit_orient_adj_array_post_swap) = (
        entanglement_swap(prob_succ_swap, prob_swap, time_cutoff,
                          max_swap_dist, virtual_link_adj_array_post_gen,
                          swap_count, swap_fail_count,
                          swap_dist_adj_array_post_gen,
                          qubit_orient_adj_array_post_gen))

    return (virtual_link_adj_array_post_swap, swap_count, swap_fail_count,
            swap_dist_adj_array_post_swap, qubit_orient_adj_array_post_swap)


def entanglement_generation(prob_succ_gen, virtual_link_adj_array,
                            swap_dist_adj_array, qubit_orient_adj_array):
    """
    Perform a round of entanglement generation step of the CD protocol
    where all nodes attempt to generate entanglement with the node
    neighbouring top and right. Each attempt succeeds with a probability
    'prob_succ_gen'. The newly generated links will be stored in the
    lowest-numbered empty qubit of 'virtual_link_adj_array',
    'swap_dist_adj_array' and 'qubit_orient_adj_array'.

    Parameters
    ----------
    prob_succ_gen: float
        The probability of successful entanglement generation
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
        before a step of entanglement generation
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links before entanglement
        generation
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links before entanglement
        generation, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.

    Returns
    -------
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
        after a step of entanglement generation
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links after entanglement
        generation
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links after entanglement
        generation, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.

    """
    number_of_nodes = np.shape(virtual_link_adj_array)[0]

    # Entanglement generation is attempted with the node positioned to
    # the top (node_index_hor, (node_index_vert + 1) % number_of_nodes)
    # and right ((node_index_hor + 1) % number_of_nodes, node_index_vert)
    # [by periodicity of the chain].

    cart_prod_node_indices = product(np.arange(0, number_of_nodes),
                                     np.arange(0, 1))

    for node_index_hor, node_index_vert in cart_prod_node_indices:
        if prob_succ_gen >= np.random.random():
            free_qubit = smallest_free_qubit(
                node_index_hor, node_index_vert,
                (node_index_hor + 1) % number_of_nodes, node_index_vert,
                virtual_link_adj_array)

            virtual_link_adj_array[
                node_index_hor, node_index_vert,
                (node_index_hor + 1) % number_of_nodes, node_index_vert,
                free_qubit] = 0
            virtual_link_adj_array[
                (node_index_hor + 1) % number_of_nodes, node_index_vert,
                node_index_hor, node_index_vert,
                free_qubit] = 0

            swap_dist_adj_array[
                node_index_hor, node_index_vert,
                (node_index_hor + 1) % number_of_nodes, node_index_vert,
                free_qubit] = 1
            swap_dist_adj_array[
                (node_index_hor + 1) % number_of_nodes, node_index_vert,
                node_index_hor, node_index_vert,
                free_qubit] = 1

            qubit_orient_adj_array[
                node_index_hor, node_index_vert,
                (node_index_hor + 1) % number_of_nodes, node_index_vert,
                free_qubit] = 2
            qubit_orient_adj_array[
                (node_index_hor + 1) % number_of_nodes, node_index_vert,
                node_index_hor, node_index_vert,
                free_qubit] = 4

    return virtual_link_adj_array, swap_dist_adj_array, qubit_orient_adj_array


def entanglement_swap(prob_succ_swap, prob_swap, time_cutoff, max_swap_dist,
                      virtual_link_adj_array, swap_count, swap_fail_count,
                      swap_dist_adj_array, qubit_orient_adj_array):
    """
    Perform a round of entanglement swapping, where a central node
    attempts entanglement swapping a pair of links — one link stored in
    a qubit with a random orientation, and the other in a qubit with a
    different orientation — with a probabality 'prob_swap', and succeeds
    with a probability 'prob_swap_succ'. The adjacency arrays
    'virtual_link_adj_array', 'swap_dist_adj_array' and
    'qubit_orient_adj_array' are updated to reflect the results of the
    swaps, including the removal of virtual links where swapping failed
    or a link longer than 'max_swap_dist' was generated.

    Parameters
    ----------
    prob_succ_swap : float
        The probability of a successful swap operation
    prob_swap : float
        The probability of performing a swap
    time_cutoff : int
        The cutoff time
    max_swap_dist : int
        The maximum number of swaps allowed to generate a virtual link
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
        before a step of entanglement swapping
    swap_count : int
        The number of swaps that have been attempted up to the start of
        the swapping procedure
    swap_fail_count : int
        The number of swaps that have failed up to the start of the
        swapping procedure
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links before entanglement
        swapping
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links before entanglement
        swapping, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.

    Returns
    -------
    virtual_link_adj_array_post_cutoff: numpy.ndarray
        number_of_nodes × number_of_nodes × qubits_per_node-dimensional
        adjacency array containing the virtual links existing between
        two nodes in the network and their ages after the entanglement
        swap step
    swap_count : int
        The number of swaps that have been attempted at the end of a
        round of the CD protocol
    swap_fail_count : int
        The number of swaps that have failed at the end of a round of
        the CD protocol
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links after swapping
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links after entanglement
        swapping, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.
    """
    number_of_nodes = np.shape(virtual_link_adj_array)[0]
    qubits_per_node = np.shape(virtual_link_adj_array)[4]

    cart_prod_node_indices = product(np.arange(0, number_of_nodes),
                                     np.arange(0, 1))

    # A Link between qubits within a node is resembled by the two initial links
    # who are assigned a unique identifier – the swap_count at the time of
    # generating the link – which is stored in link_within_node_adj_array; if
    # a link is not within a node, the adjacency array entry is infinite
    link_within_node_adj_array = np.empty((number_of_nodes, 1,
                                           number_of_nodes, 1,
                                           qubits_per_node))
    link_within_node_adj_array[:] = np.inf

    # The virtual link set contains the indices of the central link in
    # the swap procedure, the indices of the virtual neighbours, the
    # qubit storing the virtual links, the virtual link ages, the number
    # of swaps required to generate the link and the orientation of the
    # qubit the link is stored in at both the central and virtual
    # neigbour node. The virtual neighbours are found by 1. transforming
    # the slice of virtual_link_adj_array associated to the central node
    # to a Boolean array conditioned on finite entries and 2. retrieving
    # the indices of nonzero (= True) elements.
    for node_index_hor, node_index_vert in cart_prod_node_indices:
        link_indices = np.nonzero(
            np.isfinite(virtual_link_adj_array[
                node_index_hor, node_index_vert, :, :, :]))
        node_index_array = np.empty((np.shape(link_indices)[1], 2),
                                    dtype=int)
        node_index_array[:] = [int(node_index_hor), int(node_index_vert)]
        node_index_array = np.transpose(node_index_array)

        node_indices = (node_index_array[0][:], node_index_array[1][:],
                        link_indices[0][:], link_indices[1][:],
                        link_indices[2][:])
        node_indices_inv = (link_indices[0][:], link_indices[1][:],
                            node_index_array[0][:], node_index_array[1][:],
                            link_indices[2][:])

        virtual_link_set = (np.transpose(np.array(
            [node_index_array[0][:], node_index_array[1][:],
             link_indices[0][:], link_indices[1][:],
             link_indices[2][:],
             virtual_link_adj_array[node_indices],
             swap_dist_adj_array[node_indices],
             qubit_orient_adj_array[node_indices],
             qubit_orient_adj_array[node_indices_inv]], dtype=int)))

        # While the virtual link set contains at least two links, a link
        # is chosen randomly to be swapped. The links stored in a qubit
        # with a different orientation – which are potential swapping
        # partners – are retrieved, and if there is at least one, a swap
        # is attempted randomly with one of the potential swap links and
        # the links involved in the attempted swap are removed from the
        # link set. If there was no potential swapping link, the first
        # virtual link will be removed from the link set.
        while np.shape(virtual_link_set)[0] >= 2:
            np.random.shuffle(virtual_link_set)
            virtual_link_orientation = virtual_link_set[0, 7]
            virtual_link_set_pot_swap = virtual_link_set[
                virtual_link_set[:, 7] != virtual_link_orientation]

            if np.shape(virtual_link_set_pot_swap)[0] >= 1:
                np.random.shuffle(virtual_link_set_pot_swap)

                node_indices = (virtual_link_set[0, 0],
                                virtual_link_set[0, 1],
                                virtual_link_set[0, 2],
                                virtual_link_set[0, 3],
                                virtual_link_set[0, 4])
                node_pot_indices = (virtual_link_set_pot_swap[0, 0],
                                    virtual_link_set_pot_swap[0, 1],
                                    virtual_link_set_pot_swap[0, 2],
                                    virtual_link_set_pot_swap[0, 3],
                                    virtual_link_set_pot_swap[0, 4])

                if link_within_node_adj_array[node_indices] == np.inf:
                    # Both links don't exist between qubits within a node
                    if link_within_node_adj_array[node_pot_indices] == np.inf:

                        # Generate a link between qubits within a node
                        if np.array_equal(virtual_link_set[0, 0:4],
                                          virtual_link_set_pot_swap[0, 0:4]):

                            if prob_swap >= np.random.random():
                                swap_count += 1

                                (virtual_link_adj_array, swap_fail_count,
                                 link_within_node_adj_array) = (
                                    swap_impl_gen_within_node(
                                        prob_succ_swap, time_cutoff,
                                        virtual_link_adj_array,
                                        virtual_link_set[0, :],
                                        virtual_link_set_pot_swap[0, :],
                                        swap_count, swap_fail_count,
                                        link_within_node_adj_array))

                                virtual_link_set = np.delete(virtual_link_set,
                                                             0, 0)
                                link_index = np.nonzero(
                                    np.all(virtual_link_set == (
                                           virtual_link_set_pot_swap[0, :]),
                                           axis=1))
                                virtual_link_set = np.delete(virtual_link_set,
                                                             link_index, 0)

                            else:
                                virtual_link_set = np.delete(virtual_link_set,
                                                             0, 0)
                                link_index = np.nonzero(
                                    np.all(virtual_link_set == (
                                           virtual_link_set_pot_swap[0, :]),
                                           axis=1))
                                virtual_link_set = np.delete(virtual_link_set,
                                                             link_index, 0)

                        # Normal swap situation
                        else:
                            if prob_swap >= np.random.random():
                                swap_count += 1

                                (virtual_link_adj_array, swap_fail_count,
                                 swap_dist_adj_array,
                                 qubit_orient_adj_array) = (
                                    swap_implementation(
                                        prob_succ_swap, time_cutoff,
                                        virtual_link_adj_array,
                                        virtual_link_set[0, :],
                                        virtual_link_set_pot_swap[0, :],
                                        swap_fail_count,
                                        swap_dist_adj_array,
                                        qubit_orient_adj_array))

                                virtual_link_set = np.delete(virtual_link_set,
                                                             0, 0)
                                link_index = np.nonzero(
                                    np.all(virtual_link_set == (
                                        virtual_link_set_pot_swap[0, :]),
                                        axis=1))
                                virtual_link_set = np.delete(virtual_link_set,
                                                             link_index, 0)

                            else:
                                virtual_link_set = np.delete(virtual_link_set,
                                                             0, 0)
                                link_index = np.nonzero(
                                    np.all(virtual_link_set == (
                                           virtual_link_set_pot_swap[0, :]),
                                           axis=1))
                                virtual_link_set = np.delete(virtual_link_set,
                                                             link_index, 0)

                    # Only the second link exists between qubits within a node
                    else:
                        if prob_swap >= np.random.random():
                            swap_count += 1

                            (virtual_link_adj_array, swap_fail_count,
                             swap_dist_adj_array, qubit_orient_adj_array,
                             link_within_node_adj_array,
                             self_link, new_link) = (
                                swap_impl_one_within_node(
                                    prob_succ_swap, time_cutoff,
                                    virtual_link_adj_array,
                                    virtual_link_set_pot_swap[0, :],
                                    virtual_link_set[0, :],
                                    swap_fail_count,
                                    swap_dist_adj_array,
                                    qubit_orient_adj_array,
                                    link_within_node_adj_array))

                            # Delete both initial links – keep in mind that the
                            # link within a node is stored as two links – and
                            # add the link still allowed to be used in swaps
                            virtual_link_set = np.delete(virtual_link_set,
                                                         0, 0)
                            link_index = np.nonzero(
                                np.all(virtual_link_set == (
                                    virtual_link_set_pot_swap[0, :]), axis=1))
                            virtual_link_set = np.delete(virtual_link_set,
                                                         link_index, 0)
                            link_index = np.nonzero(
                                np.all(virtual_link_set == self_link, axis=1))
                            virtual_link_set = np.delete(virtual_link_set,
                                                         link_index, 0)
                            virtual_link_set = np.vstack((virtual_link_set,
                                                          new_link))

                        else:
                            virtual_link_set = np.delete(virtual_link_set,
                                                         0, 0)
                            link_index = np.nonzero(
                                np.all(virtual_link_set == (
                                       virtual_link_set_pot_swap[0, :]),
                                       axis=1))
                            virtual_link_set = np.delete(virtual_link_set,
                                                         link_index, 0)

                else:
                    # Only the first link is between qubits within a node
                    if link_within_node_adj_array[node_pot_indices] == np.inf:
                        if prob_swap >= np.random.random():
                            swap_count += 1

                            (virtual_link_adj_array, swap_fail_count,
                             swap_dist_adj_array, qubit_orient_adj_array,
                             link_within_node_adj_array,
                             self_link, new_link) = (
                                swap_impl_one_within_node(
                                    prob_succ_swap, time_cutoff,
                                    virtual_link_adj_array,
                                    virtual_link_set[0, :],
                                    virtual_link_set_pot_swap[0, :],
                                    swap_fail_count,
                                    swap_dist_adj_array,
                                    qubit_orient_adj_array,
                                    link_within_node_adj_array))

                            # Delete both initial links – keep in mind that the
                            # link within a node is stored as two links – and
                            # add the link still allowed to be used in swaps
                            virtual_link_set = np.delete(virtual_link_set,
                                                         0, 0)
                            link_index = np.nonzero(
                                np.all(virtual_link_set == (
                                       virtual_link_set_pot_swap[0, :]),
                                       axis=1))
                            virtual_link_set = np.delete(virtual_link_set,
                                                         link_index, 0)
                            link_index = np.nonzero(
                                np.all(virtual_link_set == self_link, axis=1))
                            virtual_link_set = np.delete(virtual_link_set,
                                                         link_index, 0)
                            virtual_link_set = np.vstack((virtual_link_set,
                                                         new_link))

                        else:
                            virtual_link_set = np.delete(virtual_link_set,
                                                         0, 0)
                            link_index = np.nonzero(
                                np.all(virtual_link_set == (
                                       virtual_link_set_pot_swap[0, :]),
                                       axis=1))
                            virtual_link_set = np.delete(virtual_link_set,
                                                         link_index, 0)

                    # Both virtual links are between qubits within a node
                    else:
                        # The links have been generated in different swaps
                        if link_within_node_adj_array[node_pot_indices] != (
                           -link_within_node_adj_array[node_indices]):

                            if prob_swap >= np.random.random():
                                swap_count += 1

                                (virtual_link_adj_array, swap_fail_count,
                                 swap_dist_adj_array, qubit_orient_adj_array,
                                 link_within_node_adj_array, self_link_1,
                                 self_link_2, new_link_1, new_link_2) = (
                                    swap_impl_both_within_node(
                                        prob_succ_swap, time_cutoff,
                                        virtual_link_adj_array,
                                        virtual_link_set[0, :],
                                        virtual_link_set_pot_swap[0, :],
                                        swap_count, swap_fail_count,
                                        swap_dist_adj_array,
                                        qubit_orient_adj_array,
                                        link_within_node_adj_array))

                                # Delete initial links – but keep in mind that
                                # a link within a node is stored as two links –
                                # and add the  new links still allowed to be
                                # used in swaps
                                virtual_link_set = np.delete(virtual_link_set,
                                                             0, 0)
                                link_index = np.nonzero(
                                    np.all(virtual_link_set == (
                                           virtual_link_set_pot_swap[0, :]),
                                           axis=1))
                                virtual_link_set = np.delete(virtual_link_set,
                                                             link_index, 0)
                                link_index = np.nonzero(
                                    np.all(virtual_link_set == (
                                           self_link_1), axis=1))
                                virtual_link_set = np.delete(virtual_link_set,
                                                             link_index, 0)
                                link_index = np.nonzero(
                                    np.all(virtual_link_set == (
                                           self_link_2), axis=1))
                                virtual_link_set = np.delete(virtual_link_set,
                                                             link_index, 0)
                                virtual_link_set = np.vstack((virtual_link_set,
                                                              new_link_1))
                                virtual_link_set = np.vstack((virtual_link_set,
                                                              new_link_2))

                            else:
                                virtual_link_set = np.delete(virtual_link_set,
                                                             0, 0)
                                link_index = np.nonzero(
                                    np.all(virtual_link_set == (
                                           virtual_link_set_pot_swap[0, :]),
                                           axis=1))
                                virtual_link_set = np.delete(virtual_link_set,
                                                             link_index, 0)

                        # The links have been generated in the same swap
                        else:
                            if prob_swap >= np.random.random():
                                swap_count += 1

                                (virtual_link_adj_array, swap_fail_count,
                                 swap_dist_adj_array, qubit_orient_adj_array,
                                 link_within_node_adj_array) = (
                                    swap_impl_same_within_node(
                                        prob_succ_swap,
                                        virtual_link_adj_array,
                                        virtual_link_set[0, :],
                                        virtual_link_set_pot_swap[0, :],
                                        swap_fail_count,
                                        swap_dist_adj_array,
                                        qubit_orient_adj_array,
                                        link_within_node_adj_array))

                                virtual_link_set = np.delete(virtual_link_set,
                                                             0, 0)
                                link_index = np.nonzero(
                                    np.all(virtual_link_set == (
                                           virtual_link_set_pot_swap[0, :]),
                                           axis=1))
                                virtual_link_set = np.delete(virtual_link_set,
                                                             link_index, 0)

                            else:
                                virtual_link_set = np.delete(virtual_link_set,
                                                             0, 0)
                                link_index = np.nonzero(
                                    np.all(virtual_link_set == (
                                           virtual_link_set_pot_swap[0, :]),
                                           axis=1))
                                virtual_link_set = np.delete(virtual_link_set,
                                                             link_index, 0)
            else:
                virtual_link_set = np.delete(virtual_link_set, 0, 0)

    # Virtual links that (1) have been involved in swaps that have failed (and
    # have been assigned the fail age), (2) are links between qubits within a
    # node or (3) require more swaps than max_swap_dist to be created are
    # removed (i.e., the adjacency arrays are changed accordingly) by finding
    # the node indices where the statement that (1) the age is equal to the
    # fail age, (2) the link_within_node_adj_array entry is finite or (3)
    # (swap distance - 1) is larger than the max_swap_dist is True (-1 as each
    # link starts with a swap distance of 1, resulting in a swap distance of 2
    # after the first swap).
    fail_age = 2*time_cutoff
    failed_link_indices = np.nonzero(virtual_link_adj_array == fail_age)
    virtual_link_adj_array[failed_link_indices] = np.inf
    swap_dist_adj_array[failed_link_indices] = 0
    qubit_orient_adj_array[failed_link_indices] = 0

    self_link_indices = np.nonzero(np.isfinite(link_within_node_adj_array))
    virtual_link_adj_array[self_link_indices] = np.inf
    swap_dist_adj_array[self_link_indices] = 0
    qubit_orient_adj_array[self_link_indices] = 0

    too_long_link_indices = np.nonzero(swap_dist_adj_array - 1 > max_swap_dist)
    virtual_link_adj_array[too_long_link_indices] = np.inf
    swap_dist_adj_array[too_long_link_indices] = 0
    qubit_orient_adj_array[too_long_link_indices] = 0

    return (virtual_link_adj_array, swap_count, swap_fail_count,
            swap_dist_adj_array, qubit_orient_adj_array)


def swap_implementation(prob_succ_swap, time_cutoff, virtual_link_adj_array,
                        virtual_link_1, virtual_link_2, swap_fail_count,
                        swap_dist_adj_array, qubit_orient_adj_array):
    """
    Attempt an entanglement swap of two virtual links, 'virtual_link_1'
    and 'virtual_link_2' (stored in qubits with different orientations
    of a central node) and succeed with a probability 'prob_succ_swap'.
    The number of failed swaps is tallied in 'swap_count_fail'. All
    adjacency arrays are changed accordingly.

    Parameters
    ----------
    prob_succ_swap : float
        The probability of a successful swap operation
    time_cutoff : int
        The cutoff time
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
        before a swap is implemented
    virtual_link_1: numpy.ndarray
    virtual_link_2: numpy.ndarray
        The two virtual links that are attempted to swap, characterised
        by the indices of the central link in the swap procedure, the
        indices of the nodes on the edges of swap, the qubits storing the
        virtual links, the virtual link ages, the number of swaps
        required to generate the link and the orientation of the qubit
        the link is stored in at both the central and side nodes
    swap_fail_count : int
        The number of swaps that have failed up to the start of the
        swapping procedure
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links before swapping
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links before entanglement
        swapping, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.

    Returns
    -------
    virtual_link_adj_array_post_cutoff: numpy.ndarray
        number_of_nodes × number_of_nodes × qubits_per_node-dimensional
        adjacency array containing the virtual links existing between
        two nodes in the network and their ages after swap implemenation
    swap_fail_count : int
        The number of swaps that have failed after swapping
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links after swapping
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links after entanglement
        swapping, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.
    """
    node_index_hor_1 = virtual_link_1[2]
    node_index_vert_1 = virtual_link_1[3]
    node_index_hor_2 = virtual_link_2[2]
    node_index_vert_2 = virtual_link_2[3]
    node_index_hor_cent = virtual_link_1[0]
    node_index_vert_cent = virtual_link_1[1]

    qubit_index_1 = virtual_link_1[4]
    qubit_index_2 = virtual_link_2[4]

    link_age_1 = virtual_link_1[5]
    link_age_2 = virtual_link_2[5]

    swap_dist_1 = virtual_link_1[6]
    swap_dist_2 = virtual_link_2[6]

    qubit_orient_1 = virtual_link_1[8]
    qubit_orient_2 = virtual_link_2[8]

    virtual_link_1_indices = (node_index_hor_cent, node_index_vert_cent,
                              node_index_hor_1, node_index_vert_1,
                              qubit_index_1)
    virtual_link_1_indices_inv = (node_index_hor_1, node_index_vert_1,
                                  node_index_hor_cent, node_index_vert_cent,
                                  qubit_index_1)

    virtual_link_2_indices = (node_index_hor_cent, node_index_vert_cent,
                              node_index_hor_2, node_index_vert_2,
                              qubit_index_2)
    virtual_link_2_indices_inv = (node_index_hor_2, node_index_vert_2,
                                  node_index_hor_cent, node_index_vert_cent,
                                  qubit_index_2)

    max_age = max(link_age_1, link_age_2)

    virtual_link_adj_array[virtual_link_1_indices] = np.inf
    virtual_link_adj_array[virtual_link_1_indices_inv] = np.inf
    virtual_link_adj_array[virtual_link_2_indices] = np.inf
    virtual_link_adj_array[virtual_link_2_indices_inv] = np.inf

    swap_dist_adj_array[virtual_link_1_indices] = 0
    swap_dist_adj_array[virtual_link_1_indices_inv] = 0
    swap_dist_adj_array[virtual_link_2_indices] = 0
    swap_dist_adj_array[virtual_link_2_indices_inv] = 0

    qubit_orient_adj_array[virtual_link_1_indices] = 0
    qubit_orient_adj_array[virtual_link_1_indices_inv] = 0
    qubit_orient_adj_array[virtual_link_2_indices] = 0
    qubit_orient_adj_array[virtual_link_2_indices_inv] = 0

    free_qubit = smallest_free_qubit(node_index_hor_1, node_index_vert_1,
                                     node_index_hor_2, node_index_vert_2,
                                     virtual_link_adj_array)

    new_link_indices = (node_index_hor_1, node_index_vert_1,
                        node_index_hor_2, node_index_vert_2,
                        free_qubit)
    new_link_indices_inv = (node_index_hor_2, node_index_vert_2,
                            node_index_hor_1, node_index_vert_1,
                            free_qubit)

    if prob_succ_swap >= np.random.random():
        virtual_link_adj_array[new_link_indices] = max_age
        virtual_link_adj_array[new_link_indices_inv] = max_age

    # If a swap fails, pretend that the swap succeeded but set the
    # age of the new virtual links to a "fail age": this way, these
    # links can be used in swapping attempts of other nodes, but can
    # also be filtered after all nodes have attempted swaps
    else:
        swap_fail_count += 1
        fail_age = 2*time_cutoff

        virtual_link_adj_array[new_link_indices] = fail_age
        virtual_link_adj_array[new_link_indices_inv] = fail_age

    swap_dist_adj_array[new_link_indices] = swap_dist_1 + swap_dist_2
    swap_dist_adj_array[new_link_indices_inv] = swap_dist_1 + swap_dist_2

    qubit_orient_adj_array[new_link_indices] = qubit_orient_1
    qubit_orient_adj_array[new_link_indices_inv] = qubit_orient_2

    return (virtual_link_adj_array, swap_fail_count, swap_dist_adj_array,
            qubit_orient_adj_array)


def swap_impl_gen_within_node(prob_succ_swap, time_cutoff,
                              virtual_link_adj_array, virtual_link_1,
                              virtual_link_2, swap_count, swap_fail_count,
                              link_within_node_adj_array):
    """
    Attempt an entanglement swap of two virtual links, 'virtual_link_1'
    and'virtual_link_2', existing between the same two nodes (stored in
    qubits with different orientations of the central node) and succeed
    with a probability 'prob_succ_swap'. If successful, the initial links
    continue to exist but the fact that they are used in swap resulting
    in a link within a node is stored with a unique identifier
    (swap_count) in 'link_within_node_adj_array'. The number of failed
    swaps is counted in 'swap_count_fail'.

    Parameters
    ----------
    prob_succ_swap : float
        The probability of a successful swap operation
    time_cutoff : int
        The cutoff time
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
        before a swap is implemented
    virtual_link_1: numpy.ndarray
    virtual_link_2: numpy.ndarray
        The two virtual links that are attempted to swap, characterised
        by the indices of the central link in the swap procedure, the
        indices of the nodes on the edges of swap, the qubits storing the
        virtual links, the virtual link ages, the number of swaps
        required to generate the link and the orientation of the qubit
        the link is stored in at both the central and side nodes
    swap_count : int
        The number of swaps that have been attempted up to the start of
        the swapping procedure
    swap_fail_count : int
        The number of swaps that have failed up to the start of the
        swapping procedure
    link_within_node_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the swap count (a unique
        identifier) of links between qubits within a node before
        swapping (that is, the swap count when these links were
        generated). These links are stored as two links but should
        be treated as one: the other half of a pair can be found by
        retrieving the indices of swap_count with opposite sign in
        link_within_node_adj_array

    Returns
    -------
    virtual_link_adj_array_post_cutoff: numpy.ndarray
        number_of_nodes × number_of_nodes × qubits_per_node-dimensional
        adjacency array containing the virtual links existing between
        two nodes in the network and their ages after swap implemenation
    swap_fail_count : int
        The number of swaps that have failed after the swap attempt
    link_within_node_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the swap count (a unique
        identifier) of links between qubits within a node after
        swapping (that is, the swap count when these links were
        generated). These links are stored as two links but should
        be treated as one: the other half of a pair can be found by
        retrieving the indices of swap_count with opposite sign in
        link_within_node_adj_array
    """
    node_index_hor_cent = virtual_link_1[0]
    node_index_vert_cent = virtual_link_1[1]
    node_index_hor_1 = virtual_link_1[2]
    node_index_vert_1 = virtual_link_1[3]
    node_index_hor_2 = virtual_link_2[2]
    node_index_vert_2 = virtual_link_2[3]

    qubit_index_1 = virtual_link_1[4]
    qubit_index_2 = virtual_link_2[4]

    virtual_link_1_indices = (node_index_hor_cent, node_index_vert_cent,
                              node_index_hor_1, node_index_vert_1,
                              qubit_index_1)
    virtual_link_1_indices_inv = (node_index_hor_1, node_index_vert_1,
                                  node_index_hor_cent, node_index_vert_cent,
                                  qubit_index_1)

    virtual_link_2_indices = (node_index_hor_cent, node_index_vert_cent,
                              node_index_hor_2, node_index_vert_2,
                              qubit_index_2)
    virtual_link_2_indices_inv = (node_index_hor_2, node_index_vert_2,
                                  node_index_hor_cent, node_index_vert_cent,
                                  qubit_index_2)

    # The unique identifiers are only assigned to the indices lead by the non-
    # central node as only they can be used in future swap rounds (and hence
    # have to be identified again); the others get non-unique finite values
    if prob_succ_swap >= np.random.random():
        link_within_node_adj_array[virtual_link_1_indices] = 0
        link_within_node_adj_array[virtual_link_1_indices_inv] = swap_count
        link_within_node_adj_array[virtual_link_2_indices] = 0
        link_within_node_adj_array[virtual_link_2_indices_inv] = -swap_count

    # If a swap fails, the initial links are given a "fail age" so that they
    # can be filtered after all nodes have attempted swaps
    else:
        swap_fail_count += 1
        fail_age = 2*time_cutoff

        virtual_link_adj_array[virtual_link_1_indices] = fail_age
        virtual_link_adj_array[virtual_link_1_indices_inv] = fail_age
        virtual_link_adj_array[virtual_link_2_indices] = fail_age
        virtual_link_adj_array[virtual_link_2_indices_inv] = fail_age

    return virtual_link_adj_array, swap_fail_count, link_within_node_adj_array


def swap_impl_one_within_node(prob_succ_swap, time_cutoff,
                              virtual_link_adj_array, virtual_link_1,
                              virtual_link_2, swap_fail_count,
                              swap_dist_adj_array, qubit_orient_adj_array,
                              link_within_node_adj_array):
    """
    Attempt an entanglement swap of two virtual links, 'virtual_link_1'
    and 'virtual_link_2' (stored in qubits with different orientations of
    the central node) and succeed with a probability 'prob_succ_swap'.
    virtual_link_1 is half of a pair of links to be considered a link
    between qubits within a node; the other link can be found using the
    "link_within_node_adj_array". If successful, the initial two (three
    in this method) links are discarded and a new link is generated.
    This new link can still be used by the node during this swap round,
    as the node does not know its two qubits were connected and the other
    end of the link-within-a-node has not been accessed yet (there would
    not be a link_within_node_adj_array trigger otherwise). The number
    of failed swaps is counted in 'swap_count_fail'. All adjacency arrays
    are adjusted accordingly.

    Parameters
    ----------
    prob_succ_swap : float
        The probability of a successful swap operation
    time_cutoff : int
        The cutoff time
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
        before a swap is implemented
    virtual_link_1: numpy.ndarray
    virtual_link_2: numpy.ndarray
        The two virtual links that are attempted to swap, characterised
        by the indices of the central link in the swap procedure, the
        indices of the nodes on the edges of swap, the qubits storing the
        virtual links, the virtual link ages, the number of swaps
        required to generate the link and the orientation of the qubit
        the link is stored in at both the central and side nodes.
        virtual_link_1 is one half of a pair of links resembling a link
        between qubits within a node
    swap_fail_count : int
        The number of swaps that have failed up to the start of the
        swapping procedure
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links before swapping
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links before entanglement
        swapping, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.
    link_within_node_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the swap count (a unique
        identifier) of links between qubits within a node after
        swapping (that is, the swap count when these links were
        generated). These links are stored as two links but should
        be treated as one: the other half of a pair can be found by
        retrieving the indices of swap_count with opposite sign in
        link_within_node_adj_array

    Returns
    -------
    virtual_link_adj_array_post_cutoff: numpy.ndarray
        number_of_nodes × number_of_nodes × qubits_per_node-dimensional
        adjacency array containing the virtual links existing between
        two nodes in the network and their ages after swap implemenation
    swap_fail_count : int
        The number of swaps that have failed after swapping
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links after swapping
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links after entanglement
        swapping, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.
    link_within_node_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the swap count (a unique
        identifier) of links between qubits within a node after
        swapping (that is, the swap count when these links were
        generated). These links are stored as two links but should
        be treated as one: the other half of a pair can be found by
        retrieving the indices of swap_count with opposite sign in
        link_within_node_adj_array
    self_link : numpy.ndarray
        the other half to virtual_link_1 which together resemble a link
        between qubits within a node
    new_link : numpy.ndarray
        the newly generated virtual link which can still be used in the
        same swap round
    """
    node_index_hor_cent = virtual_link_1[0]
    node_index_vert_cent = virtual_link_1[1]
    node_index_hor_1 = virtual_link_1[2]
    node_index_vert_1 = virtual_link_1[3]
    node_index_hor_2 = virtual_link_2[2]
    node_index_vert_2 = virtual_link_2[3]

    qubit_index_1 = virtual_link_1[4]
    qubit_index_2 = virtual_link_2[4]

    virtual_link_1_indices = (node_index_hor_cent, node_index_vert_cent,
                              node_index_hor_1, node_index_vert_1,
                              qubit_index_1)
    virtual_link_1_indices_inv = (node_index_hor_1, node_index_vert_1,
                                  node_index_hor_cent, node_index_vert_cent,
                                  qubit_index_1)

    virtual_link_2_indices = (node_index_hor_cent, node_index_vert_cent,
                              node_index_hor_2, node_index_vert_2,
                              qubit_index_2)
    virtual_link_2_indices_inv = (node_index_hor_2, node_index_vert_2,
                                  node_index_hor_cent, node_index_vert_cent,
                                  qubit_index_2)

    # Find the other half of the pair of links resembling a link within a node
    # to virtual_link_1 ('self_link') by looking for the indices of the element
    # of "link_within_node_adj_array" with the opposite-sign unique identifier
    node_self_indices = (
        np.nonzero(link_within_node_adj_array == (
                   -link_within_node_adj_array[virtual_link_1_indices])))
    node_self_indices = (
        node_self_indices[0][0], node_self_indices[1][0],
        node_self_indices[2][0], node_self_indices[3][0],
        node_self_indices[4][0])
    node_self_indices_inv = (
        node_self_indices[2], node_self_indices[3],
        node_self_indices[0], node_self_indices[1],
        node_self_indices[4])

    self_link = np.array(
        [node_self_indices[0], node_self_indices[1],
         node_self_indices[2], node_self_indices[3],
         node_self_indices[4],
         virtual_link_adj_array[node_self_indices],
         swap_dist_adj_array[node_self_indices],
         qubit_orient_adj_array[node_self_indices],
         qubit_orient_adj_array[node_self_indices_inv]], dtype=int)

    max_age = max(virtual_link_adj_array[virtual_link_1_indices],
                  virtual_link_adj_array[virtual_link_2_indices],
                  virtual_link_adj_array[node_self_indices])

    new_swap_dist = (swap_dist_adj_array[virtual_link_1_indices]
                     + swap_dist_adj_array[virtual_link_2_indices]
                     + swap_dist_adj_array[node_self_indices])

    qubit_orient = qubit_orient_adj_array[node_self_indices]
    qubit_orient_inv = qubit_orient_adj_array[virtual_link_2_indices_inv]

    virtual_link_adj_array[virtual_link_1_indices] = np.inf
    virtual_link_adj_array[virtual_link_1_indices_inv] = np.inf
    virtual_link_adj_array[virtual_link_2_indices] = np.inf
    virtual_link_adj_array[virtual_link_2_indices_inv] = np.inf
    virtual_link_adj_array[node_self_indices] = np.inf
    virtual_link_adj_array[node_self_indices_inv] = np.inf

    swap_dist_adj_array[virtual_link_1_indices] = 0
    swap_dist_adj_array[virtual_link_1_indices_inv] = 0
    swap_dist_adj_array[virtual_link_2_indices] = 0
    swap_dist_adj_array[virtual_link_2_indices_inv] = 0
    swap_dist_adj_array[node_self_indices] = 0
    swap_dist_adj_array[node_self_indices_inv] = 0

    qubit_orient_adj_array[virtual_link_1_indices] = 0
    qubit_orient_adj_array[virtual_link_1_indices_inv] = 0
    qubit_orient_adj_array[virtual_link_2_indices] = 0
    qubit_orient_adj_array[virtual_link_2_indices_inv] = 0
    qubit_orient_adj_array[node_self_indices] = 0
    qubit_orient_adj_array[node_self_indices_inv] = 0

    link_within_node_adj_array[virtual_link_1_indices] = np.inf
    link_within_node_adj_array[virtual_link_1_indices_inv] = np.inf
    link_within_node_adj_array[node_self_indices] = np.inf
    link_within_node_adj_array[node_self_indices_inv] = np.inf

    free_qubit = smallest_free_qubit(
        node_self_indices[0], node_self_indices[1],
        node_index_hor_2, node_index_vert_2,
        virtual_link_adj_array)

    new_link_indices = (node_self_indices[0], node_self_indices[1],
                        node_index_hor_2, node_index_vert_2,
                        free_qubit)
    new_link_indices_inv = (node_index_hor_2, node_index_vert_2,
                            node_self_indices[0], node_self_indices[1],
                            free_qubit)

    if prob_succ_swap >= np.random.random():
        virtual_link_adj_array[new_link_indices] = max_age
        virtual_link_adj_array[new_link_indices_inv] = max_age

    # If a swap fails, pretend that the swap succeeded but set the
    # age of the new virtual links to a "fail age": this way, these
    # links can be used in swapping attempts of other nodes, but can
    # also be filtered after all nodes have attempted swaps.
    else:
        swap_fail_count += 1
        fail_age = 2*time_cutoff

        virtual_link_adj_array[new_link_indices] = fail_age
        virtual_link_adj_array[new_link_indices_inv] = fail_age

    swap_dist_adj_array[new_link_indices] = new_swap_dist
    swap_dist_adj_array[new_link_indices_inv] = new_swap_dist

    qubit_orient_adj_array[new_link_indices] = qubit_orient
    qubit_orient_adj_array[new_link_indices_inv] = qubit_orient_inv

    new_link = np.array([node_self_indices[0], node_self_indices[1],
                         node_index_hor_2, node_index_vert_2,
                         free_qubit,
                         virtual_link_adj_array[new_link_indices],
                         new_swap_dist,
                         qubit_orient_adj_array[new_link_indices],
                         qubit_orient_adj_array[new_link_indices_inv]],
                        dtype=int)

    return (virtual_link_adj_array, swap_fail_count,
            swap_dist_adj_array, qubit_orient_adj_array,
            link_within_node_adj_array, self_link, new_link)


def swap_impl_both_within_node(prob_succ_swap, time_cutoff,
                               virtual_link_adj_array, virtual_link_1,
                               virtual_link_2, swap_count, swap_fail_count,
                               swap_dist_adj_array, qubit_orient_adj_array,
                               link_within_node_adj_array):
    """
    Attempt an entanglement swap of two virtual links, 'virtual_link_1'
    and'virtual_link_2' (stored in qubits with different orientations of
    the central node) and succeed with a probability 'prob_succ_swap'.
    Both links are half of a pair of links to be considered a link
    between qubits within a node; the other link can be found using the
    "link_within_node_adj_array". If successful, the initial two (four
    in this method) links are discarded and a new link is generated,
    which is also a link between qubits within a node (i.e., we store two
    links again to resemble this single link). This new link can still be
    used by the node during this swap round, as the node does not know
    its qubits were connected and the other end of the link within a node
    has not been accessed yet (link_within_node_adj_array would not
    trigger otherwise). The number of failedswaps is counted in
    'swap_count_fail'. All adjacency arrays are adjusted accordingly.

    Parameters
    ----------
    prob_succ_swap : float
        The probability of a successful swap operation
    time_cutoff : int
        The cutoff time
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
        before a swap is implemented
    virtual_link_1: numpy.ndarray
    virtual_link_2: numpy.ndarray
        The two virtual links that are attempted to swap, characterised
        by the indices of the central link in the swap procedure, the
        indices of the nodes on the edges of swap, the qubits storing the
        virtual links, the virtual link ages, the number of swaps
        required to generate the link and the orientation of the qubit
        the link is stored in at both the central and side nodes.
        Both links are one half of a pair of links resembling a link
        between qubits within a node
    swap_count : int
        The number of swaps that have been attempted up to the start of
        the swapping procedure
    swap_fail_count : int
        The number of swaps that have failed up to the start of the
        swapping procedure
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links before swapping
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links before entanglement
        swapping, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.
    link_within_node_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the swap count (a unique
        identifier) of links between qubits within a node after
        swapping (that is, the swap count when these links were
        generated). These links are stored as two links but should
        be treated as one: the other half of a pair can be found by
        retrieving the indices of swap_count with opposite sign in
        link_within_node_adj_array

    Returns
    -------
    virtual_link_adj_array_post_cutoff: numpy.ndarray
        number_of_nodes × number_of_nodes × qubits_per_node-dimensional
        adjacency array containing the virtual links existing between
        two nodes in the network and their ages after swap implemenation

    swap_fail_count : int
        The number of swaps that have failed after swapping
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links after swapping
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links after entanglement
        swapping, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.
    link_within_node_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the swap count (a unique
        identifier) of links between qubits within a node after
        swapping (that is, the swap count when these links were
        generated). These links are stored as two links but should
        be treated as one: the other half of a pair can be found by
        retrieving the indices of swap_count with opposite sign in
        link_within_node_adj_array
    self_link_1 : numpy.ndarray
        the other half to virtual_link_1 which together resemble a link
        between qubits within a node
    self_link_2 : numpy.ndarray
        the other half to virtual_link_2 which together resemble a link
        between qubits within a node
    new_link_1 : numpy.ndarray
    new_link_2 : numpy.ndarray
        These newly generated virtual links resemble one link between
        qubits within a node and can still be used in this swap round
    """
    node_index_hor_cent = virtual_link_1[0]
    node_index_vert_cent = virtual_link_1[1]
    node_index_hor_1 = virtual_link_1[2]
    node_index_vert_1 = virtual_link_1[3]
    node_index_hor_2 = virtual_link_2[2]
    node_index_vert_2 = virtual_link_2[3]

    qubit_index_1 = virtual_link_1[4]
    qubit_index_2 = virtual_link_2[4]

    virtual_link_1_indices = (node_index_hor_cent, node_index_vert_cent,
                              node_index_hor_1, node_index_vert_1,
                              qubit_index_1)
    virtual_link_1_indices_inv = (node_index_hor_1, node_index_vert_1,
                                  node_index_hor_cent, node_index_vert_cent,
                                  qubit_index_1)

    virtual_link_2_indices = (node_index_hor_cent, node_index_vert_cent,
                              node_index_hor_2, node_index_vert_2,
                              qubit_index_2)
    virtual_link_2_indices_inv = (node_index_hor_2, node_index_vert_2,
                                  node_index_hor_cent, node_index_vert_cent,
                                  qubit_index_2)

    # Find the other half of the pair of links resembling a link within a node
    # to virtual_link_1/2 by looking for the indices of the element of
    # "link_within_node_adj_array" with the opposite-sign unique identifier
    node_self_indices_1 = (
        np.nonzero(link_within_node_adj_array == (
                   -link_within_node_adj_array[virtual_link_1_indices])))
    node_self_indices_1 = (
        node_self_indices_1[0][0], node_self_indices_1[1][0],
        node_self_indices_1[2][0], node_self_indices_1[3][0],
        node_self_indices_1[4][0])
    node_self_indices_inv_1 = (
        node_self_indices_1[2], node_self_indices_1[3],
        node_self_indices_1[0], node_self_indices_1[1],
        node_self_indices_1[4])

    node_self_indices_2 = (
        np.nonzero(link_within_node_adj_array == (
                   -link_within_node_adj_array[virtual_link_2_indices])))
    node_self_indices_2 = (
        node_self_indices_2[0][0], node_self_indices_2[1][0],
        node_self_indices_2[2][0], node_self_indices_2[3][0],
        node_self_indices_2[4][0])
    node_self_indices_inv_2 = (
        node_self_indices_2[2], node_self_indices_2[3],
        node_self_indices_2[0], node_self_indices_2[1],
        node_self_indices_2[4])

    self_link_1 = np.array(
        [node_self_indices_1[0], node_self_indices_1[1],
         node_self_indices_1[2], node_self_indices_1[3],
         node_self_indices_1[4],
         virtual_link_adj_array[node_self_indices_1],
         swap_dist_adj_array[node_self_indices_1],
         qubit_orient_adj_array[node_self_indices_1],
         qubit_orient_adj_array[node_self_indices_inv_1]], dtype=int)

    self_link_2 = np.array(
        [node_self_indices_2[0], node_self_indices_2[1],
         node_self_indices_2[2], node_self_indices_2[3],
         node_self_indices_2[4],
         virtual_link_adj_array[node_self_indices_2],
         swap_dist_adj_array[node_self_indices_2],
         qubit_orient_adj_array[node_self_indices_2],
         qubit_orient_adj_array[node_self_indices_inv_2]], dtype=int)

    max_age = max(virtual_link_adj_array[virtual_link_1_indices],
                  virtual_link_adj_array[virtual_link_2_indices],
                  virtual_link_adj_array[node_self_indices_1],
                  virtual_link_adj_array[node_self_indices_2])

    new_swap_dist_1 = (swap_dist_adj_array[virtual_link_1_indices]
                       + swap_dist_adj_array[node_self_indices_1])
    new_swap_dist_2 = (swap_dist_adj_array[virtual_link_2_indices]
                       + swap_dist_adj_array[node_self_indices_2])

    qubit_orient_1 = qubit_orient_adj_array[node_self_indices_1]
    qubit_orient_1_inv = qubit_orient_adj_array[node_self_indices_inv_1]
    qubit_orient_2 = qubit_orient_adj_array[node_self_indices_2]
    qubit_orient_2_inv = qubit_orient_adj_array[node_self_indices_inv_2]

    virtual_link_adj_array[virtual_link_1_indices] = np.inf
    virtual_link_adj_array[virtual_link_1_indices_inv] = np.inf
    virtual_link_adj_array[virtual_link_2_indices] = np.inf
    virtual_link_adj_array[virtual_link_2_indices_inv] = np.inf
    virtual_link_adj_array[node_self_indices_1] = np.inf
    virtual_link_adj_array[node_self_indices_inv_1] = np.inf
    virtual_link_adj_array[node_self_indices_2] = np.inf
    virtual_link_adj_array[node_self_indices_inv_2] = np.inf

    swap_dist_adj_array[virtual_link_1_indices] = 0
    swap_dist_adj_array[virtual_link_1_indices_inv] = 0
    swap_dist_adj_array[virtual_link_2_indices] = 0
    swap_dist_adj_array[virtual_link_2_indices_inv] = 0
    swap_dist_adj_array[node_self_indices_1] = 0
    swap_dist_adj_array[node_self_indices_inv_1] = 0
    swap_dist_adj_array[node_self_indices_2] = 0
    swap_dist_adj_array[node_self_indices_inv_2] = 0

    qubit_orient_adj_array[virtual_link_1_indices] = 0
    qubit_orient_adj_array[virtual_link_1_indices_inv] = 0
    qubit_orient_adj_array[virtual_link_2_indices] = 0
    qubit_orient_adj_array[virtual_link_2_indices_inv] = 0
    qubit_orient_adj_array[node_self_indices_1] = 0
    qubit_orient_adj_array[node_self_indices_inv_1] = 0
    qubit_orient_adj_array[node_self_indices_2] = 0
    qubit_orient_adj_array[node_self_indices_inv_2] = 0

    link_within_node_adj_array[virtual_link_1_indices] = np.inf
    link_within_node_adj_array[virtual_link_1_indices_inv] = np.inf
    link_within_node_adj_array[virtual_link_2_indices] = np.inf
    link_within_node_adj_array[virtual_link_2_indices_inv] = np.inf
    link_within_node_adj_array[node_self_indices_1] = np.inf
    link_within_node_adj_array[node_self_indices_inv_1] = np.inf
    link_within_node_adj_array[node_self_indices_2] = np.inf
    link_within_node_adj_array[node_self_indices_inv_2] = np.inf

    free_qubit_1 = smallest_free_qubit(
        node_self_indices_1[0], node_self_indices_1[1],
        node_self_indices_1[2], node_self_indices_1[3],
        virtual_link_adj_array)

    new_link_indices_1 = (node_self_indices_1[0], node_self_indices_1[1],
                          node_self_indices_1[2], node_self_indices_1[3],
                          free_qubit_1)
    new_link_indices_inv_1 = (node_self_indices_1[2], node_self_indices_1[3],
                              node_self_indices_1[0], node_self_indices_1[1],
                              free_qubit_1)

    # Temporary value: guarantees that free_qubit_2 is not the same as
    # free_qubit_1 in case the link-within-a-node indices are the same
    virtual_link_adj_array[new_link_indices_1] = 0

    free_qubit_2 = smallest_free_qubit(
        node_self_indices_2[0], node_self_indices_2[1],
        node_self_indices_2[2], node_self_indices_2[3],
        virtual_link_adj_array)

    new_link_indices_2 = (node_self_indices_2[0], node_self_indices_2[1],
                          node_self_indices_2[2], node_self_indices_2[3],
                          free_qubit_2)
    new_link_indices_inv_2 = (node_self_indices_2[2], node_self_indices_2[3],
                              node_self_indices_2[0], node_self_indices_2[1],
                              free_qubit_2)

    if prob_succ_swap >= np.random.random():
        virtual_link_adj_array[new_link_indices_1] = max_age
        virtual_link_adj_array[new_link_indices_inv_1] = max_age
        virtual_link_adj_array[new_link_indices_2] = max_age
        virtual_link_adj_array[new_link_indices_inv_2] = max_age

    # If a swap fails, pretend that the swap succeeded but set the
    # age of the new virtual links to a "fail age": this way, these
    # links can be used in swapping attempts of other nodes, but can
    # also be filtered after all nodes have attempted swaps.
    else:
        swap_fail_count += 1
        fail_age = 2*time_cutoff

        virtual_link_adj_array[new_link_indices_1] = fail_age
        virtual_link_adj_array[new_link_indices_inv_1] = fail_age
        virtual_link_adj_array[new_link_indices_2] = fail_age
        virtual_link_adj_array[new_link_indices_inv_2] = fail_age

    swap_dist_adj_array[new_link_indices_1] = new_swap_dist_1
    swap_dist_adj_array[new_link_indices_inv_1] = new_swap_dist_1
    swap_dist_adj_array[new_link_indices_2] = new_swap_dist_2
    swap_dist_adj_array[new_link_indices_inv_2] = new_swap_dist_2

    qubit_orient_adj_array[new_link_indices_1] = qubit_orient_1
    qubit_orient_adj_array[new_link_indices_inv_1] = qubit_orient_1_inv
    qubit_orient_adj_array[new_link_indices_2] = qubit_orient_2
    qubit_orient_adj_array[new_link_indices_inv_2] = qubit_orient_2_inv

    new_link_1 = np.array([new_link_indices_1[0], new_link_indices_1[1],
                           new_link_indices_1[2], new_link_indices_1[3],
                           free_qubit_1,
                           virtual_link_adj_array[new_link_indices_1],
                           new_swap_dist_1,
                           qubit_orient_adj_array[new_link_indices_1],
                           qubit_orient_adj_array[new_link_indices_inv_1]],
                          dtype=int)

    new_link_2 = np.array([new_link_indices_2[0], new_link_indices_2[1],
                           new_link_indices_2[2], new_link_indices_2[3],
                           free_qubit_2,
                           virtual_link_adj_array[new_link_indices_2],
                           new_swap_dist_2,
                           qubit_orient_adj_array[new_link_indices_2],
                           qubit_orient_adj_array[new_link_indices_inv_2]],
                          dtype=int)

    # The unique identifiers are only assigned to the indices lead by the
    # central node as only they can be used in future swap rounds (and hence
    # have to be identified again); the others get non-unique finite values
    link_within_node_adj_array[new_link_indices_1] = swap_count
    link_within_node_adj_array[new_link_indices_inv_1] = 0
    link_within_node_adj_array[new_link_indices_2] = -swap_count
    link_within_node_adj_array[new_link_indices_inv_2] = 0

    return (virtual_link_adj_array, swap_fail_count, swap_dist_adj_array,
            qubit_orient_adj_array, link_within_node_adj_array, self_link_1,
            self_link_2, new_link_1, new_link_2)


def swap_impl_same_within_node(prob_succ_swap, virtual_link_adj_array,
                               virtual_link_1, virtual_link_2, swap_fail_count,
                               swap_dist_adj_array, qubit_orient_adj_array,
                               link_within_node_adj_array):
    """
    Attempt an entanglement swap of two virtual links, 'virtual_link_1'
    and'virtual_link_2' (stored in qubits with different orientations of
    the central node). The links are half of a pair of links to be
    considered a link between qubits within a node generated in the same
    swap. Hence the swap will always discard the links regardless of
    whether the swap succeeds or fails (success happens with probability
    'prob_succ_swap' and failures are counted in 'swap_fail_count'). All
    adjacency arrays are adjusted accordingly.

    Parameters
    ----------
    prob_succ_swap : float
        The probability of a successful swap operation
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
        before a swap is implemented
    virtual_link_1: numpy.ndarray
    virtual_link_2: numpy.ndarray
        The two virtual links that are attempted to swap, characterised
        by the indices of the central link in the swap procedure, the
        indices of the nodes on the edges of swap, the qubits storing the
        virtual links, the virtual link ages, the number of swaps
        required to generate the link and the orientation of the qubit
        the link is stored in at both the central and side nodes.
        Both links are one half of a pair of links resembling a link
        between qubits within a node
    swap_fail_count : int
        The number of swaps that have failed up to the start of the
        swapping procedure
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links before swapping
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links before entanglement
        swapping, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.
    link_within_node_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the swap count (a unique
        identifier) of links between qubits within a node after
        swapping (that is, the swap count when these links were
        generated). These links are stored as two links but should
        be treated as one: the other half of a pair can be found by
        retrieving the indices of swap_count with opposite sign in
        link_within_node_adj_array

    Returns
    -------
    virtual_link_adj_array_post_cutoff: numpy.ndarray
        number_of_nodes × number_of_nodes × qubits_per_node-dimensional
        adjacency array containing the virtual links existing between
        two nodes in the network and their ages after swap implemenation
    swap_fail_count : int
        The number of swaps that have failed after swapping
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links after swapping
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links after entanglement
        swapping, specifically of the node defined by the first two
        node indices. 0 denotes an unused qubit, 1 a top-oriented qubit,
        2 right, 3 bottom, and 4 left.
    link_within_node_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the swap count (a unique
        identifier) of links between qubits within a node after
        swapping (that is, the swap count when these links were
        generated). These links are stored as two links but should
        be treated as one: the other half of a pair can be found by
        retrieving the indices of swap_count with opposite sign in
        link_within_node_adj_array
    """
    node_index_hor_cent = virtual_link_1[0]
    node_index_vert_cent = virtual_link_1[1]
    node_index_hor_1 = virtual_link_1[2]
    node_index_vert_1 = virtual_link_1[3]
    node_index_hor_2 = virtual_link_2[2]
    node_index_vert_2 = virtual_link_2[3]

    qubit_index_1 = virtual_link_1[4]
    qubit_index_2 = virtual_link_2[4]

    virtual_link_1_indices = (node_index_hor_cent, node_index_vert_cent,
                              node_index_hor_1, node_index_vert_1,
                              qubit_index_1)
    virtual_link_1_indices_inv = (node_index_hor_1, node_index_vert_1,
                                  node_index_hor_cent, node_index_vert_cent,
                                  qubit_index_1)

    virtual_link_2_indices = (node_index_hor_cent, node_index_vert_cent,
                              node_index_hor_2, node_index_vert_2,
                              qubit_index_2)
    virtual_link_2_indices_inv = (node_index_hor_2, node_index_vert_2,
                                  node_index_hor_cent, node_index_vert_cent,
                                  qubit_index_2)

    if prob_succ_swap >= np.random.random():
        None

    # Links are always discarded, but the number of fails is counted
    else:
        swap_fail_count += 1

    virtual_link_adj_array[virtual_link_1_indices] = np.inf
    virtual_link_adj_array[virtual_link_1_indices_inv] = np.inf
    virtual_link_adj_array[virtual_link_2_indices] = np.inf
    virtual_link_adj_array[virtual_link_2_indices_inv] = np.inf

    swap_dist_adj_array[virtual_link_1_indices] = 0
    swap_dist_adj_array[virtual_link_1_indices_inv] = 0
    swap_dist_adj_array[virtual_link_2_indices] = 0
    swap_dist_adj_array[virtual_link_2_indices_inv] = 0

    qubit_orient_adj_array[virtual_link_1_indices] = 0
    qubit_orient_adj_array[virtual_link_1_indices_inv] = 0
    qubit_orient_adj_array[virtual_link_2_indices] = 0
    qubit_orient_adj_array[virtual_link_2_indices_inv] = 0

    link_within_node_adj_array[virtual_link_1_indices] = np.inf
    link_within_node_adj_array[virtual_link_1_indices_inv] = np.inf
    link_within_node_adj_array[virtual_link_2_indices] = np.inf
    link_within_node_adj_array[virtual_link_2_indices_inv] = np.inf

    return (virtual_link_adj_array, swap_fail_count, swap_dist_adj_array,
            qubit_orient_adj_array, link_within_node_adj_array)


def cutoff_application(time_cutoff, virtual_link_adj_array,
                       swap_dist_adj_array, qubit_orient_adj_array):
    """
    Perform a round of applying cutoffs, where links equal to the
    'time_cutoff' are removed from the 'virtual_link_adj_array'. The
    age of a link is the number of time steps since the creation of
    the link, and links older than 'time_cutoff'  will have fidelities
    so low that they are not useful.

    Parameters
    ----------
    time_cutoff : int
        The cutoff time
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages
        before performing cutoffs
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links before cutoffs
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links before performing
        cutoffs, specifically of the node defined by the first two node
        indices. 0 denotes an unused qubit, 1 a top-oriented qubit, 2
        right, 3 bottom, and 4 left.

    Returns
    -------
    virtual_link_adj_array_post_cutoff: numpy.ndarray
        number_of_nodes × number_of_nodes × qubits_per_node-dimensional
        adjacency array containing the virtual links existing between
        two nodes in the network and their ages after performing cutoffs
    swap_dist_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the number of swaps required
        (+ 1) to generate each of the virtual links after cutoffs
    qubit_orient_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array storing the orientation of the
        qubits storing each of the virtual links after performing
        cutoffs, specifically of the node defined by the first two node
        indices. 0 denotes an unused qubit, 1 a top-oriented qubit, 2
        right, 3 bottom, and 4 left.
    """
    swap_dist_adj_array[virtual_link_adj_array == time_cutoff] = 0
    qubit_orient_adj_array[virtual_link_adj_array == time_cutoff] = 0
    virtual_link_adj_array[virtual_link_adj_array == time_cutoff] = np.inf

    return virtual_link_adj_array, swap_dist_adj_array, qubit_orient_adj_array


def smallest_free_qubit(node_index_hor_1, node_index_vert_1,
                        node_index_hor_2, node_index_vert_2,
                        virtual_link_adj_array):
    """
    Return the lowest-numbered index of a qubit that is free for storing
    a virtual link ('free_qubit') between '(node_index_hor_1,
    node_index_vert_1)' and '(node_index_hor_2, node_index_vert_2)' in
    'virtual_link_adj_array'

    Parameters
    ---------------
    node_index_hor_1 : int
    node_index_vert_1 : int
    node_index_hor_2 : int
    node_index_vert_2 : int
        The two nodes characterized by their horizontal and vertical
        indices between which a virtual link has been created that needs
        a free qubit in 'virtual_link_adj_array' to store the link
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages

    Returns
    ---------------
    free_qubit : int
        The smallest-numbered index of a qubit that is free for storing
        a virtual link between (node_index_hor_1, node_index_vert_1) and
        (node_index_hor_2, node_index_vert_2) in virtual_link_adj_array

    """
    # Retrieve the indices of the elements equal to infinity (i.e., no
    # link is stored) and take the minimum of the free elements in the
    # qubit register (note that np.where returns a tuple so we need to
    # call [0] to retrieve the qubit index).
    free_qubit = min(np.where(
        virtual_link_adj_array[node_index_hor_1, node_index_vert_1,
                               node_index_hor_2, node_index_vert_2,
                               :] == np.inf)[0])

    return free_qubit


def virtual_metrics(virtual_link_adj_array):
    """
    Return the virtual_node_degree and virtual_neighbourhood_size of all
    the nodes given a 'virtual_link_adj_array'.

    Parameters
    ----------
    virtual_link_adj_array : numpy.ndarray
        number_of_nodes^2 × number_of_nodes^2 × qubits_per_node-
        dimensional adjacency array characterizing virtual links by the
        horizontal and vertical indices of the two nodes it connects
        and the qubit index it is stored in, and storing their ages

    Returns
    -------
    virtual_node_degree: numpy.ndarray
        The number of virtual links the examined node has stored
    virtual_neighbourhood_size: numpy.ndarray
        Size of the set of all nodes that share a virtual link with the
        examined node
    """
    # Use product to speedup for-loops, meshgrid to be subscriptable.
    number_of_nodes = np.shape(virtual_link_adj_array)[0]
    cart_prod_node_indices = product(np.arange(0, number_of_nodes),
                                     np.arange(0, 1))
    cart_prod_node_indices_array = (
        np.array(np.meshgrid(np.arange(0, number_of_nodes),
                             np.arange(0, 1))).T.reshape(-1, 2))

    virtual_node_degree = np.empty((number_of_nodes, 1))
    virtual_neighbourhood_size = np.empty((number_of_nodes, 1))

    # Retrieve a Boolean array conditioned on the investigated node
    # having virtual links (finite entries) with another node, where
    # the nonzero (True) values are counted and its indices retrieved.
    for node_index_hor, node_index_vert in cart_prod_node_indices:
        virtual_node_degree[node_index_hor, node_index_vert] = (
            np.count_nonzero(np.isfinite(
                virtual_link_adj_array[
                    node_index_hor, node_index_vert, :, :, :])))
        virtual_neighbour_indices = (
            np.transpose(np.nonzero(np.isfinite(
                    virtual_link_adj_array[
                        node_index_hor, node_index_vert, :, :, :]))[:2]))
        # Retrieve a Boolean array conditioned on which node indices are
        # in virtual_neighbour_indices (which may contain duplicates):
        # if a combination of indices (check all permutations seperately
        # using [:, None]) match all (both) indices of any of the
        # entries of virtual_neighbour_indices, return True for this
        # combination of indices and count all True results.
        unique_virtual_neighbours = np.any(np.all(
            cart_prod_node_indices_array[:, None] == virtual_neighbour_indices,
            axis=-1), axis=-1)
        virtual_neighbourhood_size[node_index_hor, node_index_vert] = (
            np.count_nonzero(unique_virtual_neighbours))

    return virtual_node_degree, virtual_neighbourhood_size
