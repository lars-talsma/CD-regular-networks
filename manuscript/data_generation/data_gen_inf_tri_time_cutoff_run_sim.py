from timeit import default_timer as timer
import numpy as np
import skeleton_inf_tri


def run_simulation(params):
    """
    Given a simulation time, a steady-state window and the required network parameters, this function runs N_sample
    realizations of network simulation for a specific network parameter and swap probability and returns:
    - The sample mean and standard error of the mean of both the virtual neighborhood size and the virtual node degree;
    - The error required for the steady state algorithm and the maximum absolute difference between sample means for all
      time steps in the steady state window.
    """
    start = timer()
    test = False
    N_sample = 100

    # Network parameters
    prob_succ_gen = 1
    prob_succ_swap = 1
    max_swap_dist = 2
    
    # Investigated (and associated) parameters
    time_cutoff, prob_swap = params 
    number_of_nodes = 2*(max_swap_dist+1)+1
    sim_time = 3*time_cutoff
    qubits_per_node = 6*time_cutoff  # Represents an infinite memory for an infinite chain
    steady_state_window = time_cutoff

    # Reset random number seed for reproducibility 
    np.random.seed(0)
    
    virtual_node_degree_sample = np.empty((N_sample, sim_time))
    virtual_neighbourhood_size_sample = np.empty((N_sample, sim_time))
    
    for n in range(N_sample):
        (virtual_link_age_array, virtual_node_degree, virtual_neighbourhood_size, swap_count, link_fail_count) = (
            skeleton_inf_tri.simulation_infinite_triangular(prob_succ_gen, prob_succ_swap, prob_swap, time_cutoff, 
                                                        qubits_per_node, sim_time, max_swap_dist, number_of_nodes, test))
        
        # We sample data from node at position [0, 0] (arbitrary choice)
        virtual_node_degree_sample[n, :] = virtual_node_degree[0, 0, :]
        virtual_neighbourhood_size_sample[n, :] = virtual_neighbourhood_size[0, 0, :]
        
    ## Steady state routine
    virtual_node_degree_mean = np.mean(virtual_node_degree_sample, axis=0)
    virtual_node_degree_std_error = np.std(virtual_node_degree_sample, axis=0)/np.sqrt(N_sample)

    virtual_neighbourhood_size_mean = np.mean(virtual_neighbourhood_size_sample, axis=0)
    virtual_neighbourhood_size_std_error = np.std(virtual_neighbourhood_size_sample, axis=0)/np.sqrt(N_sample)

    virtual_node_degree_diff = np.empty(steady_state_window-1)
    virtual_neighbourhood_size_diff = np.empty(steady_state_window-1)
    
    # The error is defined as 2*(b-a)/sqrt(N_sample), where a (b) is the minimum (maximum) value of the performance metric
    virtual_node_degree_error = (6*time_cutoff-0)/np.sqrt(N_sample)
    virtual_neighbourhood_size_error = (6*np.minimum(time_cutoff, (max_swap_dist+1)*(max_swap_dist+2)/2)-0)/np.sqrt(N_sample)
    
    for k in range(steady_state_window-1):
        virtual_node_degree_diff[k] = np.abs(virtual_node_degree_mean[sim_time-steady_state_window]-
                                                virtual_node_degree_mean[sim_time-steady_state_window+(k+1)])
        virtual_neighbourhood_size_diff[k] = np.abs(virtual_neighbourhood_size_mean[sim_time-steady_state_window]-
                                                virtual_neighbourhood_size_mean[sim_time-steady_state_window+(k+1)])
    
    virtual_node_degree_steady_state = virtual_node_degree_mean[-1]
    virtual_node_degree_steady_state_std_error = virtual_node_degree_std_error[-1]
    
    virtual_neighbourhood_size_steady_state = virtual_neighbourhood_size_mean[-1]
    virtual_neighbourhood_size_steady_state_std_error = virtual_neighbourhood_size_std_error[-1]

    end = timer()
    print(time_cutoff, prob_swap, end - start)

    return (virtual_neighbourhood_size_steady_state, virtual_neighbourhood_size_steady_state_std_error, 
        virtual_neighbourhood_size_error, max(virtual_neighbourhood_size_diff), virtual_node_degree_steady_state,
        virtual_node_degree_steady_state_std_error, virtual_node_degree_error, max(virtual_node_degree_diff))
