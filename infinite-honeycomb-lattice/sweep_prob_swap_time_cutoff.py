import skeleton
import numpy as np

def run_simulation(params):
    """This is the main processing function. It will contain whatever
    code should be run on multiple processors.

    """
    prob_succ_gen = 1
    prob_succ_swap = 1
    max_swap_dist = 4
    number_of_nodes = 2*(max_swap_dist+1)+1
    time = 5000
    test = False

    time_cutoff, prob_swap = params 
    
    qubits_per_node = 3*time_cutoff  # Represents an infinite memory
    
    np.random.seed(0)
    
    virtual_link_age_array, virtual_node_degree, virtual_neighbourhood_size, swap_count, link_fail_count = skeleton.simulation_infinite_honeycomb(prob_succ_gen, prob_succ_swap, prob_swap, time_cutoff, qubits_per_node, time, max_swap_dist, number_of_nodes, test)
    
    virtual_node_degree_mean = np.mean(virtual_node_degree[:, :, 10*time_cutoff:])
    virtual_node_degree_std = np.std(virtual_node_degree[:, :, 10*time_cutoff:])
    virtual_neighbourhood_size_mean = np.mean(virtual_neighbourhood_size[:, :, 10*time_cutoff:])
    virtual_neighbourhood_size_std = np.std(virtual_neighbourhood_size[:, :, 10*time_cutoff:])
    
    virtual_node_degree_5pct = np.percentile(virtual_node_degree[:, :, 10*time_cutoff:], 5)
    virtual_node_degree_10pct = np.percentile(virtual_node_degree[:, :, 10*time_cutoff:], 10)
    virtual_node_degree_25pct = np.percentile(virtual_node_degree[:, :, 10*time_cutoff:], 25)
    virtual_node_degree_50pct = np.percentile(virtual_node_degree[:, :, 10*time_cutoff:], 50)
    virtual_node_degree_75pct = np.percentile(virtual_node_degree[:, :, 10*time_cutoff:], 75)
    virtual_node_degree_90pct = np.percentile(virtual_node_degree[:, :, 10*time_cutoff:], 90)
    virtual_node_degree_95pct = np.percentile(virtual_node_degree[:, :, 10*time_cutoff:], 95)
    
    virtual_neighbourhood_size_5pct = np.percentile(virtual_neighbourhood_size[:, :, 10*time_cutoff:], 5)    
    virtual_neighbourhood_size_10pct = np.percentile(virtual_neighbourhood_size[:, :, 10*time_cutoff:], 10)
    virtual_neighbourhood_size_25pct = np.percentile(virtual_neighbourhood_size[:, :, 10*time_cutoff:], 25)
    virtual_neighbourhood_size_50pct = np.percentile(virtual_neighbourhood_size[:, :, 10*time_cutoff:], 50)
    virtual_neighbourhood_size_75pct = np.percentile(virtual_neighbourhood_size[:, :, 10*time_cutoff:], 75)
    virtual_neighbourhood_size_90pct = np.percentile(virtual_neighbourhood_size[:, :, 10*time_cutoff:], 90)
    virtual_neighbourhood_size_95pct = np.percentile(virtual_neighbourhood_size[:, :, 10*time_cutoff:], 95)
    
    return virtual_node_degree_mean, virtual_node_degree_std, virtual_neighbourhood_size_mean, virtual_neighbourhood_size_std, virtual_node_degree_5pct, virtual_node_degree_10pct, virtual_node_degree_25pct, virtual_node_degree_50pct, virtual_node_degree_75pct, virtual_node_degree_90pct, virtual_node_degree_95pct, virtual_neighbourhood_size_5pct, virtual_neighbourhood_size_10pct, virtual_neighbourhood_size_25pct, virtual_neighbourhood_size_50pct, virtual_neighbourhood_size_75pct, virtual_neighbourhood_size_90pct, virtual_neighbourhood_size_95pct