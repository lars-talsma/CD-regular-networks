"""
Given a set of swap probabilities and network parameters, runs network simulations on multiple processors calling the 
run_sim function to run specific instances of a swap probability and network parameter value. Returns:
    - The sample mean and standard error of the mean of both the virtual neighborhood size and the virtual node degree; 
    - The error required for the steady state algorithm and the maximum absolute difference between sample means for all
      time steps in the steady state window, for all input swap probabilities and network parameters. 
"""
import multiprocessing
from itertools import product
from timeit import default_timer as timer
import numpy as np
import data_gen_inf_sq_time_cutoff_run_sim


prob_swap_sweep_size = 100+1
prob_swap_range = np.arange(0, prob_swap_sweep_size)/(prob_swap_sweep_size-1) # Only run probabilistic swaps
time_cutoff_range = [21, 10, 3]

start = timer()

if __name__ == '__main__':
    # Take the cartesian product as the parameters because pool.map() takes only one iterable
    params = product(time_cutoff_range, prob_swap_range)
    pool = multiprocessing.Pool(38)
    results = pool.map(data_gen_inf_sq_time_cutoff_run_sim.run_simulation, params, chunksize=1)

end = timer()
print(end - start) # Time in seconds

np.save('data/data_inf_sq_time_cutoff_ext_data', results)