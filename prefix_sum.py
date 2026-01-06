from mpi4py import MPI
import numpy as np

def prefix_mpi(comm, local_data):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # --- Step 1 (Parallel): Sum of local block ---
    local_sum = np.sum(local_data)
    
    # --- Step 2 (Sequential logic via Root): Calculate Offsets ---
    # Gather all local sums to root
    all_block_sums = comm.gather(local_sum, root=0)
    
    local_offset = 0
    if rank == 0:
        # Calculate prefix sum of the block sums to create offsets
        offsets = np.zeros(size, dtype='i') # Explicitly use 'i'
        current_acc = 0
        for i in range(size):
            offsets[i] = current_acc
            current_acc += all_block_sums[i]
    else:
        offsets = None
        
    # Scatter the offsets back to specific processes
    local_offset = comm.scatter(offsets, root=0)
    
    # --- Step 3 (Parallel): Local Prefix Sum + Offset ---
    local_prefix = np.cumsum(local_data)
    
    final_local_prefix = local_prefix + local_offset
    
    # Ensure the returned array matches the expected dtype (int32)
    return final_local_prefix.astype('i')

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N = 20
    if N % size != 0:
        if rank == 0: print("Error: N must be divisible by size.")
        return

    local_n = N // size
    
    if rank == 0:
        data = np.random.randint(0, 11, N).astype('i')
        print(f"Original Array (Rank 0): \n{data}")
    else:
        data = None
        
    local_data = np.empty(local_n, dtype='i')
    comm.Scatter(data, local_data, root=0)
    
    local_result = prefix_mpi(comm, local_data)
    
    final_result = None
    if rank == 0:
        # Receiver expects 'i' (int32)
        final_result = np.empty(N, dtype='i')
        
    comm.Gather(local_result, final_result, root=0)
    
    if rank == 0:
        print(f"MPI Prefix Sum Result: \n{final_result}")
        
        sequential_check = np.cumsum(data)
        if np.array_equal(final_result, sequential_check):
            print("SUCCESS: The MPI result matches the sequential calculation.")
        else:
            print("FAILURE: Results do not match.")

if __name__ == "__main__":
    main()