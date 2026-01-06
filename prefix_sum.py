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
        # Example: sums=[10, 20, 30] -> offsets=[0, 10, 30]
        offsets = np.zeros(size, dtype=int)
        current_acc = 0
        for i in range(size):
            offsets[i] = current_acc
            current_acc += all_block_sums[i]
    else:
        offsets = None
        
    # Scatter the offsets back to specific processes
    local_offset = comm.scatter(offsets, root=0)
    
    # --- Step 3 (Parallel): Local Prefix Sum + Offset ---
    # Calculate standard local cumulative sum
    local_prefix = np.cumsum(local_data)
    
    # Add the offset (which is the sum of all previous blocks)
    final_local_prefix = local_prefix + local_offset
    
    return final_local_prefix

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N = 20 # Total size of array (Must be divisible by size for this simple example)
    if N % size != 0:
        if rank == 0: print("Error: N must be divisible by number of processors for this example.")
        return

    local_n = N // size
    
    # 1. Rank 0 generates random array
    if rank == 0:
        data = np.random.randint(0, 11, N).astype('i')
        print(f"Original Array (Rank 0): \n{data}")
    else:
        data = None
        
    # 2. Scatter data using MPI_Scatter
    local_data = np.empty(local_n, dtype='i')
    comm.Scatter(data, local_data, root=0)
    
    # 3. Call the logic function
    local_result = prefix_mpi(comm, local_data)
    
    # 4. Gather results using MPI_Gather
    final_result = None
    if rank == 0:
        final_result = np.empty(N, dtype='i')
        
    comm.Gather(local_result, final_result, root=0)
    
    # 5. Verify correctness
    if rank == 0:
        print(f"MPI Prefix Sum Result: \n{final_result}")
        
        # Verification using Numpy's sequential cumsum
        sequential_check = np.cumsum(data)
        if np.array_equal(final_result, sequential_check):
            print("SUCCESS: The MPI result matches the sequential calculation.")
        else:
            print("FAILURE: Results do not match.")

if __name__ == "__main__":
    main()