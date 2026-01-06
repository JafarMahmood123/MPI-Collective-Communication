from mpi4py import MPI
import numpy as np

def tree_reduce(comm, local_matrix):
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    
    # Define tree relationships
    left_child = 2 * rank + 1
    right_child = 2 * rank + 2
    parent = (rank - 1) // 2
    
    # Work on a copy of local data to accumulate results
    accumulated_matrix = local_matrix.copy()
    
    # 1. Receive from children (if they exist in the communicator)
    # Check Left Child
    if left_child < size:
        recv_buf = np.empty_like(local_matrix)
        comm.Recv(recv_buf, source=left_child, tag=11)
        accumulated_matrix += recv_buf
        
    # Check Right Child
    if right_child < size:
        recv_buf = np.empty_like(local_matrix)
        comm.Recv(recv_buf, source=right_child, tag=11)
        accumulated_matrix += recv_buf
        
    # 2. Send to Parent (if not root)
    if rank != 0:
        comm.Send(accumulated_matrix, dest=parent, tag=11)
        return None # Non-root nodes don't necessarily need the final result in this specific logic
    else:
        # Root holds the final result
        return accumulated_matrix

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Matrix dimensions
    rows, cols = 500, 500
    
    # Initialize local matrix with 1s (so final result should be matrix of value = size)
    local_matrix = np.ones((rows, cols), dtype='i')
    
    # Start Timing
    t_start = MPI.Wtime()
    
    # Perform Tree Reduce
    result_matrix = tree_reduce(comm, local_matrix)
    
    # End Timing
    t_end = MPI.Wtime()
    
    if rank == 0:
        # Verification
        expected_value = size
        if result_matrix[0,0] == expected_value:
             print(f"SUCCESS: Reduced Value is {result_matrix[0,0]} (Expected: {size})")
        else:
             print(f"FAILURE: Got {result_matrix[0,0]}, expected {size}")
             
        print(f"Tree Reduce Time: {t_end - t_start:.6f} seconds")
        
        # --- Sequential Comparison Part (for the table in Q2) ---
        # A purely sequential sum would involve iterating N times adding matrices
        t_seq_start = MPI.Wtime()
        seq_acc = np.zeros((rows, cols), dtype='i')
        for _ in range(size):
            seq_acc += np.ones((rows, cols), dtype='i')
        t_seq_end = MPI.Wtime()
        print(f"Sequential Time (Approx): {t_seq_end - t_seq_start:.6f} seconds")

if __name__ == "__main__":
    main()