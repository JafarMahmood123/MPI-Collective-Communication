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
    
    # Test cases: Small, Medium, Large
    test_sizes = [500, 2000, 4000] 
    
    if rank == 0:
        print(f"{'Size':<10} | {'Processes':<10} | {'Parallel (s)':<15} | {'Sequential (s)':<15} | {'Speedup':<10}")
        print("-" * 75)

    for N in test_sizes:
        rows, cols = N, N
        
        # 1. Setup Data
        local_matrix = np.ones((rows, cols), dtype='i')
        comm.Barrier()
        
        # 2. Measure Parallel Tree Reduce
        t_start = MPI.Wtime()
        result_matrix = tree_reduce(comm, local_matrix)
        t_end = MPI.Wtime()
        parallel_time = t_end - t_start
        
        # 3. Measure Sequential (Rank 0 only) and Print
        if rank == 0:
            t_seq_start = MPI.Wtime()
            seq_acc = np.zeros((rows, cols), dtype='i')
            # Simulate adding 'size' matrices
            dummy = np.ones((rows, cols), dtype='i')
            for _ in range(size):
                seq_acc += dummy
            t_seq_end = MPI.Wtime()
            sequential_time = t_seq_end - t_seq_start
            
            speedup = sequential_time / parallel_time
            print(f"{str(N)+'x'+str(N):<10} | {size:<10} | {parallel_time:.6f}        | {sequential_time:.6f}        | {speedup:.2f}x")

if __name__ == "__main__":
    main()