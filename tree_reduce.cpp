#include <mpi.h>
#include <vector>
#include <algorithm>

// Your Parallel Implementation
void reduce_tree(const int* send_data, int* recv_data, int count, MPI_Comm communicator) {
    int rank, size;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);

    // 1. Initialize local accumulator with our own data
    // We use std::vector for safe memory management in C++
    std::vector<int> accumulated_data(send_data, send_data + count);

    // 2. Define Tree Topology
    int left_child = 2 * rank + 1;
    int right_child = 2 * rank + 2;
    int parent = (rank - 1) / 2;

    MPI_Status status;
    std::vector<int> temp_recv_buffer(count); // Temp buffer for incoming child data

    // 3. Receive from Left Child (if exists)
    if (left_child < size) {
        MPI_Recv(temp_recv_buffer.data(), count, MPI_INT, left_child, 0, communicator, &status);
        for (int i = 0; i < count; ++i) {
            accumulated_data[i] += temp_recv_buffer[i];
        }
    }

    // 4. Receive from Right Child (if exists)
    if (right_child < size) {
        MPI_Recv(temp_recv_buffer.data(), count, MPI_INT, right_child, 0, communicator, &status);
        for (int i = 0; i < count; ++i) {
            accumulated_data[i] += temp_recv_buffer[i];
        }
    }

    // 5. Send to Parent OR Store Result (if Root)
    if (rank == 0) {
        // We are Root: Copy final result to the user's output buffer
        std::copy(accumulated_data.begin(), accumulated_data.end(), recv_data);
    } else {
        // We are a Child: Send partial sum to Parent
        MPI_Send(accumulated_data.data(), count, MPI_INT, parent, 0, communicator);
    }
}