#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cstring> // for memset
#include <iomanip> // for std::setw

// --- Function Prototypes ---

// 1. Declare your parallel function (implemented in the other file)
void reduce_tree(const int* send_data, int* recv_data, int count, MPI_Comm communicator);

// 2. The Teacher's Sequential Implementation (Pasted here for comparison)
void reduce_sequential(const int* send_data, int* recv_data, int count, MPI_Comm communicator)
{
    int my_rank;
    int com_size;
    MPI_Comm_rank(communicator, &my_rank);
    MPI_Comm_size(communicator, &com_size);

    int* gather_buffer = NULL;
    if (my_rank == 0)
    {
        gather_buffer = (int*) calloc(count * com_size, sizeof(int));
    }

    // Note: const_cast needed because MPI expects void* but we have const int*
    MPI_Gather(const_cast<int*>(send_data), count, MPI_INT, gather_buffer, count, MPI_INT, 0, communicator);

    if (my_rank == 0)
    {
        memset(recv_data, 0, count * sizeof(int));
        for (int p = 0; p < com_size; p++)
            for (int i = 0; i < count; i++)
                recv_data[i] += gather_buffer[count * p + i];
        free(gather_buffer);
    }
}

// --- Main Testing Logic ---
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Configuration for the test
    // We will test 3 different sizes to show the difference
    std::vector<int> test_sizes = {500, 1000, 10000, 100000 , 1000000, 5000000}; 

    if (rank == 0) {
        std::cout << "=========================================================\n";
        std::cout << "   Parallel (Tree) vs Sequential (Gather) Comparison \n";
        std::cout << "=========================================================\n";
        std::cout << std::left << std::setw(15) << "Data Count" 
                  << std::left << std::setw(15) << "Tree Time" 
                  << std::left << std::setw(15) << "Seq Time" 
                  << "Result Match?" << std::endl;
        std::cout << "---------------------------------------------------------\n";
    }

    for (int count : test_sizes) {
        // 1. Generate Data (Everyone sends 1s)
        std::vector<int> send_data(count, 1);
        std::vector<int> recv_tree(count, 0);
        std::vector<int> recv_seq(count, 0);

        // 2. Run & Time YOUR Parallel Implementation
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        
        reduce_tree(send_data.data(), recv_tree.data(), count, MPI_COMM_WORLD);
        
        double t2 = MPI_Wtime();
        double time_tree = t2 - t1;

        // 3. Run & Time TEACHER'S Sequential Implementation
        MPI_Barrier(MPI_COMM_WORLD);
        double t3 = MPI_Wtime();
        
        reduce_sequential(send_data.data(), recv_seq.data(), count, MPI_COMM_WORLD);
        
        double t4 = MPI_Wtime();
        double time_seq = t4 - t3;

        // 4. Verify & Print
        if (rank == 0) {
            bool correct = true;
            // Check first and last element to be sure
            if (recv_tree[0] != recv_seq[0] || recv_tree[count-1] != recv_seq[count-1]) {
                correct = false;
            }

            std::cout << std::left << std::setw(15) << count 
                      << std::left << std::setw(15) << std::fixed << std::setprecision(5) << time_tree 
                      << std::left << std::setw(15) << time_seq 
                      << (correct ? "YES" : "NO") << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}