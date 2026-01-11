#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>  
#include <stdbool.h>

void reduce_tree(
    int* send_data,
    int* recv_data,
    int count,
    MPI_Comm communicator)
{
    /* 
    Add your code here
    */
}


void reduce_sequential(
    int* send_data,
    int* recv_data,
    int count,
    MPI_Comm communicator)
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

    MPI_Gather(send_data, count, MPI_INT, gather_buffer, count, MPI_INT, 0, communicator);

    if (my_rank == 0)
    {
        memset(recv_data, 0, count * sizeof(int));
        for (int p = 0; p < com_size; p++)
            for (int i = 0; i < count; i++)
                recv_data[i] += gather_buffer[count * p + i];
        free(gather_buffer);
    }
}



int main(int argc, char** args)
{
    MPI_Init(&argc, &args);
    int count = 10;
    int max_value = 64;
    int* recv_array_tree = NULL;
    int* recv_array_sequential = NULL;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0)
    {
        recv_array_tree = (int*) malloc(count * sizeof(int));
        recv_array_sequential = (int*) malloc(count * sizeof(int));
    }

    int* send_array = (int*)malloc(count * sizeof(int));
    for (int i = 0; i < count; i++)
        send_array[i] = my_rank;

    reduce_tree(send_array, recv_array_tree, count, MPI_COMM_WORLD);
    reduce_sequential(send_array, recv_array_sequential, count, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        for (int i = 0; i < count; i++)
            if (recv_array_tree[i] == recv_array_sequential[i])
                printf("At index %i: reduce_tree is %i, reduce_sequential is %i\n",
                    i, recv_array_tree[i], recv_array_sequential[i]);

        free(recv_array_tree);
        free(recv_array_sequential);
    }
    free(send_array);
    MPI_Finalize();
    return 0;
}

