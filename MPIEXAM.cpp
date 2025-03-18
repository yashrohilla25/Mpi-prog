//Ass2Q1
/*#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, size, i;
    long long int num_points = 1000000;
    long long int local_num_points, local_count = 0, global_count = 0;
    double x, y;
    double pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_num_points = num_points / size;
    srand(time(NULL) + rank);

    for (i = 0; i < local_num_points; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        if ((x * x + y * y) <= 1) {
            local_count++;
        }
    }

    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi = (4.0 * global_count) / num_points;
        printf("Estimated Pi value: %f\n", pi);
    }

    MPI_Finalize();
    return 0;
}*/

//Ass2Q2
/*
#include <stdio.h>
#include <omp.h>

#define SIZE 70

void serialMatrixMultiply(int A[SIZE][SIZE], int B[SIZE][SIZE], int C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];

    // Initialize matrices A and B
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = i + j;
            B[i][j] = i + j;
        }
    }

    double start_time = omp_get_wtime();
    serialMatrixMultiply(A, B, C);
    double run_time = omp_get_wtime() - start_time;

    printf("Serial execution time: %f seconds\n", run_time);

    return 0;
}*/

//Ass 2 Q3
/*
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100 // Total number of elements to sort

// Function to compare and swap elements
void compareSwap(int* data, int rank, int partner_rank, int size, MPI_Comm comm) {
    int partner_data[N / size];
    MPI_Request request;

    // Send data to partner
    MPI_Isend(data, N / size, MPI_INT, partner_rank, 0, comm, &request);

    // Receive partner's data
    MPI_Recv(partner_data, N / size, MPI_INT, partner_rank, 0, comm, MPI_STATUS_IGNORE);

    // Wait for send to finish
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    int temporary[N / size];

    if (rank < partner_rank) {
        int ours = 0;
        int theirs = 0;

        for (int i = 0; i < N / size; i++) {
            if (data[ours] < partner_data[theirs]) {
                temporary[i] = data[ours];
                ours++;
            }
            else {
                temporary[i] = partner_data[theirs];
                theirs++;
            }
        }
    }
    else {
        int ours = N / size - 1;
        int theirs = N / size - 1;

        for (int i = N / size - 1; i >= 0; i--) {
            if (data[ours] > partner_data[theirs]) {
                temporary[i] = data[ours];
                ours--;
            }
            else {
                temporary[i] = partner_data[theirs];
                theirs--;
            }
        }
    }

    // Copy back onto our array
    for (int i = 0; i < N / size; i++) {
        data[i] = temporary[i];
    }
}

void parallelOddEvenSort(int* data, int rank, int size, MPI_Comm comm) {
    for (int phase = 0; phase < size; phase++) {
        int partner_rank;

        if (phase % 2 == 0) { // Even phase
            if (rank % 2 == 0) {
                partner_rank = rank + 1;
            }
            else {
                partner_rank = rank - 1;
            }
        }
        else { // Odd phase
            if (rank % 2 == 0) {
                partner_rank = rank - 1;
            }
            else {
                partner_rank = rank + 1;
            }
        }

        // If our partner doesn't exist, move on
        if (partner_rank < 0 || partner_rank >= size) {
            continue;
        }

        compareSwap(data, rank, partner_rank, size, comm);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int data[N / size];

    // Initialize data
    for (int i = 0; i < N / size; i++) {
        data[i] = rank * (N / size) + i;
    }

    // Shuffle data for testing
    if (rank == 0) {
        printf("Before sorting:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: ", i);
            for (int j = 0; j < N / size; j++) {
                printf("%d ", i * (N / size) + j);
            }
            printf("\n");
        }
    }

    parallelOddEvenSort(data, rank, size, MPI_COMM_WORLD);

    // Gather results
    int* gathered_data = NULL;
    if (rank == 0) {
        gathered_data = (int*)malloc(N * sizeof(int));
    }

    MPI_Gather(data, N / size, MPI_INT, gathered_data, N / size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("After sorting:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", gathered_data[i]);
        }
        printf("\n");
        free(gathered_data);
    }

    MPI_Finalize();
    return 0;
}
*/

//Ass2  Q4
/*
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

#define N 100            // Grid size
#define MAX_ITER 1000    // Maximum iterations
#define TOLERANCE 0.001  // Convergence criteria

using namespace std;

void initialize_grid(vector<vector<double>>& grid, int rank, int size) {
    int rows_per_proc = N / size;
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = (i == 0 || i == rows_per_proc - 1 || j == 0 || j == N - 1) ? 100.0 : 0.0;  // Boundary condition
        }
    }
}

void exchange_boundaries(vector<vector<double>>& grid, int rank, int size) {
    int rows_per_proc = N / size;
    MPI_Status status;

    if (rank > 0) {
        MPI_Send(&grid[0][0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&grid[0][0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
    }

    if (rank < size - 1) {
        MPI_Send(&grid[rows_per_proc - 1][0], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&grid[rows_per_proc - 1][0], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
    }
}

void update_grid(vector<vector<double>>& grid, vector<vector<double>>& new_grid, int rank, int size) {
    int rows_per_proc = N / size;
    double max_diff = 0.0;

    for (int i = 1; i < rows_per_proc - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            new_grid[i][j] = 0.25 * (grid[i - 1][j] + grid[i + 1][j] + grid[i][j - 1] + grid[i][j + 1]);
            max_diff = max(max_diff, fabs(new_grid[i][j] - grid[i][j]));
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    vector<vector<double>> grid(rows_per_proc, vector<double>(N, 0));
    vector<vector<double>> new_grid(rows_per_proc, vector<double>(N, 0));

    initialize_grid(grid, rank, size);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        exchange_boundaries(grid, rank, size);
        update_grid(grid, new_grid, rank, size);
        grid.swap(new_grid);
    }

    if (rank == 0) {
        cout << "Simulation completed!" << endl;
    }

    MPI_Finalize();
    return 0;
}*/

//Ass2 Q5
/*
#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int local_value, global_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assign each process a unique local value (for example, its rank + 1)
    local_value = rank + 1;

    // Perform Reduction (Sum)
    MPI_Reduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the final sum
    if (rank == 0) {
        printf("Sum of all values: %d\n", global_sum);
    }

    MPI_Finalize();
    return 0;
}*/

//Ass2 Q6 
/*
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8  // Vector size (should be divisible by number of processes)

int main(int argc, char* argv[]) {
    int rank, size, i;
    int local_n;
    int A[N], B[N];
    int local_A[N], local_B[N];
    int local_dot = 0, global_dot = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure N is divisible by the number of processes
    if (N % size != 0) {
        if (rank == 0) {
            printf("Vector size N must be divisible by the number of processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    local_n = N / size;  // Number of elements per process

    // Initialize vectors in root process
    if (rank == 0) {
        printf("Vector A: ");
        for (i = 0; i < N; i++) {
            A[i] = i + 1;  // Example: A = [1, 2, 3, ..., N]
            B[i] = i + 1;  // Example: B = [1, 2, 3, ..., N]
            printf("%d ", A[i]);
        }
        printf("\nVector B: ");
        for (i = 0; i < N; i++) {
            printf("%d ", B[i]);
        }
        printf("\n");
    }

    // Scatter data to all processes
    MPI_Scatter(A, local_n, MPI_INT, local_A, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, local_n, MPI_INT, local_B, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local dot product
    for (i = 0; i < local_n; i++) {
        local_dot += local_A[i] * local_B[i];
    }

    // Reduce local dot products to global sum at rank 0
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print final dot product at root
    if (rank == 0) {
        printf("Dot Product = %d\n", global_dot);
    }

    MPI_Finalize();
    return 0;
}
*/
// Ass2 Q7
/*
#include <mpi.h>
#include <stdio.h>

#define N 8  // Number of elements

int main(int argc, char* argv[]) {
    int rank, size;
    int local_value, prefix_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assign each process a value (for example, its rank + 1)
    local_value = rank + 1;

    // Perform Parallel Prefix Sum (Inclusive Scan)
    MPI_Scan(&local_value, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Each process prints its prefix sum
    printf("Process %d: Prefix Sum = %d\n", rank, prefix_sum);

    MPI_Finalize();
    return 0;
}
*/

//Ass 2 Q8
/*
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to perform serial transpose of a matrix
void serial_transpose(double* src, double* dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// Function to perform parallel matrix transpose using MPI
void parallel_transpose(double* matrix, double* transposed, int rows, int cols, int rank, int size) {
    int block_size = rows / size;
    double* local_matrix = (double*)malloc(block_size * cols * sizeof(double));
    double* local_transposed = (double*)malloc(block_size * rows * sizeof(double));

    // Gather local data from each process
    MPI_Scatter(matrix, block_size * cols, MPI_DOUBLE, local_matrix, block_size * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform local transpose
    serial_transpose(local_matrix, local_transposed, block_size, cols);

    // Create a custom MPI datatype for sending/receiving blocks
    MPI_Datatype block_t;
    MPI_Type_vector(block_size, 1, rows, MPI_DOUBLE, &block_t);
    MPI_Type_commit(&block_t);

    // Use MPI_Alltoall to transpose the matrix
    MPI_Alltoall(local_transposed, 1, block_t, transposed, 1, block_t, MPI_COMM_WORLD);

    // Free the custom datatype
    MPI_Type_free(&block_t);

    // Free local memory
    free(local_matrix);
    free(local_transposed);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 100;
    int cols = 100;

    if (rank == 0) {
        // Initialize the matrix on the root process
        double* matrix = (double*)malloc(rows * cols * sizeof(double));
        double* transposed = (double*)malloc(rows * cols * sizeof(double));

        // Initialize matrix elements (example)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i * cols + j] = i * cols + j;
            }
        }

        // Perform parallel transpose
        parallel_transpose(matrix, transposed, rows, cols, rank, size);

        // Print the transposed matrix (optional)
        if (rows <= 10 && cols <= 10) {
            printf("Transposed Matrix:\n");
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < rows; j++) {
                    printf("%f ", transposed[i * rows + j]);
                }
                printf("\n");
            }
        }

        free(matrix);
        free(transposed);
    }
    else {
        // Other processes do not need to initialize data
        parallel_transpose(NULL, NULL, rows, cols, rank, size);
    }

    MPI_Finalize();
    return 0;
}
*/

//Ass 3 Q1 
/*
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N (1 << 16)  // 2^16 = 65536 elements

void daxpy_serial(double a, double* X, double* Y, int n) {
    for (int i = 0; i < n; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

void daxpy_parallel(double a, double* X, double* Y, int n, int rank, int size) {
    int local_n = n / size;
    double* X_local = (double*)malloc(local_n * sizeof(double));
    double* Y_local = (double*)malloc(local_n * sizeof(double));

    // Ensure all processes allocate memory for X and Y
    if (X_local == NULL || Y_local == NULL) {
        printf("Memory allocation failed on process %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Distribute the vectors across processes
    MPI_Scatter(X, local_n, MPI_DOUBLE, X_local, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y, local_n, MPI_DOUBLE, Y_local, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform DAXPY on local chunks
    for (int i = 0; i < local_n; i++) {
        X_local[i] = a * X_local[i] + Y_local[i];
    }

    // Gather results back to the root process
    MPI_Gather(X_local, local_n, MPI_DOUBLE, X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(X_local);
    free(Y_local);
}

int main(int argc, char* argv[]) {
    int rank, size;
    double a = 2.5;
    double* X = NULL, * Y = NULL;
    double start_time, end_time, serial_time, parallel_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = N / size;

    // Allocate memory on all processes
    X = (double*)malloc(N * sizeof(double));
    Y = (double*)malloc(N * sizeof(double));

    if (X == NULL || Y == NULL) {
        printf("Memory allocation failed on process %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize data only on rank 0
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            X[i] = 1.0;
            Y[i] = 2.0;
        }

        // Measure serial execution time
        start_time = MPI_Wtime();
        daxpy_serial(a, X, Y, N);
        end_time = MPI_Wtime();
        serial_time = end_time - start_time;

        // Reset X for parallel execution
        for (int i = 0; i < N; i++) {
            X[i] = 1.0;
        }
    }

    // Synchronize before parallel execution
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    daxpy_parallel(a, X, Y, N, rank, size);

    end_time = MPI_Wtime();
    parallel_time = end_time - start_time;

    // Print results only from rank 0
    if (rank == 0) {
        printf("Serial Execution Time: %f seconds\n", serial_time);
        printf("Parallel Execution Time: %f seconds\n", parallel_time);
        printf("Speedup: %f\n", serial_time / parallel_time);
    }

    // Free memory
    free(X);
    free(Y);

    MPI_Finalize();
    return 0;
}
*/

//Ass 3 Q2
/*
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

static long num_steps = 100000;  // Total number of steps
double step;

int main(int argc, char* argv[]) {
    int rank, size, i;
    double x, sum = 0.0, pi, local_sum = 0.0;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step size remains the same across all processes
    step = 1.0 / (double)num_steps;

    // Broadcast num_steps from root process (0) to all processes
    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // Each process computes its own portion of the sum
    for (i = rank; i < num_steps; i += size) {
        x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    // Reduce all local sums to a final sum in rank 0
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Process 0 computes final π value
    if (rank == 0) {
        pi = step * sum;
        printf("Approximated π Value: %.15f\n", pi);
    }

    MPI_Finalize();
    return 0;
}
*/

//Ass3 Q3
/*
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VALUE 100  // Change this to the desired limit
#define MASTER 0

// Function to check if a number is prime
int is_prime(int num) {
    if (num < 2) return 0;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER) {
        // Master process
        int num = 2;  // Start from the first prime
        int workers = size - 1;
        int active_workers = 0;
        int received_value;

        printf("Primes up to %d:\n", MAX_VALUE);

        // Assign initial numbers to all slaves
        for (int i = 1; i <= workers && num <= MAX_VALUE; i++) {
            MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            num++;
            active_workers++;
        }

        // Process results and continue sending numbers
        while (active_workers > 0) {
            MPI_Recv(&received_value, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (received_value > 0) {
                printf("%d ", received_value);
            }

            if (num <= MAX_VALUE) {
                // Send next number to test
                MPI_Send(&num, 1, MPI_INT, received_value > 0 ? received_value : -received_value, 0, MPI_COMM_WORLD);
                num++;
            }
            else {
                // Send termination signal (zero)
                MPI_Send(&num, 1, MPI_INT, received_value > 0 ? received_value : -received_value, 0, MPI_COMM_WORLD);
                active_workers--;
            }
        }
        printf("\n");
    }
    else {
        // Slave process
        int received_num;
        while (1) {
            // Request a number to test
            MPI_Recv(&received_num, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (received_num == 0) break; // Termination signal

            // Check primality and send result back
            int result = is_prime(received_num) ? received_num : -received_num;
            MPI_Send(&result, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
*/