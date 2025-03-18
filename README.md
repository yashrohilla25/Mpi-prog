# Mpi-prog
Some mpi basic c programs
Overview
This repository contains various MPI (Message Passing Interface) and OpenMP-based programs for solving computational problems in parallel and distributed systems. Below is a detailed explanation of each program, including its purpose, compilation instructions, and execution steps.

Table of Contents
Ass2Q1: Estimating Pi using Monte Carlo Method

Ass2Q2: Serial Matrix Multiplication

Ass2Q3: Parallel Odd-Even Sort

Ass2Q4: Heat Distribution Simulation

Ass2Q5: MPI Reduction for Sum

Ass2Q6: Parallel Dot Product

Ass2Q7: Parallel Prefix Sum

Ass2Q8: Parallel Matrix Transpose

Ass3Q1: DAXPY Operation (Serial vs Parallel)

Ass3Q2: Approximating Pi using Numerical Integration

Ass3Q3: Finding Prime Numbers using Master-Slave Model

Ass2Q1: Estimating Pi using Monte Carlo Method
Description: This program estimates the value of Pi using the Monte Carlo method in parallel using MPI.

Compilation:

bash
mpicc -o monte_carlo_pi ass2q1.c
Execution:

bash
mpirun -np <num_processes> ./monte_carlo_pi
Ass2Q2: Serial Matrix Multiplication
Description: Performs matrix multiplication serially without parallelism.

Compilation:

bash
gcc -o serial_matrix_multiply ass2q2.c -fopenmp
Execution:

bash
./serial_matrix_multiply
Ass2Q3: Parallel Odd-Even Sort
Description: Implements the odd-even sorting algorithm in parallel using MPI.

Compilation:

bash
mpicc -o parallel_sort ass2q3.c
Execution:

bash
mpirun -np <num_processes> ./parallel_sort
Ass2Q4: Heat Distribution Simulation
Description: Simulates heat distribution in a 2D grid using Jacobi iteration in parallel with MPI.

Compilation:

bash
mpicxx -o heat_simulation ass2q4.cpp
Execution:

bash
mpirun -np <num_processes> ./heat_simulation
Ass2Q5: MPI Reduction for Sum
Description: Demonstrates the use of MPI_Reduce to calculate the sum of values across processes.

Compilation:

bash
mpicc -o mpi_reduction_sum ass2q5.c
Execution:

bash
mpirun -np <num_processes> ./mpi_reduction_sum
Ass2Q6: Parallel Dot Product
Description: Computes the dot product of two vectors in parallel using MPI.

Compilation:

bash
mpicc -o parallel_dot_product ass2q6.c
Execution:

bash
mpirun -np <num_processes> ./parallel_dot_product
Ass2Q7: Parallel Prefix Sum
Description: Implements a parallel prefix sum (inclusive scan) operation using MPI_Scan.

Compilation:

bash
mpicc -o prefix_sum ass2q7.c
Execution:

bash
mpirun -np <num_processes> ./prefix_sum
Ass2Q8: Parallel Matrix Transpose
Description: Transposes a matrix in parallel using MPI.

Compilation:

bash
mpicc -o matrix_transpose ass2q8.c
Execution:

bash
mpirun -np <num_processes> ./matrix_transpose
Ass3Q1: DAXPY Operation (Serial vs Parallel)
Description: Compares the performance of serial and parallel implementations of the DAXPY operation (Y = a*X + Y).

Compilation:

bash
mpicc -o daxpy_operation ass3q1.c -lm
Execution:

bash
mpirun -np <num_processes> ./daxpy_operation
Ass3Q2: Approximating Pi using Numerical Integration
Description: Approximates the value of Pi using numerical integration in parallel with MPI.

Compilation:

bash
mpicc -o pi_integration ass3q2.c -lm
Execution:

bash
mpirun -np <num_processes> ./pi_integration
Ass3Q3: Finding Prime Numbers using Master-Slave Model
Description:
This program finds all prime numbers up to a maximum value (MAX_VALUE) using the master-slave model in MPI.

The master assigns numbers to test for primality.

Slaves test these numbers and return results to the master.

Compilation:
bash
mpicc -o find_primes ass3q3.c -lm 
Execution:
bash
mpirun -np <num_processes> ./find_primes 
Replace <num_processes> with the desired number of processes.

Sample Output:
For MAX_VALUE = 100, the output will list all primes up to 100.
