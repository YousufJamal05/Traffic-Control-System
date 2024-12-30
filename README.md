# Parallel Traffic Management System

This repository contains two implementations of a Traffic Management System using parallel computing paradigms:

1. **MPI Implementation** (`MPI.cpp`): Utilizes the Message Passing Interface (MPI) for distributed parallelism.
2. **OpenMP Implementation** (`OpenMP.cpp`): Employs OpenMP for shared-memory parallelism.

## Features

The project focuses on simulating various aspects of traffic management, including:

- **Traffic Flow Monitoring**: Monitors the flow of vehicles across different locations.
- **Incident Detection**: Identifies traffic incidents based on sensor data.
- **Congestion Monitoring**: Tracks traffic density in real-time.
- **Vehicle Counting**: Counts vehicles passing through specific sections.
- **Adaptive Signal Control**: Adjusts traffic signals dynamically based on traffic flow.
- **Predictive Analytics**: Predicts future traffic patterns using historical data.
- **Air Quality Monitoring**: Tracks air quality indices using sensor data.
- **Noise Pollution Monitoring**: Monitors noise pollution levels in urban areas.
- **Green Wave System**: Simulates synchronization of traffic signals for smoother traffic flow.
- **EV Charging Integration**: Manages charging station data and prioritization for electric vehicles.
- **Public Transport Integration**: Processes data from public transport systems.
- **Traffic Simulation**: Simulates traffic flow and incidents.

## Prerequisites

### MPI Implementation

- **MPI Library** (e.g., MPICH, OpenMPI)
- Compiler supporting MPI (e.g., `mpic++`)

### OpenMP Implementation

- Compiler with OpenMP support (e.g., GCC, Clang)
- Multi-core processor for optimal performance

## Compilation

### MPI

To compile the MPI implementation:
```bash
mpic++ MPI.cpp -o mpi_traffic_management
```

### OpenMP

To compile the OpenMP implementation:
```bash
g++ -fopenmp OpenMP.cpp -o openmp_traffic_management
```

## Execution

#### MPI

Run the MPI executable with the desired number of processes:
```bash
mpirun -np <number_of_processes> ./mpi_traffic_management
```
### OpenMP


Run the OpenMP executable:
```bash
./openmp_traffic_management
```

## Code Overview

### MPI Implementation (MPI.cpp)
**Implements distributed parallelism across multiple processes.
**Uses MPI communication primitives (MPI_Init, MPI_Gather, MPI_Reduce) to manage data sharing.
**Each process handles a subset of the workload, and the results are aggregated at rank 0.

### OpenMP Implementation (OpenMP.cpp)
**Uses OpenMP directives (#pragma omp parallel, #pragma omp for, etc.) to parallelize computations.
**Leverages shared memory for efficient data access among threads.
**Implements fine-grained parallelism for tasks like traffic simulations.

## Performance Notes
**MPI: Suitable for distributed systems where each node has its own memory. Scales well for large datasets.
**OpenMP: Designed for shared-memory architectures. Provides ease of implementation with moderate scalability.
