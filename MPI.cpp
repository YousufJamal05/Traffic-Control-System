#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <mpi.h>

using namespace std;

#define NUM_SENSORS 100
#define NUM_CAMERAS 50
#define NUM_VEHICLES 10000
#define NUM_INTERSECTIONS 50
#define NUM_EV_STATIONS 50
#define NUM_PEDESTRIANS 200
#define NUM_DRONES 10
#define NUM_USER_PREFERENCES 50

// Function prototypes
void trafficFlowMonitoring(vector<int>& vehicle_data, int rank, int size);
void incidentDetection(vector<int>& incidents, int rank, int size);
void congestionMonitoring(vector<int>& traffic_density, int rank, int size);
void vehicleCounting(vector<int>& vehicle_data, int num_sections, int rank, int size);
void adaptiveSignalControl(vector<vector<int>>& traffic_lights, vector<int>& traffic_flow, int rank, int size);
void predictiveAnalytics(vector<int>& historical_data, vector<int>& future_traffic, int rank, int size);
void airQualityMonitoring(vector<int>& air_quality_data, int rank, int size);
void noisePollutionMonitoring(vector<int>& noise_data, int rank, int size);
void greenWaveSystem(vector<vector<int>>& traffic_lights, int rank, int size);
void evChargingIntegration(vector<int>& charging_stations, vector<int>& ev_prioritization, int rank, int size);
void publicTransportIntegration(vector<int>& public_transport_data, int rank, int size);
void trafficSimulation(vector<int>& traffic_flow, vector<int>& incidents, int rank, int size);

int main(int argc, char* argv[]) {
    int rank, size;
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> vehicle_data(NUM_VEHICLES, 0);
    vector<int> incidents(NUM_SENSORS, 0);
    vector<int> traffic_density(NUM_CAMERAS, 0);
    vector<int> air_quality_data(NUM_SENSORS, 50);
    vector<int> noise_data(NUM_SENSORS, 30);
    vector<int> historical_data(365, 0);
    vector<int> future_traffic(7, 0);
    vector<int> ev_prioritization(NUM_VEHICLES, 0);
    vector<int> public_transport_data(100, 0);
    vector<int> charging_stations(NUM_EV_STATIONS, 1);
    vector<vector<int>> traffic_lights(NUM_INTERSECTIONS, vector<int>(4, 0));

    auto start = chrono::high_resolution_clock::now();

    // Call functions sequentially instead of using OpenMP
    trafficFlowMonitoring(vehicle_data, rank, size);
    incidentDetection(incidents, rank, size);
    congestionMonitoring(traffic_density, rank, size);
    vehicleCounting(vehicle_data, NUM_SENSORS, rank, size);
    adaptiveSignalControl(traffic_lights, traffic_density, rank, size);
    predictiveAnalytics(historical_data, future_traffic, rank, size);
    airQualityMonitoring(air_quality_data, rank, size);
    noisePollutionMonitoring(noise_data, rank, size);
    greenWaveSystem(traffic_lights, rank, size);
    evChargingIntegration(charging_stations, ev_prioritization, rank, size);
    publicTransportIntegration(public_transport_data, rank, size);
    trafficSimulation(vehicle_data, incidents, rank, size);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    if (rank == 0) {
        cout << "Execution Time: " << elapsed.count() << " seconds" << endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

// Function implementations with MPI communication and print statements

void trafficFlowMonitoring(vector<int>& vehicle_data, int rank, int size) {
    int chunk_size = vehicle_data.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? vehicle_data.size() : start + chunk_size;

    srand(rank); // Seed random number generator for consistency
    for (int i = start; i < end; ++i) {
        vehicle_data[i] = rand() % 100;
    }

    // Gather data at rank 0 and print it
    if (rank == 0) {
        vector<int> global_data(vehicle_data.size());
        MPI_Gather(vehicle_data.data() + start, chunk_size, MPI_INT, global_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Traffic Flow Monitoring Data: " << endl;
        for (int i = 0; i < NUM_VEHICLES; ++i) {
            cout << "Vehicle " << i << ": " << global_data[i] << " vehicles detected." << endl;
        }
    } else {
        MPI_Gather(vehicle_data.data() + start, chunk_size, MPI_INT, nullptr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void incidentDetection(vector<int>& incidents, int rank, int size) {
    int chunk_size = incidents.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? incidents.size() : start + chunk_size;

    srand(rank); // Seed random number generator
    for (int i = start; i < end; ++i) {
        incidents[i] = rand() % 2; // Randomly detect incident (0 or 1)
    }

    // Reduce to get the sum of incidents across all processes
    int local_sum = accumulate(incidents.begin() + start, incidents.begin() + end, 0);
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Total incidents detected: " << global_sum << endl;
    }
}

void congestionMonitoring(vector<int>& traffic_density, int rank, int size) {
    int chunk_size = traffic_density.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? traffic_density.size() : start + chunk_size;

    srand(rank); // Seed random number generator
    for (int i = start; i < end; ++i) {
        traffic_density[i] = rand() % 100;
    }

    // Gather data at rank 0 and print it
    if (rank == 0) {
        vector<int> global_data(traffic_density.size());
        MPI_Gather(traffic_density.data() + start, chunk_size, MPI_INT, global_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Traffic Congestion Data: " << endl;
        for (int i = 0; i < NUM_CAMERAS; ++i) {
            cout << "Camera " << i << ": " << global_data[i] << " traffic density." << endl;
        }
    } else {
        MPI_Gather(traffic_density.data() + start, chunk_size, MPI_INT, nullptr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void vehicleCounting(vector<int>& vehicle_data, int num_sections, int rank, int size) {
    int chunk_size = vehicle_data.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? vehicle_data.size() : start + chunk_size;

    srand(rank); // Seed random number generator
    for (int i = start; i < end; ++i) {
        vehicle_data[i] = rand() % 100;
    }

    // Gather data at rank 0 and print it
    if (rank == 0) {
        vector<int> global_data(vehicle_data.size());
        MPI_Gather(vehicle_data.data() + start, chunk_size, MPI_INT, global_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Vehicle Counting Data: " << endl;
        for (int i = 0; i < NUM_SENSORS; ++i) {
            cout << "Section " << i << ": " << global_data[i] << " vehicles." << endl;
        }
    } else {
        MPI_Gather(vehicle_data.data() + start, chunk_size, MPI_INT, nullptr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void adaptiveSignalControl(vector<vector<int>>& traffic_lights, vector<int>& traffic_flow, int rank, int size) {
    int chunk_size = traffic_lights.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? traffic_lights.size() : start + chunk_size;

    for (int i = start; i < end; ++i) {
        for (int j = 0; j < 4; ++j) {
            traffic_lights[i][j] = traffic_flow[i % traffic_flow.size()];
        }
    }

    // Gather data at rank 0 and print it
    if (rank == 0) {
        vector<vector<int>> global_data(traffic_lights.size(), vector<int>(4, 0));
        MPI_Gather(traffic_lights.data() + start, chunk_size * 4, MPI_INT, global_data.data(), chunk_size * 4, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Adaptive Signal Control Data: " << endl;
        for (int i = 0; i < NUM_INTERSECTIONS; ++i) {
            cout << "Intersection " << i << ": ";
            for (int j = 0; j < 4; ++j) {
                cout << global_data[i][j] << " ";
            }
            cout << endl;
        }
    } else {
        MPI_Gather(traffic_lights.data() + start, chunk_size * 4, MPI_INT, nullptr, chunk_size * 4, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void predictiveAnalytics(vector<int>& historical_data, vector<int>& future_traffic, int rank, int size) {
    int chunk_size = historical_data.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? historical_data.size() : start + chunk_size;

    srand(rank);
    for (int i = start; i < end; ++i) {
        historical_data[i] = rand() % 100;
    }

    // Gather data at rank 0 and print it
    if (rank == 0) {
        vector<int> global_data(historical_data.size());
        MPI_Gather(historical_data.data() + start, chunk_size, MPI_INT, global_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Historical Data: " << endl;
        for (int i = 0; i < 365; ++i) {
            cout << "Day " << i << ": " << global_data[i] << " vehicles." << endl;
        }
    } else {
        MPI_Gather(historical_data.data() + start, chunk_size, MPI_INT, nullptr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void airQualityMonitoring(vector<int>& air_quality_data, int rank, int size) {
    int chunk_size = air_quality_data.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? air_quality_data.size() : start + chunk_size;

    srand(rank); 
    for (int i = start; i < end; ++i) {
        air_quality_data[i] = rand() % 200; // Random air quality index
    }

    if (rank == 0) {
        vector<int> global_data(air_quality_data.size());
        MPI_Gather(air_quality_data.data() + start, chunk_size, MPI_INT, global_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Air Quality Monitoring Data: " << endl;
        for (int i = 0; i < NUM_SENSORS; ++i) {
            cout << "Sensor " << i << ": " << global_data[i] << " AQI." << endl;
        }
    } else {
        MPI_Gather(air_quality_data.data() + start, chunk_size, MPI_INT, nullptr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void noisePollutionMonitoring(vector<int>& noise_data, int rank, int size) {
    int chunk_size = noise_data.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? noise_data.size() : start + chunk_size;

    srand(rank); 
    for (int i = start; i < end; ++i) {
        noise_data[i] = rand() % 100; // Random noise level
    }

    if (rank == 0) {
        vector<int> global_data(noise_data.size());
        MPI_Gather(noise_data.data() + start, chunk_size, MPI_INT, global_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Noise Pollution Monitoring Data: " << endl;
        for (int i = 0; i < NUM_SENSORS; ++i) {
            cout << "Sensor " << i << ": " << global_data[i] << " dB." << endl;
        }
    } else {
        MPI_Gather(noise_data.data() + start, chunk_size, MPI_INT, nullptr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void greenWaveSystem(vector<vector<int>>& traffic_lights, int rank, int size) {
    int chunk_size = traffic_lights.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? traffic_lights.size() : start + chunk_size;

    for (int i = start; i < end; ++i) {
        for (int j = 0; j < 4; ++j) {
            traffic_lights[i][j] = (rand() % 2); // Random green wave activation
        }
    }

    if (rank == 0) {
        vector<vector<int>> global_data(traffic_lights.size(), vector<int>(4, 0));
        MPI_Gather(traffic_lights.data() + start, chunk_size * 4, MPI_INT, global_data.data(), chunk_size * 4, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Green Wave System Data: " << endl;
        for (int i = 0; i < NUM_INTERSECTIONS; ++i) {
            cout << "Intersection " << i << ": ";
            for (int j = 0; j < 4; ++j) {
                cout << global_data[i][j] << " ";
            }
            cout << endl;
        }
    } else {
        MPI_Gather(traffic_lights.data() + start, chunk_size * 4, MPI_INT, nullptr, chunk_size * 4, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void evChargingIntegration(vector<int>& charging_stations, vector<int>& ev_prioritization, int rank, int size) {
    int chunk_size = charging_stations.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? charging_stations.size() : start + chunk_size;

    srand(rank); 
    for (int i = start; i < end; ++i) {
        charging_stations[i] = rand() % 2; // Random charging station status
    }

    if (rank == 0) {
        vector<int> global_data(charging_stations.size());
        MPI_Gather(charging_stations.data() + start, chunk_size, MPI_INT, global_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "EV Charging Integration Data: " << endl;
        for (int i = 0; i < NUM_EV_STATIONS; ++i) {
            cout << "Charging Station " << i << ": " << (global_data[i] == 0 ? "Available" : "Occupied") << endl;
        }
    } else {
        MPI_Gather(charging_stations.data() + start, chunk_size, MPI_INT, nullptr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void publicTransportIntegration(vector<int>& public_transport_data, int rank, int size) {
    int chunk_size = public_transport_data.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? public_transport_data.size() : start + chunk_size;

    srand(rank); 
    for (int i = start; i < end; ++i) {
        public_transport_data[i] = rand() % 100; // Random number of passengers
    }

    if (rank == 0) {
        vector<int> global_data(public_transport_data.size());
        MPI_Gather(public_transport_data.data() + start, chunk_size, MPI_INT, global_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Public Transport Integration Data: " << endl;
        for (int i = 0; i < public_transport_data.size(); ++i) {
            cout << "Stop " << i << ": " << global_data[i] << " passengers." << endl;
        }
    } else {
        MPI_Gather(public_transport_data.data() + start, chunk_size, MPI_INT, nullptr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void trafficSimulation(vector<int>& traffic_flow, vector<int>& incidents, int rank, int size) {
    int chunk_size = traffic_flow.size() / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? traffic_flow.size() : start + chunk_size;

    srand(rank); 
    for (int i = start; i < end; ++i) {
        traffic_flow[i] = rand() % 100;
        incidents[i] = rand() % 2;
    }

    // Gather data at rank 0 and print it
    if (rank == 0) {
        vector<int> global_traffic(traffic_flow.size());
        vector<int> global_incidents(incidents.size());
        MPI_Gather(traffic_flow.data() + start, chunk_size, MPI_INT, global_traffic.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(incidents.data() + start, chunk_size, MPI_INT, global_incidents.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Traffic Simulation Data: " << endl;
        for (int i = 0; i < traffic_flow.size(); ++i) {
            cout << "Location " << i << ": Traffic Flow = " << global_traffic[i] << ", Incidents = " << global_incidents[i] << endl;
        }
    } else {
        MPI_Gather(traffic_flow.data() + start, chunk_size, MPI_INT, nullptr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(incidents.data() + start, chunk_size, MPI_INT, nullptr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

