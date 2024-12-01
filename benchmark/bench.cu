#include <cstdio>
#include <vector>
#include <random>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pyro.hxx"

using thermo = pyro::pyro<double>;

struct Cell {
    double T;
    double rho;
    thermo::SpeciesT Y;
};


__global__ void bench(Cell* cells, int num_cells) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = x + y * gridDim.x + z * gridDim.x * gridDim.y;
    if (idx >= num_cells) return;

    auto& cell = cells[idx];
    cell.Y = thermo::get_net_production_rates(cell.rho, cell.T, cell.Y);
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <sidelength>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int sidelength = std::atoi(argv[1]);
    const int num_cells  = sidelength * sidelength * sidelength;

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::printf("Configuration:\n");
    std::printf("> Solution: %s (%d species and %d reactions)\n", thermo::name, thermo::num_species, thermo::num_reactions);
    std::printf("> Grid:     %d x %d x %d\n", sidelength, sidelength, sidelength);
    std::printf("> Device:   %s (%d SMs, %d GB)\n", prop.name, prop.multiProcessorCount, (int)(prop.totalGlobalMem / (1024 * 1024 * 1024)));

    std::printf("Setup:\n");

    std::printf("> Host Buffer Allocation\n");
    std::vector<Cell> cells(num_cells);

    auto rng = std::mt19937(42);
    auto T_dist = std::uniform_real_distribution<double>(300, 3000);
    auto rho_dist = std::uniform_real_distribution<double>(0.1, 10);
    auto Y_dist = std::uniform_real_distribution<double>(0, 1);

    std::printf("> Generating random state vectors\n");
    for (int i = 0; i < num_cells; ++i) {
        cells[i].T = T_dist(rng);
        cells[i].rho = rho_dist(rng);
        for (int j = 0; j < thermo::num_species; ++j) {
            cells[i].Y[j] = Y_dist(rng);
        }
    }

    std::printf("> Allocating GPU memory\n");
    Cell* gpu_cells; cudaMalloc(&gpu_cells, num_cells * sizeof(Cell));
    
    std::printf("> Transfering CPU Buffers to GPU\n");
    cudaMemcpy(gpu_cells, cells.data(), num_cells * sizeof(Cell), cudaMemcpyHostToDevice);

    dim3 block(32, 32, 1);
    dim3 grid((sidelength + block.x - 1) / block.x, (sidelength + block.y - 1) / block.y, (sidelength + block.z - 1) / block.z);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double fom_avg = 0.0; // ns per cell
    double ms_avg = 0.0;

    constexpr int num_runs = 10;

    std::printf("> Benchmarking %02d times\n", num_runs);
    for (int i = 0; i < num_runs; ++i) {
        std::printf("\r  Run %02d / %02d", i + 1, num_runs);
        std::fflush(stdout);
        cudaEventRecord(start);
        bench<<<grid, block>>>(gpu_cells, num_cells);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);

        fom_avg += ms * 1e6 / num_cells;
        ms_avg += ms;
    }
    std::printf("\n");

    fom_avg /= num_runs;
    ms_avg /= num_runs;

    std::printf("> Time:     %.3f ms\n", ms_avg);
    std::printf("> FOM:      %.3f ns/cell\n", fom_avg);

    cudaMemcpy(cells.data(), gpu_cells, num_cells * sizeof(Cell), cudaMemcpyDeviceToHost);

    cudaFree(gpu_cells);

    return EXIT_SUCCESS;
}
