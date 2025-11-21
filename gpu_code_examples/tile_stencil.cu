#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCKSIZE 8

__global__ void stencil4pt(double* u, double* u_new, int nx, int ny){
    // Global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local thread indices within the block
    int ti = threadIdx.x;
    int tj = threadIdx.y;
    
    // Shared memory with padding for halos
    __shared__ double block_data[BLOCKSIZE + 2][BLOCKSIZE + 2];
    
    // Load main tile (each thread loads its element)
    if (i < nx && j < ny) {
        block_data[ti + 1][tj + 1] = u[i * ny + j];
    }
    
    // Load halo cells (only edge threads do this)
    if (ti == 0 && i > 0) { // Top edge
        block_data[0][tj + 1] = u[(i-1) * ny + j];
    }
    if (ti == BLOCKSIZE - 1 && i < nx - 1) { // Bottom edge
        block_data[BLOCKSIZE + 1][tj + 1] = u[(i+1) * ny + j];
    }
    if (tj == 0 && j > 0) { // Left edge
        block_data[ti + 1][0] = u[i * ny + (j-1)];
    }
    if (tj == BLOCKSIZE - 1 && j < ny - 1) { // Right edge
        block_data[ti + 1][BLOCKSIZE + 1] = u[i * ny + (j+1)];
    }
    
    __syncthreads();
    
    // Compute stencil
    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1) {
        u_new[i * ny + j] = block_data[ti + 1][tj + 1] + 
                            block_data[ti][tj + 1] + 
                            block_data[ti + 2][tj + 1] + 
                            block_data[ti + 1][tj] + 
                            block_data[ti + 1][tj + 2];
    }
}

int main(int argc, char** argv){
    int nx = 6400;
    int ny = 10000;
    
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, 
                 (ny + blockDim.y - 1) / blockDim.y);
    
    double *h_u, *h_unew, *d_u, *d_unew;
    int size = nx * ny * sizeof(double);
    
    h_u = (double*) malloc(size);
    h_unew = (double*) malloc(size);
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_unew, size);
    
    // Initialize
    for(int i = 0; i < nx; i++){
        for(int j = 0; j < ny; j++){
            if(i == 0 || j == 0 || i == nx-1 || j == ny-1){
                h_u[i * ny + j] = 1.0;
            } else {
                h_u[i * ny + j] = 0.0;
            }
        }
    }
    
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
        
    stencil4pt<<<gridDim, blockDim>>>(d_u, d_unew, nx, ny);

    // Add error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    cudaFree(d_u);
    cudaFree(d_unew);
    free(h_u);
    free(h_unew);
    
    return 0;
}