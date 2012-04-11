/*
 * Null model based on cell link strength
 * ranged between 0 and 1
 * 
 * CUDA version
 * 
 * compile with nvcc nco.cu -o nco -lcuda
 */

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

__global__ void nullmodel(float *M, int *out, curandState *states)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(threadIdx.x, 0, 0, &states[tid]);
	float tar = (float)curand_uniform(&states[tid]);
	out[tid] = (tar < M[tid])? 1 : 0;
}

int main(int argc, char **argv)
{
	srand ( time(NULL) );
	clock_t start, stop;

	// PRNG
	curandState *devStates;

	// Network shape
	int nrow = 100;
	int ncol = 100;
	double connec = 0.6;
	int net_size = nrow * ncol; 

	// Memory allocation on host
	const size_t nbytes_ref= net_size * sizeof(float);
	const size_t nbytes_out= net_size * sizeof(int);
	float *h_ref = (float*)malloc(nbytes_ref);
	int *h_out = (int*)malloc(nbytes_out);

	// create a random betwork -------
	for(int row = 0; row < nrow; ++row)
	{
		for(int col = 0; col < ncol; ++col)
		{
			if((double)(rand() / (float)RAND_MAX) < connec)
			{
				h_ref[col + row*nrow] = (float)(rand() / (float)RAND_MAX);
			}
		}
	}
	// rand() / (float)RAND_MAX;	

	// Memory allocation on device
	float *d_ref;
	int *d_out;
	cudaMalloc((void**)&d_ref, nbytes_ref);
	cudaMalloc((void**)&d_out, nbytes_out);

	// Memory transfer from host to device
	cudaMemcpy(d_ref, h_ref, nbytes_ref, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, h_out, nbytes_out, cudaMemcpyHostToDevice);
	cudaMalloc( (void **)&devStates, net_size * sizeof(curandState) );

	// Record initial time and start doing the null model
	start = clock();

	for(int repl = 0; repl < 1000; ++repl)
	{
		nullmodel<<<ncol,nrow>>>(d_ref, d_out, devStates);
		// Memory transfer from the device to the host
		cudaMemcpy(h_out, d_out, nbytes_ref, cudaMemcpyDeviceToHost);
	}
	
	stop = clock();
	printf("1000 null network generated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	// Free memory space
	free(h_ref);
	free(h_out);
	cudaFree(d_ref);
	cudaFree(d_out);

	return EXIT_SUCCESS;
}