// nvcc gdist.cu -o gdist -lcuda -use_fast_math compiler

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
  
#define NS 50
#define NBASES 100
#define seqpos(s_,b_)  ((b_) + (s_) * NBASES)

__global__ void pwdist(char *se, float *dist)
{
	int tid = blockIdx.x;
	int M = blockDim.x - 1;
	int ii = (M * (M - 1)) / 2 - tid - 1;
	int K = (sqrtf(8 * ii + 1) - 1) / 2;
	
	int FSidx = M + K;
	int SCidx = tid - M*(M+1)/2 + (K+1)*(K+2)/2; 
	
	int Dissim = 0;
	for(int base = 0; base < NBASES; ++base)
	{
		if(!(se[seqpos(FSidx,base)] == se[seqpos(SCidx,base)]))
		{
			++Dissim;
		}
	}
	
	dist[blockIdx.x] = Dissim;
	
    //int tid = threadIdx.x + blockDim.x * blockIdx.x; // blockDim.x := 1000
    
}

int main (int argc, char *argv[])
{	
	// generate a bunch of test sequences
	int ds = NS * NBASES;	
	const unsigned int bytes = ds * sizeof(char);
	char *se = (char*)malloc(bytes);
	
	// create cyclic sequences
	for(int i = 0; i < ds; ++i)
	{
		se[i] = 'a';
		//if (i % 3 == 4){se[i] = 'c';}
		//if (i % 7 == 4){se[i] = 't';}
		//if (i % 19 == 8){se[i] = '-';}
		//if (i % 13 == 1){se[i] = 'g';}
	}
	
	// print sequences to screen just to check
	for(int i = 0; i < NBASES; i++)
	{
		for(int j = 0; j < NS; j++)
		{
			printf("%c",se[seqpos(j,i)]);
		}
		printf("\n");
	}
	
	// number of distances to compute
	const unsigned int ndist = (int)(NS*(NS-1)/2);
	// memory allocation for the distances
	float *h_dist = (float*)malloc(ndist*sizeof(float));
	
	// allocate memory on the devices
	char *d_se;
	float *d_dist;
	cudaMalloc((void**)&d_se, bytes);
	cudaMalloc((void**)&d_dist, ndist * sizeof(float));
	
	/*
	 * HERE BE THE MAIN PROGRAM
	 */
	
	// copy the sequences from the host to the device
	cudaMemcpy(d_se, se, bytes, cudaMemcpyHostToDevice);
	
	// call CUDA function pwdist
	pwdist<<<ndist,1>>>(d_se, d_dist);
	
	// copy the distances from the device to the host
	cudaMemcpy(h_dist, d_dist, ndist * sizeof(float), cudaMemcpyDeviceToHost);
	
	/*
	 * END OF THE MAIN PROGRAM
	 */
	
	// Output of the results in a text form
	for(int i = 0; i < ndist; ++i)
	{
		printf("%f \t",h_dist[i]);
	}
	
	// free the vectors
	cudaFree(d_se);
	cudaFree(d_dist);
	free(se);
	free(h_dist);
	
	return EXIT_SUCCESS;
}