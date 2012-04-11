// nvcc gdist.cu -o gdist -lcuda -use_fast_math compiler

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
  
#define NS              3 // Number of sequences
#define NBASES          100 // Nucleotides / sequence
#define seq(s,i,n)      (s[((n)+(i)*(NBASES+1))]) // nth position of the ith sequence in string 's'.
#define getx0(i)        (((unsigned int)(-0.5+0.5*sqrtf(1+8*(i-1))))+2)
#define gety0(i)        ((unsigned int)((getx0(i))*(3-(getx0(i)))/2+i-1))
#define getx(i)         (getx0(i+1)-1)
#define gety(i)         (gety0(i+1)-1)

__global__ void pwdist(char *se, float *dist)
{
    int tid = blockIdx.x;
    const unsigned int x = getx(tid);
    const unsigned int y = gety(tid);
    unsigned int diff = 0;
    for (int base = 0; base < NBASES; ++base)
    {
        if (seq(se,x,base) != seq(se,y,base))
        {
            ++diff;
        }
    }
    dist[tid] = (float)diff / NBASES;
}

int main (int argc, char *argv[])
{
    // generate a bunch of test sequences
    const unsigned int ds = NS * (NBASES + 1); // Length of the array.
    const unsigned int bytes = ds * sizeof(char);
    char *se = (char*)malloc(bytes);

    // create sequences
    for (int i = 0; i < NS; ++i)
    {
        for (int j = 0; j < NBASES; ++j)
        {
            seq(se,i,j) = 'a';
        }
        seq(se,i,NBASES) = '\0';
    }

    // Mutations!!!!!!!
    seq(se,0,5) = 'c';
    seq(se,2,5) = 'c';
    seq(se,2,3) = 'g';

    // print sequences to screen just to check
    for (int i = 0; i < NS; ++i)
    {
        printf("%s\n", &seq(se,i,0));
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
    printf("Total of %u pairwise distances\n",ndist);

    // copy the distances from the device to the host
    cudaMemcpy(h_dist, d_dist, ndist * sizeof(float), cudaMemcpyDeviceToHost);

    /*
    * END OF THE MAIN PROGRAM
    */

    // Output of the results in a text form
    for (unsigned int i = 0; i < ndist; ++i)
    {
        printf("(%u,%u) \t %.4f\n", getx(i), gety(i), h_dist[i]);
    }

    // free the vectors
    cudaFree(d_se);
    cudaFree(d_dist);
    free(se);
    free(h_dist);

    return EXIT_SUCCESS;
}
