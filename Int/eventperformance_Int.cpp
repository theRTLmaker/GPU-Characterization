#include "hip/hip_runtime.h"
/*
 * Copyright 2011-2015 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain metric values
 * using callbacks for CUDA runtime APIs
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../lcutil.h"


#define COMP_ITERATIONS (4096) //512
#define THREADS (1024)
#define BLOCKS (32760)
#define REGBLOCK_SIZE (4)
#define UNROLL_ITERATIONS (32)
#define deviceNum (0)


//CODE
__global__ void warmup(int aux){

	__shared__ float shared[THREADS];

	short r0 = 1.0,
	  r1 = r0+(short)(31),
	  r2 = r0+(short)(37),
	  r3 = r0+(short)(41);

	for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			// Each iteration maps to floating point 8 operations (4 multiplies + 4 additions)
			r0 = r0 * r0 + r1;//r0;
			r1 = r1 * r1 + r2;//r1;
			r2 = r2 * r2 + r3;//r2;
			r3 = r3 * r3 + r0;//r3;
		}
	}
	shared[threadIdx.x] = r0;
}

template <class T> __global__ void benchmark(int initValue, T* cdout){

	register T r0 = initValue,
	           r1 = r0,
	           r2 = r0,
	           r3 = r0;

	for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			// Each iteration maps to floating point 8 operations (4 multiplies + 4 additions)
			r0 = r0 * r0 + r1;//r0;
			r1 = r1 * r1 + r2;//r1;
			r2 = r2 * r2 + r3;//r2;
			r3 = r3 * r3 + r0;//r3;
		}
	}

	cdout[blockIdx.x * THREADS + threadIdx.x] = r0;
}

void initializeEvents(hipEvent_t *start, hipEvent_t *stop){
	HIP_SAFE_CALL( hipEventCreate(start) );
	HIP_SAFE_CALL( hipEventCreate(stop) );
	HIP_SAFE_CALL( hipEventRecord(*start, 0) );
}

float finalizeEvents(hipEvent_t start, hipEvent_t stop){
	HIP_SAFE_CALL( hipGetLastError() );
	HIP_SAFE_CALL( hipEventRecord(stop, 0) );
	HIP_SAFE_CALL( hipEventSynchronize(stop) );
	float kernel_time;
	HIP_SAFE_CALL( hipEventElapsedTime(&kernel_time, start, stop) );
	HIP_SAFE_CALL( hipEventDestroy(start) );
	HIP_SAFE_CALL( hipEventDestroy(stop) );
	return kernel_time;
}

void runbench_warmup(){
	const int BLOCK_SIZE = 256;
	const int TOTAL_REDUCED_BLOCKS = 256;
    int aux=0;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	hipLaunchKernelGGL((warmup), dim3(dimReducedGrid), dim3(dimBlock ), 0, 0, aux);
	HIP_SAFE_CALL( hipGetLastError() );
	HIP_SAFE_CALL( hipDeviceSynchronize() );
}

void runbench(double* kernel_time, double* flops, int * hostOut){

	const long long computations = 2*(long long)(COMP_ITERATIONS)*REGBLOCK_SIZE*THREADS*BLOCKS;
	
    dim3 dimBlock(THREADS, 1, 1);
    dim3 dimGrid(BLOCKS, 1, 1);
	hipEvent_t start, stop;

	// Generates random init Value
	int initValue = ((unsigned)rand() << 17) | ((unsigned)rand() << 2) | ((unsigned)rand() & 3);

	initializeEvents(&start, &stop);

    hipLaunchKernelGGL((benchmark<int>), dim3(dimGrid), dim3(dimBlock ), 0, 0, initValue, (int *) hostOut);

	hipDeviceSynchronize();

	double time = finalizeEvents(start, stop);
	double result = ((double)computations)/(double)time*1000./(double)(1000*1000*1000);

	*kernel_time = time;
	*flops=result;

}

int main(int argc, char *argv[]){

    srand((unsigned) time(NULL));
	int device = 0;
	// int deviceCount;
	// char deviceName[32];

	hipDeviceProp_t deviceProp;

	printf("Usage: %s [ntries]\n", argv[0]);
	int ntries;
	if (argc>1){
		ntries = atoi(argv[1]);
	}else{
		ntries = 1;
	}

	hipSetDevice(deviceNum);

	double time[ntries][2],value[ntries][4];

	HIP_SAFE_CALL(hipGetDeviceProperties(&deviceProp, device));
	
	int i;

	for (i=0;i<1;i++){
		runbench_warmup();
	}

	int * hostOut = (int *) calloc(BLOCKS * THREADS, sizeof(int));
	int * deviceOut;
	HIP_SAFE_CALL(hipMalloc((void**)&deviceOut, BLOCKS * THREADS  * sizeof(int)));

	int benchmarkValue = 0;
	int error = 0;

	for (i=0;i<ntries;i++){
		runbench(&time[0][0],&value[0][0], deviceOut);
		printf("Registered time: %f ms\n",time[0][0]);
		
		// Verify
		// Transfer data from device to host
		HIP_SAFE_CALL(hipMemcpy(hostOut, deviceOut, BLOCKS * THREADS * sizeof(int), hipMemcpyDeviceToHost));

		benchmarkValue = hostOut[0];
		for (int i = 1; i < BLOCKS * THREADS; ++i)
			if(benchmarkValue != hostOut[i])
				error = 1;
		if(error == 0)
			printf("Result: True\n");
		else
			printf("Result: False\n");
	}
	HIP_SAFE_CALL( hipDeviceReset());
	return 0;
}
