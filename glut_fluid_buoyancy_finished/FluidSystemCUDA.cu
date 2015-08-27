#include <GL/freeglut.h>

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "Kernel_implement.cuh"
extern "C"
{
	void cudaGLInit(int argc, char **argv) { findCudaGLDevice(argc, (const char **)argv); }

	void allocateArray(void **devPtr, size_t size) { checkCudaErrors(cudaMalloc(devPtr, size)); }

	void freeArray(void *devPtr) { checkCudaErrors(cudaFree(devPtr)); }

	void threadSync() { checkCudaErrors(cudaDeviceSynchronize()); }

	void copyArrayToDevice(void *device, const void *host, int offset, int size) { checkCudaErrors(cudaMemcpy((char *)device + offset, host, size, cudaMemcpyHostToDevice)); }

	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource) { checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone)); }

	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource) { checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource)); }

	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource) {
		void *ptr; 
		checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
		size_t num_bytes; 
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, *cuda_vbo_resource));
		return ptr;
	}

	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource) { checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0)); }

	void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size)
	{
		if (cuda_vbo_resource)
		{
			device = mapGLBufferObject(cuda_vbo_resource);
		}

		checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

		if (cuda_vbo_resource)
		{
			unmapGLBufferObject(*cuda_vbo_resource);
		}
	}

	void setParameters(KernelParams *hostParams) { checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(KernelParams))); }

	uint iDivUp(uint a, uint b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}

	void integrateSystem(float *pos, float *vel, float *den, float *pre, float deltaTime, uint numParticles, float* buoyancyForce)
	{
		thrust::device_ptr<float4> d_pos4((float4 *)pos);
		thrust::device_ptr<float4> d_vel4((float4 *)vel);
		//thrust::device_ptr<float3> d_bForce3((float3 *)buoyancyForce);
		//thrust::device_ptr<float> d_den1((float *)den);
		//thrust::device_ptr<float> d_pre1((float *)pre);

		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4 + numParticles, d_vel4 + numParticles)),
			integrate_functor(deltaTime));
	}

	void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos, int numParticles)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 1024, numBlocks, numThreads);

		calcHashD <<< numBlocks, numThreads >>>(gridParticleHash,
			gridParticleIndex,
			(float4 *)pos,
			numParticles);

		getLastCudaError("Kernel execution failed");
	}

	void reorderDataAndFindCellStart(uint *cellStart, uint *cellEnd, float *sortedPos, float *sortedVel, uint *gridParticleHash,
		uint *gridParticleIndex, float *oldPos, float *oldVel, uint numParticles, uint numCells, float *sortedDen, float *sortedPre, float *oldDen, float *oldPre,float* sortedColorf,float* oldColorf)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 1024, numBlocks, numThreads);
		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

		uint smemSize = sizeof(uint)*(numThreads + 1);
		reorderDataAndFindCellStartD << < numBlocks, numThreads, smemSize >> >(
			cellStart,
			cellEnd,
			(float4 *)sortedPos,
			(float4 *)sortedVel,
			gridParticleHash,
			gridParticleIndex,
			(float4 *)oldPos,
			(float4 *)oldVel,
			numParticles,
			(float *)sortedDen,
			(float *)sortedPre,
			(float *)oldDen,
			(float *)oldPre,
			(float *)sortedColorf,
			(float *)oldColorf);
		getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
	}

	void calculateDensity(float *sortedPos,uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, float *newDen)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 1024, numBlocks, numThreads);

		calculateDensityD << < numBlocks, numThreads >> >(
			(float4 *)sortedPos,
			gridParticleIndex,
			cellStart,
			cellEnd,
			numParticles,
			(float *)newDen
			);

		getLastCudaError("Kernel execution failed");
	}

	void collide(float *newVel, float *sortedPos, float *sortedVel, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, uint numCells,
		float *newDen, float *newPre, float *sortedDen, float *sortedPre, float* oldPos, float* buoyancyForce)
	{
		uint numThreads, numBlocks;

		computeGridSize(numParticles, 1024, numBlocks, numThreads);

		collideD << < numBlocks, numThreads >> >((float4 *)newVel,
			(float4 *)sortedPos,
			(float4 *)sortedVel,
			gridParticleIndex,
			cellStart,
			cellEnd,
			numParticles,
			(float *)newDen,
			(float *)newPre,
			(float *)sortedDen,
			(float *)sortedPre,
			(float4 *)oldPos,
			(float3 *)buoyancyForce);

		getLastCudaError("Kernel execution failed");
    }

	void plusBuoyancyforce(float* buoyancyForce, uint numParticles){
		uint numThreads, numBlocks;

		computeGridSize(numParticles, 1024, numBlocks, numThreads);

		plusBuoyancyforceD << <numBlocks, numThreads >> >(
			(float3*)buoyancyForce,
			numParticles
			);

		getLastCudaError("Kernel execution failed");
	}


    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }

	void colorfield(float *sortedPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, uint numCells, float *colorfield, float *sortedcolorfield)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 1024, numBlocks, numThreads);

		colorfieldD << <numBlocks, numThreads >> >((float4 *)sortedPos,
			gridParticleIndex,
			cellStart,
			cellEnd,
			numParticles,
			(float *)colorfield,
			(float *)sortedcolorfield
			);
		getLastCudaError("Kernel execution failed");
	}

	void calcDepth(float  *pos,
		float   *keys,        // output
		uint    *indices,     // output
		float3   sortVector,
		int      numParticles)
	{
		thrust::device_ptr<float4> d_pos((float4 *)pos);
		thrust::device_ptr<float> d_keys(keys);
		thrust::device_ptr<uint> d_indices(indices);

		/*thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_keys)),
			thrust::make_zip_iterator(thrust::make_tuple(d_pos + numParticles, d_keys + numParticles)),
			calcDepth_functor(sortVector));*/

		thrust::sequence(d_indices, d_indices + numParticles);
	}

	void sortParticlesKey(float *sortKeys, uint *indices, uint numParticles)
	{
		thrust::sort_by_key(thrust::device_ptr<float>(sortKeys),
			thrust::device_ptr<float>(sortKeys + numParticles),
			thrust::device_ptr<uint>(indices));
	}

}  
