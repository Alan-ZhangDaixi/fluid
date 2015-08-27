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
#include "tables.h"
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

	void integrateSystem(float *pos, float *vel, float *den, float *pre,float deltaTime, uint numParticles)
	{
		thrust::device_ptr<float4> d_pos4((float4 *)pos);
		thrust::device_ptr<float4> d_vel4((float4 *)vel);
		thrust::device_ptr<float> d_den1((float *)den);
		thrust::device_ptr<float> d_pre1((float *)pre);

		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4,d_den1,d_pre1)),
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4 + numParticles, d_vel4 + numParticles, d_den1 + numParticles, d_pre1 + numParticles)),
			integrate_functor(deltaTime));
	}

	void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos, int numParticles, float* oldDen, float* oldColorf, float* d_surfaceNormal, float* metaballtable, float* normaltable,int pow)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 1024, numBlocks, numThreads);

		calcHashD <<< numBlocks, numThreads >>>(gridParticleHash,
			gridParticleIndex,
			(float4 *)pos,
			numParticles,
			(float *)oldDen,
			(float *)oldColorf,
			(float3 *)d_surfaceNormal,
			(float*) metaballtable,
			(float3*) normaltable,
			pow);

		getLastCudaError("Kernel execution failed");
	}


	void calcMarchingcubeandmetaball(uint *gridParticleHash, uint *gridParticleIndex, float *pos, int numParticles, 
		float* oldDen, float* oldColorf, float* d_surfaceNormal, float* metaballtable, float* normaltable, int pow, int fornum)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 1024, numBlocks, numThreads);

		calcMarchingcubeandmetaballD << < numBlocks, numThreads >> >(gridParticleHash,
			gridParticleIndex,
			(float4 *)pos,
			numParticles,
			(float *)oldDen,
			(float *)oldColorf,
			(float3 *)d_surfaceNormal,
			(float*)metaballtable,
			(float3*)normaltable,
			pow,
			fornum);

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
		float *newDen, float *newPre, float *sortedDen, float *sortedPre,float* oldPos)
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
			(float4 *)oldPos);

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

	////////////////////////////////////////////////////////////////////Marching cube
	void allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable)
	{
		checkCudaErrors(cudaMalloc((void **)d_edgeTable, 256 * sizeof(uint)));
		checkCudaErrors(cudaMemcpy((void *)*d_edgeTable, (void *)edgeTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
		checkCudaErrors(cudaBindTexture(0, edgeTex, *d_edgeTable, channelDesc));

		checkCudaErrors(cudaMalloc((void **)d_triTable, 256 * 16 * sizeof(uint)));
		checkCudaErrors(cudaMemcpy((void *)*d_triTable, (void *)triTable, 256 * 16 * sizeof(uint), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, triTex, *d_triTable, channelDesc));

		checkCudaErrors(cudaMalloc((void **)d_numVertsTable, 256 * sizeof(uint)));
		checkCudaErrors(cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, numVertsTex, *d_numVertsTable, channelDesc));
	}

	void build_surfacePosition(dim3 grid, dim3 threads, float *surfacePosition,float *d_surfaceNormal)
	{
		initsurfacePosition << <grid, threads >> >(surfacePosition, (float3 *)d_surfaceNormal);
		getLastCudaError("classifyVoxel failed");
	}

	void build_metaballtable(dim3 grid, dim3 threads, float *metaballtable, float *normaltable)
	{
		initmetaballtable << <grid, threads >> >(metaballtable,(float3*)normaltable);
		getLastCudaError("classifyVoxel failed");
	}

	void launch_classifyVoxel(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float * surfacePos,
		uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
		float3 voxelSize, float isoValue, float* d_surfaceNormal)
	{
		classifyVoxel << <grid, threads >> >(voxelVerts, voxelOccupied, (float *)surfacePos,
			gridSize, gridSizeShift, gridSizeMask,
			numVoxels, voxelSize, isoValue, (float3 *)d_surfaceNormal);
		getLastCudaError("classifyVoxel failed");
	}





	void launch_compactVoxels(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
	{
		compactVoxels << <grid, threads >> >(compactedVoxelArray, voxelOccupied,
			voxelOccupiedScan, numVoxels);
		getLastCudaError("compactVoxels failed");
	}


	void launch_generateTriangles(dim3 grid, dim3 threads,
		float *pos, float *norm, uint *compactedVoxelArray, uint *numVertsScanned, float * surfacePos,
		uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
		float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts, float* d_surfaceNormal)
	{
		generateTriangles << <grid, threads >> >((float4 *)pos, (float4 *)norm,
			compactedVoxelArray,
			numVertsScanned, surfacePos,
			gridSize, gridSizeShift, gridSizeMask,
			voxelSize, isoValue, activeVoxels,
			maxVerts, (float3 *)d_surfaceNormal);
		getLastCudaError("generateTriangles2 failed");
	}

	void Normalize(dim3 grid, dim3 threads,
	float *norm, uint *numVertsScanned)
	{
		Normalize << <grid, threads >> >((float4 *)norm, numVertsScanned);
		getLastCudaError("generateTriangles2 failed");
	}


	void ThrustScanWrapper(unsigned int *output, unsigned int *input, unsigned int numElements)
	{
		thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),  //exclusive_scan     data [6] = {1, 0, 2, 2, 1, 3};   data is now {0, 1, 1, 3, 5, 6} 
			thrust::device_ptr<unsigned int>(input + numElements),
			thrust::device_ptr<unsigned int>(output));
	}


}
