#include "Kernel.cuh"
typedef unsigned char uchar;
extern "C"
{
    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

    void setParameters(KernelParams *hostParams);

	void integrateSystem(float *pos, float *vel,float *den,float *pre, float deltaTime, uint numParticles);

	void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos, int numParticles, float* oldDen, float* oldColorf, float* d_surfaceNormal, float* metaballtable, float* normaltable,int pow);

	void calcMarchingcubeandmetaball(uint *gridParticleHash, uint *gridParticleIndex, float *pos, int numParticles, float* oldDen, 
		float* oldColorf, float* d_surfaceNormal, float* metaballtable, float* normaltable, int pow,int fornum);

	void reorderDataAndFindCellStart(uint *cellStart, uint *cellEnd, float *sortedPos, float *sortedVel, uint *gridParticleHash, uint *gridParticleIndex,
		float *oldPos, float *oldVel, uint numParticles, uint numCells, float *sortedDen, float *sortedPre, float *oldDen, float *oldPre, float* sortedColorf, float* oldColorf);

	void calculateDensity(float *oldPos,uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, float *newDen);

	void collide(float *newVel, float *sortedPos, float *sortedVel, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, uint numCells,
		float *newDen, float *newPre, float *sortedDen, float *sortedPre,float *oldPos);

	void colorfield(float *sortedPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, uint numCells, float *colorfield, float *sortedcolorfield);

	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

	void calcDepth(float *pos, float *keys, uint *indices, float3 sortVector, int numParticles);

	void sortParticlesKey(float *sortKeys, uint *indices, uint numParticles);
	//////////////////////////////////////////////////////////////Marching cube
	void allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable);

	void build_surfacePosition(dim3 grid, dim3 threads, float *surfacePosition, float *d_surfaceNormal);

	void launch_classifyVoxel(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *surfacePos,
		uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
		float3 voxelSize, float isoValue, float* d_surfaceNormal);
	void ThrustScanWrapper(unsigned int *output, unsigned int *input, unsigned int numElements);
	void launch_compactVoxels(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels);
	void launch_generateTriangles(dim3 grid, dim3 threads,
		float *pos, float *norm, uint *compactedVoxelArray, uint *numVertsScanned, float * surfacePos,
		uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
		float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts, float* d_surfaceNormal);
	void Normalize(dim3 grid, dim3 threads, float *norm, uint *numVertsScanned);

	void build_metaballtable(dim3 grid, dim3 threads, float *metaballtable, float *normaltable);
}
