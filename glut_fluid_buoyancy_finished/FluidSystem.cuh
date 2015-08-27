#include "Kernel.cuh"
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

	void integrateSystem(float *pos, float *vel, float *den, float *pre, float deltaTime, uint numParticles, float* buoyancyForce);

	void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos, int numParticles);

	void reorderDataAndFindCellStart(uint *cellStart, uint *cellEnd, float *sortedPos, float *sortedVel, uint *gridParticleHash, uint *gridParticleIndex,
		float *oldPos, float *oldVel, uint numParticles, uint numCells, float *sortedDen, float *sortedPre, float *oldDen, float *oldPre, float* sortedColorf, float* oldColorf);

	void calculateDensity(float *oldPos,uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, float *newDen);

	void collide(float *newVel, float *sortedPos, float *sortedVel, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, uint numCells,
		float *newDen, float *newPre, float *sortedDen, float *sortedPre, float *oldPos, float* buoyancyForce);

	void colorfield(float *sortedPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, uint numCells, float *colorfield, float *sortedcolorfield);

	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

	void calcDepth(float *pos, float *keys, uint *indices, float3 sortVector, int numParticles);

	void sortParticlesKey(float *sortKeys, uint *indices, uint numParticles);

	void plusBuoyancyforce(float* buoyancyForce, uint numParticles);

}
