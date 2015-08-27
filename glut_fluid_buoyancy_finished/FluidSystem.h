#ifndef __FLUIDSYSTEM_H__
#define __FLUIDSYSTEM_H__

#include <helper_functions.h>
#include "Kernel.cuh"
#include "vector_functions.h"
#include "nvMath.h"
#include "GpuArray.h"
using namespace std;
using namespace nv;
class FluidSystem
{
public:
	FluidSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL);
	~FluidSystem();

	enum FluidConfig
	{
		CONFIG_RANDOM,
		CONFIG_GRID,
	};

	enum FluidArray
	{
		POSITION,
		VELOCITY,
		DENSITY,
		PRESSURE,
		COLORFIELD,
	};

	void update(float deltaTime);
	void reset(FluidConfig config);
	void depthSort();

	float *getArray(FluidArray array);
	void   setArray(FluidArray array, const float *data, int start, int count);

	int    getNumParticles() const { return m_numParticles; }

	unsigned int getCurrentReadBuffer() const { return m_posVbo; }
	unsigned int getColorBuffer() const { return m_colorVBO; }
	unsigned int getVelBuffer() const { return m_velVBO; }
	unsigned int getobstacleBuffer(int i) const { return obstaclePosVbo[i]; }

	void *getCudaPosVBO() const { return (void *)m_cudaPosVBO; }
	void *getCudaColorVBO() const { return (void *)m_cudaColorVBO; }
	void *getCudaVelVBO() const { return (void *)m_cudaVelVBO; }

	void setGravity(float x) { m_params.gravity = make_float3(0.0f, x, 0.0f); }

	float getParticleRadius() { return m_params.particleRadius; }
	uint3 getGridSize() { return m_params.gridSize; }
	float3 getWorldOrigin() { return m_params.worldOrigin; }
	float3 getCellSize() { return m_params.cellSize; }

	void dumpParticles(uint start, uint count);
	void dumpBin(float4 **posData, float4 **velData);

	uint getSortedIndexBuffer()	{ return m_indices.getVbo(); }
	void setSortVector(float3 v) { m_sortVector = v; }
	void setSorting(bool x)	{ m_doDepthSort = x; }
	float3 getColliderPos()	{ return m_params.colliderPOS; }
	float3 getboardPos()	{ return m_params.colliderboard; }
	float getColliderRadius() { return m_params.colliderRadius; }
	void setColliderPos(float3 x) { m_params.colliderPOS = x; }
	void setboardPos(float3 x) { m_params.colliderboard = x; }
	float3 getColliderSpherePos()	{ return m_params.colliderSphere; }
	float getColliderSphereRadius() { return m_params.SphereRadius; }
	void setColliderShperePos(float3 x) { m_params.colliderSphere = x; }

	float3 getBuoyancyPos()	{ return m_params.buoyancyPOS; }
	void setBuoyancyPos(float3 x) { m_params.buoyancyPOS = x; }
	float getBuoyancyRadius() { return m_params.buoyancyRadius; }

	void ShootSphere();
	void calculateBuoyancy();
protected:
	FluidSystem() {}
	uint createVBO(uint size);

	void _initialize(int numParticles);
	void _finalize();

	void initGrid(uint *size, float spacing, float jitter, uint numParticles);

protected: 
	bool m_bInitialized, m_bUseOpenGL;
	uint m_numParticles;

	float *m_hPos;      
	float *m_hVel; 
	float *m_hDen;//new host density pointer
	float *m_hPre;//new host pressure pointer
	float *m_hColorf;//host color field pointer
	float *buoyancyPos;
	float3 tempvel;
	int buoyancycount;   //this is a number to sum how many surface particle around buoyancy sphere 
	float storefactor; //this is a factor to judge the buoyancy sphere is under fluid surface or not.

	uint  *m_hParticleHash;
	uint  *m_hCellStart;
	uint  *m_hCellEnd;

	float *m_dPos;
	float *m_dVel;
	float *m_dDen;//new device density pointer
	float *m_dPre;//new device pressure pointer
	float *m_dColorf;//new device color field pointer
	float *buoyancyForce;

	float *m_dSortedPos;
	float *m_dSortedVel;
	float *m_dSortedDen;//new device sorted density pointer
	float *m_dSortedPre;//new device sorted pressure pointer
	float *m_dSortedColorf;

	uint  *m_dGridParticleHash;
	uint  *m_dGridParticleIndex;
	uint  *m_dCellStart;   
	uint  *m_dCellEnd;       

	uint   m_gridSortBits;

	uint   m_posVbo;     
	uint   m_colorVBO; 
	uint   m_velVBO;
	uint   obstaclePosVbo[5];

	float *m_cudaPosVBO;    
	float *m_cudaColorVBO; 
	float *m_cudaVelVBO;

	struct cudaGraphicsResource *m_cuda_posvbo_resource; 
	struct cudaGraphicsResource *m_cuda_colorvbo_resource; 
	struct cudaGraphicsResource *m_cuda_velvbo_resource;

	KernelParams m_params;
	uint3 m_gridSize;
	uint m_numGridCells;

	StopWatchInterface *m_timer;

	float3 m_sortVector;
	bool m_doDepthSort;
	GpuArray<uint> m_indices;   // sorted indices for rendering
	GpuArray<float> m_sortKeys;
};

#endif 
