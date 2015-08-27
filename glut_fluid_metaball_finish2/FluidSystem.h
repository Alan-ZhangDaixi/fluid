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
	unsigned int getMarchingCubePosBuffer() const { return m_marchingcubeposVbo;}
	unsigned int getMarchingCubeNormalBuffer() const { return m_normalVbo; }
	unsigned int getColorBuffer() const { return m_colorVBO; }
	unsigned int getMarchingcubeColorBuffer() const { return m_marchingcubecolorVBO; }
	unsigned int getVelBuffer() const { return m_velVBO; }

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
	float getColliderRadius() { return m_params.colliderRadius; }
	void setColliderPos(float3 x) { m_params.colliderPOS = x; }
	float3 getColliderSpherePos()	{ return m_params.colliderSphere; }
	float getColliderSphereRadius() { return m_params.SphereRadius; }
	void setColliderShperePos(float3 x) { m_params.colliderSphere = x; }

	void ShootSphere();
	//////////////////////////////////////////////////////////////////////////  Marching cube
	void computeIsosurface();

	uint gettotalVerts(){ return totalVerts; }

	void setStep(int i){ step0 = i; }


	///////////////////////////////////////////////////////////////////////////
protected:
	FluidSystem() {}
	uint createVBO(uint size);

	void _initialize(int numParticles);
	void initMC();
	void _finalize();

	void initGrid(uint *size, float spacing, float jitter, uint numParticles);

protected: 
	bool m_bInitialized, m_bUseOpenGL;
	int step0;
	uint m_numParticles;

	float *m_hPos;      
	float *m_hVel; 
	float *m_hDen;//new host density pointer
	float *m_hPre;//new host pressure pointer
	float *m_hColorf;//host color field pointer

	uint  *m_hParticleHash;
	uint  *m_hCellStart;
	uint  *m_hCellEnd;

	float *m_dPos;
	float *m_dVel;
	float *m_dDen;//new device density pointer
	float *m_dPre;//new device pressure pointer
	float *m_dColorf;//new device color field pointer

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
	uint   m_marchingcubecolorVBO;
	uint   m_velVBO;
	uint   m_marchingcubeposVbo;
	uint   m_normalVbo;

	float *m_cudaPosVBO;    
	float *m_cudaColorVBO; 
	float *m_cudaVelVBO;

	struct cudaGraphicsResource *m_cuda_posvbo_resource; 
	struct cudaGraphicsResource *m_cuda_colorvbo_resource; 
	struct cudaGraphicsResource *m_cuda_velvbo_resource;
	struct cudaGraphicsResource *m_cuda_marchcubeposvbo_resource;
	struct cudaGraphicsResource *cuda_normalvbo_resource;

	KernelParams m_params;
	uint3 m_gridSize;
	uint m_numGridCells;

	StopWatchInterface *m_timer;

	float3 m_sortVector;
	bool m_doDepthSort;
	GpuArray<uint> m_indices;   // sorted indices for rendering
	GpuArray<float> m_sortKeys;
	/////////////////////////////////////////////////////////////////////////////// Marching cube
	uint3 gridSizeLog2;
	uint3 cube_gridSize;
	uint3 cube_gridSizeShift;
	uint3 cube_gridSizeMask;
	float3 voxelSize;
	uint numVoxels;
	uint maxVerts;
	uint activeVoxels;
	uint totalVerts;
	float* h_verPos;
	float *h_surfacePos = 0;
	float *h_fixparticlePos = 0;
	/////////////device data
	//float* d_surfacePos=0, *d_surfaceNormal=0;
	float *d_surfacePos = 0;
	float *d_surfaceNormal = 0;
	float *d_verPos;
	float *d_fixparticlePos = 0;
	uint *d_voxelVerts = 0;
	uint *d_voxelVertsScan = 0;
	uint *d_voxelOccupied = 0;
	uint *d_voxelOccupiedScan = 0;
	uint *d_compVoxelArray;
	////////////// tables
	float* normaltable = 0;
	float* metaballtable = 0;
	uint *d_numVertsTable = 0;
	uint *d_edgeTable = 0;
	uint *d_triTable = 0;
	//////////////////////////////////////////////////////////////////////////////////
};

#endif 
