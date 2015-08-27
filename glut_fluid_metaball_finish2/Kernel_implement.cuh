#ifndef _KERNEL_IMPLEMENT_H_
#define _KERNEL_IMPLEMENT_H_

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#include "Kernel.cuh"
#define NTHREADS 128
texture<uint, 1, cudaReadModeElementType> edgeTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;
typedef unsigned char uchar;
__constant__ KernelParams params;

struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

	template <typename Tuple>
	__device__
		void operator()(Tuple t)
	{
		volatile float4 posData = thrust::get<0>(t);
		volatile float4 velData = thrust::get<1>(t);
		//volatile float denData = thrust::get<2>(t);
		float3 pos = make_float3(posData.x, posData.y, posData.z);
		float3 vel = make_float3(velData.x, velData.y, velData.z);
		//float den = denData;

		vel += params.gravity * deltaTime;
		
		pos += vel * deltaTime;
#if 1

		if (pos.x > 30.0f - params.particleRadius)
		{
			pos.x = 30.0f - params.particleRadius;
			vel.x *= params.boundaryDamping;
		}

		if (pos.x < -30.0f + params.particleRadius)
		{
			pos.x = -30.0f + params.particleRadius;
			vel.x *= params.boundaryDamping;
		}

		if (pos.y > 14.0f - params.particleRadius)
		{
			pos.y = 14.0f - params.particleRadius;
			vel.y *= params.boundaryDamping;
		}

		if (pos.z > 30.0f - params.particleRadius)
		{
			pos.z = 30.0f - params.particleRadius;
			vel.z *= params.boundaryDamping;
		}

		if (pos.z < -30.0f + params.particleRadius)
		{
			pos.z = -30.0f + params.particleRadius;
			vel.z *= params.boundaryDamping;
		}
#endif

		if (pos.y < -14.0f + params.particleRadius)
		{
			pos.y = -14.0f + params.particleRadius;
			vel.y*=params.boundaryDamping;
		}

		thrust::get<0>(t) = make_float4(pos, posData.w);
		thrust::get<1>(t) = make_float4(vel, velData.w);

	}
};

__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.gridSize.y - 1);
	gridPos.z = gridPos.z & (params.gridSize.z - 1);
	return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

__device__ float4 sampleVolume(float *data,float3* normal, uint3 p, uint3 gridSize,int boolnum)
{
	p.x = min(p.x, gridSize.x - 1);
	p.y = min(p.y, gridSize.y - 1);
	p.z = min(p.z, gridSize.z - 1);
	uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
	if ((float)data[i] == 0){
		return make_float4(0.0f,0.0f,0.0f,0.0f);
	}
	else{
		//return (float)data[i] / 50.0f;
		float3 temp1 = normal[i];
		float temp2 = (float)data[i];

		return make_float4(temp1, temp2);
	}

	//return tex1Dfetch(volumeTex, i);
}

__device__ uint3 calcMarchingGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
{
	uint3 gridPos;
	gridPos.x = i & gridSizeMask.x;
	gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
	gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;    //在算HASH KEY的时候是正算，这个是反着算，从而得出网格位置

	return gridPos;
}

__global__ void calcHashD(uint *gridParticleHash, uint *gridParticleIndex, float4 *pos, uint numParticles, float* oldDen, 
	float* oldColorf, float3 *d_surfaceNormal, float* metaballtable, float3* normaltable,int power)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

__global__ void calcMarchingcubeandmetaballD(uint *gridParticleHash, uint *gridParticleIndex, float4 *pos, uint numParticles, float* oldDen,
	float* oldColorf, float3 *d_surfaceNormal, float* metaballtable, float3* normaltable, int power,int fornum)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	volatile float4 p = pos[index];

	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);

	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;

	uint3 marchingPos;
	marchingPos.x = floor((p.x - params.worldOrigin.x) *4.0f);
	marchingPos.y = floor((p.y - params.worldOrigin.y) *4.0f);
	marchingPos.z = floor((p.z - params.worldOrigin.z) *4.0f);
	float3 mn;

	mn.x = (p.x - params.worldOrigin.x)*4.0f;
	mn.y = (p.y - params.worldOrigin.y)*4.0f;
	mn.z = (p.z - params.worldOrigin.z)*4.0f;
	////////////////////////////////////////////////////////// new color field

	for (int i = -fornum; i < (fornum+1); i++){
		float temp1 = (float)i;
		for (int j = -fornum; j < (fornum+1); j++){
			float temp2 = (float)j;
			for (int k = -fornum; k < (fornum + 1); k++){
				float temp3 = (float)k;

				int tempPositon = (marchingPos.x + i) + (marchingPos.y + j) * 256 + (marchingPos.z + k) * 256 * 256;

				//////////////////////////////////////////////////////////////////metaball

				float3 p = make_float3(temp1 + floor(mn.x), temp2 + floor(mn.y), temp3 + floor(mn.z));

				//float div0 = temp1*temp1 + temp2*temp2 + temp3*temp3;
				float div0 = length(mn - p);
				float div = sqrt(div0);
				float3 normalvec = mn - p;

				if (oldColorf[tempPositon] < 0.006f){
					oldColorf[tempPositon] += 0.25f / pow(div0, power);
					d_surfaceNormal[tempPositon] += -0.5f*(normalvec / pow(div0, power + 2));
				}

			}
		}
	}
}

__global__ void reorderDataAndFindCellStartD(uint *cellStart, uint *cellEnd, float4 *sortedPos, float4 *sortedVel, uint *gridParticleHash, uint *gridParticleIndex, 
	float4 *oldPos, float4 *oldVel, uint numParticles, float *sortedDen, float *sortedPre, float *oldDen, float *oldPre, float* sortedColorf, float* oldColorf)
{
    extern __shared__ uint sharedHash[]; 
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    uint hash;

    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    __syncthreads();

    if (index < numParticles)
    {
        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        uint sortedIndex = gridParticleIndex[index];
        float4 pos = FETCH(oldPos, sortedIndex);   
        float4 vel = FETCH(oldVel, sortedIndex);    
		float den = FETCH(oldDen, sortedIndex);
		float pre = FETCH(oldPre, sortedIndex);
		//float colorf = FETCH(oldColorf, sortedIndex);

        sortedPos[index] = pos;
        sortedVel[index] = vel;
		sortedDen[index] = den;
		sortedPre[index] = pre;
		//sortedColorf[index] = colorf; 
    }
}



__device__ float caculateDenSecond(float3 posA, float3 posB)   //caculate each particle density
{
	float3 relPos = posB - posA;

	//float dist = length(relPos)*32;
	float dist = length(relPos);
	float den0 = 0.0f;//new
	if (dist <= 0.05f || dist >= 1.5f){
		//return den0;
	}
	else{
		float temp = 2.25f - dist*dist;
		//float temp = 0.03125 - dist*dist;
		den0 = 1.27f*temp*temp*temp;//1.27=4/(pi*h8)
	}
	return den0;

}

__device__ float caculateDenFirst(int3 gridPos, uint index, float3 pos,float4 *oldPos,uint *cellStart, uint *cellEnd)
{
	uint gridHash = calcGridHash(gridPos);

	uint startIndex = FETCH(cellStart, gridHash);

	float den0 = 0.0f;//new

	if (startIndex != 0xffffffff)
	{

		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			if (j != index)
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				den0 += caculateDenSecond(pos, pos2);
			}
		}
	}
	return den0;
}



__device__ float3 calculatePreforceSecond(float3 posA, float3 posB, float radiusA, float radiusB,  float preA, float denB, float preB)
{
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);

	if (0.02f<dist && dist < collideDist)
	{	
		if (1.26f<denB){
			force += 9.55*(collideDist - dist)*(collideDist - dist)/dist*((preA + preB) / (2.0f * denB))*(-relPos);
			//*(9.55f*(((collideDist - dist)*(collideDist - dist)) / dist))*relPos;//-9.55=-30/pi
		}
	}
	return force;
}

__device__ float3 calculatePreforceFirst(int3 gridPos, uint index, float3 pos,  float4 *oldPos,  uint *cellStart, uint *cellEnd,float pre, float *oldDen, float *oldPre)
{
	uint gridHash = calcGridHash(gridPos);

	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);

	if (startIndex != 0xffffffff)
	{

		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			if (j != index)
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				float den2 = FETCH(oldDen, j);
				float pre2 = FETCH(oldPre, j);

				force += calculatePreforceSecond(pos, pos2, params.particleRadius, params.particleRadius, pre, den2, pre2);
			}
		}
	}
	return force;
}

__device__ float3 calculateVisSecond(float3 posA, float3 posB, float3 velA, float3 velB, float radiusA, float radiusB, float denB, float preA,float preB)
{
	float3 relPos = posB - posA;

	float dist = length(relPos) ;
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);

	if (0.02f<dist && dist< collideDist)
	{

		float3 norm = (relPos ) / dist;
		float3 relVel = velB - velA;
		float3 relTanVel = relVel - (dot(relVel, norm)*norm);
		force = -0.7f*(collideDist - dist)*norm;
		force += 0.01f*relVel;
		force += 0.03f*relTanVel;
		if (1.91f<denB ){
			force += 0.01274f*((velA - velB) / denB)*(collideDist - dist);//12.74=m*40/pi
			force += 9.55f*(collideDist - dist)*(collideDist - dist) / dist*((preA + preB) / (2.0f * denB))*(-relPos);
		}	
	}
	return force;
}

__device__ float3 calculateVisFirst(int3 gridPos, uint index, float3 pos, float3 vel, float4 *oldPos, float4 *oldVel, uint *cellStart, uint *cellEnd, float *oldDen, float pre, float *oldPre)
{
	uint gridHash = calcGridHash(gridPos);

    uint startIndex = FETCH(cellStart, gridHash);

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff) 
    {

        uint endIndex = FETCH(cellEnd, gridHash);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index) 
            {
                float3 pos2 = make_float3(FETCH(oldPos, j));
                float3 vel2 = make_float3(FETCH(oldVel, j));
				float den2 = FETCH(oldDen, j);
				float pre2 = FETCH(oldPre, j);

				force += calculateVisSecond(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, den2,pre,pre2);
            }
        }
    }
    return force;
}

__device__ float3 collideSpheres(float3 posA, float3 posB,float3 velA, float3 velB,float radiusA, float radiusB)
{
	// calculate relative position
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);

	if (dist < collideDist)
	{
		float3 norm = relPos / dist;

		// relative velocity
		float3 relVel = velB - velA;

		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);

		// spring force
		force = -0.5f*(collideDist - dist) * norm;
		// dashpot (damping) force
		force += 0.01f*relVel;
		// tangential shear force
		force += 0.02f*tanVel;
		//force = force / 50;
	}

	return force;
}


__global__ void calculateDensityD(float4 *oldPos,  uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, float *newDen){
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;
	uint originalIndex = gridParticleIndex[index];
	float3 pos = make_float3(FETCH(oldPos, index));
	int3 gridPos = calcGridPos(pos);
	float den0 = 1.91f;

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				den0 += caculateDenFirst(neighbourPos, index, pos, oldPos, cellStart, cellEnd);
			}
		}
	}
	newDen[originalIndex] = den0;
}


__global__ void collideD(float4 *newVel, float4 *sortedPos, float4 *oldVel, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles,
	float *newDen, float *newPre, float *oldDen, float *oldPre, float4* oldPos)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	float3 pos = make_float3(FETCH(sortedPos, index));
    float3 vel = make_float3(FETCH(oldVel, index));
	float den = FETCH(oldDen,index);
	float pre = FETCH(oldPre, index);

    int3 gridPos = calcGridPos(pos);

    float3 force = make_float3(0.0f);

	float pre0 = 0.0f;//new

    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
				force += calculateVisFirst(neighbourPos, index, pos, vel, sortedPos, oldVel, cellStart, cellEnd, oldDen, pre, oldPre);
            }
        }
    }

	uint originalIndex = gridParticleIndex[index];

	//force += collideSpheres(pos, params.colliderPOS, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius);

	float den0 = newDen[originalIndex];
	pre0 = (den0 - den)*0.0668f*2.0f;

	if (fabs(den0 - den) >= 3.5f){
		den0 = (den0 + den) / 2 + 1.91;
	}

	//control the water foam
	
	if (den0 < 7.5f){
		if (oldPos[originalIndex].w == 0.0f){
			oldPos[originalIndex].w = 1.0f;
		}
	}
	if (den0 > 19.0f){
		if (oldPos[originalIndex].w == 1.0f){
			oldPos[originalIndex].w = 0.0f;
		}
	}

	newVel[originalIndex] = make_float4(vel + force, den0);
	newDen[originalIndex] = den0;
	newPre[originalIndex] = pre0;
}

__device__ float caculateColorfieldSecond(float3 posA, float3 posB, float radiusA, float radiusB)   
{
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float colorf0 = 0.0f;
	if (dist <= 0.05f || dist >= 1.5f){

	}
	else{
		float temp = 2.25f - dist*dist;
		
		colorf0 = 7.64f*temp*temp*dist;
	}
	return colorf0;

}

__device__ float caculateColorfieldFirst(int3 gridPos, uint index, float3 pos,  float4 *oldPos,uint *cellStart, uint *cellEnd,
	float colorfield, float *oldcolorfield)
{
	uint gridHash = calcGridHash(gridPos);

	uint startIndex = FETCH(cellStart, gridHash);

	float colorf0 = 0.0f;//new

	if (startIndex != 0xffffffff)
	{

		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			if (j != index)
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				//float colorf2 = FETCH(oldcolorfield, j);
				colorf0 += caculateColorfieldSecond(pos, pos2, params.particleRadius, params.particleRadius);
			}
		}
	}
	return colorf0;
}

__global__ void colorfieldD(float4 *sortedPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, float *colorfield, float *sortedcolorfield){
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float3 pos = make_float3(FETCH(sortedPos, index));
	float colorf = FETCH(colorfield, index);
	//int3 gridPos = calcGridPos(pos);
	float colorf0 = 0.0f;
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				//int3 neighbourPos = gridPos + make_int3(x, y, z);
				//colorf0 += caculateColorfieldFirst(neighbourPos, index, pos, sortedPos, cellStart, cellEnd, colorf, colorfield);
			}
		}
	}

	uint originalIndex = gridParticleIndex[index];
	colorfield[originalIndex] = colorf0;
	//__syncthreads();
}

struct calcDepth_functor
{
	float3 sortVector;

	__host__ __device__
		calcDepth_functor(float3 sort_vector) : sortVector(sort_vector) {}

	template <typename Tuple>
	__host__ __device__
		void operator()(Tuple t)
	{
		volatile float4 p = thrust::get<0>(t);
		float key = -dot(make_float3(p.x, p.y, p.z), sortVector); // project onto sort vector
		thrust::get<1>(t) = key;
	}
};

__global__ void initmetaballtable(float *metaballtable, float3 *normaltable){
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
	for (int i = -4; i < 5; i++){
		float temp1 = (float)i;
		for (int j = -4; j < 5; j++){
			float temp2 = (float)j;
			for (int k = -4; k < 5; k++){
				int temp = (k+4) + (j+4) * 8 + (i+4) * 8 * 8;
				float temp3 = (float)k;
				float div0 = temp1*temp1 + temp2*temp2 + temp3*temp3;
				float div = sqrt(div0);
				float3 normalvec = make_float3(temp1, temp2, temp3);

				metaballtable[temp] = 0.25 / (div0*div0);
				normaltable[temp] = -0.5*(normalvec / (div0*div0*div0));
			}
		}
	}
}

__global__ void initsurfacePosition(float *surfacePosition,float3 *surfaceNormal){
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
	surfacePosition[i] = 0;
	surfaceNormal[i] = make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void classifyVoxel(uint *voxelVerts, uint *voxelOccupied, float *volume,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
float3 voxelSize, float isoValue, float3* d_surfaceNormal)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	uint3 gridPos = calcMarchingGridPos(i, gridSizeShift, gridSizeMask);

	// read field values at neighbouring grid vertices
//#if SAMPLE_VOLUME
	float field[8];
	field[0] = sampleVolume(volume, d_surfaceNormal, gridPos, gridSize,0).w;
	field[1] = sampleVolume(volume, d_surfaceNormal, gridPos + make_uint3(1.0, 0, 0), gridSize,0).w;
	field[2] = sampleVolume(volume, d_surfaceNormal, gridPos + make_uint3(1.0, 1.0, 0), gridSize,0).w;
	field[3] = sampleVolume(volume, d_surfaceNormal, gridPos + make_uint3(0, 1.0, 0), gridSize,0).w;
	field[4] = sampleVolume(volume, d_surfaceNormal, gridPos + make_uint3(0, 0, 1.0), gridSize,0).w;
	field[5] = sampleVolume(volume, d_surfaceNormal, gridPos + make_uint3(1.0, 0, 1.0), gridSize,0).w;
	field[6] = sampleVolume(volume, d_surfaceNormal, gridPos + make_uint3(1.0, 1.0, 1.0), gridSize,0).w;
	field[7] = sampleVolume(volume, d_surfaceNormal, gridPos + make_uint3(0, 1.0, 1.0), gridSize,0).w;

	// calculate flag indicating if each vertex is inside or outside isosurface
	uint cubeindex;
	cubeindex = uint(field[0] > isoValue);              //返回一个field[0] < isoValue 真假值，如果是真，那就等于1，如果是假就等于0
	cubeindex += uint(field[1] > isoValue) * 2;
	cubeindex += uint(field[2] > isoValue) * 4;
	cubeindex += uint(field[3] > isoValue) * 8;
	cubeindex += uint(field[4] > isoValue) * 16;
	cubeindex += uint(field[5] > isoValue) * 32;
	cubeindex += uint(field[6] > isoValue) * 64;
	cubeindex += uint(field[7] > isoValue) * 128;

	// read number of vertices from texture
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

	if (i < numVoxels)
	{
		voxelVerts[i] = numVerts;
		voxelOccupied[i] = (numVerts > 0);
	}
}


__global__ void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (voxelOccupied[i] && (i < numVoxels))
	{
		compactedVoxelArray[voxelOccupiedScan[i]] = i;  // voxelOccupiedScan[i] 该数组通过之前的操作，里面元素挨着排列【0,1,1,2,3,4,5,5,6...】，所以compactedVoxelArray里面记录的就是 例如
		// 在该数组里面的第一个体素 0 ，对应的是真实体素 i
	}
}

__device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);
}

__device__ float3 normalInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);
}

__device__ float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
	float3 edge0 = *v1 - *v0;
	float3 edge1 = *v2 - *v0;
	// note - it's faster to perform normalization in vertex shader rather than here
	float3 temp = cross(edge0, edge1);
	
	//return cross(edge0, edge1);
	return normalize(temp);
}

__global__ void generateTriangles(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *surfacePos,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts,  float3* d_surfaceNormal)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (i > activeVoxels - 1)
	{
		i = activeVoxels - 1;
	}

	uint voxel = compactedVoxelArray[i];

	// compute position in 3d grid
	uint3 gridPos = calcMarchingGridPos(voxel, gridSizeShift, gridSizeMask);

	float3 p;

	p.x = (gridPos.x * voxelSize.x); //voxelSize(0.0625)       -1.0--1.0 == -16.0--16.0
	p.y = (gridPos.y * voxelSize.y);
	p.z = (gridPos.z * voxelSize.z);


	// calculate cell vertex positions
	float3 v[8];
	v[0] = p;
	v[1] = p + make_float3(voxelSize.x, 0, 0);
	v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
	v[3] = p + make_float3(0, voxelSize.y, 0);
	v[4] = p + make_float3(0, 0, voxelSize.z);
	v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
	v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
	v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);


	/////fix marching cube movement



//#if SAMPLE_VOLUME
	float4 wholefield[8];
	float3 fieldNormal[8];
	float field[8];   //field 存储的是正方体（长方体）的8各顶点的iossurface值

	wholefield[0] = sampleVolume(surfacePos, d_surfaceNormal, gridPos, gridSize,1);
	wholefield[1] = sampleVolume(surfacePos, d_surfaceNormal,gridPos + make_uint3(1, 0, 0), gridSize,1);
	wholefield[2] = sampleVolume(surfacePos, d_surfaceNormal,gridPos + make_uint3(1, 1, 0), gridSize,1);
	wholefield[3] = sampleVolume(surfacePos, d_surfaceNormal,gridPos + make_uint3(0, 1, 0), gridSize,1);
	wholefield[4] = sampleVolume(surfacePos, d_surfaceNormal,gridPos + make_uint3(0, 0, 1), gridSize,1);
	wholefield[5] = sampleVolume(surfacePos, d_surfaceNormal,gridPos + make_uint3(1, 0, 1), gridSize,1);
	wholefield[6] = sampleVolume(surfacePos, d_surfaceNormal,gridPos + make_uint3(1, 1, 1), gridSize,1);
	wholefield[7] = sampleVolume(surfacePos, d_surfaceNormal,gridPos + make_uint3(0, 1, 1), gridSize,1);

	fieldNormal[0] = make_float3(wholefield[0].x, wholefield[0].y, wholefield[0].z);
	fieldNormal[1] = make_float3(wholefield[1].x, wholefield[1].y, wholefield[1].z);
	fieldNormal[2] = make_float3(wholefield[2].x, wholefield[2].y, wholefield[2].z);
	fieldNormal[3] = make_float3(wholefield[3].x, wholefield[3].y, wholefield[3].z);
	fieldNormal[4] = make_float3(wholefield[4].x, wholefield[4].y, wholefield[4].z);
	fieldNormal[5] = make_float3(wholefield[5].x, wholefield[5].y, wholefield[5].z);
	fieldNormal[6] = make_float3(wholefield[6].x, wholefield[6].y, wholefield[6].z);
	fieldNormal[7] = make_float3(wholefield[7].x, wholefield[7].y, wholefield[7].z);

	field[0] = wholefield[0].w;
	field[1] = wholefield[1].w;
	field[2] = wholefield[2].w;
	field[3] = wholefield[3].w;
	field[4] = wholefield[4].w;
	field[5] = wholefield[5].w;
	field[6] = wholefield[6].w;
	field[7] = wholefield[7].w;


	// recalculate flag
	uint cubeindex;
	cubeindex = uint(field[0] > isoValue);
	cubeindex += uint(field[1] > isoValue) * 2;
	cubeindex += uint(field[2] > isoValue) * 4;
	cubeindex += uint(field[3] > isoValue) * 8;
	cubeindex += uint(field[4] > isoValue) * 16;
	cubeindex += uint(field[5] > isoValue) * 32;
	cubeindex += uint(field[6] > isoValue) * 64;
	cubeindex += uint(field[7] > isoValue) * 128;

	// find the vertices where the surface intersects the cube

//#if USE_SHARED
	// use shared memory to avoid using local
	__shared__ float3 vertlist[12 * NTHREADS];
	

	vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]) /*+ fix_move*/;
	vertlist[NTHREADS + threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]) /*+ fix_move*/;
	vertlist[(NTHREADS * 2) + threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]) /*+ fix_move*/;
	vertlist[(NTHREADS * 3) + threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]) /*+ fix_move*/;
	vertlist[(NTHREADS * 4) + threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]) /*+ fix_move*/;
	vertlist[(NTHREADS * 5) + threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]) /*+ fix_move*/;
	vertlist[(NTHREADS * 6) + threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]) /*+ fix_move*/;
	vertlist[(NTHREADS * 7) + threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]) /*+ fix_move*/;
	vertlist[(NTHREADS * 8) + threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]) /*+ fix_move*/;
	vertlist[(NTHREADS * 9) + threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]) /*+ fix_move*/;
	vertlist[(NTHREADS * 10) + threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]) /*+ fix_move*/;
	vertlist[(NTHREADS * 11) + threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]) /*+ fix_move*/;
	__syncthreads();

	__shared__ float3 normallist[12 * NTHREADS];
	normallist[threadIdx.x] = normalInterp(isoValue, fieldNormal[0], fieldNormal[1], field[0], field[1]) /*+ fix_move*/;
	normallist[NTHREADS + threadIdx.x] = normalInterp(isoValue, fieldNormal[1], fieldNormal[2], field[1], field[2]) /*+ fix_move*/;
	normallist[(NTHREADS * 2) + threadIdx.x] = normalInterp(isoValue, fieldNormal[2], fieldNormal[3], field[2], field[3]) /*+ fix_move*/;
	normallist[(NTHREADS * 3) + threadIdx.x] = normalInterp(isoValue, fieldNormal[3], fieldNormal[0], field[3], field[0]) /*+ fix_move*/;
	normallist[(NTHREADS * 4) + threadIdx.x] = normalInterp(isoValue, fieldNormal[4], fieldNormal[5], field[4], field[5]) /*+ fix_move*/;
	normallist[(NTHREADS * 5) + threadIdx.x] = normalInterp(isoValue, fieldNormal[5], fieldNormal[6], field[5], field[6]) /*+ fix_move*/;
	normallist[(NTHREADS * 6) + threadIdx.x] = normalInterp(isoValue, fieldNormal[6], fieldNormal[7], field[6], field[7]) /*+ fix_move*/;
	normallist[(NTHREADS * 7) + threadIdx.x] = normalInterp(isoValue, fieldNormal[7], fieldNormal[4], field[7], field[4]) /*+ fix_move*/;
	normallist[(NTHREADS * 8) + threadIdx.x] = normalInterp(isoValue, fieldNormal[0], fieldNormal[4], field[0], field[4]) /*+ fix_move*/;
	normallist[(NTHREADS * 9) + threadIdx.x] = normalInterp(isoValue, fieldNormal[1], fieldNormal[5], field[1], field[5]) /*+ fix_move*/;
	normallist[(NTHREADS * 10) + threadIdx.x] = normalInterp(isoValue, fieldNormal[2], fieldNormal[6], field[2], field[6]) /*+ fix_move*/;
	normallist[(NTHREADS * 11) + threadIdx.x] = normalInterp(isoValue, fieldNormal[3], fieldNormal[7], field[3], field[7]) /*+ fix_move*/;

	__syncthreads();



	// output triangle vertices
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

	for (int i = 0; i<numVerts; i += 3)
	{
		uint index = numVertsScanned[voxel] + i;

		float3 *v[3];
		float3 *normal[3];
		uint edge;
		edge = tex1Dfetch(triTex, (cubeindex * 16) + i);

		v[0] = &vertlist[(edge*NTHREADS) + threadIdx.x] ;
		normal[0] = &normallist[(edge*NTHREADS) + threadIdx.x];


		edge = tex1Dfetch(triTex, (cubeindex * 16) + i + 1);

		v[1] = &vertlist[(edge*NTHREADS) + threadIdx.x] ;
		normal[1] = &normallist[(edge*NTHREADS) + threadIdx.x];


		edge = tex1Dfetch(triTex, (cubeindex * 16) + i + 2);

		v[2] = &vertlist[(edge*NTHREADS) + threadIdx.x] ;
		normal[2] = &normallist[(edge*NTHREADS) + threadIdx.x];



		if (index < (maxVerts - 3))
		{
			pos[index] = make_float4(*v[0], 1.0f);
			norm[index] = make_float4(*normal[0], 0.0f);

			pos[index + 1] = make_float4(*v[1], 1.0f);
			norm[index + 1] = make_float4(*normal[1], 0.0f);

			pos[index + 2] = make_float4(*v[2], 1.0f);
			norm[index + 2] = make_float4(*normal[2], 0.0f);
		}
	}
}

__global__ void Normalize(float4 *norm, uint *numVertsScanned){
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	norm[i] = normalize(norm[i]);
}

#endif
