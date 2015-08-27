#ifndef _KERNEL_IMPLEMENT_H_
#define _KERNEL_IMPLEMENT_H_

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#include "Kernel.cuh"

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
		//volatile float3 bForceData = thrust::get<2>(t);
		//volatile float denData = thrust::get<2>(t);
		float3 pos = make_float3(posData.x, posData.y, posData.z);
		float3 vel = make_float3(velData.x, velData.y, velData.z);
		//float3 bForce = make_float3(bForceData.x, bForceData.y, bForceData.z);
		//float den = denData;
		//params.buoyancyPOS = make_float3(50, 50, 50);
		vel += params.gravity * deltaTime;
		
		pos += vel * deltaTime;
#if 1

		if (pos.x > 96.0f - params.particleRadius)
		{
			pos.x = 96.0f - params.particleRadius;
			vel.x *= params.boundaryDamping;
		}

		if (pos.x < -96.0f + params.particleRadius)
		{
			pos.x = -96.0f + params.particleRadius;
			vel.x *= params.boundaryDamping;
		}

		if (pos.y > 64.0f - params.particleRadius)
		{
			pos.y = 64.0f - params.particleRadius;
			vel.y *= params.boundaryDamping;
		}

		if (pos.z > 48.0f - params.particleRadius)
		{
			pos.z = 48.0f - params.particleRadius;
			vel.z *= params.boundaryDamping;
		}

		if (pos.z < -48.0f + params.particleRadius)
		{
			pos.z = -48.0f + params.particleRadius;
			vel.z *= params.boundaryDamping;
		}
#endif

		if (pos.y < -31.0f + params.particleRadius)
		{
			pos.y = -31.0f + params.particleRadius;
			vel.y*=params.boundaryDamping;
		}

		//thrust::get<0>(t) = make_float4(pos, posData.w);
		thrust::get<0>(t) = make_float4(pos, posData.w);
		//thrust::get<1>(t) = make_float4(vel, velData.w);


		/*	if (velData.w == 32.0f){
				thrust::get<1>(t) = make_float4(0.0f,0.0f,0.0f, velData.w);
				}*/

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

__global__ void calcHashD(uint *gridParticleHash, uint *gridParticleIndex, float4 *pos, uint numParticles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
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
		float colorf = FETCH(oldColorf, sortedIndex);

        sortedPos[index] = pos;
        sortedVel[index] = vel;
		sortedDen[index] = den;
		sortedPre[index] = pre;
		sortedColorf[index] = colorf; 
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

__device__ float3 collideSpheres(float3 posA, float3 posB, float3 velA, float3 velB, float radiusA, float radiusB)
{
	// calculate relative position
	float3 relPos = posB - posA;
	float dist = length(relPos);
	float collideDist = radiusA + radiusB;
	float3 force = make_float3(0.0f);
	if (dist < collideDist)
	{
		float3 norm = relPos / dist;
		float3 relVel = velB - velA;
		float3 tanVel = relVel - (dot(relVel, norm) * norm);
		force = -0.5f*(collideDist - dist) * norm;
		force += 0.01f*relVel;
		force += 0.02f*tanVel;
	}

	return force;
}

__device__ float4 buoyancySpheres(float3 posA, float3 posB, float3 velA, float3 velB, float radiusA, float radiusB,float den)
{
	// calculate relative position
	float3 relPos = posB - posA;
	float dist = length(relPos);
	float collideDist = radiusA + radiusB;
	float3 force = make_float3(0.0f);
	float height = 0.0;
	if (dist < collideDist)
	{
		float3 norm = relPos / dist;
		float3 relVel = velB - velA;
		float3 tanVel = relVel - (dot(relVel, norm) * norm);
		force = -0.5f*(collideDist - dist) * norm;
		force += 0.01f*relVel;
		force += 0.02f*tanVel;
		//force = make_float3(0, 0, 0);
		if (10.0f < den&&den < 15.0f){
			height = posA.y+32.0;
			//height = 1.0;
		}
		else{
			height = -32.0;
		}
	}
	
	return make_float4(force,height);
}

__device__ float3 collideBoard(float3 posA, float3 posB, float3 velA, float3 velB, float radiusA)
{
	// calculate relative position
	float relPos = -posB.x+posA.x-64.0f; //-64.0f can reach the board edge

	float dist = relPos;
	float collideDist = radiusA;

	float3 force = make_float3(0.0f);

	if (dist < collideDist&&0!=dist)
	{
		float3 norm = make_float3(-relPos / dist, 0, 0);

		float3 relVel = velB - velA;
		float3 tanVel = relVel - (dot(relVel, norm) * norm);
		force = -0.5f*(collideDist - dist) * norm;
		force += 0.01f*relVel;
		force += 0.02f*tanVel;
		//force = make_float3(10, 10, 10);

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
	float *newDen, float *newPre, float *oldDen, float *oldPre, float4* oldPos, float3* buoyancyForce)
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
	//float den0 = newDen[originalIndex];
	//pre0 = (den0 - den)*0.0668f*2.0f;
	float4 tempb = -buoyancySpheres(pos, params.buoyancyPOS, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.buoyancyRadius, newDen[originalIndex]);
	buoyancyForce[originalIndex].x = tempb.x;
	buoyancyForce[originalIndex].y = -tempb.w;
	buoyancyForce[originalIndex].z = tempb.z;
	force.x += -tempb.x;
	force.y += -tempb.y;
	force.z += -tempb.z;

	force += collideBoard(pos, params.colliderboard, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius);

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

__global__ void plusBuoyancyforceD(float3 *buoyancyForce,  uint numParticles){
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;
	buoyancyForce[index] = make_float3(0.0f, 0.0f, 0.0f);
}

//__device__ float caculateColorfieldSecond(float3 posA, float3 posB, float radiusA, float radiusB)   
//{
//	float3 relPos = posB - posA;
//
//	float dist = length(relPos);
//	float colorf0 = 0.0f;
//	if (dist <= 0.05f || dist >= 1.5f){
//
//	}
//	else{
//		float temp = 2.25f - dist*dist;
//		
//		colorf0 = 7.64f*temp*temp*dist;
//	}
//	return colorf0;
//
//}
//
//__device__ float caculateColorfieldFirst(int3 gridPos, uint index, float3 pos,  float4 *oldPos,uint *cellStart, uint *cellEnd,
//	float colorfield, float *oldcolorfield)
//{
//	uint gridHash = calcGridHash(gridPos);
//
//	uint startIndex = FETCH(cellStart, gridHash);
//
//	float colorf0 = 0.0f;//new
//
//	if (startIndex != 0xffffffff)
//	{
//
//		uint endIndex = FETCH(cellEnd, gridHash);
//
//		for (uint j = startIndex; j<endIndex; j++)
//		{
//			if (j != index)
//			{
//				float3 pos2 = make_float3(FETCH(oldPos, j));
//				//float colorf2 = FETCH(oldcolorfield, j);
//				colorf0 += caculateColorfieldSecond(pos, pos2, params.particleRadius, params.particleRadius);
//			}
//		}
//	}
//	return colorf0;
//}

__global__ void colorfieldD(float4 *sortedPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint numParticles, float *colorfield, float *sortedcolorfield){
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	//float3 pos = make_float3(FETCH(sortedPos, index));
	//float colorf = FETCH(colorfield, index);
	////int3 gridPos = calcGridPos(pos);
	//float colorf0 = 0.0f;
	//for (int z = -1; z <= 1; z++)
	//{
	//	for (int y = -1; y <= 1; y++)
	//	{
	//		for (int x = -1; x <= 1; x++)
	//		{
	//			//int3 neighbourPos = gridPos + make_int3(x, y, z);
	//			//colorf0 += caculateColorfieldFirst(neighbourPos, index, pos, sortedPos, cellStart, cellEnd, colorf, colorfield);
	//		}
	//	}
	//}

	//uint originalIndex = gridParticleIndex[index];
	//colorfield[originalIndex] = colorf0;
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

#endif
