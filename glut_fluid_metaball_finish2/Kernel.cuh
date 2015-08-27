#ifndef _KERNEL_H
#define _KERNEL_H

#define FETCH(t, i) t[i]

#include "vector_types.h"
typedef unsigned int uint;

struct KernelParams
{
    float3 gravity;
    float particleRadius;

    uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;

    uint numBodies;

    float spring;
    float boundaryDamping;

	float3 colliderPOS;
	float colliderRadius;
	float3 colliderSphere;
	float SphereRadius;
};

#endif
