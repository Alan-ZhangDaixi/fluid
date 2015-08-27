#include "FluidSystem.h"
#include "FluidSystem.cuh"
#include "Kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

FluidSystem::FluidSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_hPos(0),
	h_verPos(0),
    m_hVel(0),
	m_hDen(0),//new
	m_hPre(0),//new
	m_hColorf(0),
	h_surfacePos(0),
	h_fixparticlePos(0),
    m_dPos(0),
    m_dVel(0),
	m_dDen(0),//new
	m_dPre(0),//new
	m_dColorf(0),
	d_surfacePos(0),
	d_surfaceNormal(0),
	d_fixparticlePos(0),
	d_verPos(0),
    m_gridSize(gridSize),
	m_doDepthSort(false),
    m_timer(NULL)
{
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;

    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

    //m_params.particleRadius = 1.0f / 64.0f;
	m_params.particleRadius = 1.0f / 2.0f;


    m_params.worldOrigin = make_float3(-32.0f, -32.0f, -32.0f);
    float cellSize = m_params.particleRadius * 2.0f;  
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring = 0.7f;

    m_params.boundaryDamping = -0.5f;

    m_params.gravity = make_float3(0.0f, -0.0192f, 0.0f);

	m_params.colliderPOS = make_float3(0.0f, 32.0f, 0.0f);
	m_params.colliderRadius = 8.0f;

	m_params.colliderSphere = make_float3(-32.0f, 32.0f, 0.0f);
	m_params.SphereRadius = 6.0f;

	step0 = 1;

	initMC();

    _initialize(numParticles);

}

FluidSystem::~FluidSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint FluidSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
	/*glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);*/
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

void colorRamp(float t1,float t2,float t3, float *r)
{
	r[0] = t1;
	r[1] = t2;
	r[2] = t3;
}

void FluidSystem::initMC(){          //////////////////////////////// initialize marching cubes
	//gridSizeLog2 = make_uint3(5,5,5);
	//cube_gridSize = make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
	cube_gridSize = make_uint3(256,256,256);
	//cube_gridSizeMask = make_uint3(cube_gridSize.x - 1, cube_gridSize.y - 1, cube_gridSize.z - 1);
	cube_gridSizeMask = make_uint3(255,255,255);
	//cube_gridSizeShift = make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);
	cube_gridSizeShift = make_uint3(0, 8, 16);

	numVoxels = cube_gridSize.x*cube_gridSize.y*cube_gridSize.z;
	//voxelSize = make_float3(2.0f / cube_gridSize.x, 2.0f / cube_gridSize.y, 2.0f / cube_gridSize.z);
	voxelSize = make_float3(0.25f , 0.25f, 0.25f);
	maxVerts = cube_gridSize.x*cube_gridSize.y * 500;
	
	printf("grid: %d x %d x %d = %d voxels\n", cube_gridSize.x, cube_gridSize.y, cube_gridSize.z, numVoxels);
	printf("max verts = %d\n", maxVerts);

	h_surfacePos = new float[numVoxels];
	h_verPos = new float[maxVerts];

	//h_fixparticlePos = new float[numVoxels];
	memset(h_surfacePos, 0, numVoxels *sizeof(float));
	memset(h_verPos, 0, maxVerts *sizeof(float));

	//memset(h_fixparticlePos, 0, numVoxels *sizeof(float));

	allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

	//m_marchingcubecolorVBO = createVBO(maxVerts);
	//registerGLBufferObject(m_marchingcubecolorVBO, &m_cuda_colorvbo_resource);

	unsigned int memSize = sizeof(uint) * numVoxels;
	checkCudaErrors(cudaMalloc((void **)&d_voxelVerts, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelVertsScan, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelOccupied, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelOccupiedScan, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_compVoxelArray, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_surfacePos, sizeof(float) * numVoxels));
	checkCudaErrors(cudaMalloc((void **)&d_surfaceNormal, sizeof(float) * numVoxels*3));
	checkCudaErrors(cudaMalloc((void **)&d_verPos, sizeof(float) * maxVerts));
	checkCudaErrors(cudaMalloc((void **)&metaballtable, sizeof(float) * 512));
	checkCudaErrors(cudaMalloc((void **)&normaltable, sizeof(float) * 512*3));
	//checkCudaErrors(cudaMalloc((void **)&d_fixparticlePos, sizeof(float) * numVoxels ));


	{
		build_metaballtable(1, 1, metaballtable, normaltable);
	}
}

void FluidSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
	m_hPos = new float[m_numParticles * 4];
	m_hVel = new float[m_numParticles * 4];
	m_hDen = new float[m_numParticles];//new just need 1 float to store density and pressure
	m_hPre = new float[m_numParticles];//new
	m_hColorf = new float[m_numParticles];//new

	memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
	memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));
	memset(m_hDen, 0, m_numParticles * sizeof(float));//new just need 1 float to store density and pressure
	memset(m_hPre, 0, m_numParticles * sizeof(float));//new
	memset(m_hColorf, 0, m_numParticles * sizeof(float));//new

	m_sortKeys.alloc(m_numParticles);               
	m_indices.alloc(m_numParticles, true, false, true);  //create as index buffer ,to sort

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

	// allocate GPU data
	unsigned int memSize = sizeof(float) * 4 * m_numParticles;
	unsigned int memSizetwo = sizeof(float) * m_numParticles;

	if (m_bUseOpenGL)
    {
        m_posVbo = createVBO(memSize);
		m_velVBO = createVBO(memSize);
		//m_marchingcubeposVbo = createVBO(maxVerts*sizeof(float) * 4);
		//m_normalVbo = createVBO(maxVerts*sizeof(float) * 4);
		m_marchingcubeposVbo = createVBO(6553600*2*sizeof(float) * 4);
		m_normalVbo = createVBO(6553600*2*sizeof(float) * 4);
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
		registerGLBufferObject(m_velVBO, &m_cuda_velvbo_resource);
		registerGLBufferObject(m_marchingcubeposVbo, &m_cuda_marchcubeposvbo_resource);
		registerGLBufferObject(m_normalVbo, &cuda_normalvbo_resource);

    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize)) ;
		checkCudaErrors(cudaMalloc((void **)&m_cudaVelVBO, memSize));
    }

    allocateArray((void **)&m_dVel, memSize);
	allocateArray((void **)&m_dDen, memSizetwo);
	allocateArray((void **)&m_dPre, memSizetwo);
	allocateArray((void **)&m_dColorf, memSizetwo);

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);
	allocateArray((void **)&m_dSortedDen, memSizetwo);
	allocateArray((void **)&m_dSortedPre, memSizetwo);
	allocateArray((void **)&m_dSortedColorf, memSizetwo);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

    if (m_bUseOpenGL)
    {
        //m_colorVBO = createVBO(m_numParticles*4*sizeof(float));
		m_colorVBO = createVBO(memSize);
        registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;

		for (uint i = 0; i<m_numParticles; i++)
        {
			if (i < m_numParticles){
				float t = i / (float)m_numParticles;

				colorRamp(0, 0, 1, ptr);
				ptr += 3;
				*ptr++ = 1.0f;
			}
			/*else if (250000 <= i&&i < 450000){
				float t = i / (float)m_numParticles;

				colorRamp(0, 1, 0, ptr);
				ptr += 3;
				*ptr++ = 0.5f;
			}*/
			/*else{
				float t = i / (float)m_numParticles;

				colorRamp(1, 0, 0, ptr);
				ptr += 3;
				*ptr++ = 1.0f;
			}*/
        }

        glUnmapBufferARB(GL_ARRAY_BUFFER);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaColorVBO, sizeof(float)*numParticles*4));
    }

    sdkCreateTimer(&m_timer);

    setParameters(&m_params);

    m_bInitialized = true;
}

void FluidSystem::_finalize()
{
    assert(m_bInitialized);

	delete[] m_hPos;
	delete[] m_hVel;
	delete[] m_hDen;
	delete[] m_hPre;
	delete[] m_hColorf;

	delete[] m_hCellStart;
	delete[] m_hCellEnd;

    freeArray(m_dVel);
	freeArray(m_dDen);
	freeArray(m_dPre);
	freeArray(m_dColorf);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);
	freeArray(m_dSortedDen);
	freeArray(m_dSortedPre);
	freeArray(m_dSortedColorf);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);
	/////////////////////////////////////////////////////////////////Marching cube
	delete[] h_surfacePos;
	delete[] h_verPos;
	//delete[] h_fixparticlePos;

	checkCudaErrors(cudaFree(d_edgeTable));
	checkCudaErrors(cudaFree(d_triTable));
	checkCudaErrors(cudaFree(d_numVertsTable));

	checkCudaErrors(cudaFree(d_voxelVerts));
	checkCudaErrors(cudaFree(d_voxelVertsScan));
	checkCudaErrors(cudaFree(d_voxelOccupied));
	checkCudaErrors(cudaFree(d_voxelOccupiedScan));
	checkCudaErrors(cudaFree(d_compVoxelArray));
	checkCudaErrors(cudaFree(d_surfacePos));
	checkCudaErrors(cudaFree(d_surfaceNormal));
	checkCudaErrors(cudaFree(d_verPos));
	checkCudaErrors(cudaFree(metaballtable));
	//checkCudaErrors(cudaFree(d_fixparticlePos));
	///////////////////////////////////////////////////////////////////
    if (m_bUseOpenGL)
    {
        unregisterGLBufferObject(m_cuda_colorvbo_resource);
        unregisterGLBufferObject(m_cuda_posvbo_resource);
		unregisterGLBufferObject(m_cuda_velvbo_resource);
		unregisterGLBufferObject(m_cuda_marchcubeposvbo_resource);
		unregisterGLBufferObject(cuda_normalvbo_resource);
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
		glDeleteBuffers(1, (const GLuint *)&m_velVBO);
		glDeleteBuffers(1, (const GLuint *)&m_marchingcubeposVbo);
		glDeleteBuffers(1, (const GLuint *)&m_normalVbo);
		//glDeleteBuffers(1, (const GLuint *)&m_marchingcubecolorVBO);
    }
    else
    {
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
		checkCudaErrors(cudaFree(m_cudaVelVBO));
    }
} 

void FluidSystem::depthSort()
{
	float *dPos;
	m_dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);

	m_indices.map();

	// calculate depth

	calcDepth(m_dPos, m_sortKeys.getDevicePtr(), m_indices.getDevicePtr(), m_sortVector, m_numParticles);

	// radix sort

	//sortParticlesKey(m_sortKeys.getDevicePtr(), m_indices.getDevicePtr(), m_numParticles);

	unmapGLBufferObject(m_cuda_posvbo_resource);
	m_indices.unmap();
}

void FluidSystem::update(float deltaTime) //整个particle或流体的update
{
    assert(m_bInitialized);

    float *dPos;
    if (m_bUseOpenGL)
    {
		m_dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
		m_dVel = (float *)mapGLBufferObject(&m_cuda_velvbo_resource);

    }
    else
    {
		m_dPos = (float *)m_cudaPosVBO;
		m_dVel = (float *)m_cudaVelVBO;
    }

	{
		int threads = 1024;
		dim3 grid(numVoxels / threads, 1, 1);

		if (grid.x > 65535)
		{
			grid.y = grid.x / 32768;
			grid.x = 32768;
		}

		build_surfacePosition(grid, threads, d_surfacePos,d_surfaceNormal);
		//build_surfacePosition(grid, threads, d_surfacePos, NULL);
	}

    setParameters(&m_params);


	//if (step0 == 0){
		integrateSystem(
			m_dPos,
			m_dVel,
			m_dDen,
			m_dPre,
			deltaTime,
			m_numParticles);
		//step0 = 1;
	//}
		
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
		m_dPos,
        m_numParticles,
		m_dDen,
		d_surfacePos,
		d_surfaceNormal,
		metaballtable,
		normaltable,
		4);//this is the pow of metaball

    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleHash,
        m_dGridParticleIndex,
		m_dPos,
        m_dVel,
        m_numParticles,
        m_numGridCells,
		m_dSortedDen,
		m_dSortedPre,
		m_dDen,   //old density
		m_dPre,//old pressure
		m_dSortedColorf,
		m_dColorf);



	/*colorfield(
		m_dSortedPos,
		m_dGridParticleIndex,
		m_dCellStart,
		m_dCellEnd,
		m_numParticles,
		m_numGridCells,
		m_dColorf,
		m_dSortedColorf);*/
	//
	copyArrayFromDevice(m_hDen, m_dDen, 0, sizeof(float) * m_numParticles);


	/*glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
	float *data = (float *)glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;

	for (uint i = 0; i<m_numParticles; i++)
	{
		float t = i / (float)m_numParticles;
		if (  m_hDen[i]<2.5f){
			colorRamp(1, 1, 1, ptr);
			ptr += 3;
			*ptr++ = 0.1f;
		}
		else
		{
			colorRamp(0, 1, 1, ptr);
			ptr += 3;
			*ptr++ = 0.1f;
		}

	}

	glUnmapBufferARB(GL_ARRAY_BUFFER);*/


	calculateDensity(m_dSortedPos,
		m_dGridParticleIndex,
		m_dCellStart,
		m_dCellEnd,
		m_numParticles,
		m_dDen);

    collide(
        m_dVel, //new velocity
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells,
		m_dDen, //new density
		m_dPre, //new pressure
		m_dSortedDen,
		m_dSortedPre,
		m_dPos);

	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	// Start record
	cudaEventRecord(start1, 0);
	calcMarchingcubeandmetaball(
		m_dGridParticleHash,
		m_dGridParticleIndex,
		m_dPos,
		m_numParticles,
		m_dDen,
		d_surfacePos,
		d_surfaceNormal,
		metaballtable,
		normaltable,
		4,
		5);
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	float elapsedTime1;
	cudaEventElapsedTime(&elapsedTime1, start1, stop1); // that's our time!

	std::cout << "neighbour " << elapsedTime1 << endl;

	// Clean up:
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);

	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	// Start record
	cudaEventRecord(start2, 0);

	computeIsosurface();

	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);
	float elapsedTime2;
	cudaEventElapsedTime(&elapsedTime2, start2, stop2); // that's our time!

	std::cout << "confirm " << elapsedTime2 << endl;

	// Clean up:
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);

    if (m_bUseOpenGL)
    {
        unmapGLBufferObject(m_cuda_posvbo_resource);
		unmapGLBufferObject(m_cuda_velvbo_resource);
    }
}

///////////////////////////////////////////////////////////////Marching cube main function

void FluidSystem::computeIsosurface(){

	//int threads = 128;
	int threads = 1024;
	dim3 grid(numVoxels / threads, 1, 1);

	if (grid.x > 65535)
	{
		grid.y = grid.x / 32768;
		grid.x = 32768;
	}

	//build_surfacePosition(grid, threads, d_surfacePos);
	// 0.015625f,0.00025f,0.0009766,0.00006104,0.000015625
	launch_classifyVoxel(grid, threads,
		d_voxelVerts, d_voxelOccupied, d_surfacePos,
		cube_gridSize, cube_gridSizeShift, cube_gridSizeMask,
		numVoxels, voxelSize, 0.0005f, d_surfaceNormal);

	ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelOccupied + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelOccupiedScan + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		activeVoxels = lastElement + lastScanElement;               //为什么需要lastElement，因为lastScanElement是数组所有元素的和，但是不包括最后一个，所以要把最后一个加上
		//std::cout << lastElement << " " << lastScanElement<<" ";
		//std::cout << activeVoxels << " ";

	}
	if (activeVoxels == 0)
	{
		// return if there are no full voxels
		totalVerts = 0;
		return;
	}

	launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);

	ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);

	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelVerts + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelVertsScan + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		totalVerts = lastElement + lastScanElement;
		std::cout << totalVerts << " "<<endl;
	}

	float* d_verPos = (float *)mapGLBufferObject(&m_cuda_marchcubeposvbo_resource);
	float* d_normal = (float *)mapGLBufferObject(&cuda_normalvbo_resource);

	dim3 grid2((int)ceil(activeVoxels / (float)128.0f), 1, 1);

	while (grid2.x > 65535)
	{
		grid2.x /= 2;
		grid2.y *= 2;
	}

	launch_generateTriangles(grid2,128.0f, d_verPos, d_normal,
		d_compVoxelArray,
		d_voxelVertsScan, d_surfacePos,
		cube_gridSize, cube_gridSizeShift, cube_gridSizeMask,
		voxelSize, 0.0005f, activeVoxels,
		maxVerts,d_surfaceNormal);

	//Normalize(grid2, 32.0f, d_normal, d_voxelVertsScan);


	unmapGLBufferObject(m_cuda_marchcubeposvbo_resource);
	unmapGLBufferObject(cuda_normalvbo_resource);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

float *FluidSystem::getArray(FluidArray array) //复制出particle的位置或者速度array，根据传进去的array是什么，可以是POSITION和VELOCITY
{
    assert(m_bInitialized);

    float *hdata = 0;
    float *ddata = 0;
    struct cudaGraphicsResource *cuda_vbo_resource = 0;

    switch (array)
    {
        default:
        case POSITION:
            hdata = m_hPos;
            ddata = m_dPos;
            cuda_vbo_resource = m_cuda_posvbo_resource;
            break;

        case VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel;
            break;
		case DENSITY:
			hdata = m_hDen;
			ddata = m_dDen;
			break;
    }

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
    return hdata;
}

void FluidSystem::setArray(FluidArray array, const float *data, int start, int count) //重新设置particle的速度和位置数组
{
    assert(m_bInitialized);

    switch (array)
    {
        default:
        case POSITION:
            {
                if (m_bUseOpenGL)
                {
                    unregisterGLBufferObject(m_cuda_posvbo_resource);
                    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
                    glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                    registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
                }
            }
            break;

        case VELOCITY:
			unregisterGLBufferObject(m_cuda_velvbo_resource);
			glBindBuffer(GL_ARRAY_BUFFER, m_velVBO);
			glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float), count * 4 * sizeof(float), data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			registerGLBufferObject(m_velVBO, &m_cuda_velvbo_resource);

			break;
		case DENSITY:
			copyArrayToDevice(m_dDen, data, start * sizeof(float), count * sizeof(float));
			break;
		case PRESSURE:
			copyArrayToDevice(m_dPre, data, start *  sizeof(float), count *  sizeof(float));
			break;
		case COLORFIELD:
			copyArrayToDevice(m_dColorf, data, start * sizeof(float), count * sizeof(float));
			break;
	}
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void FluidSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles) //重新随机设置位置数组,通过frand（）函数，并把速度数组所有参数设为0
{
    srand(3000);

    for (uint z=0; z<size[2]; z++)
    {
        for (uint y=0; y<size[1]; y++)
        {
            for (uint x=0; x<size[0]; x++)
            {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;

                if (i < numParticles)
                {
                    m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 14.0f + (frand()*2.0f-1.0f)*jitter;
					m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius - 14.0f + (frand()*2.0f-1.0f)*jitter;
					m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius - 14.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+3] = 1.0f;

                    m_hVel[i*4] = 0.0f;
                    m_hVel[i*4+1] = 0.0f;
                    m_hVel[i*4+2] = 0.0f;
                    m_hVel[i*4+3] = 0.0f;

					m_hDen[i] = 0.0f;
					m_hPre[i] = 0.0f;
					m_hColorf[i] = 0.0f;
                }
            }
        }
    }

	/*m_hPos[(numParticles - 1) * 4] = 20.0f;
	m_hPos[(numParticles - 1) * 4 + 1] = 0.0f;
	m_hPos[(numParticles - 1) * 4 + 2] = 0.0f;
	m_hPos[(numParticles - 1) * 4 + 3] = 1.0f;

	m_hVel[(numParticles - 1) * 4] = 0.0f;
	m_hVel[(numParticles - 1) * 4 + 1] = 0.0f;
	m_hVel[(numParticles - 1) * 4 + 2] = 0.0f;
	m_hVel[(numParticles - 1) * 4 + 3] = 100.0f;*/
}

void FluidSystem::dumpParticles(uint start, uint count)  //打印出所有particle的坐标和速度
{
	// debug
	//copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float) * 4 * count);
	copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float) * 4 * count);
	copyArrayFromDevice(m_hDen, m_dDen, 0, sizeof(float) * count);
	copyArrayFromDevice(m_hPre, m_dPre, 0, sizeof(float) * count);
	copyArrayFromDevice(m_hColorf, m_dColorf, 0, sizeof(float) * count);
	//copyArrayFromDevice(h_surfacePos, d_fixparticlePos, 0, sizeof(float)*totalVerts);
	//copyArrayFromDevice(h_verPos, 0, &m_cuda_marchcubeposvbo_resource, sizeof(float) * 4 * totalVerts);
	//copyArrayFromDevice(h_fixparticlePos, d_surfacePos, 0, sizeof(float) * 800);
	//d_fixparticlePos
	/*for (uint i = start; i < start + totalVerts; i++){
		printf("h_verPos: (%.4f, %.4f, %.4f, %.4f)\n", h_verPos[i * 4 + 0], h_verPos[i * 4 + 1], h_verPos[i * 4 + 2], h_verPos[i * 4 + 3]);
		}*/
	for (uint i = 0; i < totalVerts; i++)
	//for (uint i = 0; i < 8000; i++)
	{
		if (h_surfacePos[i]>0){
			cout << "h_fixparticlePos:          " << h_surfacePos[i] << endl;
		}
		//if (h_surfacePos[i]>0){
		//	printf("%d: ", i);
		//	//printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i * 4 + 0], m_hPos[i * 4 + 1], m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
		//	//printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i * 4 + 0], m_hVel[i * 4 + 1], m_hVel[i * 4 + 2], m_hVel[i * 4 + 3]);
		//	//cout << "den:          " << m_hDen[i]<<endl;
		//	//cout << "pre:          " << m_hPre[i] << endl;
		//	//cout << "colorfield:          " << m_hColorf[i] << endl;
		//	cout << "h_surfacePos:          " << h_surfacePos[i] << endl;
		//	//printf("den: (%.7f)\n", m_hDen[i]);
		//}
	}
}

void FluidSystem::reset(FluidConfig config) //回到初始状态
{
	uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
	float jitter = m_params.particleRadius*0.01f;
	switch (config)
	{
	default:
	case CONFIG_RANDOM:
	{
		for (uint z = 0; z < s; z++)
		{
			for (uint y = 0; y < s; y++)
			{
				for (uint x = 0; x < s; x++)
				{
					uint i = (z*s * s) + (y*s) + x;

					if (i < m_numParticles)
					{
						m_hPos[i * 4] = (3 * x) + m_params.particleRadius - 32.0f /*+ (frand()*2.0f-1.0f)*jitter*/;
						m_hPos[i * 4 + 1] = (3 * y) + m_params.particleRadius - 32.0f /*+ (frand()*2.0f-1.0f)*jitter*/;
						m_hPos[i * 4 + 2] = (3 * z) + m_params.particleRadius - 32.0f /*+ (frand()*2.0f-1.0f)*jitter*/;
						m_hPos[i * 4 + 3] = 1.0f;

						m_hVel[i * 4] = 0.0f;
						m_hVel[i * 4 + 1] = 0.0f;
						m_hVel[i * 4 + 2] = 0.0f;
						m_hVel[i * 4 + 3] = 0.0f;

						m_hDen[i] = 0.0f;
						m_hPre[i] = 0.0f;
						m_hColorf[i] = 0.0f;
					}
				}
			}
		}
	}
	break;

	case CONFIG_GRID:
	{
		uint gridSize[3];
		gridSize[0] = gridSize[1] = gridSize[2] = s;
		initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_numParticles);
	}
	break;
	}

	setArray(POSITION, m_hPos, 0, m_numParticles);
	setArray(VELOCITY, m_hVel, 0, m_numParticles);
	setArray(DENSITY, m_hDen, 0, m_numParticles);
	setArray(PRESSURE, m_hPre, 0, m_numParticles);
	setArray(COLORFIELD, m_hColorf, 0, m_numParticles);
}

void FluidSystem::dumpBin(float4 **posData,
	float4 **velData)
{
	/*m_pos.copy(GpuArray<float4>::DEVICE_TO_HOST);
	*posData = m_pos.getHostPtr();

	m_vel.copy(GpuArray<float4>::DEVICE_TO_HOST);
	*velData = m_vel.getHostPtr();*/
}

void FluidSystem::ShootSphere(){

}