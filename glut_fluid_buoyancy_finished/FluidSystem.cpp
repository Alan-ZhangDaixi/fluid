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
    m_hVel(0),
	m_hDen(0),//new
	m_hPre(0),//new
	m_hColorf(0),
    m_dPos(0),
    m_dVel(0),
	m_dDen(0),//new
	m_dPre(0),//new
	m_dColorf(0),
	buoyancyForce(0),
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

	m_params.colliderPOS = make_float3(0.0f, 320.0f, 0.0f);
	m_params.colliderRadius = 8.0f;

	m_params.colliderboard = make_float3(-1600.0f, 0.0f, 0.0f);
	//m_params.colliderSphere = make_float3(-32.0f, 32.0f, 0.0f);
	//m_params.SphereRadius = 6.0f;

	m_params.buoyancyPOS = make_float3(1000.0f, 60.0f, 0.0f);
	m_params.buoyancyRadius = 8.0f;
	tempvel = make_float3(0,0,0);
	buoyancycount = 0;
	storefactor = 0.0;
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
	buoyancyPos = new float[m_numParticles*3];

	memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
	memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));
	memset(m_hDen, 0, m_numParticles * sizeof(float));//new just need 1 float to store density and pressure
	memset(m_hPre, 0, m_numParticles * sizeof(float));//new
	memset(m_hColorf, 0, m_numParticles * sizeof(float));//new
	memset(buoyancyPos, 0, m_numParticles*3 * sizeof(float));

	m_sortKeys.alloc(m_numParticles);               
	m_indices.alloc(m_numParticles, true, false, true);  //create as index buffer ,to sort

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

	// allocate GPU data
	unsigned int memSize = sizeof(float) * 4 * m_numParticles;
	unsigned int memSizetwo = sizeof(float) * m_numParticles;
	unsigned int memSizethree = sizeof(float) *3* m_numParticles;

	if (m_bUseOpenGL)
    {
        m_posVbo = createVBO(memSize);
		m_velVBO = createVBO(memSize);
		//obstaclePosVbo[0] = createVBO(memSize);
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
		registerGLBufferObject(m_velVBO, &m_cuda_velvbo_resource);
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
	allocateArray((void **)&buoyancyForce, memSizethree);

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

        for (uint i=0; i<m_numParticles; i++)
        {
			if (i < 450000){
				float t = i / (float)m_numParticles;

				colorRamp(0.1, 0.5, 0.7, ptr);
				ptr += 3;
				*ptr++ = 0.5f;
			}
			/*else if (250000 <= i&&i < 450000){
				float t = i / (float)m_numParticles;

				colorRamp(0, 1, 0, ptr);
				ptr += 3;
				*ptr++ = 0.5f;
			}*/
			else{
				float t = i / (float)m_numParticles;

				colorRamp(0, 0, 0, ptr);
				ptr += 3;
				*ptr++ = 1.0f;
			}
        }

        glUnmapBufferARB(GL_ARRAY_BUFFER);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaColorVBO, sizeof(float)*numParticles*4));
    }

	plusBuoyancyforce(buoyancyForce, m_numParticles);//initialize this value to 0;

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
	freeArray(buoyancyForce);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);
	freeArray(m_dSortedDen);
	freeArray(m_dSortedPre);
	freeArray(m_dSortedColorf);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    if (m_bUseOpenGL)
    {
        unregisterGLBufferObject(m_cuda_colorvbo_resource);
        unregisterGLBufferObject(m_cuda_posvbo_resource);
		unregisterGLBufferObject(m_cuda_velvbo_resource);
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
		glDeleteBuffers(1, (const GLuint *)&m_velVBO);
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
/////////////////////////////////////////////////////calculate time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);
///////////////////////////////////////////////////////////////////////

    setParameters(&m_params);

    integrateSystem(
		m_dPos,
        m_dVel,
		m_dDen,
		m_dPre,
        deltaTime,
        m_numParticles,
		buoyancyForce);

	//calculateBuoyancy();

	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	// Start record
	cudaEventRecord(start1, 0);

    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
		m_dPos,
        m_numParticles);

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

	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	float elapsedTime1;
	cudaEventElapsedTime(&elapsedTime1, start1, stop1); // that's our time!

	std::cout <<"neighbour " <<elapsedTime1 << endl;

	// Clean up:
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);

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


	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	// Start record
	cudaEventRecord(start2, 0);

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
		m_dPos,
		buoyancyForce);

	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);
	float elapsedTime2;
	cudaEventElapsedTime(&elapsedTime2, start2, stop2); // that's our time!

	std::cout << "renew " << elapsedTime2 << endl;

	// Clean up:
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);
	////////////////////////////////////////////////////////////stop calculate time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!

	std::cout << "whole " << elapsedTime << endl;

	// Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//////////////////////////////////////////////////////////

    if (m_bUseOpenGL)
    {
        unmapGLBufferObject(m_cuda_posvbo_resource);
		unmapGLBufferObject(m_cuda_velvbo_resource);
    }
}

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
                    m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 30.0f + (frand()*2.0f-1.0f)*jitter;
					m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius - 31.0f + (frand()*2.0f-1.0f)*jitter;
					m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius - 30.0f + (frand()*2.0f-1.0f)*jitter;
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

	/*m_hPos[(numParticles - 1) * 4] = 0.0f;
	m_hPos[(numParticles - 1) * 4 + 1] = 360.0f;
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
	for (uint i = start; i < start + count; i++)
	{
		printf("%d: ", i);
		//printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i * 4 + 0], m_hPos[i * 4 + 1], m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
		printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i * 4 + 0], m_hVel[i * 4 + 1], m_hVel[i * 4 + 2], m_hVel[i * 4 + 3]);
		cout << "den:          " << m_hDen[i]<<endl;
		cout << "pre:          " << m_hPre[i] << endl;
		cout << "colorfield:          " << m_hColorf[i] << endl;
		//printf("den: (%.7f)\n", m_hDen[i]);
		
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
						m_hPos[i * 4] = (3 * x) + m_params.particleRadius - 128.0f /*+ (frand()*2.0f-1.0f)*jitter*/;
						m_hPos[i * 4 + 1] = (3 * y) + m_params.particleRadius - 31.0f /*+ (frand()*2.0f-1.0f)*jitter*/;
						m_hPos[i * 4 + 2] = (3 * z) + m_params.particleRadius - 64.0f /*+ (frand()*2.0f-1.0f)*jitter*/;
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

void FluidSystem::calculateBuoyancy(){
	buoyancycount = 0;
	float buoyancyforce =0;  //this is a value to store each frame buoyancy force of the sphere
	float nearsphereCount1 = 0;
	copyArrayFromDevice(buoyancyPos, buoyancyForce, 0, sizeof(float) * 3*m_numParticles);     //buoyancyForce is a cuda value to store all the particles which is surface particle and 
	                                                                                          //near buoyancy sphere and store its position
	//std::cout << storefactor << endl;                                                                              
	for (int i = 0; i < m_numParticles; i++){  //sum all the force upon buoyancy sphere
		if (buoyancyPos[i * 3] != 0){
			nearsphereCount1++;
		}
		buoyancyPos[0] += buoyancyPos[i * 3];
		if (buoyancyPos[i * 3 + 1] >0.0){
			buoyancyPos[1] += buoyancyPos[i * 3 + 1];
			buoyancycount++;
		}
		else{
			buoyancyPos[1] += 0;
		}

		buoyancyPos[2] += buoyancyPos[i * 3 + 2];
	}
	//std::cout << nearsphereCount1  << endl;
	if (buoyancycount != 0){
		float surfaceheight = buoyancyPos[1] / (float)buoyancycount - 32.0;
		float minusheight = m_params.buoyancyPOS.y + m_params.buoyancyRadius - surfaceheight;
		float factor = (m_params.buoyancyRadius * 2 - minusheight)/8.0f;
		storefactor = factor;
		if (factor < 1){
			factor = factor*factor;   //y = x*x
		}
		if (1.0<=factor&&factor<2){
			factor = -factor*factor + 4*factor -2;   //y=-x*x+4*x-2
		}
		buoyancyforce = factor*0.0192f*0.5; 
		//std::cout << buoyancyPos[1] / (float)buoyancycount - 32.0 << endl;
		tempvel.y *= 0.99;
		tempvel.y += buoyancyforce;
	}
	//if (buoyancycount == 0 && storefactor > 1.75){
	if (buoyancycount == 0 && nearsphereCount1 > 300){
		buoyancyforce = 2.0f*0.0192f*0.5;
		tempvel.y *= 0.99;
		tempvel.y += buoyancyforce;
	}
	tempvel.y += -0.0192f*0.5;
	if (nearsphereCount1 != 0){
		tempvel.x += buoyancyPos[0] / nearsphereCount1;
		tempvel.z += buoyancyPos[2] / nearsphereCount1;
	}
	//std::cout << m_params.buoyancyPOS.x << " " << m_params.buoyancyPOS.y << " " << m_params.buoyancyPOS.z << " " << endl;
	m_params.buoyancyPOS.x += tempvel.x*0.5;
	m_params.buoyancyPOS.y += tempvel.y*0.5;
	m_params.buoyancyPOS.z += tempvel.z*0.5;


	if (m_params.buoyancyPOS.x > 96.0f - m_params.buoyancyRadius)
	{
		m_params.buoyancyPOS.x = 96.0f - m_params.buoyancyRadius;
		tempvel.x *= m_params.boundaryDamping;
	}

	if (m_params.buoyancyPOS.x < -96.0f + m_params.buoyancyRadius)
	{
		m_params.buoyancyPOS.x = -96.0f + m_params.buoyancyRadius;
		tempvel.x *= m_params.boundaryDamping;
	}

	if (m_params.buoyancyPOS.y > 64.0f - m_params.buoyancyRadius)
	{
		m_params.buoyancyPOS.y = 64.0f - m_params.buoyancyRadius;
		tempvel.y *= m_params.boundaryDamping;
	}

	if (m_params.buoyancyPOS.z > 48.0f - m_params.buoyancyRadius)
	{
		m_params.buoyancyPOS.z = 48.0f - m_params.buoyancyRadius;
		tempvel.z *= m_params.boundaryDamping;
	}

	if (m_params.buoyancyPOS.z < -48.0f + m_params.buoyancyRadius)
	{
		m_params.buoyancyPOS.z = -48.0f + m_params.buoyancyRadius;
		tempvel.z *= m_params.boundaryDamping;
	}

	if (m_params.buoyancyPOS.y < -31.0f + m_params.buoyancyRadius)
	{
		m_params.buoyancyPOS.y = -31.0f + m_params.buoyancyRadius;
		tempvel.y *= m_params.boundaryDamping;
	}

	plusBuoyancyforce(buoyancyForce, m_numParticles);
}