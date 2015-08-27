#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "render.h"
#include "shaders.h"
#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif
Renderer::Renderer()
	: m_pos(0),
	m_numParticles(0),
	m_pointSize(1.0f),
	m_particleRadius(0.125f * 0.5f),
	//m_program(0),
	m_vbo(0),
	m_colorVBO(0),
	mWindowW(1024),
	mWindowH(768),
	mFov(60.0f),
	m_downSample(1),
	m_imageTex(0),
	m_postprocessingTex(0),
	m_depthTex(0),
	m_imageFbo(0),
	m_postprocessingFbo(0)
	  //m_indexBuffer(0)
{
	txParticle = LoadTexture("data/water.bmp");
	m_displayTexProg = new GLSLProgram(passThruVS, texture2DPS);
	m_postprocessing = new GLSLProgram(passThruVS, postprocessingPS);
    _initGL();
}

Renderer::~Renderer()
{
    m_pos = 0;
	delete m_imageFbo;
	delete m_postprocessingFbo;
	delete m_displayTexProg;
	delete m_postprocessing;
	glDeleteTextures(1, &m_imageTex);
	glDeleteTextures(1, &m_depthTex);
	glDeleteTextures(1, &m_postprocessingTex);
}

void Renderer::setPositions(float *pos, int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}

void Renderer::setVertexBuffer(unsigned int vbo, int numParticles)
{
    m_vbo = vbo;
    m_numParticles = numParticles;
}

void Renderer::_drawPoints()
{
    if (!m_vbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;

            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo);
		//glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, m_vbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);

  //      if (m_colorVBO)
  //      {
  //          glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);
		//	//glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, m_colorVBO);
  //          glColorPointer(4, GL_FLOAT, 0, 0);
		//	glEnableClientState(GL_COLOR_ARRAY);
		//}

		//if (m_velVbo)
		//{
		//	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_velVbo);
		//	glClientActiveTexture(GL_TEXTURE0);
		//	glTexCoordPointer(4, GL_FLOAT, 0, 0);
		//	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		//}
		if (m_normal)
		{
			glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_normal);
			/*glClientActiveTexture(GL_TEXTURE1);
			glTexCoordPointer(4, GL_FLOAT, 0, 0);
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);*/
			glNormalPointer(GL_FLOAT, sizeof(float) * 4, 0);
			glEnableClientState(GL_NORMAL_ARRAY);
		}

		//glDrawArrays(GL_POINTS, 0, m_numParticles);
		glDrawArrays(GL_TRIANGLES, 0, m_numParticles);
		//glDrawElements(GL_POINTS, m_numParticles, GL_UNSIGNED_INT, (void *)(0 * sizeof(unsigned int)));

        //glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
}

void Renderer::display(DisplayMode mode, GLSLProgram *prog)
{
    switch (mode)
    {
        case PARTICLE_POINTS:
            glColor3f(1, 1, 1);
            glPointSize(m_pointSize);
            _drawPoints();
            break;
		case PARTICLE_FLUID:
			glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
			glColor4f(0.5, 0.5, 0.5, 0.1);
			glEnable(GL_DEPTH_TEST);
			glDepthMask(GL_FALSE);  // don't write depth
			glEnable(GL_BLEND);
			prog->enable();
			//prog->setUniform1f("pointRadius", mParticleRadius);
			prog->setUniform1f("pointRadius", 0.5f);
			glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo);
			glVertexPointer(4, GL_FLOAT, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);

			glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, m_indexBuffer);

			glDrawElements(GL_POINTS, m_numParticles, GL_UNSIGNED_INT, (void *)(0 * sizeof(unsigned int)));
			//glDrawArrays(GL_POINTS, 0, m_numParticles);
			glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);

			prog->disable();
			glDepthMask(GL_TRUE);
			glDisable(GL_BLEND);

        default:
        case PARTICLE_SPHERES:

			//glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
			//glColor4f(0.5, 0.5, 0.5, 0.1);
			//glEnable(GL_DEPTH_TEST);
			//glDepthMask(GL_FALSE);  // don't write depth
			//glEnable(GL_BLEND);
			//prog->enable();
			////prog->setUniform1f("pointRadius", mParticleRadius);
			//prog->setUniform1f("pointRadius", 1.5f);
			//glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo);
			//glVertexPointer(4, GL_FLOAT, 0, 0);
			//glEnableClientState(GL_VERTEX_ARRAY);

			//if (m_colorVBO){
			//	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);
			//	glColorPointer(4, GL_FLOAT, 0, 0);
			//	glEnableClientState(GL_COLOR_ARRAY);
			//}

			//if (m_velVbo)
			//{
			//	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_velVbo);
			//	glClientActiveTexture(GL_TEXTURE0);
			//	glTexCoordPointer(4, GL_FLOAT, 0, 0);
			//	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
			//}

			//if (m_indexBuffer){
			//	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, m_indexBuffer);
			//	glDrawElements(GL_POINTS, m_numParticles, GL_UNSIGNED_INT, (void *)(0 * sizeof(unsigned int)));
			//	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
			//}

			//glDisableClientState(GL_VERTEX_ARRAY);
			//glDisableClientState(GL_COLOR_ARRAY);

			//prog->disable();
			//glDepthMask(GL_TRUE);
			//glDisable(GL_BLEND);



			m_imageFbo->Bind();
			glClearColor(0.0, 0.0, 0.0, 0.0);
			glClear(GL_COLOR_BUFFER_BIT);
			m_imageFbo->Disable();

			glActiveTexture(GL_TEXTURE0);
			glMatrixMode(GL_TEXTURE);

			m_imageFbo->Bind();
			glViewport(0, 0, m_imageW, m_imageH);
			glColor4f(1.0, 1.0, 1.0, 1.0);
			glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

			drawpointsprite();

			m_imageFbo->Disable();
			glActiveTexture(GL_TEXTURE0);
			glMatrixMode(GL_TEXTURE);
			glLoadIdentity();

			//drawPostProcess();
			compositeResult();


			//m_program->enable();
			//glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
			//glVertexPointer(4, GL_FLOAT, 0, 0);
			//glEnableClientState(GL_VERTEX_ARRAY);

			///*glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalVbo);
			//glNormalPointer(GL_FLOAT, sizeof(float) * 4, 0);
			//glEnableClientState(GL_NORMAL_ARRAY);*/

			//glColor3f(1.0, 0.0, 0.0);
			//glDrawArrays(GL_TRIANGLES, 0, totalVerts);
			//glDisableClientState(GL_VERTEX_ARRAY);
			//glDisableClientState(GL_NORMAL_ARRAY);

			//glBindBuffer(GL_ARRAY_BUFFER, 0);
			//m_program->disable();
   //         break;
    }
}

void Renderer::drawpointsprite(){
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	m_program->enable();
	m_program->setUniform1f("pointScale", m_window_h / tanf(m_fov*0.5f*(float)M_PI / 180.0f));
	//m_program->setUniform1f("pointRadius", m_particleRadius);
	m_program->setUniform1f("pointRadius", 0.5f);
	m_program->setUniform3f("lightPositionIn", 32.0f,128.0f,32.0f);
	//m_program->setUniform1f("near", 0.1f);
	//m_program->setUniform1f("far", 3000.0f);

	glColor4f(1, 1, 1, 0.5);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	_drawPoints();
	glDisable(GL_BLEND);

	m_program->disable();
	glDisable(GL_POINT_SPRITE_ARB);
}

void Renderer::drawPostProcess(){
	/*m_postprocessingFbo->Bind();
	glViewport(0, 0, m_imageW, m_imageH);

	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDepthMask(GL_TRUE);
	glClear(GL_DEPTH_BUFFER_BIT );
	m_postprocessingFbo->Disable();*/

	glViewport(0, 0, mWindowW, mWindowH);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

	m_postprocessingFbo->Bind();

	glActiveTexture(GL_TEXTURE0);
	glMatrixMode(GL_TEXTURE);

	m_postprocessing->enable();

	glViewport(0, 0, m_imageW, m_imageH);

	glDisable(GL_DEPTH_TEST);
	m_postprocessing->setUniform2f("pixelSize", 1.f/(m_imageW*2.f), 1.f/(m_imageH*2.f));

	if (m_velVbo)
	{
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_velVbo);
		glClientActiveTexture(GL_TEXTURE0);
		glTexCoordPointer(4, GL_FLOAT, 0, 0);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	}

	for (int i = 0; i < 50; i++){
		m_postprocessingFbo->AttachTexture(GL_TEXTURE_2D, m_postprocessingTex, GL_COLOR_ATTACHMENT0_EXT);
		m_postprocessing->setUniform1f("isvertical",0.0);
		m_postprocessing->bindTexture("tex", m_imageTex, GL_TEXTURE_2D, 0);

		drawQuad();

		m_postprocessingFbo->AttachTexture(GL_TEXTURE_2D, m_imageTex, GL_COLOR_ATTACHMENT0_EXT);		
		m_postprocessing->setUniform1f("isvertical", 1.0);
		m_postprocessing->bindTexture("tex", m_postprocessingTex, GL_TEXTURE_2D, 0);

		drawQuad();
	}

	m_postprocessingFbo->Disable();
	m_postprocessing->disable();

	glEnable(GL_DEPTH_TEST);
}

GLuint Renderer::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void Renderer::_initGL()
{
	//m_program = _compileProgram(vertexShader, spherePixelShader);

	m_program = new GLSLProgram(vertexShader, spherePixelShader);
}

GLuint Renderer::LoadTexture(const char * filename){
	GLuint texture;

	int width, height;

	unsigned char * data;

	FILE * file;

	file = fopen(filename, "rb");

	if (file == NULL) return 0;
	width = 1024;
	height = 512;
	data = (unsigned char *)malloc(width * height * 3);
	//int size = fseek(file,);
	fread(data, width * height * 3, 1, file);
	fclose(file);

	for (int i = 0; i < width * height; ++i)
	{
		int index = i * 3;
		unsigned char B, R;
		B = data[index];
		R = data[index + 2];

		data[index] = R;
		data[index + 2] = B;

	}


	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);


	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	gluBuild2DMipmaps(GL_TEXTURE_2D, 3, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);
	free(data);

	return texture;
}

void Renderer::beginSceneRender()
{

	m_imageFbo->Bind();
	glViewport(0, 0, m_imageW, m_imageH);


	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glDepthMask(GL_TRUE);
	glClear(GL_DEPTH_BUFFER_BIT);

	/*m_imageFbo->Disable();

	glViewport(0, 0, mWindowW, mWindowH);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);*/
}

void Renderer::endSceneRender()
{
	m_imageFbo->Disable();

	glViewport(0, 0, mWindowW, mWindowH);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
}

void Renderer::setWindowSize(int w, int h)
{
	m_window_w = w;
	m_window_h = h;
	createBuffers(w, h);
}

void Renderer::createBuffers(int w, int h)
{
	if (m_imageFbo)
	{
		glDeleteTextures(1, &m_imageTex);
		glDeleteTextures(1, &m_depthTex);
		delete m_imageFbo;

	}
	if (m_postprocessingFbo){

		glDeleteTextures(1, &m_postprocessingTex);
		delete m_postprocessingFbo;
	}

	mWindowW = w;
	mWindowH = h;

	m_imageW = w / m_downSample;
	m_imageH = h / m_downSample;

	// create fbo for image buffer
	GLint format = GL_RGBA16F_ARB;
	//GLint format = GL_LUMINANCE16F_ARB;
	//GLint format = GL_RGBA8;
	m_imageTex = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, format, GL_RGBA);
	m_postprocessingTex = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, format, GL_RGBA);
	m_depthTex = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, GL_DEPTH_COMPONENT24_ARB, GL_DEPTH_COMPONENT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);//new
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);//new
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	m_imageFbo = new FramebufferObject();
	m_imageFbo->AttachTexture(GL_TEXTURE_2D, m_imageTex, GL_COLOR_ATTACHMENT0_EXT);
	m_imageFbo->AttachTexture(GL_TEXTURE_2D, m_depthTex, GL_DEPTH_ATTACHMENT_EXT);
	m_imageFbo->IsValid();

	m_postprocessingFbo = new FramebufferObject();
	m_postprocessingFbo->AttachTexture(GL_TEXTURE_2D, m_postprocessingTex, GL_COLOR_ATTACHMENT0_EXT);
	m_postprocessingFbo->AttachTexture(GL_TEXTURE_2D, m_depthTex, GL_DEPTH_ATTACHMENT_EXT);
	m_postprocessingFbo->IsValid();
}
GLuint Renderer::createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format)
{
	GLuint texid;
	glGenTextures(1, &texid);
	glBindTexture(target, texid);

	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_FLOAT, 0);
	return texid;
}
void Renderer::compositeResult()
{
	/*m_imageFbo->Disable();
	m_postprocessingFbo->Disable();
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	m_displayTexProg->enable();
	m_displayTexProg->bindTexture("tex", m_imageTex, GL_TEXTURE_2D, 0);
	drawQuad();
	m_displayTexProg->disable();*/


	glViewport(0, 0, mWindowW, mWindowH);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	//glEnable(GL_BLEND); GL_SRC_ALPHA
	displayTexture(m_imageTex);
	//m_postprocessingTex m_imageTex
	glDisable(GL_BLEND);
}
void Renderer::displayTexture(GLuint tex)
{
	m_displayTexProg->enable();
	m_displayTexProg->bindTexture("tex", tex, GL_TEXTURE_2D, 0);
	drawQuad();
	m_displayTexProg->disable();
}
void Renderer::drawQuad()
{
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glEnd();
}