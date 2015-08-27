#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>

#include "FluidSystem.h"
#include "render.h"
#include "shaders.h"
#include "GLSLProgram.h"


using namespace std;

#define GRID_SIZE       64
#define NUM_PARTICLES   10000
const uint width = 1024, height = 768;

static int mode = 0;
enum { M_VIEW = 0, M_MOVE=1 };
static int ox = 0;
static int	oy = 0;
static int buttonState = 0;
static float camera_trans[3] = { 0, -32, -100 };
static float camera_rot[3] = { 0, 0, 0 };
static float camera_trans_lag[3] = { 0, 0, -3 };
static float camera_rot_lag[3] = { 0, 0, 0 };
static bool keyDown[256];
static float walkSpeed = 0.05f;
static Renderer::DisplayMode displayMode = Renderer::PARTICLE_SPHERES;
//PARTICLE_POINTS  PARTICLE_SPHERES
static uint num_Particles = NUM_PARTICLES;
static uint3 grid_Size=make_uint3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
static const int numIterations = 0;
static const float inertia=0.1f;

static float timestep=0.5f;
static int fpsCount = 0;
static int fpsLimit = 1;

static FluidSystem *psystem =0;
static StopWatchInterface *timer=NULL;
static Renderer *renderer=0 ;
static float modelView[16];

static GLSLProgram *m_particleProg=0;
static GLSLProgram *floorProg = 0;
static GLuint floorTex = 0;
static vec3f lightPos(0.0, 32.0, 0.0);
static vec3f lightColor(1.0f, 1.0f, 0.8f);
static vec3f colorAttenuation(0.5f, 0.75f, 1.0f);
static float blurRadius = 48.0f;

static float3 colliderSphereVel = make_float3(0.1f, -0.1f, 0.0f);


static GLuint createTexture(GLenum target, GLint internalformat, GLenum format, int w, int h, void *data)
{
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(target, tex);
	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(target, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_UNSIGNED_BYTE, data);
	return tex;
}

static GLuint loadTexture(char *filename)
{
	unsigned char *data = 0;
	unsigned int width, height;
	sdkLoadPPM4ub(filename, &data, &width, &height);
	//sdkLoadPGM(filename, &data, &width, &height);
	if (!data)
	{
		printf("Error opening file '%s'\n", filename);
		return 0;
	}

	printf("Loaded '%s', %d x %d pixels\n", filename, width, height);

	return createTexture(GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, width, height, data);
}


static const char* LoadShaderFile(string from){
	string into;
	ifstream	file;
	string		temp;

	cout << "Loading shader text from " << from << endl << endl;

	file.open(from.c_str());
	if (!file.is_open()){
		cout << "File does not exist!" << endl;
	}

	while (!file.eof()){
		getline(file, temp);
		into += temp + "\n";
	}

	cout << into << endl << endl;

	file.close();
	cout << "Loaded shader text!" << endl << endl;

	const char* chars = into.c_str();

	return chars;

}

static void initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("CUDA Particles");

	glewInit();

	if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
	{
		fprintf(stderr, "Required OpenGL extensions missing.");
		exit(EXIT_FAILURE);
	}

	glEnable(GL_DEPTH_TEST);
	//load floor texture
	//char *imagePath = sdkFindFilePath("floortile.ppm", argv[0]);
	char *imagePath = sdkFindFilePath("reference.ppm", argv[0]);
	if (imagePath == NULL)
	{
		fprintf(stderr, "Error finding floor image file\n");
		exit(EXIT_FAILURE);
	}
	floorTex = loadTexture(imagePath);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f);


	floorProg = new GLSLProgram(floorVS, floorPS);
	//floorProg = new GLSLProgram(LoadShaderFile("data/TexturedVertex.glsl"), LoadShaderFile("data/TexturedFragment.glsl"));

	m_particleProg = new GLSLProgram(mblurVS, mblurGS, particlePS);
	//load end
	glClearColor(0.25, 0.25, 0.25, 1.0);
}

static void glutloop(bool bUseOpenGL){

	psystem = new FluidSystem(num_Particles, grid_Size, true);
	psystem->reset(FluidSystem::CONFIG_GRID);
	if (bUseOpenGL)
	{
		renderer = new Renderer;
		renderer->setParticleRadius(psystem->getParticleRadius());
		renderer->setColorBuffer(psystem->getColorBuffer());
	}

	sdkCreateTimer(&timer);
}


static void cleanup(){
	sdkDeleteTimer(&timer);

	if (psystem)
	{
		delete psystem;
	}
	delete floorProg;
	delete m_particleProg;
	cudaDeviceReset();
	return;
}

static void computeFPS()
{
	//frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, " %3.1f fps",  ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;

		fpsLimit = (int)MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}
}

static void renderFloor(){
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	// draw floor
	floorProg->enable();
	//floorProg->bindTexture("texCoord", floorTex, GL_TEXTURE_2D, 0);

	floorProg->bindTexture("tex", floorTex, GL_TEXTURE_2D, 0);
	//floorProg->bindTexture("tex", renderer->GetTexture(), GL_TEXTURE_2D, 0);

	//floorProg->setUniformfv("lightPosEye", renderer->getLightPositionEyeSpace(), 3);
	//floorProg->setUniformfv("lightColor", lightColor, 3);

	glColor3f(1.0, 1.0, 1.0);
	glNormal3f(0.0, 1.0, 0.0);
	glBegin(GL_QUADS);
	{
		float s = 48.f;
		float rep = 1.f;
		glTexCoord2f(0.f, 0.5f);
		glVertex3f(-s, -32.f, -s);
		glTexCoord2f(rep, 0.5f);
		glVertex3f(s, -32.f, -s);
		glTexCoord2f(rep, 0.7f);
		glVertex3f(s, -32.f, s);
		glTexCoord2f(0.f, 0.7f);
		glVertex3f(-s, -32.f, s);
	}
	glEnd();
	floorProg->disable();

	glMatrixMode(GL_TEXTURE);
	glLoadIdentity();

	// draw light
	/*glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glTranslatef(lightPos.x, lightPos.y, lightPos.z);
	glColor3fv(&lightColor[0]);
	glutSolidSphere(0.1, 10, 5);
	glPopMatrix();*/
}

static void renderscene(){
	sdkStartTimer(&timer);

	psystem->update(timestep);


	psystem->depthSort(); //sort the depth 


	if (renderer)
	{
		//renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
		renderer->setVertexBuffer(psystem->getMarchingCubePosBuffer(), psystem->gettotalVerts());
		renderer->setIndexBuffer(psystem->getSortedIndexBuffer());
		renderer->setVelocityBuffer(psystem->getVelBuffer());
		renderer->settotalVerts(psystem->gettotalVerts());
		renderer->setNormalBuffer(psystem->getMarchingCubeNormalBuffer());
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	for (int c = 0; c < 3; ++c)
	{
		camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
		camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
	}

	glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
	glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
	glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);



	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glClearColor(0.5, 0.5, 0.5, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// cube
	//glColor3f(1.0, 1.0, 1.0);
	//glutWireCube(2.0);

	// create collider
	//renderer->beginSceneRender();

	/*glPushMatrix();
	float3 p = psystem->getColliderPos();
	glTranslatef(p.x, p.y, p.z);
	glColor3f(1.0, 0.5, 0.5);
	glutSolidSphere(psystem->getColliderRadius(), 50, 50);
	glPopMatrix();
	glPushMatrix();
	float3 ps = psystem->getColliderSpherePos();
	glTranslatef(ps.x, ps.y, ps.z);
	glColor3f(1.0, 0.0, 0.0);
	glutSolidSphere(psystem->getColliderSphereRadius(), 50, 50);
	glPopMatrix();*/

	//renderer->endSceneRender();

	renderer->beginSceneRender();
	//renderFloor();
	renderer->endSceneRender();


	if (renderer /*&& displayEnabled*/)
	{
		renderer->display(Renderer::PARTICLE_SPHERES, m_particleProg);
	}

	sdkStopTimer(&timer);

	glutSwapBuffers();
	glutReportErrors();

	computeFPS();
	/*for (int i = 0; i < 4; i++){
		cout << modelView[i*4] << " " << modelView[i*4+1] << " " << modelView[i*4+2] << " " << modelView[i*4+3] << " " << endl;
	}*/
}


static void reshape(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float)w / (float)h, 0.1, 3000.0);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);

	if (renderer)
	{
		renderer->setFOV(60.0);
		renderer->setWindowSize(w, h);
		//renderer->setFOV(60.0);
	}
}

static void idle(void)
{

	glutPostRedisplay();
}

static void ixform(float *v, float *r, GLfloat *m)
{
	r[0] = v[0] * m[0] + v[1] * m[1] + v[2] * m[2];
	r[1] = v[0] * m[4] + v[1] * m[5] + v[2] * m[6];
	r[2] = v[0] * m[8] + v[1] * m[9] + v[2] * m[10];
}

static void key(unsigned char key, int, int)
{
	switch (key)
	{
	case 'v':
		mode = M_VIEW;
		break;

	case 'm':
		mode = M_MOVE;
		break;
	case 'd':
		camera_trans[0] -= 3.0f;
		break;
	case 'a':
		camera_trans[0] += 3.0f;
		break;
	case 'q':
		camera_trans[1] += 3.0f;
		break;
	case 'e':
		camera_trans[1] -= 3.0f;
		break;
	case 'w':
		camera_trans[2] += 3.0f;
		break;
	case 's':
		camera_trans[2] -= 3.0f;
		break;
	case 'g':
		psystem->setStep(0);
		break;
	case 'u':
		psystem->dumpParticles(0, num_Particles - 1);
		break;
	case 'r':
		psystem->reset(FluidSystem::CONFIG_RANDOM);
		break;
	case 'i':
		for (int i = 0; i < 4; i++){
			cout << modelView[i * 4] << " " << modelView[i * 4 + 1] << " " << modelView[i * 4 + 2] << " " << modelView[i * 4 + 3] << " " << endl;
		}
		break;

	}
	glutPostRedisplay();
}

static void mouse(int button, int state, int x, int y)
{
	int mods;

	if (state == GLUT_DOWN)
	{
		buttonState |= 1 << button;
	}

	ox = x;
	oy = y;

	glutPostRedisplay();
}

static void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);
	if (mode == M_VIEW){
		if (buttonState == 1){
			camera_rot[0] += dy / 5.0f;
			camera_rot[1] += dx / 5.0f;
		}
	}
	if (mode == M_MOVE){
		float translateSpeed = 0.2f;
		float3 p = psystem->getColliderPos();

		if (buttonState == 1)
		{
			float v[3], r[3];
			v[0] = dx*translateSpeed;
			v[1] = -dy*translateSpeed;
			v[2] = 0.0f;
			ixform(v, r, modelView);
			p.x += r[0];
			p.y += r[1];
			p.z += r[2];
		}
		else if (buttonState == 2)
		{
			float v[3], r[3];
			v[0] = 0.0f;
			v[1] = 0.0f;
			v[2] = dy*translateSpeed;
			ixform(v, r, modelView);
			p.x += r[0];
			p.y += r[1];
			p.z += r[2];
		}
		//p.x += 0.1f;
		psystem->setColliderPos(p);
	}
	ox = x;
	oy = y;


	

	/*float3 ps = psystem->getColliderSpherePos();
	float vs[3], rs[3];
	if (ps.y < -32.0f){
		colliderSphereVel.y = -colliderSphereVel.y;
	}
	if (ps.y > 32.0f){
		colliderSphereVel.y = -colliderSphereVel.y;
	}
	vs[0] = colliderSphereVel.x;
	vs[1] = colliderSphereVel.y;
	vs[2] = 0.0f;
	ixform(vs, rs, modelView);
	ps.x += rs[0];
	ps.y += rs[1];
	ps.z += rs[2];

	psystem->setColliderShperePos(ps);*/

	glutPostRedisplay();
}

