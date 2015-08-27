#ifndef __RENDER__
#define __RENDER__
#include <GL/glew.h>
#include "nvMath.h"
#include "framebufferObject.h"
#include "GLSLProgram.h"
using namespace nv;
class Renderer
{
    public:
        Renderer();
        ~Renderer();

        void setPositions(float *pos, int numParticles);
        void setVertexBuffer(unsigned int vbo, int numParticles);
        void setColorBuffer(unsigned int vbo)
        {
            m_colorVBO = vbo;
        }

        enum DisplayMode
        {
            PARTICLE_POINTS,
            PARTICLE_SPHERES,
			PARTICLE_FLUID,
            PARTICLE_NUM_MODES
        };
		enum Target
		{
			LIGHT_BUFFER,
			SCENE_BUFFER
		};
		void display(DisplayMode mode , GLSLProgram *prog);
        void displayGrid();

        void setPointSize(float size)
        {
            m_pointSize = size;
        }
        void setParticleRadius(float r)
        {
            m_particleRadius = r;
        }
        void setFOV(float fov)
        {
            m_fov = fov;
        }
		void setWindowSize(int w, int h);

		vec4f getLightPositionEyeSpace()
		{
			return m_lightPosEye;
		}
		GLuint getShadowTexture

			()
		{
			return m_lightTexture[m_srcLightTexture];
		}
		void setPositionBuffer(GLuint vbo)
		{
			m_vbo = vbo;
		}
		void setVelocityBuffer(GLuint vbo)
		{
			m_velVbo = vbo;
		}
		void setIndexBuffer(GLuint ib)
		{
			m_indexBuffer = ib;
		}
		void setNormalBuffer(GLuint vbo)
		{
			m_normal = vbo;
		}
		GLuint LoadTexture(const char * filename);
		GLuint GetTexture(){ return txParticle; }

		void beginSceneRender();
		void endSceneRender();

		void createBuffers(int w, int h);
		void compositeResult();
		void displayTexture(GLuint tex);
		void drawQuad();
		void drawpointsprite();
		void drawPostProcess();
		void settotalVerts(unsigned int x){ totalVerts = x; }
    protected: // methods
        void _initGL();
        void _drawPoints();
        GLuint _compileProgram(const char *vsource, const char *fsource);
		GLuint createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format);
    protected: // data
        float *m_pos;
        int m_numParticles;

        float m_pointSize;
		float m_particleRadius;
		float m_fov;
		int m_window_w, m_window_h;

		matrix4f m_modelView, m_lightView, m_lightProj, m_shadowMatrix;
		vec3f m_viewVector, m_halfVector;
		bool m_invertedView;
		vec4f m_eyePos;
		vec4f m_halfVectorEye;
		vec4f m_lightPosEye;

		int                 m_lightBufferSize;
		GLuint              m_lightTexture[2];
		int                 m_srcLightTexture;
		GLuint              m_lightDepthTexture;
		FramebufferObject   *m_lightFbo;

		//GLSLProgram         *m_particleProg;

		//GLuint m_program;
		GLSLProgram *m_program;
		GLSLProgram *m_displayTexProg;
		GLSLProgram *m_postprocessing;
		GLuint m_vbo;
		GLuint m_velVbo;
        GLuint m_colorVBO;
		GLuint m_indexBuffer;
		GLuint m_normal;

		GLuint txParticle;
		GLuint m_imageTex, m_depthTex,m_postprocessingTex;

		FramebufferObject *m_imageFbo;
		FramebufferObject *m_postprocessingFbo;
		int m_imageW, m_imageH;
		unsigned int mWindowW, mWindowH;
		float mAspect, mInvFocalLen;
		float mFov;
		int m_downSample;
		unsigned int totalVerts;
};

#endif //__ RENDER_PARTICLES__
