#ifndef UCDAVIS_FRAMEBUFFER_OBJECT_H
#define UCDAVIS_FRAMEBUFFER_OBJECT_H

#include <GL/glew.h>
#include <iostream>

class FramebufferObject
{
public:
	FramebufferObject();
	virtual ~FramebufferObject();

	void Bind();

	virtual void AttachTexture(GLenum texTarget,
		GLuint texId,
		GLenum attachment = GL_COLOR_ATTACHMENT0_EXT,
		int mipLevel = 0,
		int zSlice = 0);
	virtual void AttachTextures(int numTextures,
		GLenum texTarget[],
		GLuint texId[],
		GLenum attachment[] = NULL,
		int mipLevel[] = NULL,
		int zSlice[] = NULL);

	virtual void AttachRenderBuffer(GLuint buffId,
		GLenum attachment = GL_COLOR_ATTACHMENT0_EXT);
	virtual void AttachRenderBuffers(int numBuffers, GLuint buffId[],
		GLenum attachment[] = NULL);

	void Unattach(GLenum attachment);

	void UnattachAll();

#ifndef NDEBUG
	bool IsValid(std::ostream &ostr = std::cerr);
#else
	bool IsValid(std::ostream &ostr = std::cerr)
	{
		return true;
	}
#endif

	GLenum GetAttachedType(GLenum attachment);

	GLuint GetAttachedId(GLenum attachment);

	GLint  GetAttachedMipLevel(GLenum attachment);

	GLint  GetAttachedCubeFace(GLenum attachment);

	GLint  GetAttachedZSlice(GLenum attachment);

	static int GetMaxColorAttachments();

	static void Disable();
protected:
	void  _GuardedBind();
	void  _GuardedUnbind();
	void  _FramebufferTextureND(GLenum attachment, GLenum texTarget,
		GLuint texId, int mipLevel, int zSlice);
	static GLuint _GenerateFboId();

private:
	GLuint m_fboId;
	GLint  m_savedFboId;
};

#endif

