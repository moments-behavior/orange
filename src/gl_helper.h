#ifndef GL_HELPER
#define GL_HELPER
#include <stdio.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

static void glew_error_callback(GLenum glew_error)
{
    if (GLEW_OK != glew_error) {
        printf("GLEW error: %s\n", glewGetErrorString(glew_error));
    }
}

static void create_texture(GLuint *texture) {
        // Create a OpenGL texture identifier
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

static void bind_texture(GLuint *texture)
{
    glBindTexture(GL_TEXTURE_2D, *texture);
}

static void unbind_texture()
{
    glBindTexture(GL_TEXTURE_2D, 0);
}

static void create_pbo(GLuint *pbo, int image_width, int img_height)
{
    glGenBuffers(1, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, image_width * img_height * 4 * sizeof(unsigned char), 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

static void bind_pbo(GLuint *pbo)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
}

static void unbind_pbo()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

static void register_pbo_to_cuda(GLuint *pbo, cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsGLRegisterBuffer(cuda_resource, *pbo, cudaGraphicsRegisterFlagsNone);
}

static void map_cuda_resource(cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsMapResources(1, cuda_resource);
}

static void cuda_pointer_from_resource(unsigned char **cuda_buffer_p, size_t *size_p, cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsResourceGetMappedPointer((void **) cuda_buffer_p, size_p, *cuda_resource);
}

static void unmap_cuda_resource(cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsUnmapResources(1, cuda_resource);
}

static void upload_image_pbo_to_texture(int image_width, int image_height)
{
    // Assume PBO is bound before this, therefore the last
    // argument is an offset into the PBO, not a pointer to a
    // buffer stored in CPU memory
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}


#endif