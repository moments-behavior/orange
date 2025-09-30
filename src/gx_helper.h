#ifndef GX_HELPER
#define GX_HELPER

#include "IconsForkAwesome.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "types.h"
#include "utils.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <stdio.h>

typedef struct gx_context {
    u32 swap_interval;
    u32 width;
    u32 height;
    GLFWwindow *render_target;
    char *render_target_title;
    char *glsl_version;
} gx_context;

void gx_glfw_error_callback(int error, const char *description);

void gx_glew_error_callback(GLenum glew_error);

void gx_init(gx_context *context, GLFWwindow *render_target);

GLFWwindow *gx_glfw_init_render_target(u32 marjor_version, u32 minor_version,
                                       u32 width, u32 height, const char *title,
                                       char *glsl_version);

void gx_imgui_init(gx_context *context);

void gx_delete_buffer(GLuint *texture);

void create_texture(GLuint *texture, int image_width, int image_height);

void bind_texture(GLuint *texture);

void unbind_texture();

void create_pbo(GLuint *pbo, int image_width, int img_height);

void upload_image_pbo_to_texture(int image_width, int image_height);

void bind_pbo(GLuint *pbo);

void unbind_pbo();

void register_pbo_to_cuda(GLuint *pbo, cudaGraphicsResource_t *cuda_resource);

void cuda_unregister_pbo(cudaGraphicsResource_t cuda_resource);

void map_cuda_resource(cudaGraphicsResource_t *cuda_resource);

void cuda_pointer_from_resource(unsigned char **cuda_buffer_p, size_t *size_p,
                                cudaGraphicsResource_t *cuda_resource);

void unmap_cuda_resource(cudaGraphicsResource_t *cuda_resource);

void render_initialize_target(gx_context *window);

void create_new_frame();

void render_a_frame(gx_context *window);

void gx_cleanup(gx_context *window);

#endif
