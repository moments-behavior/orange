#ifndef GX_HELPER
#define GX_HELPER

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "implot.h"
#include "IconsForkAwesome.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "types.h"


typedef struct gx_context
{
    u32 swap_interval;
    u32 width;
    u32 height;
    GLFWwindow *render_target;
    char *render_target_title;
    char glsl_version[32];
} gx_context;


static inline void gx_glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

static inline void gx_glew_error_callback(GLenum glew_error)
{
    if (GLEW_OK != glew_error)
    {
        printf("GLEW error: %s\n", glewGetErrorString(glew_error));
    }
}

static inline GLFWwindow *gx_glfw_init_render_target(u32 major_version, u32 minor_version, 
    u32 width, u32 height, const char *title, char *glsl_version)
{
    // GLFW should already be initialized in main()
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major_version);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor_version);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create window with graphics context
    GLFWwindow *window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window) {
        const char* error_msg;
        glfwGetError(&error_msg);
        fprintf(stderr, "Failed to create GLFW window: %s\n", error_msg ? error_msg : "Unknown error");
        return nullptr;
    }
    
    glfwMakeContextCurrent(window);
    
    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(err));
        glfwDestroyWindow(window);
        return nullptr;
    }
    
    // Copy GLSL version if pointer is valid
    if (glsl_version) {
        strncpy(glsl_version, "#version 130", 31);
        glsl_version[31] = '\0';
    }
    
    return window;
}

static inline bool gx_init(gx_context *context, GLFWwindow *render_target)
{
    if (!context || !render_target) {
        return false;
    }
    
    context->render_target = render_target;
    glfwMakeContextCurrent(render_target);
    gx_glew_error_callback(glewInit());
    glfwSwapInterval(1); // Enable vsync
    
    return true;
}

static inline bool gx_imgui_init(gx_context *context)
{
    if (!context || !context->render_target) {
        return false;
    }

    // ************* Dear Imgui ********************//
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlotContext *implotCtx = ImPlot::CreateContext();
    if (!implotCtx) {
        return false;
    }

    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Setup Dear ImGui style
    ImGui::StyleColorsClassic();

    // When viewports are enabled we tweak WindowRounding/WindowBg
    ImGuiStyle &style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    bool glfw_init_ok = ImGui_ImplGlfw_InitForOpenGL(context->render_target, true);
    bool gl3_init_ok = ImGui_ImplOpenGL3_Init(context->glsl_version);
    
    if (!glfw_init_ok || !gl3_init_ok) {
        return false;
    }

    // Load fonts
    try {
        if (!io.Fonts->AddFontFromFileTTF("fonts/Roboto-Regular.ttf", 15.0f)) {
            return false;
        }
        
        // merge in icons from Font Awesome
        static const ImWchar icons_ranges[] = {ICON_MIN_FK, ICON_MAX_16_FK, 0};
        ImFontConfig icons_config;
        icons_config.MergeMode = true;
        icons_config.PixelSnapH = true;
        if (!io.Fonts->AddFontFromFileTTF("fonts/forkawesome-webfont.ttf", 15.0f, 
                                         &icons_config, icons_ranges)) {
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

static inline void gx_delete_buffer(GLuint *texture) 
{
	glDeleteBuffers(1, texture);
}

static inline void create_texture(GLuint *texture, int image_width, int image_height)
{
    // Create a OpenGL texture identifier
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    glEnable(GL_TEXTURE_2D);
}

static inline void bind_texture(GLuint *texture)
{
    glBindTexture(GL_TEXTURE_2D, *texture);
}

static inline void unbind_texture()
{
    glBindTexture(GL_TEXTURE_2D, 0);
}

static inline void create_pbo(GLuint *pbo, int image_width, int img_height)
{
    glGenBuffers(1, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, image_width * img_height * 4 * sizeof(unsigned char), 0, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}


static inline void upload_image_pbo_to_texture(int image_width, int image_height)
{
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}


static inline void bind_pbo(GLuint *pbo)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
}

static inline void unbind_pbo()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

static inline void register_pbo_to_cuda(GLuint *pbo, cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsGLRegisterBuffer(cuda_resource, *pbo, cudaGraphicsRegisterFlagsNone);
}

static inline void cuda_unregister_pbo(cudaGraphicsResource_t cuda_resource)
{
    cudaGraphicsUnregisterResource(cuda_resource);
}

static inline void map_cuda_resource(cudaGraphicsResource_t *cuda_resource, cudaStream_t stream)
{
    cudaGraphicsMapResources(1, cuda_resource, stream);
}

static inline void cuda_pointer_from_resource(unsigned char **cuda_buffer_p, size_t *size_p, cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsResourceGetMappedPointer((void **)cuda_buffer_p, size_p, *cuda_resource);
}

static inline void unmap_cuda_resource(cudaGraphicsResource_t *cuda_resource)
{
    cudaGraphicsUnmapResources(1, cuda_resource);
}


static inline void render_initialize_target(gx_context *window)
{
    GLFWwindow *render_target = gx_glfw_init_render_target(3, 3, window->width, window->height, "Orange", window->glsl_version);
    gx_init(window, render_target);
    gx_imgui_init(window);
}

static inline void create_new_frame()
{
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

static inline void render_a_frame(gx_context *window)
{
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window->render_target, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Update and Render additional Platform Windows
    // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
    //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow *backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }

    glfwSwapBuffers(window->render_target);
}

static inline void gx_cleanup(gx_context *window)
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window->render_target);
    glfwTerminate();
}


#endif