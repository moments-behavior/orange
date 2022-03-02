#include <GLFW/glfw3.h>

int main(void) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);


    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL", NULL, NULL);

    glfwMakeContextCurrent(window);

    while(!glfwWindowShouldClose(window)) {
            glfwSwapBuffers(window);
            glfwPollEvents();
    }

    glfwTerminate();
}