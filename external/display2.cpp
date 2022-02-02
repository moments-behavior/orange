#include <iostream>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

int main(void)
{
   glewExperimental = GL_TRUE;

   if(glewInit() != GLEW_OK)
   {
      std::cerr<<"GLEW Initialization failed"<<std::endl;
      return -1;
   }

   return 0;
}