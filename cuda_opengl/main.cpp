//myVBO.cpp
#include <glew.h>
#include <freeglut.h>
#include "cuda_gl_interop.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

unsigned int window_width = 512;
unsigned int window_height = 512;
unsigned int mesh_width = 256;
unsigned int mesh_height = 256;

unsigned int timer = 0;

int animFlag = 1;
float animTime = 0.0f;
float animInc = 0.01f;

GLuint vbo = NULL;

float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

struct cudaGraphicsResource *cuda_vbo_resource;

extern "C" void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);

void createVBO(GLuint *vbo)
{
	if (vbo)
	{
		glGenBuffers(1, vbo);
		glBindBuffer(GL_ARRAY_BUFFER, *vbo);

		unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
		glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, *vbo, cudaGraphicsMapFlagsWriteDiscard);
	}
}


void runCuda()
{
	float4 *dptr = NULL;
	size_t num_bytes;

	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo_resource);

	launch_kernel(dptr, mesh_width, mesh_height, animTime);

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}

void initCuda(int argc, char **argv)
{
	cudaSetDevice(0);

	createVBO(&vbo);

	runCuda();
}


void display()
{
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, mesh_width*mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	if (animFlag)
	{
		glutPostRedisplay();
		animTime += animInc;
	}


}

void fpsDisplay()
{

	display();

}



void initGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop Demo (adapted from NVDIA's simpleGL)");

	glutDisplayFunc(fpsDisplay);

	glewInit();

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	glViewport(0, 0, window_width, window_height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

}


int main(int argc, char **argv)
{
	initGL(argc, argv);
	initCuda(argc, argv);

	glutDisplayFunc(fpsDisplay);


	glutMainLoop();

	cudaThreadExit();			///////////////////////////
}