
/* Copyright (c) Mark J. Kilgard, 1994.  */

/* This program is freely distributable without licensing fees
and is provided without guarantee or warrantee expressed or
implied. This program is -not- in the public domain. */

/* New GLUT 3.0 glutGetModifiers() functionality used to make Shift-Left
   mouse scale the dinosaur's size. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>       /* for cos(), sin(), and sqrt() */
#include <GL/glut.h>
#include "trackball.h"
#include "file_processing.h"
#include "finite-difference.h"

typedef enum {
	RESERVED, BODY_SIDE, BODY_EDGE, BODY_WHOLE, ARM_SIDE, ARM_EDGE, ARM_WHOLE,
	LEG_SIDE, LEG_EDGE, LEG_WHOLE, EYE_SIDE, EYE_EDGE, EYE_WHOLE, DINOSAUR
} displayLists;

GLfloat angle = -150;   /* in degrees */
GLboolean doubleBuffer = GL_TRUE, iconic = GL_FALSE, keepAspect = GL_FALSE;
int spinning = 0, moving = 0;
int pleaseStop = 0;
int beginx, beginy;
int W = 300, H = 300;
float curquat[4];
float lastquat[4];
GLdouble bodyWidth = 3.0;
int newModel = 1;
int scaling;
float scalefactor = 1.0;

/* *INDENT-OFF* */
GLfloat body[][2] = { {0, 3}, {1, 1}, {5, 1}, {8, 4}, {10, 4}, {11, 5},
	{11, 11.5}, {13, 12}, {13, 13}, {10, 13.5}, {13, 14}, {13, 15}, {11, 16},
	{8, 16}, {7, 15}, {7, 13}, {8, 12}, {7, 11}, {6, 6}, {4, 3}, {3, 2},
	{1, 2} };
GLfloat arm[][2] = { {8, 10}, {9, 9}, {10, 9}, {13, 8}, {14, 9}, {16, 9},
	{15, 9.5}, {16, 10}, {15, 10}, {15.5, 11}, {14.5, 10}, {14, 11}, {14, 10},
	{13, 9}, {11, 11}, {9, 11} };
GLfloat leg[][2] = { {8, 6}, {8, 4}, {9, 3}, {9, 2}, {8, 1}, {8, 0.5}, {9, 0},
	{12, 0}, {10, 1}, {10, 2}, {12, 4}, {11, 6}, {10, 7}, {9, 7} };
GLfloat eye[][2] = { {8.75, 15}, {9, 14.7}, {9.6, 14.7}, {10.1, 15},
	{9.6, 15.25}, {9, 15.25} };
GLfloat skinColor[] = {0.1, 1.0, 0.1, 1.0}, eyeColor[] = {1.0, 0.2, 0.2, 1.0};
	

#define MX 176
#define MY 96
#define MZ 44
	
GLfloat lightZeroPosition[] = {0.0, 4.0, -MZ*2.0, 1.0};
//GLfloat lightZeroColor[] = {0.8, 1.0, 0.8, 1.0}; /* green-tinted */
GLfloat lightZeroColor[] = {1.0, 1.0, 1.0, 1.0};

GLfloat lightOnePosition[] = {0.0, 0.0, MZ*5.0, 1.0};
//GLfloat lightOneColor[] = {0.6, 0.3, 0.2, 1.0}; /* red-tinted */
GLfloat lightOneColor[] = {1.0, 1.0, 1.0, 1.0}; 

GLfloat lightTwoPosition[] = {MX*2.0, MY*2.0, MZ*4.0, 1.0};
//GLfloat lightOneColor[] = {0.6, 0.3, 0.2, 1.0}; /* red-tinted */
GLfloat lightTwoColor[] = {1.0, 1.0, 1.0, 1.0}; 
/* *INDENT-ON* */
#if 0
GLfloat flatSpace[] = {
			  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
			  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			  2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
			  3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
			  4.0f, 4.0f, 4.0f, 4.0f, 4.0f,

			  0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
			  1.1f, 1.1f, 1.1f, 1.1f, 1.1f,
			  2.1f, 2.1f, 2.1f, 2.1f, 2.1f,
			  3.1f, 3.1f, 3.1f, 3.1f, 3.1f,
			  4.1f, 4.1f, 4.1f, 4.1f, 4.1f,

			  0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
			  1.2f, 1.2f, 1.2f, 1.2f, 1.2f,
			  2.2f, 2.2f, 2.2f, 2.2f, 2.2f,
			  3.2f, 3.2f, 3.2f, 3.2f, 3.2f,
			  4.2f, 4.2f, 4.2f, 4.2f, 4.2f,

			  0.3f, 0.3f, 0.3f, 0.3f, 0.3f,
			  1.3f, 1.3f, 1.3f, 1.3f, 1.3f,
			  2.3f, 2.3f, 2.3f, 2.3f, 2.3f,
			  3.3f, 3.3f, 3.3f, 3.3f, 3.3f,
			  4.3f, 4.3f, 4.3f, 4.3f, 4.3f,

			  0.4f, 0.4f, 0.4f, 0.4f, 0.4f,
			  1.4f, 1.4f, 1.4f, 1.4f, 1.4f,
			  2.4f, 2.4f, 2.4f, 2.4f, 2.4f,
			  3.4f, 3.4f, 3.4f, 3.4f, 3.4f,
			  4.4f, 4.4f, 4.4f, 4.4f, 4.4f
};
#else
char flatSpace[MX*MY*MZ*16*4];
#endif
#define MAX_VAL 2.0f

void makeVolume(GLuint edge)
{
	int i,  j, k;
//	FD_Init3dSpaceCos(flatSpace, 2, MX, MY, MZ, 5.0f, 1.2f );
	FP_ReadRiff("../test1/medLin.riff");
printf("%s\n", __FUNCTION__);
//extern void FP_MakeVia(int xCenter, int yCenter, int outerRadius, int innerRadius, int start, int end, char matIndex)
	FP_MakeVia(15,20, 8,4, 2, 15, 1);
	FD_Testbed(flatSpace, MX, MY, MZ);
printf("reading back\n");
	glNewList(edge, GL_COMPILE);
//		glShadeModel(GL_FLAT);  // flat shade keeps angular hands from being * * "smoothed" 
		glPointSize(10);
		glBegin(GL_POINTS);
		for(i = 0; i < MX; i++)
		{
			for(j=0; j<MY;j++)
			{
				for(k=0;k<MZ;k++)
				{	// look up the value at the space in the array then convert it to a color
					int index = MY*MZ*i + MZ*j + k;
					if( flatSpace[index] == 0 )
						continue;
					float val = flatSpace[index];
//					printf("val %3.1f\n",val);
					GLfloat color[4];
					if(val >0)
					{
						color[0] = val/MAX_VAL;
						color[1] = (5-val)/5;
						color[2] =-1.0;
						color[3] = 0.5;
					}
					else
					{
						val = -val;
						color[2] = val/MAX_VAL;
						color[1] = (val-5)/5;
						color[0] = -1.0;
						color[3] = 0.5;
					}
					glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
					glVertex3f((float)(i), (float)(j), (float)(k)); // x, y, z
				}
			}
		}
		glEnd();
	glEndList();
}

void
extrudeSolidFromPolygon(GLfloat data[][2], unsigned int dataSize, GLdouble thickness, GLuint side, GLuint edge, GLuint whole)
{
	static GLUtriangulatorObj *tobj = NULL;
	GLdouble vertex[3], dx, dy, len;
	int i;
	int count = (int) (dataSize / (2 * sizeof(GLfloat)));

	if (tobj == NULL)
	{
		tobj = gluNewTess();  // create and initialize a GLU polygon tesselation object
		gluTessCallback(tobj, GLU_BEGIN, (_GLUfuncptr)glBegin);
		gluTessCallback(tobj, GLU_VERTEX, (_GLUfuncptr)glVertex2fv);  // semi-tricky
		gluTessCallback(tobj, GLU_END, glEnd);
	}

	// make side, a polygon
	glNewList(side, GL_COMPILE);
		glShadeModel(GL_SMOOTH);  // smooth minimizes seeing tessellation 
		gluBeginPolygon(tobj);
		for (i = 0; i < count; i++)
		{
			vertex[0] = data[i][0];
			vertex[1] = data[i][1];
			vertex[2] = 0;
			gluTessVertex(tobj, vertex, data[i]);
		}
		gluEndPolygon(tobj);
	glEndList();

	// make edge a Vertex 3D float
	glNewList(edge, GL_COMPILE);
		glShadeModel(GL_FLAT);  // flat shade keeps angular hands from being * * "smoothed" 
		glBegin(GL_POINTS);
		for (i = 0; i <= count; i++)
		{	// mod function handles closing the edge 
			glVertex3f(data[i % count][0], data[i % count][1], 0.0); // x, y, z
			glVertex3f(data[i % count][0], data[i % count][1], thickness);	// x, y, z
			// Calculate a unit normal by dividing by Euclidean
			// distance. We * could be lazy and use
			//glEnable(GL_NORMALIZE) so we could pass in * arbitrary
			// normals for a very slight performance hit. 
			dx = data[(i + 1) % count][1] - data[i % count][1];
			dy = data[i % count][0] - data[(i + 1) % count][0];
			len = sqrt(dx * dx + dy * dy);
			glNormal3f(dx / len, dy / len, 0.0);
		}
		glEnd();
	glEndList();

	glNewList(whole, GL_COMPILE);
		glFrontFace(GL_CW);
		glCallList(edge);
		glNormal3f(0.0, 0.0, -1.0);  // constant normal for side 
		glCallList(side);
		glPushMatrix();
			glTranslatef(0.0, 0.0, thickness);
			glFrontFace(GL_CCW);
			glNormal3f(0.0, 0.0, 1.0);   // opposite normal for other side 
			glCallList(side);
		glPopMatrix();
	glEndList();
}


void makeDinosaur(void)
{
//	extrudeSolidFromPolygon(leg, sizeof(leg), bodyWidth / 2, LEG_SIDE, LEG_EDGE, LEG_WHOLE);
//	extrudeSolidFromPolygon(eye, sizeof(eye), bodyWidth + 0.2, EYE_SIDE, EYE_EDGE, EYE_WHOLE);
	makeVolume(BODY_EDGE);
	glNewList(DINOSAUR, GL_COMPILE);
//		glMaterialfv(GL_FRONT, GL_DIFFUSE, skinColor);
//		glCallList(BODY_WHOLE);
		glPushMatrix();
//			glTranslatef(0.0, 0.0, bodyWidth);
			glCallList(BODY_EDGE);
//			glCallList(LEG_WHOLE);
//			glTranslatef(0.0, 0.0, -bodyWidth - bodyWidth / 4);
//			glCallList(ARM_WHOLE);
//			glTranslatef(0.0, 0.0, -bodyWidth / 4);
//			glCallList(LEG_WHOLE);
//			glTranslatef(0.0, 0.0, bodyWidth / 2 - 0.1);
//			glMaterialfv(GL_FRONT, GL_DIFFUSE, eyeColor);
//			glCallList(EYE_WHOLE);
		glPopMatrix();
	glEndList();
}


void
recalcModelView(void)
{
	GLfloat m[4][4];

	glPopMatrix();
	glPushMatrix();
	build_rotmatrix(m, curquat);
	glMultMatrixf(&m[0][0]);
	if (scalefactor == 1.0)
	{
		glDisable(GL_NORMALIZE);
	}
	else
	{
		glEnable(GL_NORMALIZE);
	}
	glScalef(scalefactor, scalefactor, scalefactor);
	glTranslatef(-MX/2, -MY/2, -MZ/2);
	newModel = 0;
}

void
showMessage(GLfloat x, GLfloat y, GLfloat z, char *message)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glTranslatef(x, y, z);
	glScalef(.02, .02, .02);
	while (*message)
	{
		glutStrokeCharacter(GLUT_STROKE_ROMAN, *message);
		message++;
	}
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

void
redraw(void)
{
	if(newModel)
		recalcModelView();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glCallList(DINOSAUR);
	showMessage(2, 7.1, 4.1, "Spin me.");
	glutSwapBuffers();
}

void
myReshape(int w, int h)
{
	glViewport(0, 0, w, h);
	W = w;
	H = h;
}

void
mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		spinning = 0;
		glutIdleFunc(NULL);
		moving = 1;
		beginx = x;
		beginy = y;
		if (glutGetModifiers() & GLUT_ACTIVE_SHIFT)
		{
			scaling = 1;
		}
		else
		{
			scaling = 0;
		}
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		moving = 0;
	}
}

void
animate(void)
{
	add_quats(lastquat, curquat, curquat);
	newModel = 1;
	glutPostRedisplay();
}

void
motion(int x, int y)
{
	if (scaling)
	{
		scalefactor = scalefactor * (1.0 + (((float) (beginy - y)) / H));
		beginx = x;
		beginy = y;
		newModel = 1;
		glutPostRedisplay();
		return;
	}
	if (moving)
	{
		trackball(lastquat,
			(2.0 * beginx - W) / W,
			(H - 2.0 * beginy) / H,
			(2.0 * x - W) / W,
			(H - 2.0 * y) / H
		);
		
		if(pleaseStop !=0 )
		{
			pleaseStop = 0;
			spinning =0;
			moving = 0;
		}
		beginx = x;
		beginy = y;
		spinning = 1;
		glutIdleFunc(animate);
	}
}

GLboolean lightZeroSwitch = GL_TRUE, lightOneSwitch = GL_TRUE;

void
menuSelection(int value)
{
	switch (value)
	{
	case 1:
		lightZeroSwitch = !lightZeroSwitch;
		if (lightZeroSwitch)
		{
			glEnable(GL_LIGHT0);
		}
		else
		{
			glDisable(GL_LIGHT0);
		}
	break;
	case 2:
		lightOneSwitch = !lightOneSwitch;
		if (lightOneSwitch)
		{
			glEnable(GL_LIGHT1);
		}
		else
		{
			glDisable(GL_LIGHT1);
		}
	break;
#ifdef GL_MULTISAMPLE_SGIS
	case 3:
		if (glIsEnabled(GL_MULTISAMPLE_SGIS))
		{
			glDisable(GL_MULTISAMPLE_SGIS);
		} else {
			glEnable(GL_MULTISAMPLE_SGIS);
		}
	break;
#endif
	case 4:
		moving = 0;
		pleaseStop =1;
	break;
	case 5:
		exit(0);
	break;
	}
	glutPostRedisplay();
}


void
vis(int visible)
{
	if (visible == GLUT_VISIBLE)
	{
		if (spinning)
			glutIdleFunc(animate);
	}
	else
	{
		if (spinning)
			glutIdleFunc(NULL);
	}
}


int
main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
	trackball(curquat, 0.0, 0.0, 0.0, 0.0);
	glutCreateWindow("dinospin");
	glutDisplayFunc(redraw);
	glutReshapeFunc(myReshape);
	glutVisibilityFunc(vis);

	// setup mouse functionality
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutCreateMenu(menuSelection);
	glutAddMenuEntry("Toggle right light", 1);
	glutAddMenuEntry("Toggle left light", 2);
	if (glutGet(GLUT_WINDOW_NUM_SAMPLES) > 0)
	{
		glutAddMenuEntry("Toggle multisampling", 3);
		glutSetWindowTitle("dinospin (multisample capable)");
	}
	glutAddMenuEntry("Reset", 4);
	glutAddMenuEntry("Quit", 5);
	glutAttachMenu(GLUT_RIGHT_BUTTON);


	makeDinosaur();
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glMatrixMode(GL_PROJECTION);
	gluPerspective( /* field of view in degree */ 45.0,
		/* aspect ratio */ 1.0,
		/* Z near */ 10.0, /* Z far */ (float)500.0);
	glMatrixMode(GL_MODELVIEW);
	gluLookAt(0.0, 0.0, (float)MZ*5.0,  /* eye is at (0,0,30) */
		0.0, 0.0, 0.0,      /* center is at (0,0,0) */
		0.0, 1.0, 0.);      /* up is in positive Y direction */

//	glPushMatrix();       // dummy push so we can pop on model recalc 
	// set up lighting for model
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);

	glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor);
//	glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1);
//	glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05);

	glLightfv(GL_LIGHT1, GL_POSITION, lightOnePosition);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, lightOneColor);
	glLightfv(GL_LIGHT1, GL_POSITION, lightTwoPosition);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, lightTwoColor);

	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHT2);

	glLineWidth(2.0);

	glPushMatrix();       // dummy push so we can pop on model recalc 

	glutMainLoop();
	return 0;             /* ANSI C requires main to return int. */
}

