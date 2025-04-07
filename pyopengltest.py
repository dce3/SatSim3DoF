import sys
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Global variables for simplicity
pos_arr = None      # This will hold your simulation trajectory (Nx3 NumPy array)
current_index = 2   # Start with at least 2 points to draw a line

def initGL(width, height):
    glClearColor(0.0, 0.0, 0.0, 0.0)   # Black background
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Set up a perspective projection
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def display():
    global pos_arr, current_index
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # Position the camera: eye, center, up
    gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0)

    glColor3f(0.0, 1.0, 0.0)  # Green color for the curve
    glLineWidth(2.0)
    glBegin(GL_LINE_STRIP)
    # Draw only up to current_index points
    for i in range(current_index):
        glVertex3f(pos_arr[i, 0], pos_arr[i, 1], pos_arr[i, 2])
    glEnd()

    glutSwapBuffers()

def timer(value):
    global current_index, pos_arr
    if current_index < len(pos_arr):
        current_index += 1
        glutPostRedisplay()  # Request display update
        glutTimerFunc(50, timer, 0)  # Call timer again after 50 ms
    else:
        # Optionally, reset or finish the animation
        pass

def main():
    global pos_arr

    # Example: create sample data (replace with your simulation data)
    t = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(t)
    y = np.sin(t)
    z = (t / (2 * np.pi)) * 5  # Scale z for visibility
    pos_arr = np.column_stack((x, y, z))  # pos_arr.shape = (200, 3)

    # Initialize GLUT and create a window
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"3D Curve Animation with PyOpenGL")

    initGL(800, 600)

    # Register display and timer callbacks
    glutDisplayFunc(display)
    glutTimerFunc(50, timer, 0)

    glutMainLoop()

if __name__ == '__main__':
    main()