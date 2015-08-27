#include "Glutloop.cpp"

int main(int argc, char **argv)
{
	initGL(&argc, argv);
	glutloop(true);

	cout << "grid x,y,z size " << grid_Size.x << " " << grid_Size.y << " " <<
		grid_Size.z << " " << "number of cell " << grid_Size.x*grid_Size.y*grid_Size.z << endl;
	cout << "number of particle " << num_Particles;

	glutDisplayFunc(renderscene);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutIdleFunc(idle);
	glutCloseFunc(cleanup);
	glutMainLoop();

	if (psystem)
	{
		delete psystem;
	}

	cudaDeviceReset();
}

