//#include "RTOneW/core.h"
#include "mainWindow.h"
#include "RTOneW/vec3.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <thread>

class hitable;
class camera;
class atmosphere;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" void init_fb(vec3* fb, int nx, int ny);
extern "C" void rand_init(curandState *rand_state);
extern "C" void render_init(int nx, int ny, int tx, int ty, curandState *rand_state);
extern "C" void render_image(vec3 *fb, int nx, int ny, int tx, int ty, int ns, camera **cam, hitable **world, atmosphere **sky,curandState *rand_state);
extern "C" void create_world(hitable **d_list, hitable **d_world, camera **d_camera, atmosphere **d_atmosphere,
							 int nx, int ny, float fov, float aperture, curandState *rand_state);
extern "C" void free_world(hitable **d_list, hitable **d_world, camera **d_camera);


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


mainWindow *win;
bool rendering = true;
int spp;


void setInstanceForRenderDistribution(mainWindow* w) {
	win = w;
}


void imageOutput(vec3 *pix, int s, int width, int height) {

	win->RTOneW_process_signal(pix, s, width, height);
}

void process_image(vec3 *pix, int s, int width, int height) {
	
	QImage image(width, height, QImage::Format_RGB32);
	float s_inv = 1 / float(s);
#pragma omp parallell for
	for (int row = height - 1; row >= 0; row--) {
		for (int col = 0; col < width; col++) {
			size_t pixel_index = row * width + col;
			int ir = int(255.99*pix[pixel_index][0] * s_inv);
			int ig = int(255.99*pix[pixel_index][1] * s_inv);
			int ib = int(255.99*pix[pixel_index][2] * s_inv);

			image.setPixel(col, (height - 1) - row, qRgb(ir, ig, ib));
		}
	}
	
	win->drawImage(image, s, spp);
	win->update();
}


void render(int width, int height, int spp, float fov, float aperture, int b_size, int t_size) {


	int nx = width;
	int ny = height;
	int ns = spp;
	int tx = b_size;
	int ty = t_size;


	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	// allocate FB
	vec3 *fb;

	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
	init_fb(fb, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// allocate random state
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState *d_rand_state2;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

	// we need that 2nd random state to be initialized for the world creation
	rand_init(d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// make our world of hitables & the camera
	hitable **d_list;
	int num_hitables = 4;
	checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
	atmosphere **d_atmosphere;
	checkCudaErrors(cudaMalloc((void **)&d_atmosphere, sizeof(atmosphere *)));
	
	create_world(d_list, d_world, d_camera, d_atmosphere , nx, ny,fov,aperture, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	

	clock_t start, stop;
	start = clock();
	// Render our buffer

	render_init(nx, ny ,tx ,ty , d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	for (int s = 1; s <= spp; s++) {
		

		if (!rendering) break;

		render_image(fb, nx, ny,tx, ty, ns, d_camera, d_world, d_atmosphere, d_rand_state);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		imageOutput(fb, s, width, height);

	}


	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";


	
	// clean up
	checkCudaErrors(cudaDeviceSynchronize());
	
	/*
	free_world(d_list, d_world, d_camera);
	
	checkCudaErrors(cudaGetLastError());
	
	checkCudaErrors(cudaFree(d_camera));
	
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));
	*/
	cudaDeviceReset();

	
	rendering = true;
	//win->close();
}

void send_to_render(int width, int height, int spp, float fov, float aperture, int b_size, int t_size) {
	//qDebug() << "send to render";
	spp = spp;
	render(width, height, spp, fov, aperture, b_size,t_size);
}

void cancel_render() {
	rendering = false;
}