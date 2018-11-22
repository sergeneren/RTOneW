#include "RTOneW/core.h"
#include "mainWindow.h"
#include <cuda_runtime.h>

#include <thread>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )


mainWindow *win;
bool rendering = false;
int spp;

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__host__ void setInstanceForRenderDistribution(mainWindow* w) {
	win = w;
}


__host__ void imageOutput(vec3 *pix, int s, int width, int height) {

	win->RTOneW_process_signal(pix, s, width, height);
}

__host__ void process_image(vec3 *pix, int s, int width, int height) {
	qDebug() << "in_process_image";
	QImage image(width, height, QImage::Format_RGB32);

#pragma omp parallell for
	for (int row = height - 1; row >= 0; row--) {
		for (int col = 0; col < width; col++) {
			size_t pixel_index = row * width + col;
			int ir = int(255.99*pix[pixel_index][0] / s);
			int ig = int(255.99*pix[pixel_index][1] / s);
			int ib = int(255.99*pix[pixel_index][2] / s);

			image.setPixel(col, (height - 1) - row, qRgb(ir, ig, ib));
		}
	}

	win->drawImage(image, s, spp);
	win->update();
}

__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f*(unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render_image(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);

	float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
	float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
	ray r = (*cam)->get_ray(u, v, &local_rand_state);
	col = color(r, world, &local_rand_state);

	rand_state[pixel_index] = local_rand_state;
	//col /= float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}


__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;

		d_list[0] = new sphere(vec3(0, 0, 0), 0.5, new lambertian(vec3(0.9, 0, 0)));
		d_list[1] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.0, 0.9, 0.9)));
		d_list[2] = new sphere(vec3(0, 0, -1), 0.5, new metal(vec3(0.9, 0.9, 0.9), .1));
		d_list[3] = new sphere(vec3(0, 0, 1), 0.5, new dielectric(1.333));

		*d_world = new hitable_list(d_list, 4);

		*rand_state = local_rand_state;

		vec3 lookfrom(-1, 0, 4);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = (lookfrom - lookat).length();
		float aperture = 0.1;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			30.0f,
			float(nx) / float(ny),
			aperture,
			dist_to_focus, 0.0f, 1.0f);
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	for (int i = 0; i < 4; i++) {
		delete ((sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}


void render(int width, int height, int spp, float fov, float aperture) {

	vec3 **pix = new vec3*[width];
	for (int i = 0; i < width; i++) {
		pix[i] = new vec3[height];
	}

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {

			pix[col][row] = vec3(0, 0, 0);
		}
	}

	int nx = width;
	int ny = height;
	int ns = spp;
	int tx = 8;
	int ty = 8;


	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	// allocate FB
	vec3 *fb;
	vec3 *db;
	db = new vec3[num_pixels];
	
	for (int i = 0; i<num_pixels; i++) {
		db[i] = vec3(0,0,0);
	}
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	// allocate random state
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState *d_rand_state2;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

	// we need that 2nd random state to be initialized for the world creation
	rand_init << <1, 1 >> > (d_rand_state2);
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
	create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	clock_t start, stop;
	start = clock();
	// Render our buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);



	for (int s = 1; s <= spp; s++) {

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		render_image << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
		//cudaError_t t =  cudaMemcpy(db, fb, fb_size, cudaMemcpyDeviceToHost);
		//std::cerr << t;
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		for(int i=0; i<num_pixels; i++){
			db[i] += fb[i]/s;
		}
		

		imageOutput(db, s, width, height);

	}


	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";



	// clean up
	checkCudaErrors(cudaDeviceSynchronize());

	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();


}

__host__ void send_to_render(int width, int height, int spp, float fov, float aperture) {
	//qDebug() << "send to render";
	spp = spp;
	render(width, height, spp, fov, aperture);
}
