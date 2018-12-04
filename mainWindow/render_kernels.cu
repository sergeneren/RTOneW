
/* 
* Cuda kernels that does the heavy work
*/


#include "RTOneW/core.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);


__device__ vec3 color(const ray& r, hitable **world, atmosphere **sky, curandState *local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation(1.0, 1.0, 1.0);
	vec3 cur_emmission(0, 0, 0);
	vec3 color(0, 0, 0);

	for (int i = 0; i < 50; i++) {
		hit_record rec;
		ray scattered;
		vec3 attenuation;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
			
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return emitted * attenuation;
				
			}
		}
		else {
			float t0, t1, tmax = FLT_MAX;
			vec3 orig = vec3(cur_ray.origin().x(), cur_ray.origin().y()+ (*sky)->earthRadius+2000, cur_ray.origin().z());
			if (raySphereIntersect(orig, unit_vector(cur_ray.direction()), (*sky)->earthRadius, t0, t1) && t1 > 0) tmax = ffmax(0.0f, t0);
			vec3 sky_color;
			/*if (t0 < 0)*/ sky_color = (*sky)->computeIncidentLight(orig, unit_vector(cur_ray.direction()), 0, tmax);
			//else sky_color =  vec3(.004, .002, 0);
			sky_color = vec3(0, 0, 0);
			return cur_attenuation * sky_color;
			
		}
		
	}
	return color; // exceeded recursion
}


__global__ void rand_init_kernel(curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void init_fb_kernel(vec3* fb, int max_x, int max_y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;

	fb[pixel_index] = vec3(0, 0, 0);
}

__global__ void render_init_kernel(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render_image_kernel(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, atmosphere **sky, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);

	float u = float(i + vanDerCorput(&local_rand_state)) / float(max_x);
	float v = float(j + vanDerCorput(&local_rand_state,3)) / float(max_y);
	ray r = (*cam)->get_ray(u, v, &local_rand_state);
	col = color(r, world,sky, &local_rand_state);

	rand_state[pixel_index] = local_rand_state;
	//col /= float(ns);
	col[0] = col[0] < 1.413f ? pow(col[0] * 1.38317f, 1.0f / 2.2f) : 1.0f - exp(-col[0]);
	col[1] = col[1] < 1.413f ? pow(col[1] * 1.38317f, 1.0f / 2.2f) : 1.0f - exp(-col[1]);
	col[2] = col[2] < 1.413f ? pow(col[2] * 1.38317f, 1.0f / 2.2f) : 1.0f - exp(-col[2]);
	
	fb[pixel_index] += col;
}


__global__ void create_world_kernel(hitable **d_list, hitable **d_world, camera **d_camera, atmosphere **d_atmosphere,int nx, int ny, float fov, float aperture, curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;

		constant_texture *red_texture = new constant_texture(vec3(1, 0,0));
		constant_texture *white_texture = new constant_texture(vec3(1, 1, 1));
		constant_texture *light_texture = new constant_texture(vec3(15, 15, 15));
		constant_texture *turqoise_texture = new constant_texture(vec3(0, 0.9, 0.9));
		constant_texture *green_texture = new constant_texture(vec3(0, 0.9, 0.0));
		checker_texture *checker_tex = new checker_texture(red_texture, green_texture);

		diffuse_light *light = new diffuse_light(checker_tex, 1);
		diffuse_light *white_light = new diffuse_light(light_texture, 10);
		lambertian *white_lamb = new lambertian(white_texture);

		d_list[0] = new flip_normals( new yz_rect(0,555,0,555,555,new lambertian(green_texture)));
		d_list[1] = new yz_rect(0, 555, 0, 555, 0, new lambertian(red_texture));
		d_list[2] = new xz_rect(213, 343, 227, 332, 554, white_light);
		d_list[3] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white_lamb));
		d_list[4] = new xz_rect(0, 555, 0, 555, 0, white_lamb);
		d_list[5] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white_lamb));

		*d_world = new hitable_list(d_list, 6);

		*rand_state = local_rand_state;

		vec3 lookfrom(278, 278, -800);
		vec3 lookat(278, 278, 0);
		float dist_to_focus = (lookfrom - lookat).length();

		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			fov,
			float(nx) / float(ny),
			aperture,
			dist_to_focus, 0.0f, 1.0f);

		*d_atmosphere = new atmosphere(vec3(0, 1, 0));
	}
}

__global__ void free_world_kernel(hitable **d_list, hitable **d_world, camera **d_camera) {
	for (int i = 0; i < 5; i++) {
		delete ((sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}



////////////////////////////////////////////////////////////////////////////////////////
//////// WRAPPER FUNCTIONS


extern "C" void init_fb(vec3* fb, int nx, int ny) {

	init_fb_kernel <<< 1, 1 >>> (fb, nx, ny);

}

extern "C" void rand_init(curandState *rand_state) {

	rand_init_kernel << <1, 1 >> > (rand_state);

}




extern "C" void render_init(int nx, int ny, int tx, int ty, curandState *rand_state) {

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init_kernel << <blocks, threads >> > (nx, ny, rand_state);


}



extern "C" void render_image(vec3 *fb, int nx, int ny, int tx, int ty, int ns, camera **cam, hitable **world, atmosphere **sky,curandState *rand_state) {
	
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	render_image_kernel << <blocks, threads >> > (fb, nx, ny, ns, cam, world, sky, rand_state);


}


extern "C" void create_world(hitable **d_list, hitable **d_world, camera **d_camera, atmosphere **d_atmosphere, int nx, int ny, float fov, float aperture, curandState *rand_state) {

	create_world_kernel <<<1, 1 >>> (d_list, d_world, d_camera, d_atmosphere ,nx, ny, fov, aperture, rand_state);

}

extern "C" void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {

	free_world_kernel <<<1, 1 >>> (d_list, d_world, d_camera);

}