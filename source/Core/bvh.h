#pragma once

#ifndef BVHH
#define BVHH
#include "hitable.h"


class bvh_node : public hitable {

public:
	__device__ bvh_node() {};

	__device__ bvh_node(hitable **l, int n, float time0, float time1, curandState *local_rand_state);
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

	hitable *left;
	hitable *right;
	aabb box;

};


__device__ void buble_sort_x(hitable **l, int n) {

	aabb box_left, box_right;
	hitable *a;
	hitable *b;
	bool swapped;
	
	for (int i=0 ; i<n-1; i++)
	{
		swapped = false;
		for (int j=0; j<n-i-1;j++)
		{
			a = l[j];
			b = l[j + 1];
			a->bounding_box(0, 0, box_left);
			b->bounding_box(0, 0, box_right);

			if (box_left.min().x() > box_right.min().x())
			{
				hitable *temp; 
				temp = l[j];
				l[j] = l[j+1];
				l[j + 1] = temp;

				swapped = true;
			}
		}
		if (!swapped) break;
	}
}

__device__ void buble_sort_y(hitable **l, int n) {

	aabb box_left, box_right;
	hitable *a;
	hitable *b;
	bool swapped;

	for (int i = 0; i < n - 1; i++)
	{
		swapped = false;
		for (int j = 0; j < n - i - 1; j++)
		{
			a = l[j];
			b = l[j + 1];
			a->bounding_box(0, 0, box_left);
			b->bounding_box(0, 0, box_right);

			if (box_left.min().y() > box_right.min().y())
			{
				hitable *temp;
				temp = l[j];
				l[j] = l[j + 1];
				l[j + 1] = temp;

				swapped = true;
			}
		}
		if (!swapped) break;
	}
}
__device__ void buble_sort_z(hitable **l, int n) {

	aabb box_left, box_right;
	hitable *a;
	hitable *b;
	bool swapped;

	for (int i = 0; i < n - 1; i++)
	{
		swapped = false;
		for (int j = 0; j < n - i - 1; j++)
		{
			a = l[j];
			b = l[j + 1];
			a->bounding_box(0, 0, box_left);
			b->bounding_box(0, 0, box_right);

			if (box_left.min().z() > box_right.min().z())
			{
				hitable *temp;
				temp = l[j];
				l[j] = l[j + 1];
				l[j + 1] = temp;

				swapped = true;
			}
		}
		if (!swapped) break;
	}
}


__device__ bvh_node::bvh_node(hitable **l, int n, float time0, float time1, curandState *local_rand_state) {

	int axis = int(3 * curand_uniform(local_rand_state));

	if (axis == 0) buble_sort_x(l, n);
	else if(axis==1) buble_sort_y(l, n);
	else buble_sort_z(l, n);

	if (n == 1) {
		left = right = l[0];
	}
	else if (n == 2) {

		left = l[0];
		right = l[1];

	}
	else
	{
		left = new bvh_node(l, n / 2, time0, time1, local_rand_state);
		right = new bvh_node(l + n / 2, n - n / 2, time0, time1, local_rand_state);
	}

	aabb box_left, box_right;
	//if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right)) CUDA_ERROR_NOT_FOUND;
	box = surrounding_box(box_left, box_right);


}



__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& b) const {

	b = box;
	return true;

}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

	if(box.hit(r,t_min,t_max)){
		
		hit_record left_rec, right_rec;
		bool hit_left = left->hit(r, t_min, t_max, left_rec);
		bool hit_right = right->hit(r, t_min, t_max, right_rec);
		if (hit_left&&hit_right) {

			if (left_rec.t < right_rec.t) rec = left_rec;
			else rec = right_rec;
			return true;
		}
		else if(hit_left)
		{
			rec = left_rec;
			return true;
		}
		else if(hit_right)
		{
			rec = right_rec;
			return true;
		}
		else
		{
			return false;
		}
	}

	else
	{
		return false;
	}
}



#endif // !BVHH
