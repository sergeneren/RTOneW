#pragma once

#ifndef RECTANGLEH
#define RECTANGLEH

#include "hitable.h"


class xy_rect:public hitable {

public:

	__device__ xy_rect(){}
	__device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material *mat):x0(_x0), y0(_y0), x1(_x1), y1(_y1), k(_k),mp(mat) {}
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const {
		box = aabb(vec3(x0, y0, k-0.001f), vec3(x1,y1,k+0.0001f));
		return true;
	}

	material * mp;
	float x0, y0, x1, y1, k;

};

__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
	float t = (k - r.origin().z()) / r.direction().z();
	if (t<t_min || t>t_max) return false;
	float x = r.origin().x() + t * r.direction().x();
	float y = r.origin().y() + t * r.direction().y();

	if (x<x0 || x>x1 || y<y0 || y>y1) return false;
	
	rec.u = (x - x0) / (x1 - x0);
	rec.v = (y -y0) / (y1 - y0);
	rec.t = t;
	rec.mat_ptr = mp;
	rec.p = r.point_at_parameter(t);
	rec.normal = vec3(0, 0, 1);
	return true;

}

class xz_rect :public hitable {

public:

	__device__ xz_rect() {}
	__device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material *mat) :x0(_x0), z0(_z0), x1(_x1), z1(_z1), k(_k), mp(mat) {}
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const {
		box = aabb(vec3(x0, k - 0.001f, z0), vec3(x1, k + 0.0001f, z1));
		return true;
	}

	material * mp;
	float x0, z0, x1, z1, k;

};

__device__ bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
	float t = (k - r.origin().y()) / r.direction().y();
	if (t<t_min || t>t_max) return false;
	float x = r.origin().x() + t * r.direction().x();
	float z = r.origin().z() + t * r.direction().z();

	if (x<x0 || x>x1 || z<z0 || z>z1) return false;

	rec.u = (x - x0) / (x1 - x0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	rec.mat_ptr = mp;
	rec.p = r.point_at_parameter(t);
	rec.normal = vec3(0, 1, 0);
	return true;

}

class yz_rect :public hitable {

public:

	__device__ yz_rect() {}
	__device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material *mat) :y0(_y0), z0(_z0), y1(_y1), z1(_z1), k(_k), mp(mat) {}
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const {
		box = aabb(vec3(k - 0.001f,y0, z0), vec3(k + 0.0001f, y1, z1));
		return true;
	}

	material * mp;
	float y0, z0, y1, z1, k;

};

__device__ bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
	float t = (k - r.origin().x()) / r.direction().x();
	if (t<t_min || t>t_max) return false;
	float y = r.origin().y() + t * r.direction().y();
	float z = r.origin().z() + t * r.direction().z();

	if (y<y0 || y>y1 || z<z0 || z>z1) return false;

	rec.u = (y - y0) / (y1 - y0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	rec.mat_ptr = mp;
	rec.p = r.point_at_parameter(t);
	rec.normal = vec3(1, 0, 0);
	return true;

}




#endif