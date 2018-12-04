#pragma once

#ifndef TEXTUREH
#define TEXTUREH
#include "vec3.h"

class base_texture {

public:
	__device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};


class constant_texture : public base_texture {
public:

	__device__ constant_texture(vec3 c) :color(c) {};
	__device__ virtual vec3 value(float u, float v, const vec3& p) const{ return color; }


	vec3 color;

};


class checker_texture : public base_texture
{
public:
	__device__ checker_texture() {};
	__device__ checker_texture(base_texture *t0, base_texture *t1):even(t0), odd(t1) {};

	__device__ vec3 value(float u, float v, const vec3& p) const {

		float sines = sin(10 * p.x())*sin(10 * p.y())*sin(10 * p.z());
		if (sines < 0) return odd->value(u, v, p);
		else return even->value(u, v, p);
	}

	base_texture *odd;
	base_texture *even;
};



#endif // !TEXTUREH
