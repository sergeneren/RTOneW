#pragma once

#ifndef TEXTUREH
#define TEXTUREH

#include "vec3.h"


class texture {

public:

	__device__ virtual vec3 value(float u, float v, const vec3& p) const =0 ;

};


class constant_texture : public texture
{
public:
	
	__device__ constant_texture() {};
	__device__ constant_texture(vec3 c) : color(c) {};
	__device__ virtual vec3 value(float u, float v, const vec3& p) const;

	vec3 color;

};

__device__ vec3 constant_texture::value(float u, float v, const vec3& p) const
{
	return color;
}

#endif