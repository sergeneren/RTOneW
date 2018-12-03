#pragma once
#include "vec3.h"

#ifndef TEXTUREH
#define TEXTUREH


class texture {

public:
	virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public texture {
public:

	constant_texture() {  };
	constant_texture(vec3 c) :color(c) {};
	vec3 value(float u, float v, const vec3& p)const override {

		return color;

	}


	vec3 color;
	
};




#endif // !TEXTUREH
