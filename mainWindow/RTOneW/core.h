#pragma once


#ifndef COREH
#define COREH


#include "ray.h"
#include "vec3.h"
#include "bvh.h"
#include "hitable.h"
#include "sphere.h"
#include "rectangle.h"
#include "box.h"
#include "medium.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "atmosphere.h"

void imageOutput(vec3 *pix, int s, int width, int height);
void process_image(vec3 *pix, int s, int width, int height);


#endif // !COREH