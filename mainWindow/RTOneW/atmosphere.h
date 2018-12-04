#pragma once

#ifndef ATMOSPHEREH
#define ATMOSPHEREH



// Atmosphere implementation from Scratch A Pixel

#include "hitable.h"


class atmosphere {


public:

	__device__ atmosphere(
		vec3 sd = vec3(0.0f, 1.0f, 0.0f),
		float er = 6360e3, float ar = 6420e3,
		float hr = 7994, float hm = 1200, 
		vec3 betar = vec3(3.8e-6f, 13.5e-6f, 33.1e-6f),
		vec3 betam = vec3(21e-6f)):
		sunDirection(sd),
		earthRadius(er),
		atmosphereRadius(ar),
		Hr(hr),
		Hm(hm),
		betaR(betar),
		betaM(betam)
	 {}

	__device__ vec3 computeIncidentLight(const vec3& orig, const vec3& dir, float tmin, float tmax) const;
	__device__ bool bounding_box(float t0, float t1, aabb& box) const;

	vec3 sunDirection;     // The sun direction (normalized)
	float earthRadius;      // In the paper this is usually Rg or Re (radius ground, earth)
	float atmosphereRadius; // In the paper this is usually R or Ra (radius atmosphere)
	float Hr;               // Thickness of the atmosphere if density was uniform (Hr)
	float Hm;               // Same as above but for Mie scattering (Hm)

	const vec3 betaR;
	const vec3 betaM;

};

__device__ bool atmosphere::bounding_box(float t0, float t1, aabb& box) const {

	box = aabb(vec3(0,0,0) - vec3(atmosphereRadius), vec3(0, 0, 0) + vec3(atmosphereRadius));

	return true;
}



__device__ bool solveQuadratic(float a, float b, float c, float& x1, float& x2)
{
	if (b == 0) {
		// Handle special case where the the two vector ray.dir and V are perpendicular
		// with V = ray.orig - sphere.centre
		if (a == 0) return false;
		x1 = 0; x2 = sqrt(-c / a);
		return true;
	}
	float discr = b * b - 4 * a * c;

	if (discr < 0) return false;

	float q = (b < 0.f) ? -0.5f * (b - sqrt(discr)) : -0.5f * (b + sqrt(discr));
	x1 = q / a;
	x2 = c / q;

	return true;
}



__device__ bool raySphereIntersect(const vec3& orig, const vec3& dir, const float& radius, float& t0, float& t1)  {

	float A = dir.squared_length();
	float B = 2 * (dir.x() * orig.x() + dir.y() * orig.y() + dir.z() * orig.z());
	float C = orig.x() * orig.x() + orig.y() * orig.y() + orig.z() * orig.z() - radius * radius;

	if (!solveQuadratic(A, B, C, t0, t1)) return false;

	if (t0 > t1) {
		float tempt = t1;
		t1 = t0;
		t0 = tempt;	
	} 
	return true;
}

__device__ vec3 atmosphere::computeIncidentLight(const vec3& orig, const vec3& dir, float tmin, float tmax) const
{
	float t0, t1;
	if (!raySphereIntersect(orig, dir, atmosphereRadius, t0, t1) || t1 < 0) return vec3(1,0,0);
	if (t0 > tmin && t0 > 0) tmin = t0;
	if (t1 < tmax) tmax = t1;
	uint32_t numSamples = 16;
	uint32_t numSamplesLight = 8;
	float segmentLength = (tmax - tmin) / numSamples;
	float tCurrent = tmin;
	vec3 sumR(0), sumM(0); // mie and Rayleigh contribution
	float opticalDepthR = 0, opticalDepthM = 0;
	float mu = dot(dir, sunDirection); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction
	float phaseR = 3.f / (16.f * M_PI) * (1 + mu * mu);
	float g = 0.76f;
	float phaseM = 3.f / (8.f * M_PI) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));
	for (uint32_t i = 0; i < numSamples; ++i) {
		vec3 samplePosition = orig + (tCurrent + segmentLength * 0.5f) * dir;
		float height = samplePosition.length() - earthRadius;
		// compute optical depth for light
		float hr = exp(-height / Hr) * segmentLength;
		float hm = exp(-height / Hm) * segmentLength;
		opticalDepthR += hr;
		opticalDepthM += hm;
		// light optical depth
		float t0Light, t1Light;
		raySphereIntersect(samplePosition, sunDirection, atmosphereRadius, t0Light, t1Light);
		float segmentLengthLight = t1Light / numSamplesLight, tCurrentLight = 0;
		float opticalDepthLightR = 0, opticalDepthLightM = 0;
		uint32_t j;
		for (j = 0; j < numSamplesLight; ++j) {
			vec3 samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5f) * sunDirection;
			float heightLight = samplePositionLight.length() - earthRadius;
			if (heightLight < 0) break;
			opticalDepthLightR += exp(-heightLight / Hr) * segmentLengthLight;
			opticalDepthLightM += exp(-heightLight / Hm) * segmentLengthLight;
			tCurrentLight += segmentLengthLight;
		}
		if (j == numSamplesLight) {
			vec3 tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1f * (opticalDepthM + opticalDepthLightM);
			vec3 attenuation(exp(-tau.x()), exp(-tau.y()), exp(-tau.z()));
			sumR += attenuation * hr;
			sumM += attenuation * hm;
		}
		tCurrent += segmentLength;
	}

	// [comment]
	// We use a magic number here for the intensity of the sun (20). We will make it more
	// scientific in a future revision of this lesson/code
	// [/comment]
	return (sumR * betaR * phaseR + sumM * betaM * phaseM) * 1;
}


#endif // !ATMOSPHEREH
