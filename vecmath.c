#include "vecmath.h"

float VecMult(float* vec1, float* vec2, int dimensions) {
	float sum = 0.f;
	for (int i = 0; i < dimensions; i++) {
		sum += (vec1[i] * vec2[i]);
	}
	return sum;
}
