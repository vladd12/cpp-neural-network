#ifndef RAND_TIME_H
#define RAND_TIME_H

#include <climits>
#include <random>
#include "nntypes.h"

inline int rand_num(int start, int end) {
	std::random_device random_device;
	std::mt19937 generator(random_device());
	std::uniform_int_distribution<> distr(start, end);
	int res = distr(generator);
	return res;
}

inline float32 rand_num(float32) {
	return float32(rand_num(INT32_MIN, INT32_MAX)) / float32(INT32_MAX);
}

inline float64 rand_num(float64) {
	return float64(rand_num(INT32_MIN, INT32_MAX)) / float64(INT32_MAX);
}

inline float128 rand_num(float128) {
	return float128(rand_num(INT32_MIN, INT32_MAX)) / float128(INT32_MAX);
}

template <typename Type=float32>
inline void rand_vectors(std::vector<Type>& in, std::vector<Type>& out) {
	const int small = 200000000;
	uint size = in.size();
	for (uint i = 0; i < size; i++) {
		in[i] = Type(rand_num(INT32_MIN / small, INT32_MAX / small))
			+ rand_num(Type());
		out[i] = Type(std::cin(in[i]));
		//std::cout << "x: " << in[i] << "\t\ty: " << out[i] << '\n';
	}
}

template <typename Type = float32>
inline void nonrand_vectors(std::vector<Type>& in, std::vector<Type>& out) {
	uint size = in.size();
	for (uint i = 0; i < size; i++) {
		in[i] = Type(-10) + Type(2 + i * 5);
		out[i] = CelsiumToFahrenheit(in[i]);
		std::cout << "x: " << in[i] << "\t\ty: " << out[i] << '\n';
	}
}

template <typename Type = float32>
inline Type CelsiumToFahrenheit(Type& in) {
	return (in * Type(1.8)) + Type(32);
}

#endif // RAND_TIME_H
