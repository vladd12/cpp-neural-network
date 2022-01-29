#ifndef ACT_FUNCS_H
#define ACT_FUNCS_H

#include <cmath>

/// <summary>
/// Activation Function ReLU 
/// </summary>
template <typename Type = float32>
inline Type ReLU(Type& in) {
	if (in < Type(0)) return Type(0);
	else return in;
}

/// <summary>
/// Derivative Activation Function ReLU.
/// in - result of ReLU function
/// </summary>
template <typename Type = float32>
inline Type derivateReLU(Type& in) {
	if (in == Type(0)) return Type(0);
	else return Type(1);
}

/// <summary>
/// Activation Function Sigmoid 
/// </summary>
template <typename Type = float32>
inline Type Sigmoid(Type& in) {
	return Type(1.0 / (1.0 + std::exp(in * (-1.0))));
}

/// <summary>
/// Derivative Activation Function Sigmoid.
/// in - result of Sigmoid function
/// </summary>
template <typename Type = float32>
inline Type derivateSigmoid(Type& in) {
	return in * (Type(1) - in);
}

/// <summary>
/// Activation Function Hyper Tang
/// </summary>
template <typename Type = float32>
inline Type HyperTan(Type& in) {
	return Type(std::tanh(in));
}

/// <summary>
/// Derivative Activation Function Hyper Tang.
/// in - result of HyperTan function
/// </summary>
template <typename Type = float32>
inline Type derivateHyperTan(Type& in) {
	return Type(1) - (in * in);
}

#endif // ACT_FUNCS_H
