#ifndef NNTYPES_H
#define NNTYPES_H

typedef unsigned int uint;		// Unsigned integer number
typedef float float32;			// Float-point 32bit number
typedef double float64;			// Float-point 64bit number
typedef long double float128;	// Float-point 128bit number

/// <summary>
/// Type of Activation Function
/// </summary>
enum act_func_type {
	relu,						// ReLU function
	sigmoid,					// Sigmoid function
	hypertan,					// Tanh function
	identity					// Identity function
};

/// <summary>
/// Type of Loss Function
/// </summary>
enum loss_func_type {
	mse,						// Mean Squared Error
	ls,							// Least Squres
	ce,							// Cross-Entropy
	bce							// Binary Cross-Entropy
};

#endif // NNTYPES_H
