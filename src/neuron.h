#ifndef NEURON_H
#define NEURON_H

#include <ctime>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "nntypes.h"
#include "activation_funcs.h"
#include "rand_time.h"

/// <summary>
/// Single Neuron Class
/// </summary>
template <typename Type = float32>
class Neuron {
private:
	uint inputSize;
	act_func_type funcType;

	/// <summary>
	/// Xavier initialisation
	/// </summary>
	inline void WeightsInit() {
		for (uint i = 0; i < inputSize; i++) {
			weights[i] = rand_num(Type());
		}
		bias = rand_num(Type());
	}

public:
	std::vector<Type> weights;
	Type bias, output;

	/// <summary>
	/// Default Constructor
	/// </summary>
	explicit inline Neuron(): inputSize(1),
		funcType(act_func_type::sigmoid),
		weights(std::vector<Type>(inputSize, Type(0))),
		bias(Type(0)), output(Type(0)) { }

	/// <summary>
	/// Constructor with Parameters
	/// </summary>
	explicit inline Neuron(const uint _inputSize, 
		const act_func_type _funcType) :
		inputSize(_inputSize), funcType(_funcType),
		weights(std::vector<Type>(inputSize, Type(0))),
		bias(Type(0)), output(Type(0)) 
	{
		WeightsInit();
	}

	/// <summary>
	/// Set Neuron Global Preferences (input size, activation function type)
	/// </summary>
	inline void SetNeuronPrefs(uint _inputSize, 
		act_func_type _funcType) 
	{
		inputSize = _inputSize;
		funcType = _funcType;
		weights.resize(inputSize, Type());
		XavierInit();
	}

	/// <summary>
	/// Set Neuron Local Data (weights, bias, output)
	/// </summary>
	inline void SetNeuronData(std::vector<Type>& _weights,
		Type _bias, Type _output) throw(...)
	{
		if (weights.size() != _weights.size())
			throw std::length_error("The size of input Weights Vector must be same size Weights Vector in Neuron Object");
		else for (uint i = 0; i < inputSize; i++) {
			weights[i] = _weights[i];
		}
		bias = _bias;
		output = _output;
	}

	/// <summary>
	/// Calculation Neuron Output
	/// </summary>
	inline Type CalcOutput(const std::vector<Type>& inputSrc) throw(...) {
		output = Type(0);
		if (inputSrc.size() != inputSize)
			throw std::length_error("The size of input vector must be same input size in Neuron Object");
		else for (uint i = 0; i < inputSize; i++) {
			output += weights[i] * inputSrc[i];
		}
		switch (funcType) {
		case act_func_type::relu:
			output = ReLU(output);
			break;
		case act_func_type::sigmoid:
			output = Sigmoid(output);
			break;
		case act_func_type::hypertan:
			output = HyperTan(output);
			break;
		default:
			throw std::invalid_argument("Activation Function not defined");
			break;
		}
		return output;
	}

	/// <summary>
	/// Neuron Object Data Output
	/// </summary>
	template<typename Type = float32>
	friend std::ostream& operator<<(std::ostream& os, 
		const Neuron<Type>& rhs) throw(...);
	
};

/// <summary>
/// Neuron Object Data Output
/// </summary>
template<typename Type = float32>
inline std::ostream& operator<<(std::ostream& os, 
	const Neuron<Type>& rhs) throw(...) 
{
	for (uint i = 0; i < rhs.inputSize; i++) {
		os << 'w' << i << '=' << rhs.weights[i] << ", ";
	}
	os << "b=" << rhs.bias << ", y=" << rhs.output << ", act_func=";
	switch (rhs.funcType) {
	case act_func_type::relu:
		os << "relu";
		break;
	case act_func_type::sigmoid:
		os << "sigmoid";
		break;
	case act_func_type::hypertan:
		os << "tanh";
		break;
	default:
		throw std::invalid_argument("Activation Function not defined");
		break;
	}
	return os;
}

#endif // NEURON_H
