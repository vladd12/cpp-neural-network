#ifndef NLAYER_H
#define NLAYER_H

#include "neuron.h"

/// <summary>
/// Neural Layer Class
/// </summary>
template <typename Type = float32>
class NeuralLayer {
private:
	uint layerSize;
	uint inputSize;
	act_func_type aFuncType;

public:
	std::vector<Neuron<Type>> neurons;
	std::vector<Type> errors;

	/// <summary>
	/// Default Constructor
	/// </summary>
	explicit inline NeuralLayer() :
		layerSize(1), inputSize(1),
		aFuncType(act_func_type::sigmoid),
		neurons(std::vector<Neuron<Type>>(layerSize,
			Neuron<Type>(inputSize, aFuncType))),
		errors(std::vector<Type>(layerSize * inputSize, Type(0))) { }

	/// <summary>
	/// Constructor with Parameters
	/// </summary>
	explicit inline NeuralLayer(const uint _layerSize,
		const uint _inputSize, const act_func_type _aFuncType):
		layerSize(_layerSize), inputSize(_inputSize),
		aFuncType(_aFuncType), neurons(std::vector<Neuron<Type>>()),
		errors(std::vector<Type>(layerSize * inputSize, Type(0)))
	{
		neurons.reserve(layerSize);
		for (uint i = 0; i < layerSize; i++) {
			neurons.push_back(Neuron<Type>(inputSize, aFuncType));
		}
	}

	/// <summary>
	/// Set Neuron Layer Global Preferences (layer size, 
	/// input size, activation function type, loss function type)
	/// </summary>
	inline void SetNLayerPrefs(const uint _layerSize,
		const uint _inputSize, const act_func_type _aFuncType)
	{
		layerSize = _layerSize;
		inputSize = _inputSize;
		aFuncType = _aFuncType;
		neurons.resize(layerSize, Neuron<Type>(inputSize, aFuncType));
	}

	/// <summary>
	/// Set Neuron Local Data (neurons array)
	/// </summary>
	inline void SetNLayerData(const std::vector<Neuron<Type>>& _neurons) throw(...) {
		for (uint i = 0; i < layerSize; i++) {
			neurons[i] = _neurons[i];
		}
	}

	/// <summary>
	/// Calculation Output for 
	/// Each Neuron in Layer
	/// </summary>
	inline std::vector<Type> CalcLayer(const std::vector<Type>& inputSrc) throw(...) {
		std::vector<Type> results(layerSize, Type());
		for (uint i = 0; i < layerSize; i++) {
			results[i] = neurons[i].CalcOutput(inputSrc);
		}
		return results;
	}

	/// <summary>
	/// Neural Layer Object Data Output
	/// </summary>
	template <typename Type = float32>
	friend std::ostream& operator<<(std::ostream& os,
		const NeuralLayer<Type>& rhs) throw(...);

	template <typename Type>
	friend class NeuralNetwork;
};

/// <summary>
/// Neural Layer Object Data Output
/// </summary>
template <typename Type>
inline std::ostream& operator<<(std::ostream &os,
	const NeuralLayer<Type>& rhs) throw(...)
{
	for (uint i = 0; i < rhs.layerSize; i++) {
		os << "Neuron" << i << ": " << rhs.neurons[i] << '\n';
	}
	os << "NeuralLayer";
	return os;
}

#endif // NLAYER_H
