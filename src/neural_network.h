#ifndef NNETWORK_H
#define NNETWORK_H

#include "neural_layer.h"
#include "loss_funcs.h"

/// <summary>
/// Neural Network Class
/// </summary>
template <typename Type = float32>
class NeuralNetwork {
private:
	uint inputSize, gradSize, prevSize;
	loss_func_type lFuncType;
	bool allocIn, allocOut;
	std::vector<std::vector<Type>>* ptrInTrainData;		// Pointer to Input Dataset Vector
	std::vector<std::vector<Type>>* ptrOutTrainData;	// Pointer to Output Dataset Vector

	/// <summary>
	/// Calculation Loss Function
	/// between Result of Model and Target Values
	/// </summary>
	inline Type CalcLossFunc(const std::vector<Type>& pred,
		const std::vector<Type>& target) throw(...)
	{
		Type result;
		switch (lFuncType) {
		case loss_func_type::mse:
			result = MSE(pred, target);
			break;
		case loss_func_type::ls:
			result = LS(pred, target);
			break;
		case loss_func_type::ce:
			result = CE(pred, target);
			break;
		case loss_func_type::bce:
			result = BCE(pred, target);
			break;
		default:
			// throw()
			break;
		}
		return result;
	}

	/// <summary>
	/// Calculation Derivate of
	/// Loss Function from Output
	/// </summary>
	inline Type GetDerivateLossFunc(const Type& pred,
		const Type& real, const uint& N)
	{
		Type result;
		switch (lFuncType) {
		case loss_func_type::mse:
			result = DerivateMSE(pred, real, N);
			break;
		case loss_func_type::ls:
			/// TODO: Do it later
			break;
		case loss_func_type::bce:
			/// TODO: Do it later
			break;
		case loss_func_type::ce:
			/// TODO: Do it later
			break;
		default:
			// throw()
			break;
		}
		return result;
	}

	/// <summary>
	/// Calculation Gradient
	/// of Loss Function from Output
	/// </summary>
	inline void CalcGradient(const std::vector<Type>& inTrData,
		const std::vector<Type>& outTrData, const Type& trueLoss,
		const Type& delta)
	{
		uint netSize = layers.size();
		Type temp, deltaLoss, derivateLoss;
		for (uint i = 0; i < netSize; i++) {
			for (uint j = 0; j < layers[i].layerSize; j++) {
				for (uint k = 0; k < layers[i].neurons[j].inputSize; k++) {
					temp = layers[i].neurons[j].weights[k];
					layers[i].neurons[j].weights[k] += delta;
					std::vector<Type> error(CalcOutputModel(inTrData));
					deltaLoss = CalcLossFunc(error, outTrData);
					derivateLoss = (deltaLoss - trueLoss) / delta;
					gradient.push_back(derivateLoss);
					layers[i].neurons[j].weights[k] = temp;
				}
				temp = layers[i].neurons[j].bias;
				layers[i].neurons[j].bias += delta;
				std::vector<Type> error(CalcOutputModel(inTrData));
				deltaLoss = CalcLossFunc(error, outTrData);
				derivateLoss = (deltaLoss - trueLoss) / delta;
				gradient.push_back(derivateLoss);
				layers[i].neurons[j].bias = temp;
			}
		}
	}

	/// <summary>
	/// Update Weights with
	/// Current Gradient
	/// </summary>
	inline void MakeGradStep(const Type& lRate) {
		uint netSize = layers.size(), offset = 0;
		for (uint i = 0; i < netSize; i++) {
			for (uint j = 0; j < layers[i].layerSize; j++) {
				for (uint k = 0; k < layers[i].neurons[j].inputSize; k++) {
					std::cout << layers[i].neurons[j].weights[k] << " - (";
					layers[i].neurons[j].weights[k] -= lRate * gradient[offset];
					std::cout << gradient[offset] << " * " << lRate << ") = "
						<< layers[i].neurons[j].weights[k] << std::endl;
					offset++;
				}
				std::cout << layers[i].neurons[j].bias << " - (";
				layers[i].neurons[j].bias -= lRate * gradient[offset];
				std::cout << gradient[offset] << " * " << lRate << ") = "
					<< layers[i].neurons[j].bias << std::endl;
				offset++;
			}
		}
		gradient.clear();
	}
	
public:
	std::vector<NeuralLayer<Type>> layers;
	std::vector<Type> gradient;

	/// <summary>
	/// Default Constructor
	/// </summary>
	explicit inline NeuralNetwork() : inputSize(1),
		gradSize(0), prevSize(inputSize),
		lFuncType(loss_func_type::mse),
		allocIn(false), allocOut(false),
		ptrInTrainData(nullptr), ptrOutTrainData(nullptr),
		layers(std::vector<NeuralLayer<Type>>()),
		gradient(std::vector<Type>()) { }

	/// <summary>
	/// Constructor with Parameters
	/// </summary>
	explicit inline NeuralNetwork(const uint _inputSize,
		const loss_func_type _lFuncType) :
		inputSize(_inputSize), gradSize(0),
		prevSize(inputSize), lFuncType(_lFuncType),
		allocIn(false), allocOut(false),
		ptrInTrainData(nullptr), ptrOutTrainData(nullptr),
		layers(std::vector<NeuralLayer<Type>>()),
		gradient(std::vector<Type>()) { }

	/// <summary>
	/// Function Adding Neural Layer in Network Model
	/// </summary>
	inline void AddLayer(uint layerSize, act_func_type aFuncType) {
		// Num of inputs = num neurons in prev layer
		// For 1st layer num of inputs = inputSize
		layers.push_back(NeuralLayer<Type>(layerSize, prevSize, aFuncType));
		gradSize += layerSize * (prevSize + 1);
		prevSize = layerSize;
	}

	/// <summary>
	/// Set Train Input Dataset. 1st version
	/// </summary>
	inline void SetTrainDataIn(const std::vector<Type>& inAtr)
	{
		using std::vector;
		uint size = inAtr.size();
		ptrInTrainData = new vector<vector<Type>>
			(size, vector<Type>(1, Type()));
		allocIn = true;
		for (uint i = 0; i < size; i++) {
			(*ptrInTrainData)[i][0] = inAtr[i];
		}
	}

	/// <summary>
	/// Set Train Input Dataset. 2nd version
	/// </summary>
	inline void SetTrainDataIn(const std::vector<std::vector<Type>>& inAtr) {
		using std::vector;
		ptrInTrainData = const_cast<vector<vector<Type>>*>(&inAtr);
	}

	/// <summary>
	/// Set Train Output Dataset. 1st version
	/// </summary>
	inline void SetTrainDataOut(const std::vector<Type>& outVal) {
		using std::vector;
		uint size = outVal.size();
		ptrOutTrainData = new vector<vector<Type>>
			(size, vector<Type>(1, Type()));
		allocOut = true;
		for (uint i = 0; i < size; i++) {
			(*ptrOutTrainData)[i][0] = outVal[i];
		}
	}

	/// <summary>
	/// Set Train Output Dataset. 2nd version
	/// </summary>
	inline void SetTrainDataOut(const std::vector<std::vector<Type>>& outVal) {
		using std::vector;
		ptrOutTrainData = const_cast<vector<vector<Type>>*>(&outVal);
	}

	/// <summary>
	/// Calculation Output Vector for Network Model
	/// </summary>
	inline std::vector<Type> CalcOutputModel(const std::vector<Type>& inTrData)
	{
		uint layersNum = layers.size();
		std::vector<Type> result(layers[0].CalcLayer(inTrData));
		for (uint i = 1; i < layersNum; i++) {
			result = layers[i].CalcLayer(result);
		}
		return result;
	}

	/// <summary>
	/// Main Model Function, that Train Model on Dataset
	/// </summary>
	inline void TrainModel(const uint epochs, const Type lRate) {
		// Train Model Start
		uint inSize = ptrInTrainData->size();
		gradient.reserve(gradSize);
		for (uint epo = 0; epo < epochs; epo++) {
			for (uint data_i = 0; data_i < inSize; data_i++) {
				// Calculate Output of Model for Current Input Data
				std::vector<Type> result(CalcOutputModel((*ptrInTrainData)[data_i]));
				// Calculate Loss of Model
				Type loss(CalcLossFunc(result, (*ptrOutTrainData)[data_i]));
				// Calculate Gradient of Current State Model
				CalcGradient((*ptrInTrainData)[data_i],
					(*ptrOutTrainData)[data_i], loss, lRate);
				// Update Weights of Current Model
				MakeGradStep(lRate);
			}
			// if (epo % 10 == 0) std::cout << epo << std::endl;
		}

		// Delete Dynamic Vectors
		if (allocIn) delete ptrInTrainData;
		if (allocOut) delete ptrOutTrainData;
		gradient.shrink_to_fit();
	}

	/// <summary>
	/// Neural Network Object Data Output
	/// </summary>
	template<typename Type = float32>
	friend std::ostream& operator<<(std::ostream& os,
		const NeuralNetwork<Type>& rhs) throw(...);

};

/// <summary>
/// Neural Network Object Data Output
/// </summary>
template<typename Type = float32>
inline std::ostream& operator<<(std::ostream& os, const NeuralNetwork<Type>& rhs) throw(...) {
	uint nwSize = rhs.layers.size();
	for (uint i = 0; i < nwSize; i++) {
		os << rhs.layers[i] << i << '\n';
	}
	return os;
}

#endif // NNETWORK_H
