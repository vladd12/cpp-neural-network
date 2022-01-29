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
	/// Calculation Back Propagation Algorithm
	/// </summary>
	inline void BackPropagation(const std::vector<Type>& inTrData, 
		const std::vector<Type>& outTrData, const Type& lRate)
	{
		uint lSize = layers.size();
		CalcErrorOutputLayer(layers[lSize - 1], outTrData);
		for (int i = lSize - 2; i >= 0; i--) {
			CalcErrorHideLayer(layers[i], layers[i + 1]);
		}
		for (uint i = lSize - 1; i > 0; i--) {
			//UpdateWeights(layers[i], layers[i - 1], lRate);
		}
		//UpdateWeightsFirstLayer(layers[0], inTrData, lRate);
	}

	/// <summary>
	/// Calculation Error Vector
	/// In Output (Last) Vector
	/// </summary>
	inline void CalcErrorOutputLayer(NeuralLayer<Type>& outLayer,
		const std::vector<Type>& outTrData)
	{
		Type error_i;
		uint lSize = outLayer.neurons.size(), nSize;
		for (uint i = 0; i < lSize; i++) {
			nSize = outLayer.neurons[i].weights.size();
			error_i = Type(0);
			error_i = GetDerivateLossFunc(outLayer.neurons[i].output, outTrData[i], lSize);
			error_i *= outLayer.GetDerivateActiveFunc(i);
			for (uint j = 0; j < nSize; j++) {
				outLayer.errors[i * nSize + j] = error_i;
			}
		}
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
			result = derivateMSE(pred, real, N);
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

	// t1
	inline void CalcErrorHideLayer(NeuralLayer<Type>& curLayer, NeuralLayer<Type>& prevLayer) {
		Type error_i = Type(0);
		uint curLSize = curLayer.neurons.size(), curNSize;
		uint prevLSize = prevLayer.neurons.size(), prevNSize;
		for (uint i = 0; i < curLSize; i++) {
			for (uint j = 0; j < prevLSize; j++) {
				error_i += prevLayer.neurons[j].output * prevLayer.errors[curLSize * j + i];
			}
			error_i *= prevLayer.GetDerivateActiveFunc(i);
		}
	}

public:
	std::vector<NeuralLayer<Type>> layers;

	/// <summary>
	/// Default Constructor
	/// </summary>
	explicit inline NeuralNetwork() : inputSize(1),
		gradSize(0), prevSize(inputSize),
		lFuncType(loss_func_type::mse),
		allocIn(false), allocOut(false),
		ptrInTrainData(nullptr), ptrOutTrainData(nullptr),
		layers(std::vector<NeuralLayer<Type>>()) { }

	/// <summary>
	/// Constructor with Parameters
	/// </summary>
	explicit inline NeuralNetwork(const uint _inputSize,
		const loss_func_type _lFuncType) :
		inputSize(_inputSize), gradSize(0),
		prevSize(inputSize), lFuncType(_lFuncType),
		allocIn(false), allocOut(false),
		ptrInTrainData(nullptr), ptrOutTrainData(nullptr),
		layers(std::vector<NeuralLayer<Type>>()) { }

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
		for (uint epo = 0; epo < epochs; epo++) {
			for (uint data_i = 0; data_i < inSize; data_i++) {
				// Calculate Output of Model for Current Input Data
				CalcOutputModel((*ptrInTrainData)[data_i]);
				BackPropagation((*ptrInTrainData)[data_i], (*ptrOutTrainData)[data_i], lRate);
			}
			if (epo % 10 == 0) std::cout << epo << std::endl;
		}

		// Delete Dynamic Vectors
		if (allocIn) delete ptrInTrainData;
		if (allocOut) delete ptrOutTrainData;
	}

	template<typename Type = float32>
	friend std::ostream& operator<<(std::ostream& os,
		const NeuralNetwork<Type>& rhs) throw(...);

};

template<typename Type = float32>
inline std::ostream& operator<<(std::ostream& os, const NeuralNetwork<Type>& rhs) throw(...) {
	uint nwSize = rhs.layers.size();
	for (uint i = 0; i < nwSize; i++) {
		os << rhs.layers[i] << i << '\n';
	}
	return os;
}

#endif // NNETWORK_H
