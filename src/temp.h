#pragma once

/*--- Non working Algo #1 ---

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
		UpdateWeights(layers[i], layers[i - 1], lRate);
	}
	UpdateWeightsFirstLayer(layers[0], inTrData, lRate);
}

/// <summary>
/// Calculation Error Vector
/// In Output (Last) Vector
/// </summary>
inline void CalcErrorOutputLayer(NeuralLayer<Type>& outLayer,
	const std::vector<Type>& outTrData)
{
	Type error_i;
	uint lSize = outLayer.neurons.size();
	for (uint i = 0; i < lSize; i++) {
		error_i = Type(0);
		error_i = GetDerivateLossFunc(outLayer.neurons[i].output, outTrData[i], lSize);
		error_i *= outLayer.GetDerivateActiveFunc(i);
		outLayer.errors[i] = error_i;
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

// t4
inline void CalcErrorHideLayer(NeuralLayer<Type>& curLayer,
	NeuralLayer<Type>& prevLayer)
{
	uint curLayerSize = curLayer.neurons.size();	// current layer size
	uint prevLayerSize = prevLayer.neurons.size();	// previous layer size
	for (uint i = 0; i < curLayerSize; i++) {
		for (uint j = 0; j < prevLayerSize; j++) {
			curLayer.errors[i] += prevLayer.errors[j] * prevLayer.neurons[j].weights[i];
		}
		curLayer.errors[i] *= curLayer.GetDerivateActiveFunc(i);
	}
}

// t5
inline void UpdateWeights(NeuralLayer<Type>& curLayer,
	NeuralLayer<Type>& prevLayer, const Type& lRate)
{
	uint curLayerSize = curLayer.neurons.size();	// current layer size
	uint prevLayerSize = prevLayer.neurons.size();	// previous layer size
	for (uint i = 0; i < curLayerSize; i++) {
		for (uint j = 0; j < prevLayerSize; j++) {
			curLayer.neurons[i].weights[j] += (lRate * curLayer.errors[i] * prevLayer.neurons[j].output);
		}
		curLayer.neurons[i].bias += (lRate * curLayer.errors[i]);
		// std::cout << curLayer.errors[i] << std::endl;
	}
}

// t6
inline void UpdateWeightsFirstLayer(NeuralLayer<Type>& curLayer, const std::vector<Type>& inTrData, const Type& lRate) {
	uint neuronsNum = curLayer.neurons.size();
	uint inputsNum = inTrData.size();
	for (uint i = 0; i < neuronsNum; i++) {
		for (uint j = 0; j < inputsNum; j++) {
			curLayer.neurons[i].weights[j] += (lRate * curLayer.errors[i] * inTrData[j]);
		}
		curLayer.neurons[i].bias += (lRate * curLayer.errors[i]);
	}
}

*/

/*--- Non working Algo #2 ---

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
		CalcInputLayer(layers[0], inTrData);
		for (uint i = 0; i < lSize; i++) {
			UpdateWeights(layers[i], lRate);
		}
	}

	
	/// <summary>
	/// Calculation Error Vector
	/// In Output (Last) Layer
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
			error_i *= outLayer.neurons[i].GetDerivateActiveFunc();
			for (uint j = 0; j < nSize; j++) {
				outLayer.errors[i * nSize + j] = error_i;
			}
		}
	}

	/// <summary>
	/// Calculation Error Vector
	/// In Hide Layers
	/// </summary>
	inline void CalcErrorHideLayer(NeuralLayer<Type>& curLayer, NeuralLayer<Type>& prevLayer) {
		Type error_i = Type(0);
		uint curLSize = curLayer.neurons.size(),
			curNSize = curLayer.neurons[0].weights.size(),
			prevLSize = prevLayer.neurons.size(),
			prevNSize = prevLayer.neurons[0].weights.size();
		// Current deltas
		for (uint i = 0; i < curLSize; i++) {
			for (uint j = 0; j < prevLSize; j++) {
				error_i += prevLayer.neurons[j].output * prevLayer.errors[curLSize * j + i];
			}
			error_i *= curLayer.neurons[i].GetDerivateActiveFunc();
			for (uint j = 0; j < curNSize; j++) {
				curLayer.errors[i * curNSize + j] = error_i;
			}
		}
		// Prev deltas
		for (uint i = 0; i < prevLSize; i++) {
			for (uint j = 0; j < prevNSize; j++) {
				prevLayer.errors[i * prevNSize + j] *= curLayer.neurons[j].output;
			}
		}
	}

	/// <summary>
	/// Calculation Error Vector
	/// In Input (First) Layer
	/// </summary>
	inline void CalcInputLayer(NeuralLayer<Type>& inLayer, const std::vector<Type>& inTrData) {
		uint lSize = inLayer.neurons.size();
		uint nSize = inLayer.neurons[0].weights.size();
		for (uint i = 0; i < lSize; i++) {
			for (uint j = 0; j < nSize; j++) {
				inLayer.errors[i * nSize + j] *= inTrData[j];
			}
		}
	}

	/// <summary>
	/// Update all Weights with New Values
	/// </summary>
	inline void UpdateWeights(NeuralLayer<Type>& inLayer, const Type& lRate) {
		uint lSize = inLayer.neurons.size();
		uint nSize = inLayer.neurons[0].weights.size();
		for (uint i = 0; i < lSize; i++) {
			for (uint j = 0; j < nSize; j++) {
				inLayer.neurons[i].weights[j] += lRate * inLayer.errors[i * nSize + j];
			}
		}
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
				std::vector<Type> result(CalcOutputModel((*ptrInTrainData)[data_i]));
				BackPropagation((*ptrInTrainData)[data_i], (*ptrOutTrainData)[data_i], lRate);
				Type loss = MSE(result, (*ptrOutTrainData)[data_i]);
				std::cout << "epo" << epo + 1 << ": error is " << loss << std::endl;
			}
			if (epo % 10 == 0) std::cout << epo << std::endl;
		}

		// Delete Dynamic Vectors
		if (allocIn) delete ptrInTrainData;
		if (allocOut) delete ptrOutTrainData;
	}

*/
