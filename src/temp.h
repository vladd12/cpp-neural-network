#pragma once

/*--- Non working Trash ---

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