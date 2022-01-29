#ifndef LOSS_FUNCS_H
#define LOSS_FUNCS_H

template <typename Type = float32>
inline Type MSE(const std::vector<Type>& pred,
	const std::vector<Type>& real) throw(...) 
{
	uint size = pred.size();
	if (size != real.size())
		throw std::invalid_argument("Different sizes");
	Type result = Type(0);
	for (uint i = 0; i < size; i++) {
		result += std::pow(pred[i] - real[i], Type(2));
	}
	return result / Type(size);
}

template <typename Type = float32>
inline Type derivateMSE(const Type& pred, const Type& real, const uint& N) {
	if (N == 1) return (real - pred);
	else return Type(2)*(pred - real) / Type(N);
}

template <typename Type = float32>
inline Type CE(const std::vector<Type>& pred,
	const std::vector<Type>& real) throw(...) 
{
	uint size = pred.size();
	if (size != real.size())
		throw std::invalid_argument("Different sizes");
	Type result = Type(0);
	for (uint i = 0; i < size; i++) {
		result += real[i] * std::log(pred[i]);
	}
	return result * Type(-1);
}

template <typename Type = float32>
inline Type BCE(const std::vector<Type>& pred,
	const std::vector<Type>& real) throw(...) 
{
	if (pred.size() != 1 || real.size() != 1)
		throw std::invalid_argument("Many arguments for BCE");
	Type y = real[0], p = pred[0], result = Type(0);
	result = y * std::log(p) + (Type(1) - y)
		* std::log(Type(1) - p);
	return result * Type(-1);
}



#endif // LOSS_FUNCS_H
