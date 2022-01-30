#ifndef LOSS_FUNCS_H
#define LOSS_FUNCS_H

/// <summary>
/// Mean Squared Error Loss Function
/// </summary>
template <typename Type = float32>
inline Type MSE(const std::vector<Type>& pred,
	const std::vector<Type>& target) throw(...) 
{
	uint size = pred.size();
	if (size != target.size())
		throw std::invalid_argument("Different sizes");
	Type result = Type(0);
	for (uint i = 0; i < size; i++) {
		result += std::pow(pred[i] - target[i], Type(2));
	}
	return result / Type(size);
}

/// <summary>
/// Derivate of MSE Loss Function
/// </summary>
template <typename Type = float32>
inline Type DerivateMSE(const Type& pred,
	const Type& target, const uint& N)
{
	return Type(2)*(pred - target) / Type(N);
}

/// <summary>
/// Least Squares Loss Function
/// </summary>
template <typename Type = float32>
inline Type LS(const std::vector<Type>& pred,
	const std::vector<Type>& target) throw(...)
{
	uint size = pred.size();
	if (size != target.size())
		throw std::invalid_argument("Different sizes");
	Type result = Type(0);
	for (uint i = 0; i < size; i++) {
		result += std::pow(target[i] - pred[i], Type(2));
	}
	return result / Type(2);
}

/// <summary>
/// Cross-Entropy Loss Function
/// </summary>
template <typename Type = float32>
inline Type CE(const std::vector<Type>& pred,
	const std::vector<Type>& target) throw(...)
{
	uint size = pred.size();
	if (size != target.size())
		throw std::invalid_argument("Different sizes");
	Type result = Type(0);
	for (uint i = 0; i < size; i++) {
		result += target[i] * std::log(pred[i]);
	}
	return result * Type(-1);
}

/// <summary>
/// Binary Cross-Entropy Loss Function
/// </summary>
template <typename Type = float32>
inline Type BCE(const std::vector<Type>& pred,
	const std::vector<Type>& target) throw(...)
{
	if (pred.size() != 1 || target.size() != 1)
		throw std::invalid_argument("Many arguments for BCE");
	Type y = target[0], p = pred[0], result = Type(0);
	result = y * std::log(p) + (Type(1) - y)
		* std::log(Type(1) - p);
	return result * Type(-1);
}

#endif // LOSS_FUNCS_H
