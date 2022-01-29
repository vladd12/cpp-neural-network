#include "neural_network.h"
#include <algorithm>

int main(void) {
	using namespace std;

	NeuralNetwork<float32> nnmodel(1, loss_func_type::mse);
	nnmodel.AddLayer(100, act_func_type::sigmoid);
	nnmodel.AddLayer(1, act_func_type::sigmoid);
	
	vector<float32> in(500, float32());
	vector<float32> out(500, float32());
	rand_vectors(in, out);

	nnmodel.SetTrainDataIn(in);
	nnmodel.SetTrainDataOut(out);
	cout << nnmodel << endl;
	nnmodel.TrainModel(500, 0.1f);
	cout << nnmodel << endl;

	float32 test_x = -0.5, test_y = std::sin(test_x);
	cout << "Test X: " << test_x << "\nTest Y: " << test_y;
	float32 pred_Y = nnmodel.CalcOutputModel(vector<float32>(1, test_x))[0];
	cout << "\nPred Y: " << pred_Y << endl;

	sort(in.begin(), in.end());
	for (auto& in_i : in) {
		cout << "X: " << in_i << "\t\tY: " <<
			nnmodel.CalcOutputModel(vector<float32>(1, in_i))[0] << endl;
	}

	system("pause");
	return 0;
}
