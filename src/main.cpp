#include "neural_network.h"
#include <algorithm>

int main(void) {
	using namespace std;

	NeuralNetwork<float32> nnmodel(1, loss_func_type::ls);
	nnmodel.AddLayer(1, act_func_type::identity);
	
	vector<float32> in(5, float32());
	vector<float32> out(5, float32());
	rand_vectors(in, out);

	nnmodel.SetTrainDataIn(in);
	nnmodel.SetTrainDataOut(out);
	cout << nnmodel << endl;
	nnmodel.TrainModel(10, 1.0f);
	cout << nnmodel << endl;

	float32 test_x = 1.0f, test_y = CelsiumToFahrenheit(test_x);
	cout << "Test X: " << test_x << "\nTest Y: " << test_y;
	float32 pred_Y = nnmodel.CalcOutputModel(vector<float32>(1, test_x))[0];
	cout << "\nPred Y: " << pred_Y << endl;

	system("pause");
	return 0;
}
