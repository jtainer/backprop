#include <stdio.h>
#include "backprop.h"

int main() {
	// Create network
	const int numOfLayers = 32;
	InputLayer input = CreateInputs(32);
	HiddenLayer hidden[numOfLayers];
	for (int i = 0; i < numOfLayers; i++)
		hidden[i] = CreateHidden(32);
	OutputLayer output;
	AttachToInput(&hidden[0], &input);
	for (int i = 0; i < numOfLayers - 1; i++)
		AttachToLayer(&hidden[i + 1], &hidden[i]);
	AttachOutputToLayer(&output, &hidden[numOfLayers - 1]);
	
	// Setup input values and target value
	input.val[0] = 0.1f;
	input.val[1] = 0.7f;
	SetTarget(0.1f);

	for (int i = 0; i < 10000; i++) {
		ForwardPass(&input, hidden, numOfLayers, &output);
		if (i % 1000 == 0) printf("%f\n", Prediction());
		BackwardPass(&input, hidden, numOfLayers, &output);
	}

	// Delete network
	DeleteInputs(&input);
	for (int i = 0; i < numOfLayers; i++)
		DeleteHidden(&hidden[i]);
	DeleteOutput(&output);
	return 0;
}
