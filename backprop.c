#include "backprop.h"

static float netPredicted, netTarget, netError;

float SetTarget(float target) {
	netTarget = target;
}

float Prediction() {
	return netPredicted;
}

float Sigmoid(float sop) {
	return 1.f / (1.f + exp(-1.f * sop));
}

float Error(float predicted, float target) {
	return (predicted - target) * (predicted - target);
}

float ErrorPredictedDeriv(float predicted, float target) {
	return 2.f * (predicted - target);
}

float SigmoidSopDeriv(float sop) {
	return Sigmoid(sop) * (1.f - Sigmoid(sop));
}

float SopWDeriv(float x) {
	return x;
}

float UpdateWeight(float w, float grad, float learningRate) {
	return w - (learningRate * grad);
}

InputLayer CreateInputs(int numOfInputs) {
	InputLayer input = (InputLayer) { numOfInputs, (float*)calloc(numOfInputs + 1, sizeof(float)) };
	input.val[numOfInputs] = 1.f;
	return input;
}

void DeleteInputs(InputLayer* input) {
	free(input->val);
	input->val = NULL;
}

HiddenLayer CreateHidden(int numOfNodes) {
	HiddenLayer hidden = (HiddenLayer) { numOfNodes, (Node*)calloc(numOfNodes, sizeof(Node)), NULL, NULL, (float*)calloc(numOfNodes + 1, sizeof(float)) };
	hidden.outputVector[numOfNodes] = 1.f;
	return hidden;
}

void AttachToInput(HiddenLayer* layer, InputLayer* input) {
	layer->upstreamInput = input;
	for (int i = 0; i < layer->size; ++i)
		layer->node[i] = (Node) { input->size + 1, (float*)calloc(input->size + 1, sizeof(float)), 0 };
}

void AttachToLayer(HiddenLayer* layer, HiddenLayer* upstream) {
	layer->upstreamLayer = upstream;
	for (int i = 0; i < layer->size; ++i)
		layer->node[i] = (Node) { upstream->size + 1, (float*)calloc(upstream->size + 1, sizeof(float)), 0 };
}

void DeleteHidden(HiddenLayer* layer) {
	for (int i = 0; i < layer->size; ++i) {
		free(layer->node[i].weight);
		layer->node[i].weight = NULL;
	}
	free(layer->node);
	layer->node = NULL;
	free(layer->outputVector);
	layer->outputVector = NULL;
}

void AttachOutputToLayer(OutputLayer* output, HiddenLayer* upstream) {
	output->upstreamInput = NULL;
	output->upstreamLayer = upstream;
	output->output = (Node) { upstream->size + 1, (float*)calloc(upstream->size + 1, sizeof(float)), 0 };
}

void AttachOutputToInput(OutputLayer* output, InputLayer* upstream) {
	output->upstreamInput = upstream;
	output->upstreamLayer = NULL;
	output->output = (Node) { upstream->size + 1, (float*)calloc(upstream->size + 1, sizeof(float)), 0 };
}

void DeleteOutput(OutputLayer* output) {
	free (output->output.weight);
	output->output.weight = NULL;
}

float ForwardPass(InputLayer* input, HiddenLayer* hidden, int hiddenLayerCount, OutputLayer* out) {
	
	// Copy address of input vector
	float* inputVector = input->val;
	
	// Propagate forward through hidden layers
	for (int layer = 0; layer < hiddenLayerCount; layer++) {
		
		// Multiply the weight vector of each node by the input vector
		for (int n = 0; n < hidden[layer].size; n++) {

			hidden[layer].outputVector[n] = hidden[layer].node[n].val = Sigmoid(VecMult(hidden[layer].node[n].weight, inputVector, hidden[layer].node[n].size));
//			hidden[layer].outputVector[n] = product;
//			hidden[layer].node[n].val = product;
		}
		
		// Get address of new input vector
		inputVector = hidden[layer].outputVector;
	}
	
	// Compute result of forward pass
	out->output.val = VecMult(inputVector, out->output.weight, out->output.size);

	// Compute network error
	netPredicted = Sigmoid(out->output.val);
	netError = Error(netPredicted, netTarget);

	return out->output.val;
}

void UpdateNode(Node* node, float* input) {
	float g1 = ErrorPredictedDeriv(netPredicted, netTarget);
	float g2 = SigmoidSopDeriv(node->val);
	float g = g1 * g2;

	// Compute new weights (input vector must include bias input)
	for (int i = 0; i < node->size; i++) {
		float grad = input[i] * g;
		node->weight[i] = UpdateWeight(node->weight[i], grad, LEARNING_RATE);
	}
/*
	// Compute new bias weight
	float grad = 1.f * g;
	node->weight[node->size - 1] = UpdateWeight(node->weight[node->size - 1], grad, LEARNING_RATE);*/
}

void BackwardPass(InputLayer* input, HiddenLayer* hidden, int hiddenLayerCount, OutputLayer* out) {
	// Compute input vector for output node
	float* inputVector = NULL;
	if (hiddenLayerCount > 0) {
		inputVector = hidden[hiddenLayerCount - 1].outputVector;
	}
	else {
		inputVector = input->val;
	}

	// Update weights for output layer node
	UpdateNode(&out->output, inputVector);

	if (hiddenLayerCount == 0) {
		return;
	}

	// Propagate backwards through hidden layers
	for (int i = hiddenLayerCount - 1; i > 0; i--) {

		// Get address of new input vector from upstream hidden layer
		inputVector = hidden[i - 1].outputVector;

		// Iterate through each node within the current layer
		for (int n = 0; n < hidden[i].size; n++)
			UpdateNode(&hidden[i].node[n], inputVector);
	}

	// Get address of input vector from input layer
	inputVector = input->val;

	// Update weights for first hidden layer
	for (int n = 0; n < hidden[0].size; n++)
		UpdateNode(&hidden[0].node[n], inputVector);
}
