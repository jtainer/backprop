#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vecmath.h"

#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01f
#endif

#ifndef BACKPROP_H
#define BACKPROP_H

float SetTarget(float target);

float Prediction();

float Sigmoid(float sop);

float Error(float predicted, float target);

float ErrorPredictedDeriv(float predicted, float target);

float SigmoidSopDeriv(float sop);

float SopWDeriv(float x);

float UpdateWeight(float w, float grad, float learningRate);

struct Node {
	int size;
	float* weight;
	float val;
};

struct InputLayer {
	int size;
	float* val;
};

struct HiddenLayer {
	int size;
	struct Node* node;
	struct InputLayer* upstreamInput;
	struct HiddenLayer* upstreamLayer;
	float* outputVector;
};

struct OutputLayer {
	struct Node output;
	struct InputLayer* upstreamInput;
	struct HiddenLayer* upstreamLayer;
};

typedef struct Node Node;

typedef struct InputLayer InputLayer;

typedef struct HiddenLayer HiddenLayer;

typedef struct OutputLayer OutputLayer;

InputLayer CreateInputs(int numOfInputs);

void DeleteInputs(InputLayer* input);

HiddenLayer CreateHidden(int numOfNodes);

void AttachToInput(HiddenLayer* layer, InputLayer* input);

void AttachToLayer(HiddenLayer* layer, HiddenLayer* upstream);

void DeleteHidden(HiddenLayer* layer);

void AttachOutputToLayer(OutputLayer* output, HiddenLayer* upstream);

void AttachOutputToInput(OutputLayer* output, InputLayer* upstream);

void DeleteOutput(OutputLayer* output);

float ForwardPass(InputLayer* input, HiddenLayer* hidden, int hiddenLayerCount, OutputLayer* out);

void UpdateNode(Node* node, float* input);

void BackwardPass(InputLayer* input, HiddenLayer* hidden, int hiddenLayerCount, OutputLayer* out);

#endif
