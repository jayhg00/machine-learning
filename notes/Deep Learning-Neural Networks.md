Artificial Neural Networks (ANN) try to replicate learning similar to neurons in human brain. 

## Perceptron (Single Layer Neuron) Intuition
<img width="484" height="402" alt="image" src="https://github.com/user-attachments/assets/04391b6e-db9f-4ae7-ac70-67aa7269486b" />

* Input Layer = No of independent variables.
* Hidden Layer = 1 Neuron
* Output Layer = 1 Output for Binary Classification, N output for N-Multiclass Classification, 1 Output for Regression
* Each Input connected to neuron with some **WEIGHTS** and some **BIAS**. STEP 1 of Neuron does WiXi+b. Its Intermediate Output is passed to STEP 2 **Activation function** to keep the output between a specific range (0 to 1 for Sigmoid function or -1 to 1 for Tanh function, etc). For binary Classification problem, we would keep threshold = 0.5 for Sigmoid and threshold = 0 for Tanh. If Activiation Func Output < Threshold, then Final Output = 0 else it is 1. This flow from Input to Output is called **FORWARD PROPAGATION**
* Perceptron is nothing but a linear classifier
* Training of Perceptron (Learning of Perceptron) means to determine the optimal Weights & Bias to correctly predict Output

## ANN (Multi-layer Neuron) Concepts
- Artifical Neural Network is made up of multiple layers of multiple neurons called **HIDDEN or DENSE Layers**. Each layer is **FULLY CONNECTED** to Next Layer --> Each neuron in Layer I is connected to every neuron in Layer I+1 through Weights
- So, there are many Weights and Biases that need to be determined. To speed up training an ANN, following concepts are used-
  - FORWARD PROPAGATION
  - BACK PROPAGATION
  - LOSS FUNCTION
  - ACTIVATION FUNCTION
  - OPTIMIZERS
- 
