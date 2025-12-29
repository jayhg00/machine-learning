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

### How ANN Learns/updates Weights
Assume we have a small dataset as below with 3 independent features - IQ, Study hours, Play hours & 1 Output feature Pass(1)/Fail(0). So, this is a binary classification problem.

<img width="300" height="auto" alt="image" src="https://github.com/user-attachments/assets/630168d9-48d0-4999-b6a2-f2f171348810" />
<img width="600" height="auto" alt="image" src="https://github.com/user-attachments/assets/adc09f5b-af3c-4d15-84f1-426066a16bea" />

And we have the ANN with 1 Input Layer (3 inputs), 1 Hidden Layer (1 neuron) & 1 Output Layer (1 Neuron since Binary Classification). 
- Weights are w1, w2, w3, w4, Bias are b1 & b2.
- Output of HL1 is O1 & Output Layer is O2.
- Let Weights, bias be initialized as w1=0.01, w2=0.02, w3=0.03; b1=0.001, b2=2

So, we provide 1st datapoint to inputs i.e. x1=95, x2=4, x3=4 and do the math as per formula $` z = \sum_{i=1}^n w_i x_i + b `$
$` z = 95*0.01 + 4*0.02 + 4*0.03 + 0.001 `$

$` z = 1.151 `$

Then, we apply Activation function to z. Consider Sigmoid Activation function $` \sigma(x) = \frac{1}{1 + e^{-x}} `$ which gives value 0 to 1

<img width="600" height="auto" alt="image" src="https://github.com/user-attachments/assets/aaeae2dc-9ba4-4456-afcc-efc2848e3212" />

$` \sigma(z) = \frac{1}{1 + e^{-1.151}} `$

$` O1 = \sigma(z) = 0.759 `$

```math
z = \sum_{i=1}^n w_i x_i + b
```


