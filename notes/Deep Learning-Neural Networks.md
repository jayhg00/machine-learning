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

And we have the ANN with 1 Input Layer (3 inputs), 2 Hidden Layer (1 neuron each) & 1 Output Layer (1 Neuron since Binary Classification). 
- Weights are w1, w2, w3, w4, Bias are b1 & b2.
- Output of HL1 is O1 & HL2 is O2.
- Let Weights, bias be initialized as w1=0.01, w2=0.02, w3=0.03, w4=0.02; b1=0.001, b2=0.03

So, we provide 1st datapoint to inputs i.e. x1=95, x2=4, x3=4 and do the math as per formula $` z = \sum_{i=1}^n w_i x_i + b `$
$` z = 95*0.01 + 4*0.02 + 4*0.03 + 0.001 `$

$` z = 1.151 `$

Then, we apply Activation function to z. Consider Sigmoid Activation function $` \sigma(x) = \frac{1}{1 + e^{-x}} `$ which gives value 0 to 1

<img width="600" height="auto" alt="image" src="https://github.com/user-attachments/assets/aaeae2dc-9ba4-4456-afcc-efc2848e3212" />

$` \sigma(z) = \frac{1}{1 + e^{-1.151}} `$

$` O1 = \sigma(z) = 0.759 `$

**For HL2**, $` z = O1*w4 + b2 `$

$` z = 0.759*0.02 + 0.03 = 0.04518 `$

Consider Sigmoid Activation function for O2

$` O2 = \sigma(z) = \frac{1}{1 + e^{-0.04518}}  = 0.51129`$

Now, O2 is Predicted Output $` y_{pred} `$  for DataPoint 1.

Compute the **LOSS FUNCTION** **$` = (y-y_{pred}) = 1-0.51129 = approx  0.49 `$** 

_Note: for simplicity of LOSS Function, we are just taking difference  between Actual & Predicted Output; In reality, the actual Loss Function depends on the Problem that we are solving. For Regression, it could be MSE, MAE; For Classification it could be Binary Cross Entropy Or Categorical Cross Entropy_

- Till now, the network performed the **FORWARD PROPAGATION Step** (Input -> Hidden -> Output) to check the value of Loss Function for current Weights/Bias for DataPoint 1.
- Now, Neural Network needs to **Minimize this Loss Function value** which it does by **updating the weights & bias using BACK PROPAGATION** (i.e. Output -> Hidden -> Input) for Datapoint 1.
- Then, the next datapoint 2 is fed to NN with updated weights/biases, evaluate Loss function for dataPoint2 and do Back Propagation to update weight/bias.
This Forward-Loss Function Eval-Back cycle is done for each Datapoint to get the Min Loss Func Value and final Weights/Bias. 
- **How quickly and efficiently** (min steps) the Weights/Bias Converge through the **Loss Func Eval & Back Propagation** is controlled by the use of **OPTIMIZERS**
- This is the Way **NN learns/trains.**

### Backpropagation & Weight updation
- Plot of Loss (assume $` = (y-y_{pred})^2 `$) VS Weight is Gradient Descent Curve
<img width="400" height="auto" alt="image" src="https://github.com/user-attachments/assets/f8dad88e-3968-4049-a066-471091616473" />

so, we start at some initial Weight value with some Loss Value. Depending on the slope at that point, Weight is updated to move towards Global Minima where Slope = 0.

- Backpropagation uses Derivatives of Loss Func wrt Weights to update the Weights/Bias. This is the **Weight updation Formula**

$` w_{\text{new}}=w_{\text{old}}-\eta \cdot \frac{\partial Loss}{\partial w} `$

where $` w_{\text{new}} `$: The updated weight.

$` w_{\text{old}} `$: The current weight.

$` \eta (eta) `$: The **learning rate**, a small positive constant (e.g., 0.01) that controls the step size.

- Learning Rate needs to be optimally set. If too low, then updation occurs in smaller steps and hence takes longer to reach Global minima. If too high, then w-value oscillates on both sides of GLobal Minima and can lead to Exploding Gradient problem and never converge
- When Derivative(Loss)/Derivative(Weight) = 0 , then w-new = w-old and hence no weight updation occurs. The final weight/bias has been reached
