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

- Now, Derivative(Loss)/Derivative(Weight) is got by **CHAIN RULE OF DERIVATIVES**. Consider the following ANN-
  
  <img width="993" height="auto" alt="image" src="https://github.com/user-attachments/assets/ef0395d3-adde-4284-952d-3d7b05872d80" />
- **To update w4**, formula is $` w_{\text{4new}}=w_{\text{4old}}-\eta \cdot \frac{\partial Loss}{\partial w_{4old}} `$

  Now, because Loss is dependent on Output O2 and O2 is dependent on w4, the Derv(Loss)/Derv(w4) $` \frac{\partial Loss}{\partial w_{4old}} `$ can be got by following Chain Rule breakdown -
  
  <img width="400" height="auto" alt="image" src="https://github.com/user-attachments/assets/43ab241d-0539-482a-9f71-7e99339dd187" />

- Similarly, **to update w1**, formula is $` w_{\text{1new}}=w_{\text{1old}}-\eta \cdot \frac{\partial Loss}{\partial w_{1old}} `$

  Now, because Loss is dependent on Output O2 and O2 is dependent on O1 and O1 is dependent on w1, the Derv(Loss)/Derv(w1) $` \frac{\partial Loss}{\partial w_{1old}} `$ can be got by following Chain Rule breakdown -

  <img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/5564376f-05ce-46f5-ba89-8f7d83933ca7" />

  Similarly, w2 & w3 have same breakdown as w1
<hr/>

- Consider the following ANN-

  <img width="1327" height="auto" alt="image" src="https://github.com/user-attachments/assets/ea18bdc5-1764-49b0-bac9-7cd4b5468a8b" />

- **To update w1**, formula is $` w_{\text{1new}}=w_{\text{1old}}-\eta \cdot \frac{\partial Loss}{\partial w_{1old}} `$

  Now, there are two paths from Loss to w1. 1st path is Loss -> O31 -> O21 -> O11 -> w1. 2nd path is Loss -> O31 -> O22 -> O12 -> w1. Then, the Derv(Loss)/Derv(w1) $` \frac{\partial Loss}{\partial w_{1old}} `$ can be got by following Chain Rule breakdown -

  <img width="600" height="auto" alt="image" src="https://github.com/user-attachments/assets/1de2d810-515a-4091-a673-070e11b6cf13" />

### Sigmoid Activation function & Vanishing Gradient Problem
- Consider an ANN like below. In each layer, we use Sigmoid Activation function-
<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/07ff4eae-8b13-4ef1-863f-513c5015954d" />

- Now, we know that to update w1, formula is $` w_{\text{1new}}=w_{\text{1old}}-\eta \cdot \frac{\partial Loss}{\partial w_{1old}} `$
- And, by Chain Rule, $` \frac{\partial Loss}{\partial w_{1old}} = \frac{\partial Loss}{\partial O_{31}} * \frac{\partial O_{31}}{\partial O_{21}} * \frac{\partial O_{21}}{\partial O_{11}} * \frac{\partial O_{11}}{\partial w_{1old}}`$
- Now, to understand, lets evaluate just one of the terms $` \frac{\partial O_{31}}{\partial O_{21}} `$.
  
  We know $` O_{31} = \sigma(w_{3}*O_{21}+b_{3}) `$

  Let $` z = w_{3}*O_{21}+b_{3} `$

  Then,  $` O_{31} = \sigma(z) `$

  And, $` \frac{\partial O_{31}}{\partial O_{21}} = \frac{\partial \sigma(z)}{\partial z} * \frac{\partial z}{\partial O_{21}}`$

  And, $` \frac{\partial O_{31}}{\partial O_{21}} = \frac{\partial \sigma(z)}{\partial z} * \frac{\partial (w_{3}*O_{21}+b_{3})}{\partial O_{21}}`$

  And, $` \frac{\partial O_{31}}{\partial O_{21}} = \frac{\partial \sigma(z)}{\partial z} * w_{3old}`$

  Now, **Derivative of Sigmoid function is always between 0 - 0.25**. And W3 is also initialized to small value (0.01). So, overall $` \frac{\partial O_{31}}{\partial O_{21}} `$ will be very small.

- So, in $` \frac{\partial Loss}{\partial w_{1old}} = \frac{\partial Loss}{\partial O_{31}} * \frac{\partial O_{31}}{\partial O_{21}} * \frac{\partial O_{21}}{\partial O_{11}} * \frac{\partial O_{11}}{\partial w_{1old}}`$ due to Sigmoid Activation function, all other terms will be 0-0.25. And overall $` \frac{\partial Loss}{\partial w_{1old}} `$ will be a very small number (almost zero) and W_new will stop updating and not converge at all ==> This is **VANISHING GRADIENT PROBLEM**
- So, **if the neural network is very deep (has many hidden layers) and each layer uses SIGMOID Activation function, this VANISHING GRADIENT Problem is Prominent and negatively affects Training of ANN** 

<img width="645" height="auto" alt="image" src="https://github.com/user-attachments/assets/dfa0d560-2a98-4296-b630-939c7de0adc3" />


  #### Advantages of Sigmoid
  - Smooth gradient, preventing jumps in output values
  - Output values bound between 0 & 1, normalizing the output of each neuron
  - Clear Prediction close to 0 or 1
  - Suitable for Binary Classification
  
  #### Disadvatages of Sigmoid
  - Prone to VANISHING GRADIENT Problem
  - Function output is not zero-centred i.e. Efficient Weight updation does not happen. Zero-centred function passes through the Origin (0,0)
    <img width="400" height="auto" alt="image" src="https://github.com/user-attachments/assets/126904b7-2b24-46c9-86a4-91c7f973b20b" />
  - Involves Exponential so mathematical computation is resource/time-consuming

- Due to these, **Sigmoid function is used only in Output Layer for Binary Classification**. And researchers started exploring other functions

### Tanh Activation function
<img width="650" height="auto" alt="image" src="https://github.com/user-attachments/assets/75dbbf36-52b5-4b7a-b96b-742e9bb9b741" />
<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/11c8774f-49e9-417f-b67d-64bfd41cd169" />
<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/1ae3eaf7-9a02-436f-91fd-57437e968ca6" />

- for Forward Propagation, output is -1 to 1 and it is Zero-centred (passes through Origin.
- For Backward Propagation, Derivative of Tanh is between 0-1 compared to 0-0.25 of Sigmoid. So, it is also susceptible to Vanishing Gradient Problem
  #### Advantages of Tanh
  - Zero-centred output so efficient weight updation

  #### Disadvantages of Tanh
  - susceptible to Vanishing Gradient Problem
  - Involves multiple Exponentials so mathematical computation is resource/time-consuming
 
### ReLU Activation Function (Rectified Linear Unit)
<img width="663" height="auto" alt="image" src="https://github.com/user-attachments/assets/9969e964-08cf-40c5-b1ad-2375281e9387" />

- For x < 0, ReLU(x) = 0; x > 0, ReLU(x) = x. So, can be written as ReLU(x) = max(0,x)
- And Derv(ReLU) = 0 for x < 0, Derv(ReLU) = 1 for x >= 0

<img width="450" height="auto" alt="image" src="https://github.com/user-attachments/assets/bf88e326-d38a-4c14-98c4-80a04a45d5f7" />
<img width="550" height="auto" alt="image" src="https://github.com/user-attachments/assets/4c313da7-04bc-4bd8-aeeb-36518478e83d" />
<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/9822ff32-e9a1-4123-9853-1b39c9eef09c" />

- If **Derv(ReLU) is 0**, then **weights will not get updated** and it will be **DEAD NEURON**
  #### Advantages of ReLU
  - Solves Vanishing Gradient Problem
  - ReLU(x) = max(0,x); Linear relationship and so superfast to calculate

  #### Disadvantages of ReLU
  - Prone to DEAD NEURON problem
  - Not zero-centred

### Parametric ReLU (Leaky ReLU)
<img width="800" height="auto" alt="image" src="https://github.com/user-attachments/assets/949bc940-f8a5-4d18-aa1e-5577253b8e83" />

- Avoids the DEAD NEURON problem of ReLU by keeping Output a small value for x < 0.
- Parametric ReLU(x) = max(alpha * x, x) where alpha is a hyperparameter with values 0.01, 0.02 etc
- In Back Propagation, Derv(PRelu) is either small Positive number (for x < 0) or 1 (x>0)

  #### Advantages of PReLU
  - Solves Dead Nueron Problem of ReLU
  - All advantages of ReLU
 
  #### Disadvantages of PReLU
  - Not Zero centred so weight updation is not efficient
 
### Exponential Linear Unit (ELU)
<img width="800" height="auto" alt="image" src="https://github.com/user-attachments/assets/b724619f-c7f0-4e34-96f5-521b35082807" />

- In Forward propagation, ELU(x) = x for x>0; ELU(x) = $` \alpha (e^x - 1) `$ for x<0.
- For Backward Propagation, Derv(ELU) is 0 - 1

  #### Advantages of ELU
  - Solves DEAD NEURON problem of RELU
  - Zero-centred so efficient weight updation

  #### Disadvatages of ELU
  - Involves Exponential so bit resource/time-consuming than PReLU

### SOFTMAX (Used in Output Layer For Multi-class classification Problem)
- Consider below ANN that uses the SKLEARN IRIS dataset that takes Petal width & Sepal width as Input and classifies the flower into one of the 3 types- Setosa, Versicolor OR Virginica
<img width="600" height="auto" alt="image" src="https://github.com/user-attachments/assets/08816c52-3413-433a-b12f-d791c5e6da2b" />

- Here, the Outputs are Raw Output values (-infinity to +infinity) and when fed to Softmax function, they will be converted to Predicted Probabilities. So, **SOFTMAX converts Raw Output values (logits) to Predicted Probabilities**. Higher the Raw Value, higher is the Probability

<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/7d48c94c-c926-499f-a855-49e49bd32539" />
<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/f74b5b72-42cf-4fe0-b5b3-d1bd28a866e4" />
<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/c66df199-fa17-404c-823d-912e0d97aa38" />
<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/f4f6ceb2-7099-4e39-be6e-25a276513137" />
<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/79a0904d-94bb-41b1-afb8-208d4f540fc8" />
<img width="500" height="auto" alt="image" src="https://github.com/user-attachments/assets/29c61e93-c321-47a4-a825-0f1bff114fed" />

- In General for K possible Classes, Softmax(Prob i belongs to K Class) = $` \Huge \frac{e^i}{\sum_{j=1}^i e^j} `$
- All Raw Outputs values get converted to Probability value between 0 - 1
- Sum of all SOFTMAX values = 1 (Since they are all probabilities)

### Which Activation/Loss functions to use and when ??
- For **HIDDEN LAYERS**, use **RELU/PReLU/ELU** (to avoid Vanishing Gradient problems of Sigmoid, Tanh)
- For **OUTPUT LAYERS**
  - Use **SIGMOID** for **BINARY Classification**
  - Use **SOFTMAX** for **MULTI-CLASS Classification**
  - Use **LINEAR or NO Function** for **REGRESSION**
- For **LOSS FUNCTION**
  - Use **BINARY CROSS ENTROPY** for **BINARY Classification**
  - Use **CATEGORICAL CROSS ENTROPY** or **SPARSE CATEGORICAL CROSS ENTROPY** for **MULTI-CLASS Classification**
  - Use **MSE, MAE, Huber Loss OR RMSE** for **REGRESSION**
<img width="900" height="auto" alt="image" src="https://github.com/user-attachments/assets/3231dfce-382b-42ae-b9c6-b0c773d7d16b" />

## OPTIMIZERS
- They determine how the model's Weights are updated based on the Gradients (Partial Derivatives) of the Loss with respect to those Weights.
- Examples
  1. Gradient Descent Optimizer
  2. Stochastic Gradient Descent Optimizer (SGD)
  3. Mini-batch SGD
  4. SGD with Momentum
  5. AdaGrad/RMSProp
  6. ADAM  (widely used)
- In ANN, when **One Input Row (1 DataPoint or 1 RECORD)** is fed to do the **FORWARD PROPAGATION->LOSS EVAL->BACK PROPAGATION->WEIGHT UPDATION cycle**, it is called an **ITERATION**.
- When the ANN has **seen all the Input Rows (All DataPoints/Records)** i.e. **FORWARD PROPAGATION->LOSS EVAL->BACK PROPAGATION->WEIGHT UPDATION cycle** has **completed for all the Input Rows**, it is called an **EPOCH**
- Depending on how many Input Rows are fed to the ANN in an iteration, the number of EPOCHS and ITERATIONS varies
 
  ### Gradient Descent optimizer
  - If no. of Input Rows=1000, and **all 1000 will be fed at once** for **FORWARD PROPAGATION->LOSS EVAL->BACK PROPAGATION->WEIGHT UPDATION cycle**, then it is **1 ITERATION** in **1 EPOCH**. You specify the no of EPOCHS to repeat this to minimize the loss function
  - Huge resources needed to compute weights for all datapoints at once

  ### Stochastic Gradient Descent SGD
  - if no. of datapoints=1000, and **each datapoint will be fed one by one** for  **FORWARD PROPAGATION->LOSS EVAL->BACK PROPAGATION->WEIGHT UPDATION cycle**. So, There will be **1000 ITERATIONS** in **1 EPOCH**. You specify the no of EPOCHS to repeat this to minimize the loss function
  - Solves resource issue of Gradient optimizer
  - High processing time since each iteration takes only one datapoint
  - Noise gets introduced (Zigzag behavior around the Gradient Descent curve while moving towards global minima)
  <img width="600" height="auto" alt="image" src="https://github.com/user-attachments/assets/c9d7914e-c7cd-4b03-afb8-04a5a47f0bda" />

  ### Mini Batch SGD
  - if no. of datapoints=1000 and **batch size=100**, **100 datapoint will be fed in batches** for **FORWARD PROPAGATION->LOSS EVAL->BACK PROPAGATION->WEIGHT UPDATION cycle**. There will be **10 ITERATIONS** in **1 EPOCH**
  - Faster convergence than SGD
  - Lesser noise than SGD but still exists

  ### SGD With Momentum
  - To smoothen the noise of SGD, Exponential Weighted/Moving Average of Past Gradients is taken. W-old and W-new are treated as Time Series i.e. W(t-1) & W(t)
  <img width="350" height="auto" alt="image" src="https://github.com/user-attachments/assets/1fceee13-8146-41d6-a637-a39f15c88a70" />
  <img width="450" height="auto" alt="image" src="https://github.com/user-attachments/assets/481b0615-be12-453d-8eac-4e62cf4f834f" />
  <img width="800" height="auto" alt="image" src="https://github.com/user-attachments/assets/dbec697b-a22f-497f-9d16-b99c0fb2e2de" />
  
  ### Adagrad (Adaptive Gradient Descent)-
  - **Dynamic Learning Rate** Eta.
  - Improves speed of convergence.
  - Initially, Eta can be high (0.01) to move quickly to global minima and then it reduces as we move closer to global minima. 
  - Disdvantage- Due to formula of Eta-new (inversely prop to Alpha(t)), as Alpha(t) becomes very large, then Eta-new will be approx. 0 and so Weight updation stops.
    
    <img width="800" height="auto" alt="image" src="https://github.com/user-attachments/assets/436d372c-7896-4507-b657-bfbe9d3e2b46" />

 
  ### AdaDelta & RMSPROP-
  - To prevent Alpha(t) of Adagrad from becoming very high, Exponential Weighted Average is applied to Alpha(t) and is called Sdw
    
  <img width="700" height="auto" alt="image" src="https://github.com/user-attachments/assets/c669cc9d-ef2e-4ade-a243-b95fcbf48740" />

  ### Adam optimizer (Most Widely used)-
  - Combines all the strengths of **SGD with Momentum + RMSPROP (Smoothening + Dynamic Learning Rate)**
  - Most widely used
    
    <img width="700" height="auto" alt="image" src="https://github.com/user-attachments/assets/9975a3a2-a9ab-462f-86ee-25efb962c310" />

### Exploding Gradient Problem
- In **Vanishing Gradient** problem, when **Weights** are initialized at **low values** and **Sigmoid Activation function is used in Hidden Layers** and it is a Deep Neural Network (lots of hidden layers), the **weights stop updating** after multiple forward/back propagations 
- In **Exploding Gradient** problem, when **Weights** are initialized at **high values**, then there is a chance that the new Weights are much bigger or much smaller which causes the **weights to go out of bounds and the Derivative never converges**
  
  <img width="900" height="auto" alt="image" src="https://github.com/user-attachments/assets/69aeec0d-e74c-4d66-9506-dfa1e4429490" />

### Weight Initialization Techniques
- Key points
  - Weights Should be initialized small
  - should NOT be same
  - should have good variance

- In Neural Network, 
  - No of inputs = Column Dimension of Input X Matrix
  - No of outputs = 1 (Binary Classification), N (multi-class classification)

- Techniques-
  1. UNIFORM DISTRIBUTION
       - Weights = UniformDistr(-1/sqrt(no of input), 1/sqrt(no of input))
  3. XAVIER/GLOROT INITIALIZATION
     - Xavier Normal dist-
          - Weights = N(0,sigma); sigma=sqrt(2/(input+output))
     - Xavier Uniform dist-
          - Weights = Uniform(-sqrt(6/(input+output)), sqrt(6/(input+output))
  4. Kaiming He Initialization
       - He Normal Dist-
         - Weights = N(0,sigma); sigma=sqrt(2/input)
       - He Uniform Dist-
         - Weights = Uniform(-sqrt(6/input),sqrt(6/input))
