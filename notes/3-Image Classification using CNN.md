https://www.youtube.com/watch?v=HGwBXDKFk9I (StatQuest josh Starmer, Neural Networks Part 8: Image Classification using CNN)

## Convolutional Neural Network
- A NN takes numbers as inputs i.e. a Vector (1-D Array)
- For a NN to work with images, the image needs to be Converted into a Input Vector (1-D Array of numbers) before it can be processed by the regular Dense, Fully Connected NN.
This Vector Representation of Image is done by CONVOLUTION + POOLING layer
- Consider a Black and white image of size 6 x 6 pixels. White pixel is denoted by 0 while black pixel is denoted by 1.
  If you just join each pixel row-wise, you will get a 36-input vector (array) which means you will have to compute atleast 36 weights/biases.
  This number of input is OK for such small image. But, in reality, images have much higher resolution (1024x300 pixels) and hence the number of weights/biases
  go into millions and the it will be resource/time-intensive to compute all of them. So, We need to represent the image with lesser number of inputs and not lose much information.
  This **Efficient Vector Representation of Image** is done by **CONVOLUTION + POOLING layer**
  
- CNNs do 3 things to make the image classification practical:-
  1. Reduce number of input nodes
  2. Tolerate small shifts in where the pixels are in the image
  3. Take advantage of the correlations that are observed in complex images (brown pixel generally is surrounded by brown pixels and can be grouped together)

### CONVOLUTION & POOLING
- Consider a Black and white image of size **6 x 6 pixels**. White pixel is denoted by 0 while black pixel is denoted by 1.
- A **Filter** is a smaller patch (**3x3 pixels**) whose pixel intensities are obtained by back-propagation.
- **CONVOLUTION** in CNN means to **Slide/Stride the filter over the Input Image**, perform the **Dot Product** (element-wise product and then sum of them) to get 1st value in **FEATURE-MAPS**. Convolution here is the Dot Product of the filter and the image. Then, the filter is moved one pixel to the right (or 2 or 3 pixels), perform the Dot Product to get next Feature-Map value. When filter reaches the right-most boundary, filter is moved one pixel down and started from left. This gives a Feature Map of 4x4 pixels
- Next, on the **4x4 Feature Map**, we slide the **POOLING filter of 2x2 pixels**. If it is **MAX pooling**, **highest value of the overlay** is selected. Pooling filter do not overlap so we will get a **2x2 POOLED FeatureMap**. These will be the **4 inputs (1-D Array) to a Regular Neural Network**. So, 36 inputs of Original image was compressed to just 4 inputs

  #### Convolution + Pooling example
  - Consider a Black and white images of size **6 x 6 pixels** of letter 'O' & 'X'. White pixel is denoted by 0 while black pixel is denoted by 1.
    
    <kbd><img width="300" height="auto" alt="image" src="https://github.com/user-attachments/assets/1afad3d0-6448-49e7-8298-d067c1f91779" /></kbd>
    <kbd><img width="300" height="auto" alt="image" src="https://github.com/user-attachments/assets/5cc4e698-fbb7-42ea-a407-679bae7fc619" /></kbd>

  - We start with trying to identify 'O'. Lets obtain a filter as shown of size 3x3 pixels and overlay it on the left most corner of the image. Perform DOT PRODUCT (Multiply pixel-wise and add) to get value of +3 and add a bias of -2. This gives value of 1 for the 1st element in FEATURE-MAP
    
    <kbd><img width="700" height="auto" alt="image" src="https://github.com/user-attachments/assets/da7a44ff-9b71-4393-870e-8e730e6d1397" /></kbd>

  - Next, slide the filter to the right by 1 pixel. Perform DOT PRODUCT to get value of +1 and add bias of -2. This gives value of -1 for 2nd element in FEATURE-MAP
    
    <kbd><img width="700" height="auto" alt="image" src="https://github.com/user-attachments/assets/cbe0c5fd-330e-4144-ab5d-a186c8a766c6" /></kbd>

  - Keep sliding the filter and repeat to get all the values of the **4x4 FEATURE-MAP**

    <kbd><img width="200" height="auto" alt="image" src="https://github.com/user-attachments/assets/b2b98e18-328a-4ff2-9a54-e731b98e4678" /></kbd>

  - Next, Pass the 4x4 FEATURE-MAP through **ReLU activation** function. All -ve values will become 0 while +ve values will retain their value

    <kbd><img width="600" height="auto" alt="image" src="https://github.com/user-attachments/assets/9e569c12-eb2f-4752-ae3e-012a5c8fcce3" /></kbd>

  - Next, Apply **POOLING filter of 2x2 pixels** on this 4x4 FEATURE-MAP Post ReLU. Pooling filter slides such that pixels of the base do not overlap. POOLING here is **MAX Pooling** which is to take the max value in the Overlay

    <img width="200" height="auto" alt="image" src="https://github.com/user-attachments/assets/4e2d6b42-fc7d-4d8f-b918-7ca816387f3f" />
    <img width="200" height="auto" alt="image" src="https://github.com/user-attachments/assets/3796410d-d088-40d4-968e-3f28c6b2eb58" />
    <img width="200" height="auto" alt="image" src="https://github.com/user-attachments/assets/85341bcd-6d1b-47a1-8cda-01e98b9b307c" />
    <img width="200" height="auto" alt="image" src="https://github.com/user-attachments/assets/f9719391-ca88-44a4-b714-790609c3371f" />

  - This gives the **2x2 POOLED FEATURE-MAP** which is the 4-input Vector to a regular Neural Network
    
    <kbd><img width="156" height="auto" alt="image" src="https://github.com/user-attachments/assets/931f256c-a72f-4bac-92d0-0f351624184d" /></kbd>

  - Consider the Neural Network. It has 4 input nodes,  1 Hidden Layer with 1 node using ReLU activation function and 2 Output nodes (one for letter 'O', one for letter 'X'). The Weights & Bias are pre-determined to correctly identify 'X' or 'O'.

    <kbd><img width="700" height="auto" alt="image" src="https://github.com/user-attachments/assets/0689da01-bd76-4962-9d4d-2d9ceca858a6" /> </kbd>

  - Applying the inputs [1,0,0,1] and weights/bias, input to Relu function is 0.34 which will output 0.34. Further weights/bias will output 1 in Node 'O' and 0 in Node 'X'. So, it correctly predicted that the input image is an 'O'

    <kbd><img width="700" height="auto" alt="image" src="https://github.com/user-attachments/assets/f8a346a9-fc8a-45e7-b5b4-dbd5317159e0" /> </kbd>

  - Similarly, if we input image of **letter 'X'** to the same NN (same weights/bias), the Feature-map, Pooled Feature-map, input to the NN & Output of NN is as below and it correctly predicts letter 'X'

    <kbd><img width="900" height="auto" alt="image" src="https://github.com/user-attachments/assets/94ffd8ba-c499-482e-95e7-c72f21bdbc1c" /></kbd>

  - Now, if the input image is of **shifted letter 'X'** (shifted to right by 1 pixel) to the same NN (same weights/bias), the Feature-map, Pooled Feature-map, input to the NN & Output of NN is as below and it correctly predicts letter 'X' (Output is 1.23 for 'X' vs -0.2 for 'O'). This is possible because of the Convolution with the filter & Pooling process

    <kbd><img width="900" height="auto" alt="image" src="https://github.com/user-attachments/assets/3ab85b6f-d408-498e-937c-42ebade60150" /></kbd>

<hr/>

## Real world use-case: Fashion Classification
- Use Case: We have a website and the user wants to create a listing in the fashion category. For example, he wants to sell a t-shirt. He uploads a picture and there is a fashion classification service which will get this picture and reply with a suggested category (here: t-shirt).
- This classification service will contain a neural network which will look at the image and predict a category for this image. (out of 10 most popular classe like T-shirts, pants, shirts etc)
- The Images to Train, Validate, Test our model is located in this Github repo "https://github.com/alexeygrigorev/clothing-dataset-small". Within this repo, you have seperate folders for Train, Validate, Test. Within Train, Validate, Test, you have the 1 folder for each class (pants, t-shirt, shirt, etc) containing respective images
- To train and use a CNN, we need to use TENSORFLOW (Open source Deep Learning Framework by Google) + KERAS (high-level API built over TensorFlow). It can be run on a powerful CPU or a GPU. Starting with Tensorflow v2.0, Keras is built into Tensorflow. So, install Tensorflow using ``` pip install tensorflow ``` cmd.

### Loading the image
- Import specific libraries to support image processing & ML
  ```Python
  import numpy as np
  import matplotlib.pyplot as plt
   
  %matplotlib inline
   
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras.preprocessing.image import load_img
  ```
  
- Load an image using PIL (Python Image Library). We resize the image to 150x150 to speed up Training
  ```Python
  path = './clothing-dataset-small/train/t-shirt'
  name = '5f0a3fa0-6a3d-4b68-b213-72766a643de7.jpg'
  fullname = f'{path}/{name}'
  img = load_img(fullname, target_size=(150, 150))  # Resize to smaller size to speed up Training
  print(img)
  img
  ```
  <img width="auto" height="250" alt="image" src="https://github.com/user-attachments/assets/2e113453-b59c-48d0-bed1-35e01d21b065" />
  
  ```Python
  # Output: <PIL.Image.Image image mode=RGB size=150x150 at 0x7F8593FA2E20>
  ```

- Translate image to Numpy array. Each pixel has 3 channels (RGB). So, each pixel is represented by 3 values between 0-255. Shape of the Numpy Array will be (150,150,3) [Height, Width, # of channels]
  ```Python
  x = np.array(img)
  x
   
  # Output:
  # array([[[179, 171,  99],
  #         [179, 171,  99],
  #         [181, 173, 101],
  #         ...,
  #         [251, 253, 248],
  #         [251, 253, 248],
  #         [251, 254, 247]],
  #
  #        [[188, 179, 112],
  #         [187, 178, 111],
  #         [186, 177, 108],
  #         ...,
  #         [251, 252, 247],
  #         [251, 252, 247],
  #         [251, 252, 246]],
  #       ...,
  #         [171, 157,  82],
  #         ...,
  #         [181, 133,  22],
  #         [179, 131,  20],
  #         [182, 134,  23]]], dtype=uint8)

  x.shape
  # Output: (150, 150, 3)
  ```

### Pre-trained Convolutional Neural Networks
- These are CNNs that are already trained on huge dataset of images by somebody else and can be used to classify into 1000s of classes. If the Pre-trained CNN has classes as per our use-case, then we can use them for inference. If the Pre-trained CNN **does not** have classes as per our use-case, then we can **re-use the base Convolution layer** of the Pre-trained CNN to convert our images to vector representation and **replace Pre-trained CNN's Dense layer/Output Classes with our custom-defined Dense layers & Output Classes**. This is called **TRANSFER LEARNING**
- Keras provides many Pre-trained models that we can use for prediction, feature extraction, and fine-tuning https://keras.io/api/applications/. One of them is Xception.

#### Using Xception to predict our images
- Import Xception model libraries
  ```Python
  from tensorflow.keras.applications.xception import Xception
  from tensorflow.keras.applications.xception import preprocess_input
  from tensorflow.keras.applications.xception import decode_predictions
   
  # weights = "imagenet" means we want to use pre-trained network that was trained on imagenet
   
  model = Xception(
      weights="imagenet",
      input_shape=(150, 150, 3)
  )
  ```
- Let us Classify our image using Xception's built-in classes. Xception model expects a bunch of images as input. So, we'll supply only one image. And, also pre-process our input image the same way as done during the Xception training
  ```Python
  X = np.array([x])
  X.shape
  # Output: (1, 150, 150, 3)

  X = preprocess_input(X)
  X[0]
  # Output:
  # array([[[ 0.4039216 ,  0.3411765 , -0.2235294 ],
  #            [ 0.4039216 ,  0.3411765 , -0.2235294 ],
  #            [ 0.41960788,  0.35686278, -0.20784312],
  #            ...,
  #            [ 0.96862745,  0.9843137 ,  0.94509804],
  #            [ 0.96862745,  0.9843137 ,  0.94509804],
  #            [ 0.96862745,  0.99215686,  0.9372549 ]],
  #
  #            [[ 0.47450984,  0.4039216 , -0.12156862],
  #            [ 0.4666667 ,  0.39607847, -0.12941176],
  #            [ 0.45882356,  0.38823533, -0.15294117],
  #             ...,
  #            [ 0.96862745,  0.9764706 ,  0.9372549 ],
  #            [ 0.96862745,  0.9764706 ,  0.9372549 ],
  #            [ 0.96862745,  0.9764706 ,  0.92941177]],
  #             ...,
  #            [ 0.41960788,  0.04313731, -0.827451  ],
  #            [ 0.4039216 ,  0.02745104, -0.84313726],
  #            [ 0.427451  ,  0.05098045, -0.81960785]]], dtype=float32)
  ```
- Predict our image's class using model.predict(). pred.Shape() is (1,1000) meaning there is 1 Prediction and Probabilities for each of the 1000 built-in classes
  ```Python
  pred = model.predict(X)
  # 1/1 [==============================] - 2s 2s/step

  pred.shape 
  # (1, 1000)

  pred 
  # Output:
  # array([[3.23712389e-04, 1.57383955e-04, 2.13493346e-04, 1.52370616e-04,
  #            2.47626507e-04, 3.05036228e-04, 3.20592342e-04, 1.47499406e-04,
  #    ...
  #            3.20827705e-04, 2.70084536e-04, 3.43746680e-04, 2.48680328e-04,
  #            2.78319319e-04, 3.25885747e-04, 1.71753796e-04, 1.73037348e-04]],
  #           dtype=float32)
  ```
- Decode the Predictions i.e. relate the 1000 probabilities with their respective classes to make output human readable
  ```Python
  decode_predictions(pred)
  # Output:
  # [[('n03595614', 'jersey', 0.6819631),
  #   ('n02916936', 'bulletproof_vest', 0.038140077),
  #   ('n04370456', 'sweatshirt', 0.034324776),
  #   ('n03710637', 'maillot', 0.011354236),
  #   ('n04525038', 'velvet', 0.0018453619)]]
  ```
  As per the default Xception model, the image is classified as a Jersey which is not right. In fact, if you see the list of the 1000 classes that ImageNet uses, there is no class like "T-Shirt", "Shirt", "Pants" etc in it and hence it does not work for our use-case. However, we need not train a new model from scratch. We can re-use the Pre-trained model and adapt it to our use-case

### TRANSFER LEARNING
- In a CNN, the CONVOLUTION+POOLING layer converts input image to 1-D VECTOR Representation and then fed to DENSE Layers that do the predictions.
- The CONVOLUTION+POOLING layer is Generic and applicable to any image irrespective to use-case.
- The DENSE Layer is use-case specific and depends on the Image data that was used for Training
- So, we can **keep the CONVOLUTION+POOLING layer** of the Pre-trained model and **train new DENSE layers**. This is the concept of **TRANSFER LEARNING**

  #### Reading training data using ImageDataGenerator
  ```Python
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
   
  train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
   
  train_ds = train_gen.flow_from_directory(
      './clothing-dataset-small/train',
      target_size=(150, 150),
      batch_size=32
  )
   
  # Output: Found 3068 images belonging to 10 classes.
  ```
    - Instantiate ImageDataGenerator by providing the preprocessing function
    - Read the training images from "train" folder in batch of 32 and resize to 150x150
    - Total of 3068 images are present among 10 classes ("pant", "shirt", t-shirt" etc)
  
  - To know the order of the classes, use train_ds.class_indices. "dress" is at index 0, hat is at 1 and so on. The names are inferred from the folder structure
  ```Python
  train_ds.class_indices
   
  # Output: 
  # {'dress': 0,
  #  'hat': 1,
  #  'longsleeve': 2,
  #  'outwear': 3,
  #  'pants': 4,
  #  'shirt': 5,
  #  'shoes': 6,
  #  'shorts': 7,
  #  'skirt': 8,
  #  't-shirt': 9}
  ```
  - Now, train_ds is an iterator which returns the 32 images of the batch and the class that it belongs to.
  ```Python
  X, y = next(train_ds)

  X
  # Output:
  # array([[[[ 0.30980396,  0.20784318,  0.13725495],
  #            [ 0.30980396,  0.16078436,  0.20784318],
  #            [ 0.22352946,  0.04313731,  0.10588241],
  #        ...,
  #            [ 0.32549024,  0.05882359, -0.70980394],
  #            [ 0.3176471 ,  0.07450986, -0.6392157 ],
  #            [ 0.3176471 ,  0.09019613, -0.6313726 ]]]], dtype=float32)
   
  X.shape     
  # Output: (32, 150, 150, 3)

  y
  # Output:
  # array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
  #             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
  #             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
  #             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
  #             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]], dtype=float32)

  y.shape
  # Output: (32, 10)
  ```
    - y is like one-hot encoded. If 1st image is a t-shirt, then index 9 is 1 while all other elements are 0. If 4th image is Pants, then index 4 is 1 and all other elements are 0

  - Similarly, we create the Validation Dataset. It contains 341 images belonging to 10 classes
  ```Python
  val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
   
  val_ds = val_gen.flow_from_directory(
      './clothing-dataset-small/validation',
      target_size=(150, 150),
      batch_size=32,
      shuffle=False
  )
   
  # Output: Found 341 images belonging to 10 classes.
  ```
  
  #### Training the Xception model for our use-case
  - We define a make_model function
  ```Python
  1   def make_model(learning_rate=0.01, size_inner=100, droprate=0.5):
  2     base_model = Xception(
  3         weights='imagenet',
  4         include_top=False,
  5         input_shape=(150, 150, 3)
  6     )
  7 
  8     base_model.trainable = False
  9 
  10     #### CUSTOM DENSE LAYER ######
  11 
  12     inputs = keras.Input(shape=(150, 150, 3))
  13     base = base_model(inputs, training=False)
  14     vectors = keras.layers.GlobalAveragePooling2D()(base)
  15     
  16     inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
  17     drop = keras.layers.Dropout(droprate)(inner)
  18     
  19     outputs = keras.layers.Dense(10)(drop)
  20     
  21     model = keras.Model(inputs, outputs)
  22     
  23     #########################################
  24 
  25     optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
  26     loss = keras.losses.CategoricalCrossentropy(from_logits=True)
  27 
  28     model.compile(
  29         optimizer=optimizer,
  30         loss=loss,
  31         metrics=['accuracy']
  32     )
  33     
  34     return model
  ```
    - Function takes **learning_rate**, size of hidden layer(no of nodes in hidden layer), **Dropout** rate.
    - base_model is the Xception model trained with weights of ImageNet. In Keras, the neural network is visualized from bottom to top with Input & convolution layer at Bottom and Dense & Output Layers on the Top.
      <kbd><img width="600" height="auto" alt="image" src="https://github.com/user-attachments/assets/93362f96-e708-4f5c-a025-1af036c7fb85" /></kbd>

      As we dont want the default Dense Layers of the Xception model, we set include_top = False. Also, we dont want to overwrite the weights of the Base Convolution Model, so we set base_model.trainable = False
    - Then, we define the model Architecture. Code above uses the **FUNCTIONAL MODEL** syntax where variables in one line are passed as arguments to the layer in next line.
      - Input is 150x150x3
      - 'input' goes to base_model and gives output 'base'
      - 'base' goes to 'vectors' base is converted to vectors by POOLING Layer GlobalAveragePooling2D().
      - 'vectors' goes to 'inner'. 'inner' is the Hidden Layer. Default number of nodes in it is 100. Each node has ReLU as Output Activation function
      - 'inner' goes to 'drop'. Dropout layer with Default droprate = 0.5 is between 'inner' and the Output layer
      - 'drop' goes to 'output. Output layer is Dense Layer with 10 nodes (since we have 10 prediction classes)
      - model builds Dense model using 'input', 'output'
      - Optimizer is Adam with default learning rate = 0.01.
      - Loss function is CategoricalCrossEntropy since we are doing Multi-class classification. from_logits=True means the Output layer gives raw outputs without applying any activation function to normalize the value between a defined range
      - Then, compile the model with metric=accuracy and return it.
  - Set learning rate, inner size, dropout rate & Call the make_model function. Then, train the model using model.fit
    ```Python
    learning_rate = 0.001
    size = 100
    droprate = 0.2
    
    model = make_model(
        learning_rate=learning_rate,
        size_inner=size,
        droprate=droprate
    )
    
    history = model.fit(train_ds, epochs=20, validation_data=val_ds)
    ```
      - You need to specify no of Epochs. 1 EPOCH is when the model has gone through all of the Training data once. EPOCH = 10 means the model will go through the Training dataset 10 times to optimize the weights/biases
      - model.fit() returns a history of all the training steps. Note the Training Accuracy, Val_loss & Val_accuracy. After each EPOCH, Val_loss should generally reduce while Training Accuracy & Validation Accuracy should increase. Training no of steps = 96 batches because Training Images were 3068, divide it by 32 will need 96 batches. Validation no of steps = 11 batches because Validation images were 341, divide it by 32 will need 11 batches
    ```Python
    Train for 96 steps, validate for 11 steps
    Epoch 1/10
    96/96 [==============================] - 21s 216ms/step - loss: 1.1353 - accuracy: 0.6170 - val_loss: 0.7258 - val_accuracy: 0.7801
    Epoch 2/10
    96/96 [==============================] - 16s 168ms/step - loss: 0.6469 - accuracy: 0.7735 - val_loss: 0.6332 - val_accuracy: 0.7859
    Epoch 3/10
    96/96 [==============================] - 16s 169ms/step - loss: 0.5182 - accuracy: 0.8243 - val_loss: 0.5905 - val_accuracy: 0.8094
    Epoch 4/10
    96/96 [==============================] - 16s 170ms/step - loss: 0.4390 - accuracy: 0.8553 - val_loss: 0.5550 - val_accuracy: 0.8152
    Epoch 5/10
    96/96 [==============================] - 16s 170ms/step - loss: 0.3827 - accuracy: 0.8827 - val_loss: 0.5437 - val_accuracy: 0.8211
    Epoch 6/10
    96/96 [==============================] - 16s 170ms/step - loss: 0.3342 - accuracy: 0.8990 - val_loss: 0.5319 - val_accuracy: 0.8358
    Epoch 7/10
    96/96 [==============================] - 16s 166ms/step - loss: 0.2978 - accuracy: 0.9149 - val_loss: 0.5169 - val_accuracy: 0.8299
    Epoch 8/10
    96/96 [==============================] - 16s 166ms/step - loss: 0.2652 - accuracy: 0.9283 - val_loss: 0.5422 - val_accuracy: 0.8299
    Epoch 9/10
    96/96 [==============================] - 16s 166ms/step - loss: 0.2419 - accuracy: 0.9387 - val_loss: 0.5234 - val_accuracy: 0.8211
    Epoch 10/10
    96/96 [==============================] - 16s 166ms/step - loss: 0.2157 - accuracy: 0.9498 - val_loss: 0.5320 - val_accuracy: 0.8328
    ```
  - You can try different values for learning rate, inner layer, dropout rate; capture the val_accuracy score for each value and compare the plots. in below example, we try different values for Dropout rate:-
    ```Python
    learning_rate = 0.001
    size = 100
    
    scores = {}
    
    for droprate in [0.0, 0.2, 0.5, 0.8]:
        model = make_model(
            learning_rate=learning_rate,
            size_inner=size,
            droprate=droprate
        )
        history = model.fit(train_ds, epochs=30, validation_data=val_ds)
        scores[droprate] = history.history

    for droprate, hist in scores.items():
      plt.plot(hist['val_accuracy'], label=('val=%s' % droprate))
  
    plt.ylim(0.78, 0.86)
    plt.legend()
    ```
    <img width="380" height="252" alt="image" src="https://github.com/user-attachments/assets/6f2f663a-7fde-4bf3-92b0-cf89e1233e47" />

    Dropout of 0.0 or 0.2 is more consistent.
