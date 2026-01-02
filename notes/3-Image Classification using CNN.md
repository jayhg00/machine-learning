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
