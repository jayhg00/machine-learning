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
