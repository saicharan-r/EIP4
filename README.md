# EIP4

Definition of the following words

Convolution
 Convolution can be defined as the mathematical opearation that is perfromed between the input layer values and kernal values to give output values or convolved layer.

Filters/Kernels
  Filters/Kernels are layers that extract specific features from the input image. Features can be varied from edges, lines, patterns, objects, etc.

Epochs
  Epochs

Feature Maps
  Feature Maps are nothing but kernels which are used to extract specific features from the input image such as edges, lines, patterns etc

1x1 convolution
 1x1 convolution is mainly used to decrease the channels of the input image so that the parameters for the next layer of convolution can be decreased.

3x3 convolution
  3x3 convolution is the convolution using 3x3 matrix. It is the basic convolution, we use 3x3 multiple times to see the entire image. Advantage of using multiple 3x3 instead of higher order matrices is it decreased the number of parameters used for the same receptive field output.

Receptive field
  Receptive field can be defined as the number of input pixels seen by a particular pixel in the output layer. Here seen can be interpreted as the features extracted by the kernel for the particular input pixels.

Activation function
