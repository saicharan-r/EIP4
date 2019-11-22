# EIP4
================================================================================================
Session 2 
================================================================================================

Logs for the 20 epochs
-------------------------------------------------------

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 7s 121us/step - loss: 0.5355 - acc: 0.8500 - val_loss: 0.0905 - val_acc: 0.9803
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 5s 91us/step - loss: 0.2642 - acc: 0.9209 - val_loss: 0.0587 - val_acc: 0.9880
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 5s 91us/step - loss: 0.2047 - acc: 0.9388 - val_loss: 0.0469 - val_acc: 0.9883
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 6s 93us/step - loss: 0.1740 - acc: 0.9460 - val_loss: 0.0385 - val_acc: 0.9898
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 6s 93us/step - loss: 0.1565 - acc: 0.9480 - val_loss: 0.0323 - val_acc: 0.9910
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 6s 92us/step - loss: 0.1427 - acc: 0.9494 - val_loss: 0.0310 - val_acc: 0.9911
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 6s 92us/step - loss: 0.1337 - acc: 0.9515 - val_loss: 0.0305 - val_acc: 0.9911
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 5s 90us/step - loss: 0.1272 - acc: 0.9525 - val_loss: 0.0276 - val_acc: 0.9929
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 5s 91us/step - loss: 0.1208 - acc: 0.9539 - val_loss: 0.0254 - val_acc: 0.9926
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 5s 91us/step - loss: 0.1134 - acc: 0.9553 - val_loss: 0.0265 - val_acc: 0.9920
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 5s 91us/step - loss: 0.1105 - acc: 0.9554 - val_loss: 0.0223 - val_acc: 0.9933
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 6s 92us/step - loss: 0.1091 - acc: 0.9562 - val_loss: 0.0228 - val_acc: 0.9935
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 5s 92us/step - loss: 0.1066 - acc: 0.9556 - val_loss: 0.0217 - val_acc: 0.9930
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 5s 91us/step - loss: 0.1051 - acc: 0.9552 - val_loss: 0.0203 - val_acc: 0.9944
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 6s 93us/step - loss: 0.1013 - acc: 0.9562 - val_loss: 0.0228 - val_acc: 0.9932
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0990 - acc: 0.9571 - val_loss: 0.0214 - val_acc: 0.9937
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0964 - acc: 0.9570 - val_loss: 0.0175 - val_acc: 0.9945
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0973 - acc: 0.9564 - val_loss: 0.0193 - val_acc: 0.9947
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 5s 91us/step - loss: 0.0963 - acc: 0.9575 - val_loss: 0.0195 - val_acc: 0.9938
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 5s 90us/step - loss: 0.0950 - acc: 0.9568 - val_loss: 0.0196 - val_acc: 0.9941
<keras.callbacks.History at 0x7f652d81a470>

------------------------------------------------------------------
Model.evaluate(score)
------------------------------------------------------------------
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
[0.019617296113586053, 0.9941]
------------------------------------------------------------------

Approach : 
Strategy for reducing the number of parameters - used 16 channel kernels to reduce the parameters to below 15K and since bias is also not used it is also reducing the number of parameters (not significantly though)
Strategy for improving the efficiency - to get the accuracy to 99.4% the techniques used are

   Scheduler - to decrease the difference between accuracies of training data set and test data set, this helps to achieve higher accuracy as the for each more and more epochs.
   Dropouts - To decrease overfitting and improve overall accuracy
   Batch normalisation - Though batch normalisation is used to increase the running time efficiency, it also helps to initialise  weights more easily and helps the activation function to give more accurate values, hence also adding to improving the efficiency.
   Activation function - Using activation function since we are filtering unwanted results we are improving the efficiency.
   
   
====================================================================================================================================









Definition of the following words

Convolution
 Convolution can be defined as the mathematical opearation that is perfromed between the input layer values and kernal values to give output values or convolved layer.

Filters/Kernels
  Filters/Kernels are layers that extract specific features from the input image. Features can be varied from edges, lines, patterns, objects, etc.

Epochs
  Epoch indicate the total input data that is fed to the network both forward and backward once. We need to run multiple epochs to properly train the data.

Feature Maps
  Feature Maps are nothing but kernels which are used to extract specific features from the input image such as edges, lines, patterns etc

1x1 convolution
 1x1 convolution is mainly used to decrease the channels of the input image so that the parameters for the next layer of convolution can be decreased.

3x3 convolution
  3x3 convolution is the convolution using 3x3 matrix. It is the basic convolution, we use 3x3 multiple times to see the entire image. Advantage of using multiple 3x3 instead of higher order matrices is it decreased the number of parameters used for the same receptive field output.

Receptive field
  Receptive field can be defined as the number of input pixels seen by a particular pixel in the output layer. Here seen can be interpreted as the features extracted by the kernel for the particular input pixels.

Activation function
 After convolution is done from Kernel there is a need to either interpret the value as maximum or minimum depending on the type of pattern, to achieve this we use Activation function.
