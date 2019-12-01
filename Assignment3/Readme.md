======================= # SESSION 3============================================
Base Model Accuracy 

Accuracy on test data is: 82.17 (base model)

Accuracy on my model is : 82.33 (50th epoch)

--------------------------------------------------------------------------

Netowrk architecture

# Define the model
model = Sequential()
model.add(SeparableConv2D(32, 3, 3, border_mode='same',  input_shape=(32, 32, 3)))#30 #R3eceptive field =
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

#model.add(Dropout(0.25))
model.add(SeparableConv2D(32,3,3))#28  #Receptive field =5
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(64,3,3))#26 #Receptive field =7
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(128,3,3))#24 #Receptive field =9
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(256,3,3))#22 #Receptive field =11
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))#11 #Receptive field =12
model.add(SeparableConv2D(32,3,3))#9 #Receptive field = 16
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(64,3,3))#7 #Receptive field =20
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(128,3,3))#5 #Receptive field =24
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(SeparableConv2D(128,3,3))#3 #Receptive field = 28
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(SeparableConv2D(10,4,4)) 
model.add(Activation('relu'))

model.add(Flatten())

model.add(Activation('softmax'))



# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

-----------------------------------------------------------------------------------------------------------------------
Model.summary()

Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_1 (Separabl (None, 32, 32, 32)        155       
_________________________________________________________________
activation_9 (Activation)    (None, 32, 32, 32)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 32)        128       
_________________________________________________________________
dropout_6 (Dropout)          (None, 32, 32, 32)        0         
_________________________________________________________________
separable_conv2d_2 (Separabl (None, 30, 30, 32)        1344      
_________________________________________________________________
activation_10 (Activation)   (None, 30, 30, 32)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 30, 30, 32)        128       
_________________________________________________________________
dropout_7 (Dropout)          (None, 30, 30, 32)        0         
_________________________________________________________________
separable_conv2d_3 (Separabl (None, 28, 28, 64)        2400      
_________________________________________________________________
activation_11 (Activation)   (None, 28, 28, 64)        0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 28, 28, 64)        256       
_________________________________________________________________
dropout_8 (Dropout)          (None, 28, 28, 64)        0         
_________________________________________________________________
separable_conv2d_4 (Separabl (None, 26, 26, 128)       8896      
_________________________________________________________________
activation_12 (Activation)   (None, 26, 26, 128)       0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 26, 26, 128)       512       
_________________________________________________________________
dropout_9 (Dropout)          (None, 26, 26, 128)       0         
_________________________________________________________________
separable_conv2d_5 (Separabl (None, 24, 24, 256)       34176     
_________________________________________________________________
activation_13 (Activation)   (None, 24, 24, 256)       0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 24, 24, 256)       1024      
_________________________________________________________________
dropout_10 (Dropout)         (None, 24, 24, 256)       0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 12, 12, 256)       0         
_________________________________________________________________
separable_conv2d_6 (Separabl (None, 10, 10, 32)        10528     
_________________________________________________________________
activation_14 (Activation)   (None, 10, 10, 32)        0         
_________________________________________________________________
batch_normalization_6 (Batch (None, 10, 10, 32)        128       
_________________________________________________________________
dropout_11 (Dropout)         (None, 10, 10, 32)        0         
_________________________________________________________________
separable_conv2d_7 (Separabl (None, 8, 8, 64)          2400      
_________________________________________________________________
activation_15 (Activation)   (None, 8, 8, 64)          0         
_________________________________________________________________
batch_normalization_7 (Batch (None, 8, 8, 64)          256       
_________________________________________________________________
dropout_12 (Dropout)         (None, 8, 8, 64)          0         
_________________________________________________________________
separable_conv2d_8 (Separabl (None, 6, 6, 128)         8896      
_________________________________________________________________
activation_16 (Activation)   (None, 6, 6, 128)         0         
_________________________________________________________________
batch_normalization_8 (Batch (None, 6, 6, 128)         512       
_________________________________________________________________
dropout_13 (Dropout)         (None, 6, 6, 128)         0         
_________________________________________________________________
separable_conv2d_9 (Separabl (None, 4, 4, 128)         17664     
_________________________________________________________________
activation_17 (Activation)   (None, 4, 4, 128)         0         
_________________________________________________________________
batch_normalization_9 (Batch (None, 4, 4, 128)         512       
_________________________________________________________________
dropout_14 (Dropout)         (None, 4, 4, 128)         0         
_________________________________________________________________
separable_conv2d_10 (Separab (None, 1, 1, 10)          3338      
_________________________________________________________________
activation_18 (Activation)   (None, 1, 1, 10)          0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_19 (Activation)   (None, 10)                0         
=================================================================
Total params: 93,253
Trainable params: 91,525
Non-trainable params: 1,728
_________________________________________________________________






------------------------------------------------------------------------------------------------------------------------
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
390/390 [==============================] - 42s 107ms/step - loss: 1.4826 - acc: 0.4651 - val_loss: 2.6908 - val_acc: 0.4002
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
390/390 [==============================] - 38s 98ms/step - loss: 1.0553 - acc: 0.6255 - val_loss: 1.2398 - val_acc: 0.6049
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
390/390 [==============================] - 38s 98ms/step - loss: 0.9102 - acc: 0.6795 - val_loss: 0.9558 - val_acc: 0.6723
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
390/390 [==============================] - 38s 98ms/step - loss: 0.8259 - acc: 0.7086 - val_loss: 0.8423 - val_acc: 0.7145
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
390/390 [==============================] - 38s 98ms/step - loss: 0.7664 - acc: 0.7296 - val_loss: 0.9490 - val_acc: 0.6714
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
390/390 [==============================] - 38s 98ms/step - loss: 0.7179 - acc: 0.7496 - val_loss: 0.7715 - val_acc: 0.7329
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
390/390 [==============================] - 38s 98ms/step - loss: 0.6830 - acc: 0.7608 - val_loss: 0.7121 - val_acc: 0.7555
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
390/390 [==============================] - 38s 98ms/step - loss: 0.6543 - acc: 0.7708 - val_loss: 0.6568 - val_acc: 0.7709
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
390/390 [==============================] - 38s 98ms/step - loss: 0.6278 - acc: 0.7796 - val_loss: 0.6514 - val_acc: 0.7795
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
390/390 [==============================] - 38s 98ms/step - loss: 0.6063 - acc: 0.7861 - val_loss: 0.6306 - val_acc: 0.7844
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
390/390 [==============================] - 38s 98ms/step - loss: 0.5887 - acc: 0.7931 - val_loss: 0.6191 - val_acc: 0.7880
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
390/390 [==============================] - 38s 98ms/step - loss: 0.5699 - acc: 0.8000 - val_loss: 0.6282 - val_acc: 0.7848
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
390/390 [==============================] - 38s 98ms/step - loss: 0.5526 - acc: 0.8070 - val_loss: 0.6218 - val_acc: 0.7866
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
390/390 [==============================] - 38s 98ms/step - loss: 0.5446 - acc: 0.8093 - val_loss: 0.6183 - val_acc: 0.7897
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
390/390 [==============================] - 38s 98ms/step - loss: 0.5311 - acc: 0.8145 - val_loss: 0.6127 - val_acc: 0.7872
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
390/390 [==============================] - 38s 97ms/step - loss: 0.5222 - acc: 0.8172 - val_loss: 0.5742 - val_acc: 0.8026
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
390/390 [==============================] - 38s 98ms/step - loss: 0.5103 - acc: 0.8214 - val_loss: 0.6695 - val_acc: 0.7738
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
390/390 [==============================] - 38s 98ms/step - loss: 0.5050 - acc: 0.8227 - val_loss: 0.5789 - val_acc: 0.8011
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
390/390 [==============================] - 38s 97ms/step - loss: 0.4942 - acc: 0.8261 - val_loss: 0.6043 - val_acc: 0.7930
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
390/390 [==============================] - 38s 97ms/step - loss: 0.4873 - acc: 0.8294 - val_loss: 0.5905 - val_acc: 0.8007
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004065041.
390/390 [==============================] - 38s 97ms/step - loss: 0.4825 - acc: 0.8317 - val_loss: 0.5901 - val_acc: 0.8024
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000389661.
390/390 [==============================] - 38s 98ms/step - loss: 0.4743 - acc: 0.8335 - val_loss: 0.5683 - val_acc: 0.8077
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003741581.
390/390 [==============================] - 38s 98ms/step - loss: 0.4715 - acc: 0.8344 - val_loss: 0.5689 - val_acc: 0.8079
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003598417.
390/390 [==============================] - 38s 98ms/step - loss: 0.4596 - acc: 0.8395 - val_loss: 0.6777 - val_acc: 0.7778
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003465804.
390/390 [==============================] - 38s 97ms/step - loss: 0.4580 - acc: 0.8384 - val_loss: 0.5573 - val_acc: 0.8120
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003342618.
390/390 [==============================] - 38s 98ms/step - loss: 0.4483 - acc: 0.8429 - val_loss: 0.5674 - val_acc: 0.8078
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003227889.
390/390 [==============================] - 38s 98ms/step - loss: 0.4473 - acc: 0.8440 - val_loss: 0.5746 - val_acc: 0.8053
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003120774.
390/390 [==============================] - 38s 97ms/step - loss: 0.4416 - acc: 0.8449 - val_loss: 0.5622 - val_acc: 0.8106
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.000302054.
390/390 [==============================] - 38s 97ms/step - loss: 0.4435 - acc: 0.8447 - val_loss: 0.5702 - val_acc: 0.8090
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002926544.
390/390 [==============================] - 38s 97ms/step - loss: 0.4345 - acc: 0.8482 - val_loss: 0.5538 - val_acc: 0.8136
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002838221.
390/390 [==============================] - 38s 97ms/step - loss: 0.4319 - acc: 0.8478 - val_loss: 0.5878 - val_acc: 0.8068
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002755074.
390/390 [==============================] - 38s 97ms/step - loss: 0.4268 - acc: 0.8506 - val_loss: 0.5497 - val_acc: 0.8157
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000267666.
390/390 [==============================] - 38s 98ms/step - loss: 0.4211 - acc: 0.8525 - val_loss: 0.5456 - val_acc: 0.8175
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002602585.
390/390 [==============================] - 38s 98ms/step - loss: 0.4181 - acc: 0.8529 - val_loss: 0.5613 - val_acc: 0.8127
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.00025325.
390/390 [==============================] - 38s 97ms/step - loss: 0.4189 - acc: 0.8547 - val_loss: 0.5832 - val_acc: 0.8073
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002466091.
390/390 [==============================] - 38s 97ms/step - loss: 0.4125 - acc: 0.8538 - val_loss: 0.5623 - val_acc: 0.8126
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002403076.
390/390 [==============================] - 38s 97ms/step - loss: 0.4090 - acc: 0.8582 - val_loss: 0.5636 - val_acc: 0.8155
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002343201.
390/390 [==============================] - 38s 97ms/step - loss: 0.4069 - acc: 0.8567 - val_loss: 0.5598 - val_acc: 0.8159
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002286237.
390/390 [==============================] - 38s 97ms/step - loss: 0.3997 - acc: 0.8582 - val_loss: 0.5719 - val_acc: 0.8116
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
390/390 [==============================] - 38s 97ms/step - loss: 0.4072 - acc: 0.8577 - val_loss: 0.5400 - val_acc: 0.8207
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002180233.
390/390 [==============================] - 38s 97ms/step - loss: 0.4005 - acc: 0.8593 - val_loss: 0.5419 - val_acc: 0.8204
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002130833.
390/390 [==============================] - 38s 97ms/step - loss: 0.3981 - acc: 0.8603 - val_loss: 0.5493 - val_acc: 0.8200
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002083623.
390/390 [==============================] - 38s 97ms/step - loss: 0.3899 - acc: 0.8620 - val_loss: 0.5658 - val_acc: 0.8146
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002038459.
390/390 [==============================] - 38s 97ms/step - loss: 0.3976 - acc: 0.8599 - val_loss: 0.5435 - val_acc: 0.8218
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0001995211.
390/390 [==============================] - 38s 97ms/step - loss: 0.3859 - acc: 0.8630 - val_loss: 0.5397 - val_acc: 0.8226
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001953761.
390/390 [==============================] - 38s 98ms/step - loss: 0.3865 - acc: 0.8640 - val_loss: 0.5426 - val_acc: 0.8188
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001913998.
390/390 [==============================] - 38s 97ms/step - loss: 0.3854 - acc: 0.8640 - val_loss: 0.5501 - val_acc: 0.8174
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001875821.
390/390 [==============================] - 38s 97ms/step - loss: 0.3811 - acc: 0.8666 - val_loss: 0.5740 - val_acc: 0.8137
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001839137.
390/390 [==============================] - 38s 97ms/step - loss: 0.3862 - acc: 0.8639 - val_loss: 0.5756 - val_acc: 0.8119
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.000180386.
390/390 [==============================] - 38s 97ms/step - loss: 0.3800 - acc: 0.8654 - val_loss: 0.5436 - val_acc: 0.8233
Model took 1907.97 seconds to train

----------------------------------------------------------------------------------------------------------------------------------
