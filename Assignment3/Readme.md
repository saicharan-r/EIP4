=======================SESSION 3============================================
Base Model Accuracy 

Accuracy on test data is: 82.41

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


------------------------------------------------------------------------------------------------------------------------

LOGS

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
390/390 [==============================] - 42s 107ms/step - loss: 1.4905 - acc: 0.4650 - val_loss: 1.8303 - val_acc: 0.4839
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022796353.
390/390 [==============================] - 36s 93ms/step - loss: 1.0759 - acc: 0.6205 - val_loss: 1.1044 - val_acc: 0.6267
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018382353.
390/390 [==============================] - 36s 93ms/step - loss: 0.9202 - acc: 0.6737 - val_loss: 0.9053 - val_acc: 0.6857
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015400411.
390/390 [==============================] - 36s 93ms/step - loss: 0.8219 - acc: 0.7098 - val_loss: 0.9058 - val_acc: 0.6861
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013250883.
390/390 [==============================] - 36s 93ms/step - loss: 0.7567 - acc: 0.7325 - val_loss: 0.7490 - val_acc: 0.7416
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011627907.
390/390 [==============================] - 37s 94ms/step - loss: 0.7077 - acc: 0.7522 - val_loss: 0.7798 - val_acc: 0.7280
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010359116.
390/390 [==============================] - 37s 94ms/step - loss: 0.6662 - acc: 0.7643 - val_loss: 0.7064 - val_acc: 0.7538
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009339975.
390/390 [==============================] - 36s 93ms/step - loss: 0.6328 - acc: 0.7775 - val_loss: 0.6903 - val_acc: 0.7643
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008503401.
390/390 [==============================] - 37s 94ms/step - loss: 0.6061 - acc: 0.7865 - val_loss: 0.6564 - val_acc: 0.7734
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.000780437.
390/390 [==============================] - 37s 94ms/step - loss: 0.5796 - acc: 0.7952 - val_loss: 0.7129 - val_acc: 0.7553
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007211538.
390/390 [==============================] - 36s 93ms/step - loss: 0.5571 - acc: 0.8033 - val_loss: 0.6598 - val_acc: 0.7709
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0006702413.
390/390 [==============================] - 36s 93ms/step - loss: 0.5465 - acc: 0.8075 - val_loss: 0.6440 - val_acc: 0.7751
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006260434.
390/390 [==============================] - 36s 93ms/step - loss: 0.5268 - acc: 0.8143 - val_loss: 0.6651 - val_acc: 0.7774
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.000587314.
390/390 [==============================] - 36s 93ms/step - loss: 0.5117 - acc: 0.8184 - val_loss: 0.6324 - val_acc: 0.7833
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005530973.
390/390 [==============================] - 36s 93ms/step - loss: 0.4990 - acc: 0.8228 - val_loss: 0.6307 - val_acc: 0.7880
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005226481.
390/390 [==============================] - 36s 93ms/step - loss: 0.4846 - acc: 0.8285 - val_loss: 0.6241 - val_acc: 0.7892
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.0004953765.
390/390 [==============================] - 36s 93ms/step - loss: 0.4739 - acc: 0.8344 - val_loss: 0.6324 - val_acc: 0.7880
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004708098.
390/390 [==============================] - 37s 94ms/step - loss: 0.4682 - acc: 0.8329 - val_loss: 0.6202 - val_acc: 0.7944
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004485646.
390/390 [==============================] - 36s 93ms/step - loss: 0.4573 - acc: 0.8386 - val_loss: 0.6522 - val_acc: 0.7812
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0004283267.
390/390 [==============================] - 36s 93ms/step - loss: 0.4504 - acc: 0.8424 - val_loss: 0.6295 - val_acc: 0.7882
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004098361.
390/390 [==============================] - 36s 93ms/step - loss: 0.4362 - acc: 0.8478 - val_loss: 0.6248 - val_acc: 0.7906
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0003928759.
390/390 [==============================] - 36s 93ms/step - loss: 0.4343 - acc: 0.8459 - val_loss: 0.6393 - val_acc: 0.7930
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003772636.
390/390 [==============================] - 37s 94ms/step - loss: 0.4228 - acc: 0.8513 - val_loss: 0.6275 - val_acc: 0.7978
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003628447.
390/390 [==============================] - 37s 94ms/step - loss: 0.4206 - acc: 0.8510 - val_loss: 0.6229 - val_acc: 0.7970
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003494874.
390/390 [==============================] - 36s 93ms/step - loss: 0.4157 - acc: 0.8530 - val_loss: 0.6273 - val_acc: 0.7938
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003370787.
390/390 [==============================] - 36s 93ms/step - loss: 0.4071 - acc: 0.8554 - val_loss: 0.6112 - val_acc: 0.8010
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003255208.
390/390 [==============================] - 36s 93ms/step - loss: 0.4037 - acc: 0.8581 - val_loss: 0.6359 - val_acc: 0.7937
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003147293.
390/390 [==============================] - 36s 93ms/step - loss: 0.3997 - acc: 0.8590 - val_loss: 0.6122 - val_acc: 0.8004
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0003046304.
390/390 [==============================] - 37s 94ms/step - loss: 0.3954 - acc: 0.8590 - val_loss: 0.6263 - val_acc: 0.7958
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002951594.
390/390 [==============================] - 36s 93ms/step - loss: 0.3860 - acc: 0.8623 - val_loss: 0.6209 - val_acc: 0.7991
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002862595.
390/390 [==============================] - 36s 93ms/step - loss: 0.3808 - acc: 0.8637 - val_loss: 0.6346 - val_acc: 0.7983
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002778807.
390/390 [==============================] - 36s 94ms/step - loss: 0.3759 - acc: 0.8651 - val_loss: 0.6291 - val_acc: 0.8020
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0002699784.
390/390 [==============================] - 36s 93ms/step - loss: 0.3759 - acc: 0.8653 - val_loss: 0.6203 - val_acc: 0.8011
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002625131.
390/390 [==============================] - 36s 93ms/step - loss: 0.3707 - acc: 0.8679 - val_loss: 0.6232 - val_acc: 0.8015
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0002554496.
390/390 [==============================] - 36s 93ms/step - loss: 0.3732 - acc: 0.8679 - val_loss: 0.6243 - val_acc: 0.8032
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002487562.
390/390 [==============================] - 36s 93ms/step - loss: 0.3632 - acc: 0.8717 - val_loss: 0.6398 - val_acc: 0.7956
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002424047.
390/390 [==============================] - 36s 94ms/step - loss: 0.3610 - acc: 0.8714 - val_loss: 0.6245 - val_acc: 0.8032
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002363694.
390/390 [==============================] - 36s 93ms/step - loss: 0.3562 - acc: 0.8718 - val_loss: 0.6441 - val_acc: 0.8017
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002306273.
390/390 [==============================] - 36s 93ms/step - loss: 0.3567 - acc: 0.8725 - val_loss: 0.6213 - val_acc: 0.8034
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002251576.
390/390 [==============================] - 37s 94ms/step - loss: 0.3521 - acc: 0.8753 - val_loss: 0.6319 - val_acc: 0.7998
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002199413.
390/390 [==============================] - 36s 93ms/step - loss: 0.3423 - acc: 0.8770 - val_loss: 0.6382 - val_acc: 0.7988
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002149613.
390/390 [==============================] - 36s 93ms/step - loss: 0.3456 - acc: 0.8765 - val_loss: 0.6286 - val_acc: 0.8034
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002102018.
390/390 [==============================] - 37s 94ms/step - loss: 0.3421 - acc: 0.8792 - val_loss: 0.6337 - val_acc: 0.8008
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002056485.
390/390 [==============================] - 37s 95ms/step - loss: 0.3419 - acc: 0.8791 - val_loss: 0.6353 - val_acc: 0.8014
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0002012882.
390/390 [==============================] - 37s 94ms/step - loss: 0.3337 - acc: 0.8798 - val_loss: 0.6265 - val_acc: 0.8017
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001971091.
390/390 [==============================] - 36s 94ms/step - loss: 0.3381 - acc: 0.8793 - val_loss: 0.6357 - val_acc: 0.8023
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001930999.
390/390 [==============================] - 37s 94ms/step - loss: 0.3346 - acc: 0.8787 - val_loss: 0.6394 - val_acc: 0.8008
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001892506.
390/390 [==============================] - 36s 94ms/step - loss: 0.3319 - acc: 0.8815 - val_loss: 0.6574 - val_acc: 0.7976
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001855517.
390/390 [==============================] - 37s 94ms/step - loss: 0.3286 - acc: 0.8834 - val_loss: 0.6465 - val_acc: 0.8007
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0001819947.
390/390 [==============================] - 36s 93ms/step - loss: 0.3250 - acc: 0.8839 - val_loss: 0.6449 - val_acc: 0.7985
Model took 1829.27 seconds to train

----------------------------------------------------------------------------------------------------------------------------------
