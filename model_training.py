import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import os

from F2_score import F2_score

tf.config.experimental_run_functions_eagerly(True)


#reading files in directory
PATH='dataset/'

SEED=42
BATCH_SIZE=5
IMG_HEIGHT=768
IMG_WIDTH=768
CHANNELS=3
TRAIN_SIZE=len(os.listdir(PATH+'train_images/images/'))
VAL_SIZE=len(os.listdir(PATH+'val_images/images/'))

image_generator_params=dict(rescale=1/255)  # Augmentation settings: just rescale pixels from 0-255 to 0-1.

image_data_generator=ImageDataGenerator(**image_generator_params) #ImageDataGenerator initialization

#Generators creation for each folder
train_image_generator = image_data_generator.flow_from_directory(directory=PATH+'train_images/',
                                                            seed=SEED,
                                                            batch_size=BATCH_SIZE,
                                                            target_size=(768,768),
                                                            class_mode=None)


train_mask_generator = image_data_generator.flow_from_directory(directory=PATH+'train_masks/',
                                                            seed=SEED,
                                                            batch_size=BATCH_SIZE,
                                                            color_mode='grayscale',
                                                            target_size=(768,768),
                                                            class_mode=None)

val_image_generator = image_data_generator.flow_from_directory(directory=PATH+'val_images/',
                                                            seed=SEED,
                                                            batch_size=BATCH_SIZE,
                                                            target_size=(768,768),
                                                            class_mode=None)


val_mask_generator = image_data_generator.flow_from_directory(directory=PATH+'val_masks/',
                                                            seed=SEED,
                                                            batch_size=BATCH_SIZE,
                                                            color_mode='grayscale',
                                                            target_size=(768,768),
                                                            class_mode=None)

train_generator=zip(train_image_generator,train_mask_generator)
val_generator=zip(val_image_generator,val_mask_generator)

# Testing input

x=train_image_generator.next()
y=train_mask_generator.next()
# for img,mask in zip(x,y):
#     plt.figure(1)
#     fig,axs=plt.subplots(1,2)
#     axs[0].imshow(img)
#     axs[1].imshow(mask,cmap='gray')
#     plt.show()



#ARCHITECTURE
#############


input=tf.keras.layers.Input(shape=(IMG_HEIGHT,IMG_WIDTH,CHANNELS))

# Encoder part
U1=tf.keras.layers.Conv2D(16,3,padding='same',activation='relu',kernel_initializer='he_normal')(input)
U1=tf.keras.layers.Dropout(0.2)(U1)
U1=tf.keras.layers.Conv2D(16,3,padding='same',activation='relu',kernel_initializer='he_normal')(U1)

U2=tf.keras.layers.MaxPooling2D()(U1)
U2=tf.keras.layers.Conv2D(32,3,padding='same',activation='relu',kernel_initializer='he_normal')(U2)
U2=tf.keras.layers.Dropout(0.2)(U2)
U2=tf.keras.layers.Conv2D(32,3,padding='same',activation='relu',kernel_initializer='he_normal')(U2)

U3=tf.keras.layers.MaxPooling2D()(U2)
U3=tf.keras.layers.Conv2D(64,3,padding='same',activation='relu',kernel_initializer='he_normal')(U3)
U3=tf.keras.layers.Dropout(0.2)(U3)
U3=tf.keras.layers.Conv2D(64,3,padding='same',activation='relu',kernel_initializer='he_normal')(U3)

U4=tf.keras.layers.MaxPooling2D()(U3)
U4=tf.keras.layers.Conv2D(128,3,padding='same',activation='relu',kernel_initializer='he_normal')(U4)
U4=tf.keras.layers.Dropout(0.2)(U4)
U4=tf.keras.layers.Conv2D(128,3,padding='same',activation='relu',kernel_initializer='he_normal')(U4)

U5=tf.keras.layers.MaxPooling2D()(U4)
U5=tf.keras.layers.Conv2D(256,3,padding='same',activation='relu',kernel_initializer='he_normal')(U5)
U5=tf.keras.layers.Dropout(0.2)(U5)
U5=tf.keras.layers.Conv2D(256,3,padding='same',activation='relu',kernel_initializer='he_normal')(U5)


# Decoder part
I1=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same',activation='relu',kernel_initializer='he_normal')(U5) #Upsampling layer
U6=tf.keras.layers.concatenate([I1,U4])
U6=tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(U6)
U6=tf.keras.layers.Dropout(0.2)(U6)
U6=tf.keras.layers.Conv2D(128,3,padding='same',activation='relu',kernel_initializer='he_normal')(U6)

I2=tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same',activation='relu',kernel_initializer='he_normal')(U6) #Upsampling layer
U7=tf.keras.layers.concatenate([I2,U3])
U7=tf.keras.layers.Conv2D(64,3,padding='same',activation='relu',kernel_initializer='he_normal')(U7)
U7=tf.keras.layers.Dropout(0.2)(U7)
U7=tf.keras.layers.Conv2D(64,3,padding='same',activation='relu',kernel_initializer='he_normal')(U7)

I3=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same',activation='relu',kernel_initializer='he_normal')(U7) #Upsampling layer
U8=tf.keras.layers.concatenate([I3,U2])
U8=tf.keras.layers.Conv2D(32,3,padding='same',activation='relu',kernel_initializer='he_normal')(U8)
U8=tf.keras.layers.Dropout(0.2)(U8)
U8=tf.keras.layers.Conv2D(32,3,padding='same',activation='relu',kernel_initializer='he_normal')(U8)

I4=tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same',activation='relu',kernel_initializer='he_normal')(U8) #Upsampling layer
U9=tf.keras.layers.concatenate([I4,U1])
U9=tf.keras.layers.Conv2D(16,3,padding='same',activation='relu',kernel_initializer='he_normal')(U9)
U9=tf.keras.layers.Dropout(0.2)(U9)
U9=tf.keras.layers.Conv2D(16,3,padding='same',activation='relu',kernel_initializer='he_normal')(U9)

outputs=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(U9)
#####################
model=tf.keras.Model(inputs=[input],outputs=[outputs])

steps_per_epoch=TRAIN_SIZE//BATCH_SIZE
steps_per_val_epoch=VAL_SIZE//BATCH_SIZE




import segmentation_models as sm
dice_loss = sm.losses.DiceLoss() #Loss initialization

opt=keras.optimizers.Adam(learning_rate=0.0001) #Optimizer initialization

thresholds=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95] #Thresholds for F2 score

metrics=[sm.metrics.IOUScore(threshold=0.5)
    # ,F2_score(threshold=thresholds,batch_size=BATCH_SIZE) #Custom F2 score not using in a training process because it increases evaluation time significantly
         ]

mc = tf.keras.callbacks.ModelCheckpoint('temp_models/dice_loss/{epoch:02d}__{loss:02f}__{val_loss:02f}__{iou_score:02f}__{val_iou_score:02f}.keras',
                                        monitor='val_loss', mode='min',verbose=1) #ModelCheckpoint initalization

def scheduler(epoch, lr):
    if epoch<2:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lrsch=tf.keras.callbacks.LearningRateScheduler(scheduler)

model.compile(optimizer=opt,loss=dice_loss,metrics=metrics,run_eagerly=True)

model.load_weights('temp_models/dice_loss/19__0.136629__0.251944__0.764009__0.639464.keras')



# Train model with dataset
history=model.fit_generator(generator=train_generator,
                    validation_data=val_generator,epochs=20,steps_per_epoch=steps_per_epoch,validation_steps=steps_per_val_epoch,
                workers=6, callbacks=[mc,lrsch])
#
pd.DataFrame.from_dict(history.history).to_csv('history.csv') #Saving a history for processing in further


#Output a history of training metrics
fig,axs=plt.subplots(ncols=3)
fig.suptitle("Graphs of metrics", fontsize=16)
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].set_title('Loss')
axs[0].legend(['Loss', 'Val_loss'])

axs[1].plot(history.history['iou_score'])
axs[1].plot(history.history['val_iou_score'])
axs[1].set_title('iou_score')
axs[1].legend(['iou_score', 'val_iou_score'])

axs[2].plot(history.history['f2-score'])
axs[2].plot(history.history['val_f2-score'])
axs[2].set_title('f2-score')
axs[2].legend(['f2-score', 'val_f2-score'])


plt.show()
model.save('u_net_segmentation_model.h5') #Save model


print('Training is completed')

