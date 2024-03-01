import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from F2_score import F2_score
import matplotlib.pyplot as plt
import segmentation_models as sm
import skimage.transform as st

from PIL import Image
import os
import re

tf.config.experimental_run_functions_eagerly(True)


image_path = 'dataset/val_images/images/'
mask_path = 'dataset/val_masks/masks/'

dice_loss = sm.losses.DiceLoss()

IOU_score = sm.metrics.IOUScore(threshold=0.5)

f2_score = sm.metrics.FScore(threshold=0.5, beta=2)
F2_score_array_768 = []

thresholds=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95] #Thresholds for F2 score. Can be used instead of just 0.5

F2_score_metric = F2_score(threshold=[0.5], batch_processing=False)

unet_model = tf.keras.models.load_model('u_net_segmentation_model.h5',
                                        custom_objects={'dice_loss': dice_loss, 'f2-score': f2_score,
                                                        'iou_score': IOU_score}) #load model

images = os.listdir(path=image_path) #read validation images' names
for batch_ind, img_temp in enumerate(images[:60:10]):
    fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(27, 20))

    plt.subplots_adjust(
        left=0.074, bottom=0.043, right=0.9, top=0.95, wspace=0.01, hspace=0
    )

    for ax in fig.axes:
        ax.axison = False

    axs[0][0].set_title(('Original Image'), fontdict={'fontsize': 10})
    axs[0][1].set_title('Ground truth', fontdict={'fontsize': 10})
    axs[0][2].set_title('Predicted mask', fontdict={'fontsize': 10})
    axs[0][3].set_title(('Original Image'), fontdict={'fontsize': 10})
    axs[0][4].set_title('Ground truth', fontdict={'fontsize': 10})
    axs[0][5].set_title('Predicted mask', fontdict={'fontsize': 10})

    for index, image in enumerate(images[(batch_ind) * 10:(batch_ind + 1) * 10]):

        img_test = plt.imread(image_path + image) / 255
        image = re.sub('.jpg', '_train_mask.jpg', image)
        img_mask_test = plt.imread(mask_path + image) / 255

        img_predict_test = unet_model.predict(
            x=tf.reshape(img_test, shape=(-1, 768, 768, 3)))

        F2_score_metric.update_state(img_mask_test, img_predict_test) #evaluate F2 score for prediction
        F2_result = F2_score_metric.result()
        F2_score_array_768.append(F2_result.numpy())
        f2_score_text = 'F2 score: ' + str(round(F2_result.numpy(), 4))

        if (index > 4):
            axs[index - 5][3].imshow(img_test)

            axs[index - 5][4].imshow(img_mask_test)

            axs[index - 5][5].imshow(np.squeeze(img_predict_test))
            axs[index - 5][5].text(img_predict_test.shape[1]/3, 50, f2_score_text, color='red', fontsize=10)
        else:
            axs[index][0].imshow(img_test)

            axs[index][1].imshow(img_mask_test)

            axs[index][2].imshow(np.squeeze(img_predict_test))
            axs[index][2].text(img_predict_test.shape[1]/3, 50, f2_score_text, color='red', fontsize=10)

    plt.show()


#Results of model trained on images with size 256x256 pixels
F2_score_array_256=[]

unet_model = tf.keras.models.load_model('u_net_segmentation_model_256pix_40ep.h5',
                                        custom_objects={'dice_loss': dice_loss, 'f2-score': f2_score,
                                                        'iou_score': IOU_score}) #load model trained on smaller images

for batch_ind, img_temp in enumerate(images[:60:10]):

    fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(27, 20))

    plt.subplots_adjust(
        left=0.074, bottom=0.043, right=0.9, top=0.95, wspace=0.01, hspace=0
    )

    for ax in fig.axes:  # Turn off axes
        ax.axison = False

    axs[0][0].set_title(('Original Image'), fontdict={'fontsize': 10})
    axs[0][1].set_title('Ground truth', fontdict={'fontsize': 10})
    axs[0][2].set_title('Predicted mask', fontdict={'fontsize': 10})
    axs[0][3].set_title(('Original Image'), fontdict={'fontsize': 10})
    axs[0][4].set_title('Ground truth', fontdict={'fontsize': 10})
    axs[0][5].set_title('Predicted mask', fontdict={'fontsize': 10})

    for index, image in enumerate(images[(batch_ind) * 10:(batch_ind + 1) * 10]):

        img_test = plt.imread(image_path + image) / 255
        image = re.sub('.jpg', '_train_mask.jpg', image)
        img_mask_test = plt.imread(mask_path + image) / 255

        img_predict_test = unet_model.predict(
            x=tf.reshape(tf.image.resize(img_test, size=(256, 256)), shape=(-1, 256, 256, 3)))  # Predict a mask

        new_img_mask_test = st.resize(img_mask_test, (256, 256), order=0, preserve_range=True,
                                      anti_aliasing=False)  # Change size of ground truth image to 256 x 256 pixels

        F2_score_metric.update_state(new_img_mask_test, img_predict_test)  # Evaluate F2 score
        F2_result = F2_score_metric.result()
        F2_score_array_256.append(F2_result.numpy())

        f2_score_text = 'F2 score: ' + str(round(F2_result.numpy(), 4))

        if (index > 4):
            axs[index - 5][3].imshow(img_test)

            axs[index - 5][4].imshow(img_mask_test)

            axs[index - 5][5].imshow(np.squeeze(img_predict_test))
            axs[index - 5][5].text(100, 50, f2_score_text, color='red', fontsize=10)
        else:
            axs[index][0].imshow(img_test)

            axs[index][1].imshow(img_mask_test)

            axs[index][2].imshow(np.squeeze(img_predict_test))
            axs[index][2].text(100, 50, f2_score_text, color='red', fontsize=10)

    plt.show()


print('Average F2 score:', np.mean(np.array(F2_score_array_768)))

print('Average F2 score:', np.mean(np.array(F2_score_array_256)))

