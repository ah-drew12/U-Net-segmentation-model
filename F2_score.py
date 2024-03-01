##F2 score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from skimage.morphology import label

tf.config.experimental_run_functions_eagerly(True)


class F2_score(keras.metrics.Metric):
    def __init__(self, threshold=[0.5], batch_size=1, batch_processing=True, name='F2_score', **kwargs):
        super(F2_score, self).__init__(**kwargs)
        self.F2_score = self.add_weight('F2_score', initializer='zeros')
        self.threshold = threshold
        self.B = 2
        self.batch_processing = batch_processing
        self.F2_score_per_threshold = self.add_weight('F2_score_per_threshold', shape=(batch_size, len(threshold)),
                                                      initializer='zeros')

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # check if array is  a numpy array
        if type(y_true).__module__ != np.__name__:
            y_true_numpy = y_true.numpy()
        else:
            y_true_numpy = y_true

        if type(y_pred).__module__ != np.__name__:
            y_pred_numpy = y_pred.numpy()
        else:
            y_pred_numpy = y_pred

        if (self.batch_processing is True):  # Branch of algorithm for proccesing a batch of images

            # Rounding pixels to execute label function correctly
            y_true_numpy = np.array(
                [[[1 if pixel > 0.5 else 0 for pixel in row[:]] for row in image[:]] for image in y_true_numpy[:]])
            y_pred_numpy = np.array(
                [[[1 if pixel > 0.5 else 0 for pixel in row[:]] for row in image[:]] for image in y_pred_numpy[:]])

            batch_size = y_true_numpy.shape[0]
            scores = []

            for img_num in range(batch_size):
                score = self.evaluate_score_for_image(y_pred_numpy[img_num], y_true_numpy[img_num],
                                                      img_ind=img_num)  # Evaluate F2 score for particular image
                scores.append(score)  # Add a score for particular image to vector for scores of all images..

            score = np.mean(np.array(scores))  # Evaluate an average F2 score among all images in the batch.
            self.F2_score.assign(score)  # Return a score

        else:
            y_pred_numpy = np.squeeze(y_pred_numpy)
            if (y_pred_numpy.shape == y_true_numpy.shape):

                # Rounding pixels to execute label function correctly
                y_true_numpy = np.array([[1 if pixel > 0.5 else 0 for pixel in row[:]] for row in y_true_numpy[:]])
                y_pred_numpy = np.array([[1 if pixel > 0.5 else 0 for pixel in row[:]] for row in y_pred_numpy[:]])

                score = self.evaluate_score_for_image(y_pred_numpy,
                                                      y_true_numpy)  # Evaluate F2 score for particular image

                self.F2_score.assign(np.squeeze(score))  # Assign F2 score to variable to show metric result
            else:
                print("Image's shapes are not the same.")

    def evaluate_score_for_image(self, predicted_image, true_image, img_ind=0):
        score = 0
        # Check if all image's pixels are black (mask is a zeros matrix) with diffrent combination. It created to output F2 score for masks without any ship on it.

        if (np.sum(true_image) == 0 and np.sum(predicted_image) == 0):
            return np.ones(len(self.threshold))  # return array of ones for each threshold
        elif (np.sum(true_image) != 0 and np.sum(predicted_image) == 0):
            return np.zeros(len(self.threshold))  # return array of zeros for each threshold
        elif np.sum(true_image) == 0 and np.sum(predicted_image) != 0:
            return np.zeros(len(self.threshold))  # return array of zeros for each threshold
        else:
            true_masks = self.separate_into_masks(
                label(true_image))  # Creating a tensor of true masks where each mask is a single object.
            pred_masks = self.separate_into_masks(
                label(predicted_image))  # Creating a tensor of predicted masks where each mask is a single object.

            iou_tensor = []  # This tensor consist of matrix which quantity is a number of true masks. Each matrix has IOU score for each predicted masks - these values written in rows. For each threshold we have a column in this matrix.

            # True mask:
            '''
            Example of matrix:


                                threshold_0.5   threshold_0.55
            predicted_mask1      IOU_score        IOU_score
            predicted_mask2      IOU_score        IOU_score
            predicted_mask3      IOU_score        IOU_score


            IOU tensor consists of those matrices.


            '''

            for true_mask in true_masks:
                iou_matrix = []
                for pred_mask in pred_masks:
                    # for pred_mask in :
                    iou_vector = self.evaluate_ious(pred_mask,
                                                    true_mask)  # get a vector of iou score for a predicted mask to the true mask. Vector has a N-dimensional size, where N is number of thresholds
                    iou_matrix.append(iou_vector)  # collect scores from all predicted mask to one true mask
                iou_tensor.append(iou_matrix)  # collect all iou scores for all true masks in a tensor

            iou_tensor = np.array(iou_tensor)

            F2_for_image = self.f2_score_eval(iou_tensor,
                                              img_ind=img_ind)  # func to calculate true positive , false positive, false negative and evaluate F2 score

            return F2_for_image

    def f2_score_eval(self, tensor, img_ind=0):

        threshold_num = len(self.threshold)
        F2_scores = []

        for threshold_ind in range(
                threshold_num):  # Iterate each threshold. So we extract only one column from all in the matrices
            tp, fp, fn = 0, 0, 0

            tp = np.sum(
                (tensor[:, :, threshold_ind] == 1) > 0)  # If IOU score higher than threshold, we mark this as a 'hit'
            for pred_mask in range(tensor.shape[1]):
                if np.sum((tensor[:,pred_mask,threshold_ind]>0)>0) > 0:
                    fp+=1
            fp -= tp
            # fp = np.sum((tensor[:, :, threshold_ind] > 0) > 0) - tp  # Counting values that less than threshold
            fn = tensor.shape[1] - tp - fp-1 # get a number of predicted masks and subtract the number of true positive masks and false positive masks.
            F2_score = (1 + self.B ** 2) * tp / ((1 + self.B ** 2) * tp + (self.B ** 2) * fn + fp + 0.0000001)  # Evaluation of F2 score, where we adding 0.0000001 in a denominator to except division by 0.
            F2_scores.append((F2_score))  # Collect F2 score for threshold
        F2_scores = np.array(F2_scores)
        if threshold_num > 1:
            self.F2_score_per_threshold[img_ind].assign(
                F2_scores)  # Assign a vector of F2 scores to vector of F2_scores
            F2_scores = np.mean(F2_scores)  # Evaluation of an average value of F2 scores.
        return F2_scores

    def evaluate_ious(self, prediction, true):
        score_vector = np.zeros(len(self.threshold))

        overlap = np.sum(prediction * true)  # calculating an overlap areas
        union = np.sum((true + prediction) == 1) + overlap  # calculating an union of areas
        iou = overlap / (union + 0.0000001)  # adding  a  0.0000001 is to except a division by zero
        for index, threshold in enumerate(self.threshold):
            if iou > 0:

                # if intersect_over_union (iou) bigger than threshold, then we mark it as 1
                if iou > threshold:
                    score_vector[index] = 1
                else:
                    # if value of iou isn't that big to be bigger than threshold, we mark it as value of metric.
                    # In future preprocessing we will say that it is a false positive prediction
                    score_vector[index] = iou
            else:
                # if value less than 0 then we say that it is a false negative prediction ( in the next steps of processing F2 score)
                score_vector[index] = 0

        return score_vector

    def separate_into_masks(self, labeled_image):

        masks_labels = np.unique(labeled_image)  # Get an array of labels to iterate it in further.
        masks = []
        for label in masks_labels:
            template = np.array([[1 if ((pixel == label) and (label != 0)) else 0 for pixel in rows] for rows in
                                 labeled_image])  # Creating a tensor of masks for a particular image
            masks.append(template)

        return np.array(masks)

    def reset_state(self):
        self.F2_score.assign(0)

    def result_for_thresholds(self):
        return self.F2_score_per_threshold

    def result(self):
        return self.F2_score

