# Image Segmentation with U-Net model

Image Segmentation model was created to localize and separate found ships on the image into masks where each mask matches one particular ship.


## Usage

You can check model performance in jupyter notebook where prediction is assessed with F2-score. For testing the model on other images, you should set a path to train images from the Airbus Ship Detection Challenge.


In the main directory, you can find train_dataset.csv from raw data that is shared on Kaggle. So you just need to create a folder with any name you want. Then inside of this folder you should create a folder named "input_data" where located two folders "train" and "masks".
You should extract raw images inside of folder "train". So, your path to data should look like this: path='folder_1/', inside of this folder is a folder "input_data" where are located 2 folders named "train" and "masks". 


So you have two directories: 

*first for RGB images: folder_1/input_data/train/

*second for mask images: folder_1/input_data/masks/

Mask creation is performed by file images_mask_creation.py. In the beginning of code you can find variable PATH, where you should put name of "folder_1" (path='folder_1/'). CSV file "train_dataset.csv" you can find in the main directory of the project.
The program iterate rows in 'train_dataset.csv' and reads RLE (Run-length encoding) for ship that located on image. It accumulate these encodings into one variable, then it fills a matrix of zeros with ones coresponding to mask's encoding. It creates one mask, that contains all ships. After creating an matrix of mask, it saves mask image into a directory with masks. So by this way program runs all images in 'train_dataset.csv' and creates masks.
RLE reading is performed by regular expression functions.

After the mask images creation is completed, you have to create train and validation datasets. You can do this with "dataset_creation.py". In this script you need to set a PATH directly to folders "train" and "masks", so your path should look like this: "path="folder_1/input_data/".
Given dataset imbalanced with images with no ships or one ship. Total size of dataset is 231723, where images with 0 ships are a majority (150k). 

### Creation of train and validation datasets
Difference in examples where images with 0 or 1 ship to 2-15 ships goes from 2000/1 to 5/1. To create a balance between classes, i decided to set a limit to the number of examples for each class to 200 images and split the generated dataset into a train and validation datasets. 

In a result, i received sets of 1960 train and 654 validation images. Some of classes received way lesser number of images than 200, because of the purpose of going in depth of the dataset to prevent a situation when 200 example images for 0_ship_images will be gotten from the first 200 images that meet a condition that this image has N ships. 

For each dataset created a particular list with image's names.

### Creation of datasets
To create a structure for training data, program split images into training and validation lists. We iterate each created list.
During iteration it saves an original photo into a directory "train_images/images/" or "val_images/images/". The same algorithm with a mask image.
The program adds to the ending of the image an ending "_train_mask" and find this mask by name in folder with masks. As it found the image, it moves the mask image to folder "train_masks/masks/" or "val_masks/masks/".

### Model architecture

For image segmentation was used U-Net model architecture. Code for model architecture was taken from https://youtu.be/68HR_eyzk00 .

As a loss function was taken a dice loss that was optimized with Adam optimizer. The task formulation requires to use F2 score, so after the ending of epoch calculation program will output a loss and f2 score (built-in function in segmentation models library) for current epoch. 


### Fit function

Training a model with that size of data requires a lot of machine resources. Adding to this we should give attention to small objects that can occupy a little space in an image. So for that purpose, the model was trained with images in the original dimension (768x768 pixels) because after an image resizing to a smaller dimension, we would lose information about tiny ships.  


As mentioned before, we split our images into four folders. For training a model with data that cannot be fitted in RAM, we can use keras built-in fit function (fit_generator) that receives as X a generator of data. As a generator of data we can use keras built-in function for data augmentation that receives a directory with images. Generator has a bunch of parameters for image rotating, flipping, resizing, but we will use only rescaling from 0-255 to 0-1. For each folder we have one generator, X and Y generators zipped by training or validation group and fitted in a model.


### Callbacks, metrics, loss function
To optimize training process i will use ModelCheckpoint which saves weights for each epoch (or best_only when indicator has a bigger or smaller value than on previous epochs). Also, i used a learning rate scheduler that changes a learning rate after a particular number of epochs. This function helps us to prevent a stucking in a local minimum.

### Training log

Total number of epochs is around 110, where each epoch takes about 15-25 minutes to calculate where the batch size is 5 images. I haven't saved a whole training history because of the error that happens because of insufficient memory (fit_generator still had epochs to train a model, so it didn't export csv with history and saved model in .h5).
But i can restore val_loss data with modelcheckpoint savings that contain epoch and loss in the name. After two times of getting an error, i specified a modelcheckpoint name to save more data about the epoch in a weight file's name. 

*A graph of training log you can see in jupyter notebook.

### Model performance

To assess the model prediction, I will use a custom F2 score metric that based on IOU score. There are already created functions of metric that compare each pixel between true and predict image, but in our case we need to compare ship's objects. The hardest part of this is to divide one mask image into several images that contain each object. For this purpose i have used label function from scikit-image library, which detects pixels that located closely and sets a mark on them. After this manipulation with image, i can compare each true mask with the predicted mask of object.

The best situation is when a true mask and predict are similar and were divided into the same number of masks. But in reality, you can receive 2-5 masks of true image and only two of predicted. So, in this situation, the algorithm will compare each true mask with predicted mask and evaluate an overlap and union. If there's some overlap we compare Intersection over Union value to settled threshold and mark our mask with 1 if evaluated value is bigger than settled threshold. Unless it bigger, it will be marked with the evaluated value.

This comparison gives us a tensor of dimension (size_true_mask x size_pred_mask x threshold). For each true mask, we have comparison to each object on the predicted  mask. This tensor will help us to calculate True Positives, False Positives, False Negatives. For exampe, true positives will be detected in the tensor as 1, false positives as value in interval (0-1). To calculate False Negatives we subtract number of TruePositives and FalsePositive from number of masks on prediction image, also we need to subtract an 1, because labeled prediction mask includes a background image that doesn't have any object. 

After those calculation, we can evaluate a F2 score.

The custom metric haven't been used in model training because its evaluation increased the time of training significantly (from 10 minutes to hour). So, it has been used after the model was trained.

You can check model prediction at the end of the jupyter notebook. There is a comparison between models trained on full-size images and images with size 256 x 256 pixels.
