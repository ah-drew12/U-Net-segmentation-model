import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import re

seed=42
random.seed(seed)
#reading files in directory
PATH='D:/test_task_winger/input_data/' #Path to raw data

dataset=pd.read_csv('edited_train_dataset.csv')#read edited csv with label of counted ships on the image

count_numbers_occurence=dataset['number_of_ships'].value_counts()
list_of_numbers=np.sort(dataset['number_of_ships'].unique())
new_dataset=[]
max_occur_numb=200
for number in list_of_numbers:
    count=0 #Count 200 images
    for row in dataset.iterrows():
        while ((count < max_occur_numb) & (count < count_numbers_occurence[number])): #If count is less than 200 and total number of occurence in dataset, then we get inside of a loop
            if row[1]['number_of_ships']==number:

                if count_numbers_occurence[number]>5*max_occur_numb:
                    if random.random()>0.9: # it's made to diversify dataset with data that take the bigger part in whole dataset (e.g. number of ships 0, 1 , 2). It helps us to not get the first 200 images. So by this way we can go deep inside dataset
                        new_dataset.append(row[1]) #Add image to new dataset
                        count+=1
                        break
                    else:
                        break
                else:
                    new_dataset.append(row[1]) #If total number of images of particular class is less than 200, then we just add image to dataset
                    count += 1
                    break
            else: break

        else:
            print('Dataset creating for [' + str(number) + '] is over.')
            break

        continue

new_dataset=pd.DataFrame(new_dataset)
new_dataset.to_csv('dataset_images.csv')

x_train,x_test=train_test_split(new_dataset) #train/test == 1/4


for row in x_train.iterrows():

    img = np.asarray(Image.open(PATH +'train/'+ row[1]['ImageId']))  #Open an RGB image from train dataset
    mask_img=np.asarray(Image.open(PATH +'masks/'+ re.sub('.jpg', '_train_mask.jpg', row[1]['ImageId']))) #Open mask image from train dataset

    Image.fromarray(img).save('dataset/train_images/images/' + row[1]['ImageId'])#Save an image into a new folder with train RGB images
    Image.fromarray(mask_img).save('dataset/train_masks/masks/' + re.sub('.jpg', '_train_mask.jpg', row[1]['ImageId']))#Save a mask image into a new folder with train masks

for row in x_test.iterrows():

    img = np.asarray(Image.open(PATH +'train/'+ row[1]['ImageId'])) #Open an RGB image from validation dataset
    mask_img=np.asarray(Image.open(PATH +'masks/'+ re.sub('.jpg', '_train_mask.jpg', row[1]['ImageId'])))#Open mask image from test dataset

    Image.fromarray(img).save('dataset/val_images/images/' + row[1]['ImageId'])
    Image.fromarray(mask_img).save('dataset/val_masks/masks/' + re.sub('.jpg', '_train_mask.jpg', row[1]['ImageId']))#Save a mask image into a new folder with train masks

print('Dataset creation completed')
