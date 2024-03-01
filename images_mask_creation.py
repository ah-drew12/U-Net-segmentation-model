import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import splitfolders
from PIL import Image

PATH='D:/test_task_winger/' #path to raw data



dataset_train = pd.read_csv('train_dataset.csv')

print(dataset_train.head())

print('\nCount of dataset:'+str(dataset_train['EncodedPixels'].size))

NanCount=dataset_train['EncodedPixels'].isna().sum()
NotNan=dataset_train['EncodedPixels'].size-NanCount
CountTrainData=[NanCount,NotNan]

plt.figure()
plt.bar(['NanCount', 'NotNan'],CountTrainData)
plt.title('Count of Nan, NotNan elements in train dataset')
plt.show()



print('\n Number of examples with N ships in image (NaN not included):')
numb_arr=(dataset_train['ImageId'].value_counts()).value_counts()
numb_arr.loc[1]=numb_arr.loc[1]-NanCount
numb_arr=pd.concat([pd.Series(NanCount,index=[0]),numb_arr])
print(numb_arr)
plt.figure()
x_temp=numb_arr.values
y_temp=np.array(numb_arr.index)
plt.bar(numb_arr.index,numb_arr,tick_label=numb_arr.index)
for index, value in enumerate(x_temp):
    plt.text(index-1,value+5,str(value),color='red',fontweight = 'bold')
plt.title('Number of examples with N ships in image (NaN not included)')
plt.show()




dataset=pd.merge(dataset_train,dataset_train['ImageId'].value_counts(),"left",'ImageId')
dataset['count']=np.array([0 if pd.isna(row[1]['EncodedPixels']) is True else row[1]['count'] for row in dataset.iterrows()])#Subtract a number of NAN values that recognised as a class with 1 ship on image.

dataset=dataset.drop_duplicates(subset=['ImageId'])
dataset=dataset.drop(columns=['EncodedPixels'])
dataset.rename(columns = {'count':'number_of_ships'}, inplace = True)
dataset.to_csv('edited_train_dataset.csv')



def create_mask_image(path, image_name, pixels, train=None):
    img = np.asarray(Image.open(path + 'train/' + image_name)) / 255

    IMG_HEIGHT=img.shape[0]
    IMG_WIDTH=img.shape[1]

    mask_image=np.zeros((IMG_HEIGHT,IMG_WIDTH))

    if(len(pixels)==0):
        #return just black picture because no mask for this photo
        print('return just black picture because no mask for this photo')
    else:
        pattern = '\d*'
        for pixel_set in pixels: #Each set in pixels represent one ship
            for pixel in pixel_set:
                match = re.split(r' +',pixel)

                column=int(match[0])//IMG_HEIGHT #Getting number of column
                start_pixel=int(match[0])%IMG_HEIGHT #Getting number of row
                current_pixel=start_pixel #Initialization for loop
                bias=0
                for i in range(int(match[1])):

                    if (current_pixel+i >= IMG_HEIGHT):#If pixel is getting out of bounds, then change column from row 0
                        bias=i
                        if ((column+1) >= IMG_WIDTH):
                            column=0
                        else: column+=1

                        current_pixel=0

                    mask_image[current_pixel+i-bias,column]=1 # set a pixel value to 1

    if train is True:
        match=re.sub('.jpg','_train_mask.jpg',image_name) #Create a new image name
    else:
        match = re.sub('.jpg', '_test_mask.jpg', image_name)

    mask_image = (mask_image*255).astype(np.uint8)
    Image.fromarray(mask_image).save(path+'masks/'+match) #Saving mask

    photo_test=plt.imread(path+'masks/'+match)
    print("Creating image's masks is finished")
def get_mask_image(dataset,PATH,train=None):
    if train is True:
        list_of_directories = np.array(os.listdir(PATH))
        PATH=PATH+'input_data/'


    edited_pixels = []
    images_dataset=[]
    pixels_dataset=[]
    saved_image=None

    for index, row in dataset.iterrows():
        image = row['ImageId']
        if index ==0: saved_image=image
        #Explaining:
        #We save pixels each step, but if images are the same, we concatenate these pixels and continue to extract new pixels. When image is changed,
        # we will see it with condition "image != saved_image" and then we append saved image name and mask pixels to general dataset and clear the variables

        if image!=saved_image:
            images_dataset.append(saved_image)
            pixels_dataset.append(edited_pixels)
            #create mask image and save it in directory in folder 'masks'

            mask_image=create_mask_image(PATH,saved_image,edited_pixels,train=True)

            edited_pixels=[] #Clearing variable

        pixels=row['EncodedPixels']
        saved_image=image

        if str(pixels) != 'nan':
            print(pixels)
            #Get pair of number with null or multiple digits with space(s) and then number with multiple or null digits and again space(s)
            pattern='(\d*\s*\d*\s*)'

            match = re.split(pattern, pixels)
            print(match)
            while ("" in match):
                match.remove("")
            else: print("Deleting of empty str completed. ")
            print(match)
            edited_pixels.append(match)
    print('Pixels convertion  is done!')



    return images_dataset,pixels_dataset

x_images_train_dataset,Y_mask_train_dataset=get_mask_image(dataset_train,PATH,train=True)

print('Masks creation is completed')

