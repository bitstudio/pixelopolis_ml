import glob, os
import random
import cv2
import math
import numpy as np
from keras.utils import Sequence
import random
import pandas as pd
from sklearn.model_selection import train_test_split


class LaneDataGenerator(Sequence):
    
    def __init__(self, images_folder, labels_folder, input_shape, batch_size):   
        self.images_folder = images_folder
        self.image_height, self.image_width, self.image_channels = input_shape
        self.batch_size = batch_size
                
        # load al label files to dataframe
        self.all_label = glob.glob(labels_folder+os.sep+'*.csv')
        data_df = []
        for label_file in self.all_label:            
            df = pd.read_csv(label_file, names=['center', 'left', 'right', 'steering'])
            data_df.append(df)
        self.data_df = pd.concat(data_df, axis=0, ignore_index=True)  
        
    
    def __iter__(self):
        return self


    def __len__(self):        
        return len(self.data_df.index) // self.batch_size
    
   
    def load_image(self, image_path):
        # load image, resize, convert to YUV, crop to 1/4 height.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)       
        image = image[int(0.75*image.shape[0]):, :, :]
        image = cv2.resize(image, (self.image_width, self.image_height), cv2.INTER_AREA)
        
        return image


    def load_sample(self, sample):      
        center, left, right, steering_target = sample
        
        # random camera position
        choice = random.randint(0,2)        
        # center
        if choice == 0:        
            image = self.load_image(os.path.join(self.images_folder, center))            
        # left
        elif choice == 1:        
            image = self.load_image(os.path.join(self.images_folder, left))
            steering_target = steering_target + 0.4        
        # right
        elif choice == 2:        
            image = self.load_image(os.path.join(self.images_folder, right))
            steering_target = steering_target - 0.4           
            
        return image, steering_target


    def __getitem__(self, item):
        input_batch = np.empty([self.batch_size, self.image_height, self.image_width, self.image_channels])        
        target_batch = np.empty(self.batch_size)      
        

        for i in range(self.batch_size):
          
            # random one sample from data frame
            chosen_sample = self.data_df.sample(n=1).iloc[0]    
            image, steering_angle = self.load_sample(chosen_sample)
            
            input_batch[i] = image
            target_batch[i] = steering_angle
        

        return input_batch, target_batch