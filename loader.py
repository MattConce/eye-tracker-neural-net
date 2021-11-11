from torch.utils.data import Dataset, DataLoader
import torch

from pathlib import Path
import numpy as np
import json
import cv2
import os

def crop_image(rect, image):
  pt1, pt2 = rect  
  w = abs(pt1[0]-pt2[0])
  h = abs(pt1[1]-pt2[1])
  new_image = image[pt1[1]:pt1[1]+h, pt1[0]:pt1[0]+w]
  return new_image

class Loader(Dataset):
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type

        with open(os.path.join(self.data_path, 'dataset.json'), 'r') as file:
            self.data_json = json.load(file)
            self.data_keys = list(self.data_json.keys())

    def __len__(self):
        return len(self.data_json)
    
    def __getitem__(self, index):
        curr_key = self.data_keys[index]

        sample_image = cv2.imread(os.path.join(self.data_path, curr_key))

        image_info = self.data_json[curr_key]

        left_eye = crop_image(image_info['LEFT_EYE'], sample_image)
        left_eye = cv2.resize(left_eye, (112, 112), interpolation = cv2.INTER_AREA) 
        left_eye = cv2.normalize(left_eye, None, 0, 1, cv2.NORM_MINMAX)
        left_eye = left_eye.transpose(2, 0, 1)

        right_eye = crop_image(image_info['RIGHT_EYE'], sample_image)
        right_eye = cv2.resize(right_eye, (112, 112), interpolation = cv2.INTER_AREA) 
        right_eye = cv2.flip(right_eye, 1)
        right_eye = cv2.normalize(right_eye, None, 0, 1, cv2.NORM_MINMAX)
        right_eye = right_eye.transpose(2, 0, 1)

        face = crop_image(image_info['FACE'], sample_image)
        face = cv2.resize(face, (224, 224), interpolation = cv2.INTER_AREA) 
        face = cv2.normalize(face, None, 0, 1, cv2.NORM_MINMAX)
        face = face.transpose(2, 0, 1)

        # Boundary box for the eyes and face plus screen resolution
        rects = image_info['LEFT_EYE'][0]  + image_info['LEFT_EYE'][1] + \
                image_info['RIGHT_EYE'][0] + image_info['RIGHT_EYE'][1] +  \
                image_info['FACE'][0]      + image_info['FACE'][1] \
                # image_info['resolution']

        # Label is the screen coordinate (x, y)
        label = np.array(image_info['label'])
        resolution = np.array(image_info['resolution'])

        return {
            "face": torch.from_numpy(face).type(torch.FloatTensor), 
            "leftEye": torch.from_numpy(left_eye).type(torch.FloatTensor),
            "rightEye": torch.from_numpy(right_eye).type(torch.FloatTensor),
            "rects": torch.from_numpy(np.array(rects)).type(torch.FloatTensor),
            "label": torch.from_numpy(label).type(torch.FloatTensor),
            "resolution": torch.from_numpy(resolution).type(torch.FloatTensor)
            }

def load_data(data_path, data_type):
    print('Starting loading data...')
    dataset = Loader(data_path, data_type=data_type)
    return dataset
