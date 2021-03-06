import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from model import Model
from loader import load_data

import cv2
import sys
import os
import copy
import json
import math
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

def main():
    abspath = Path(__file__).parent.absolute()
    data_path = os.path.join(abspath, 'data/')
    data_type = 'train'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(os.path.join(abspath, 'checkpoint')):
        os.mkdir(os.path.join(abspath, 'checkpoint'))

    # Number of epochs
    num_epochs = 100 
    # Batch size
    batch_size = 64

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Load dataset
    dataset = load_data(data_path, data_type)

    results = {}
    loss_history_epoch = {}
    loss_history = {}

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        trainLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_subsampler)
        testLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=test_subsampler)

        # Create new model
        net = Model()
        # Learning rate
        lr = 0.0001
        # Train
        net.train()
        # Load data into gpu
        net.to(device)
        # Create optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=0.0005)

        # Save the model in this path
        savepath = os.path.join(abspath,f'checkpoint/checkpoint_{fold}-fold.tar')

        print(f'Starting training {fold} fold...')
        results = {}
        results[f'fold-{fold}'] = {'mean_error': 0, 'loss': []}

        for epoch in range(num_epochs):
            if epoch > 25: 
                lr = 0.00001
            elif epoch > 50:
                lr = 0.000002
                
            print(f'Fold: {fold}, Epoch: {epoch}, lr: {lr}')
            results[f'fold-{fold}']['loss'].append([])
            for i, data in enumerate(trainLoader):
                # Execute a training step
                loss = net.training_step(data, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Register the loss along the training process
                results[f'fold-{fold}']['loss'][epoch].append(loss.item())
                print(f'loss: {loss.item()}', end='\r')

            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'lr': lr,
                }, savepath)
        # Training process is complete.
        print('Training process has finished. Saving trained model.')

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'lr': lr,
        }, savepath)

        print(f'Starting testing for {fold}-fold...')
        # Evaluation for this fold
        with torch.no_grad():
            errors = []
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testLoader, 0):
                # Generate gaze outputs
                output, labels, resolution = net.validation_step(data, device)

                for k, gaze in enumerate(output):
                    gaze = gaze.cpu().detach()
                    xyGaze = [gaze[0]*resolution[k][0], gaze[1]*resolution[k][1]]
                    xyTrue = [labels[k][0]*resolution[k][0], labels[k][1]*resolution[k][1]]
                    errors.append(dist(xyGaze, xyTrue))

            mean_error = np.mean(np.asarray(errors)) / 38 # convert from pixels to cm
            results[f'fold-{fold}']['mean_error'] = mean_error
            print(f'fold: {fold}, mean error: {mean_error}')


        with open(f'results_{fold}_fold.json', 'w') as file:
            print('Creating results json file...')
            json.dump(results, file)
        

if __name__ == '__main__':
    main()
