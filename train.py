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
    batch_size = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(os.path.join(abspath, 'checkpoint')):
        os.mkdir(os.path.join(abspath, 'checkpoint'))

    savepath = os.path.join(abspath,'checkpoint/checkpoint.tar')


    # Create the model
    net = Model()

    # Learning rate
    lr = 0.001

    if (Path(savepath).is_file()):
        checkpoint = torch.load(savepath)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        lr = checkpoint['lr']
    else:
        start_epoch = 0

    num_epochs = 17
    net.train()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=0.0005)

    print('Starting training...')

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Load dataset
    dataset = load_data(data_path, data_type)

    results = {}
    loss_history_epoch = {}
    loss_history = {}
    loss_history['all'] = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        trainLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_subsampler)
        testLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=test_subsampler)


        for epoch in range(start_epoch, num_epochs):
            if epoch >= 8:
                lr = 0.0001
            loss_history_epoch[epoch] = []
            for i, data in enumerate(trainLoader):
                # Execute a training step
                loss = net.training_step(data, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Register the loss along the training process
                loss_history['all'].append(loss.item())
                loss_history_epoch[epoch].append(loss.item())
                if i == 20:
                    break;

            if epoch == 10:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'lr': lr,
                }, savepath)
            if epoch > 1:
                break;
        # Training process is complete.
        print('Training process has finished. Saving trained model.')

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'lr': lr,
        }, savepath)

        print('Starting testing')
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

                if i == 50:
                    break

            mean_error = np.mean(np.asarray(errors)) / 38 # convert from pixels to cm
            results[fold] = mean_error
            print(f"error: {mean_error}")
        if fold > 1:
            break

    results['loss_epoch'] = loss_history_epoch
    results['loss_all'] = loss_history

    with open('results.json', 'w') as file:
        json.dump(results, file)
        

    # print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    # print('--------------------------------')
    # sum = 0.0
    # for key, value in results.items():
    #     print(f'Fold {key}: {value} cm')
    #     sum += value
    #     print(f'Average: {sum/len(results.items())} cm')


if __name__ == '__main__':
    main()
