import pandas as pd
import cv2
import csv
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, MSELoss, Sequential, Conv2d, MaxPool2d, CrossEntropyLoss, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam

from tqdm import tqdm


def video_to_frames(input_loc, train_loc, test_loc):
    """
    Converts video to frames.

    """
    try:
        os.mkdir(train_loc)
        os.mkdir(test_loc)
    except OSError:
        pass
    cap = cv2.VideoCapture(input_loc)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    count = 0
    train_count = 1

    while cap.isOpened():

        ret, frame = cap.read()
        if train_count != 5:
            cv2.imwrite(train_loc + '/%#05d.jpg' % (count + 1), frame)
        else:
            cv2.imwrite(test_loc + '/%#05d.jpg' % (count + 1), frame)
            train_count = 0

        count += 1
        train_count += 1
        if count > video_length - 1:
            cap.release()
            break

def label_training(folder):
    """
    Creates folder and labels data as training
    """
    labels = np.array(['ID', 'Label'])

    for index, image in enumerate(os.listdir(folder)):
        if index < 315 or index > 505:
            labels = np.vstack([labels, np.array([os.path.basename(image), 1])])
        else:
            labels = np.vstack([labels, np.array([os.path.basename(image), 0])])

    with open('train_data.csv', 'w+') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter = ',')
        csvWriter.writerows(labels)

def label_testing(folder):
    """
    Creates folder and labels
    """
    labels = np.array(['ID'])
    for index, image in enumerate(os.listdir(folder)):
        labels = np.vstack([labels, np.array([os.path.basename(image)])])

    with open('test_data.csv', 'w+') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter = ',')
        csvWriter.writerows(labels)


class Net(Module):
    """
    Initializes a PyTorch Convolutional NEURAL
    Net to train on the images.


    """
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(3, 4, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(4),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = 2, stride = 2),
            Conv2d(4, 4, kernel_size = 3, stride = 1 , padding = 1),
            BatchNorm2d(4),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.linear_layers = Sequential(
            Linear(5760, 1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.reshape(-1)
        return x

model = Net()


optimizer = Adam(model.parameters(), lr=0.0001)

criterion = MSELoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

def trainf(epoch):
    """
    Training function that runs through a single epoch.
    """
    model.train()
    tr_loss = 0
    x_train, y_train = Variable(train_x), Variable(train_y)

    x_val, y_val = Variable(val_x), Variable(val_y)

    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    optimizer.zero_grad()

    output_train = model(x_train)
    output_val = model(x_val)

    results = output_val.detach().clone()
    correct = 0
    for index, x in enumerate(results):
        if x < 0.5:
            results[index] = 0
        else:
            results[index] = 1
        if results[index] == y_val[index]:
            correct += 1
    print('Results:', results)
    print('Y_val:', y_val)
    print('SUCCESS PERCENTAGE: ', correct/len(output_val) * 100)


    loss_train = criterion(output_train.float(), y_train.float())
    loss_val = criterion(output_val.float(), y_val.float())
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

    if epoch%2 == 0:
        print('Epoch: ',epoch+1, '\t', 'loss:', loss_val)

if __name__ == '__main__':

    input_loc = 'Video.mkv'
    train_loc = 'train_frames/'
    test_loc = 'test_frames/'

    # label_training(train_loc)
    # label_testing(test_loc)

    train = pd.read_csv('train_data.csv', header = 0)
    test = pd.read_csv('test_data.csv', header = 0)

    pca = PCA(50)

    # test data
    # runs pca analysis on data.
    test_img = []
    for img_name in tqdm(test['ID']):

        img = imread(test_loc + str(img_name))
        blue, green, red = cv2.split(img)

        red_transformed = pca.fit_transform(red)
        blue_transformed = pca.fit_transform(blue)
        green_transformed = pca.fit_transform(green)

        img_pca = (np.dstack((blue_transformed, green_transformed, red_transformed))).astype('float32')

        test_img.append(img_pca)

    test_x = np.array(test_img)
    test_x = test_x.reshape(-1, 3, test_x.shape[1], test_x.shape[2])
    test_x = torch.from_numpy(test_x)

    # Train data
    train_img = []
    for img_name in tqdm(train['ID']):

        img = imread(train_loc + str(img_name))
        blue, green, red = cv2.split(img)

        red_transformed = pca.fit_transform(red)
        blue_transformed = pca.fit_transform(blue)
        green_transformed = pca.fit_transform(green)

        img_pca = (np.dstack((blue_transformed, green_transformed, red_transformed))).astype('float32')

        train_img.append(img_pca)

    # dataset formation
    train_x = np.array(train_img)
    train_y = np.array(train['Label'].values)
    print(train_x.shape)

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2)


    train_x = train_x.reshape(-1, 3, train_x.shape[1], train_x.shape[2])
    train_x = torch.from_numpy(train_x)

    train_y = train_y.astype(int);
    train_y = torch.from_numpy(train_y)
    print(train_y.shape)

    val_x = val_x.reshape(-1, 3, val_x.shape[1], val_x.shape[2])
    val_x = torch.from_numpy(val_x)

    val_y = val_y.astype(int);
    val_y = torch.from_numpy(val_y)


    ### RUNNING THE ACTUAL NEURAL NET ###
    n_epochs = 30

    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        trainf(epoch)

    plt.plot(train_losses, label = 'Training Loss')
    plt.plot(val_losses, label = 'Validation Loss')
    plt.legend()
    plt.show()


    #Classification

    test_x = Variable(test_x)

    with torch.no_grad():
        output = model(test_x)

    results = output.detach().clone()
    for index, x in enumerate(results):
        if x < 0.5:
            results[index] = 0
        else:
            results[index] = 1
    print('Prediction Results', results)




    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob)

    print(predictions)
