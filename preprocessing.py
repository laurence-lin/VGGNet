import tensorflow as tf
import numpy as np
import os
from PIL import Image
import math

'''Preprocessing of input data configurations'''
train_data = '/monkey10/training/training/'
validate_data = '/monkey10/validation/validation/'
#train_data = 'D:/Codepractice/Dataset/10-monkey-species/training/training'
#validate_data = 'D:/Codepractice/Dataset/10-monkey-species/validation/validation'
num_class = 10
train_batch_size = 50
test_batch_size = 50
class_label = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']

def path_to_img(img_path):
    img_set = []
    for img in img_path:
        image = Image.open(img).convert('RGB').resize([224, 224]) # fixed input images size of VGGnet
        img_set.append(np.array(image)/255)

    img_set = np.array(img_set)
    return img_set

def get_batch(data, label, batch_size):
    # one hot encoding for numpy array
    label = np.array(label)
    one_hot = np.zeros((len(label), num_class))
    one_hot[np.arange(len(label)), label] = 1
    label = one_hot # label is array now
    total_size = data.shape[0]
    data_batch = []
    label_batch = []
    for i in range(math.ceil(total_size/batch_size)): # if left sample is not enough one batch, train the left samples
        if label[i*batch_size:].shape[0] < batch_size:  # if left sample is not enough
           data_batch.append(data[i*batch_size:]) # append the left sample
           label_batch.append(label[i*batch_size:])
        else:
            data_batch.append(data[i*batch_size:(i+1)*batch_size])
            label_batch.append(label[i*batch_size:(i+1)*batch_size])
    
    return data_batch, label_batch

train_set = []
train_label = []
# training data
train_list = os.listdir(train_data) # n0 - n9
for train_class in train_list:   # for each class
    class_path = os.path.join(train_data, train_class)  # /path/n0
    class_samples = os.listdir(class_path) # all files of class n
    for sample in class_samples: # for each sample of each class
        train_set.append(os.path.join(class_path, sample)) # /path/n0/n018
        train_label.append(class_label.index(train_class)) # 0~9

# Shuffling
x_train = []
y_train = []
shuffle = np.random.permutation(len(train_label))
for i in shuffle:
    x_train.append(train_set[i])
    y_train.append(train_label[i])
        
# train_set: [train_sample_path], train_label: [train_label]
x_train = path_to_img(x_train)
x_train, y_train = get_batch(x_train, y_train, train_batch_size)

# validate data
valid_set = []
valid_label = []
valid_list = os.listdir(validate_data) # n0 - n9 files
for valid_class in valid_list:   # for each class
    class_path = os.path.join(validate_data, valid_class)  # /path/n0
    class_samples = os.listdir(class_path) # all files of class n
    for sample in class_samples: # for each sample of each class
        valid_set.append(os.path.join(class_path, sample)) # /path/n0/n018
        valid_label.append(class_label.index(valid_class))        
        
x_valid = []
y_valid = []
shuffle = np.random.permutation(len(valid_label))
for i in shuffle:
    x_valid.append(valid_set[i])
    y_valid.append(valid_label[i])

x_valid = path_to_img(valid_set)
x_valid, y_valid = get_batch(x_valid, y_valid, test_batch_size)

        
        