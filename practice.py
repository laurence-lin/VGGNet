import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import vgg16_D as vgg16
import os


#train_data = 'D:/Codepractice/Dataset/10-monkey-species/training/training'
#validate_data = 'D:/Codepractice/Dataset/10-monkey-species/validation/validation'
num_class = 10
train_batch_size = 1
test_batch_size = 1
class_label = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']

'''def path_to_img(img_path):
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

train_data = '/monkeytest/'

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

x_train = path_to_img(train_set)
x_train, y_train = get_batch(x_train, train_label, train_batch_size)'''

img = np.random.random((1, 224, 224, 3))
num_class = 10
learning_rate = 0.00001
img_y = np.zeros((1, num_class))
img_y[0,5] = 1

y = tf.placeholder(tf.float32, [None, num_class])
output = vgg16.out

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = output)
with tf.name_scope('loss'):
    loss_val = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_val)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('output: ', sess.run(vgg16.out, feed_dict = {vgg16.input_img: img}) )
    print('label: ', img_y)
    
    for i in range(5):
        loss, _ = sess.run([loss_val, optimizer], feed_dict = {vgg16.input_img:img,
                           y: img_y})
        print('Loss: ', loss)
    

















