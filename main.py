import tensorflow as tf
import numpy as np
import vgg16_D as vgg16
import preprocessing
from tqdm import tqdm


'''main.py file set up all configurations: training, validating hyperparameters'''
num_class = 10
num_of_epochs = 25
batch_size = 50
learning_rate = 0.001

# training & validate dataset
x_train, y_train = preprocessing.x_train, preprocessing.y_train
x_valid, y_valid = preprocessing.x_valid, preprocessing.y_valid
# data type = list, length = total batches of data

y = tf.placeholder(tf.float32, [None, num_class])
output = vgg16.out

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = output)
with tf.name_scope('loss'):
    loss_val = tf.reduce_mean(cross_entropy)
    
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
correct_pred = tf.cast(correct_pred, tf.float32)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(correct_pred)
    
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_val)

save_path = './model/' # path to save model
saver = tf.train.Saver()    
tf.summary.scalar('loss', loss_val)
tf.summary.scalar('accuracy', accuracy)
merge = tf.summary.merge_all()
writer = tf.summary.FileWriter(save_path)

num_of_batches = len(x_train)
test_batches = len(x_valid)
train_curve = []
valid_curve = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    print('Training batch size:', x_train[0].shape)
    print('Training label size:', len(y_train[0]))
    
    for iterate in range(num_of_epochs):
        print('Epoch start: ', iterate + 1)
        
        train_acc = 0
        train_loss = 0
        for batch in tqdm(range(num_of_batches), desc = 'Epochs: {}'.format(iterate + 1), unit = 'batch'):
            img_batch = x_train[batch]
            label_batch = y_train[batch]
            batch_loss, acc, _ = sess.run([loss_val, accuracy, optimizer], feed_dict = {vgg16.input_img: img_batch,
                               y: label_batch})
            
            print('loss = ', batch_loss)
            train_loss += batch_loss
            train_acc += acc
    
        train_acc /= num_of_batches
        train_loss /= num_of_batches
        print('Training accuracy: {},  Loss: {:.4f}'.format(train_acc, train_loss))
        train_curve.append(train_acc)
        
        valid_acc = 0
        for batch in range(test_batches):
            acc = sess.run(accuracy, feed_dict = {vgg16.input_img: x_valid[batch],
                                                    y: y_valid[batch]})
            valid_acc += acc
        
        summary = sess.run(merge, feed_dict = {vgg16.input_img: x_valid[0],
                                                    y: y_valid[0]})
        writer.add_summary(summary, iterate*batch) # summary on testing set
            
        valid_acc /= test_batches
        valid_curve.append(valid_acc)
        
        print('Validate accuracy: {:.4f}'.format(valid_acc))
        
    saver.save(sess, './model/vgg_model')
        
        
        
    
    
    
    
    
    