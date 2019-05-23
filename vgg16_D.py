import tensorflow as tf
import numpy as np


num_class = 10
input_img = tf.placeholder(tf.float32, [None, 224, 224, 3]) # fixed input image size

'''first conv. stack'''
# all filter size: 3*3, stride = 1
w1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev = 0.01), name = 'w1')
b1 = tf.Variable(tf.zeros(64), name = 'b1')
c1 = tf.nn.conv2d(input_img, w1, strides = [1, 1, 1, 1], padding = 'SAME')
c1_out = tf.nn.relu(tf.nn.bias_add(c1, b1))

w2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.01), name = 'w2')
b2 = tf.Variable(tf.zeros(64), name = 'b2')
c2 = tf.nn.conv2d(c1_out, w2, strides = [1, 1, 1, 1], padding = 'SAME')
c2_out = tf.nn.relu(tf.nn.bias_add(c2, b2))
# after conv. layer, feature map size remain unchanged. Only max pooling change spatial resolution

p1 = tf.nn.max_pool(c2_out,
                    ksize = [1, 2, 2, 1],
                    strides = [1, 2, 2, 1],
                    padding = 'VALID')
# map size / 2 
'''second conv. stack'''
w3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev = 0.01), name = 'w3')
b3 = tf.Variable(tf.zeros(128), name = 'b3')
c3 = tf.nn.conv2d(p1, w3, strides = [1, 1, 1, 1], padding = 'SAME')
c3_out = tf.nn.relu(tf.nn.bias_add(c3, b3))

w4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev = 0.01), name = 'w4')
b4 = tf.Variable(tf.zeros(128), name = 'b4')
c4 = tf.nn.conv2d(c3_out, w4, strides = [1, 1, 1, 1], padding = 'SAME')
c4_out = tf.nn.relu(tf.nn.bias_add(c4, b4))

p2 = tf.nn.max_pool(c4_out,
                    ksize = [1, 2, 2, 1],
                    strides = [1, 2, 2, 1],
                    padding = 'VALID')

'''third conv. stack'''
w5 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev = 0.01), name = 'w5')
b5 = tf.Variable(tf.zeros(256), name = 'b5')
c5 = tf.nn.conv2d(p2, w5, strides = [1, 1, 1, 1], padding = 'SAME')
c5_out = tf.nn.relu(tf.nn.bias_add(c5, b5))

w6 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev = 0.01), name = 'w6')
b6 = tf.Variable(tf.zeros(256), name = 'b6')
c6 = tf.nn.conv2d(c5_out, w6, strides = [1, 1, 1, 1], padding = 'SAME')
c6_out = tf.nn.relu(tf.nn.bias_add(c6, b6))

w7 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev = 0.01), name = 'w7')
b7 = tf.Variable(tf.zeros(256), name = 'b7')
c7 = tf.nn.conv2d(c6_out, w7, strides = [1, 1, 1, 1], padding = 'SAME')
c7_out = tf.nn.relu(tf.nn.bias_add(c7, b7))

p3 = tf.nn.max_pool(c7_out,
                    ksize = [1, 2, 2, 1],
                    strides = [1, 2, 2, 1],
                    padding = 'VALID')

'''fourth conv. stack'''
w8 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev = 0.01), name = 'w8')
b8 = tf.Variable(tf.zeros(512), name = 'b8')
c8 = tf.nn.conv2d(p3, w8, strides = [1, 1, 1, 1], padding = 'SAME')
c8_out = tf.nn.relu(tf.nn.bias_add(c8, b8))

w9 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev = 0.01), name = 'w9')
b9 = tf.Variable(tf.zeros(512), name = 'b9')
c9 = tf.nn.conv2d(c8_out, w9, strides = [1, 1, 1, 1], padding = 'SAME')
c9_out = tf.nn.relu(tf.nn.bias_add(c9, b9))

w10 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev = 0.01), name = 'w10')
b10 = tf.Variable(tf.zeros(512), name = 'b10')
c10 = tf.nn.conv2d(c9_out, w10, strides = [1, 1, 1, 1], padding = 'SAME')
c10_out = tf.nn.relu(tf.nn.bias_add(c10, b10))

p4 = tf.nn.max_pool(c10_out,
                    ksize = [1, 2, 2, 1],
                    strides = [1, 2, 2, 1],
                    padding = 'VALID')
'''fifth conv. stack'''
w11 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev = 0.01), name = 'w11')
b11 = tf.Variable(tf.zeros(512), name = 'b11')
c11 = tf.nn.conv2d(p4, w11, strides = [1, 1, 1, 1], padding = 'SAME')
c11_out = tf.nn.relu(tf.nn.bias_add(c11, b11))

w12 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev = 0.01), name = 'w12')
b12 = tf.Variable(tf.zeros(512), name = 'b12')
c12 = tf.nn.conv2d(c11_out, w12, strides = [1, 1, 1, 1], padding = 'SAME')
c12_out = tf.nn.relu(tf.nn.bias_add(c12, b12))

w13 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev = 0.01), name = 'w13')
b13 = tf.Variable(tf.zeros(512), name = 'b13')
c13 = tf.nn.conv2d(c12_out, w13, strides = [1, 1, 1, 1], padding = 'SAME')
c13_out = tf.nn.relu(tf.nn.bias_add(c13, b13))

p5 = tf.nn.max_pool(c13_out,
                    ksize = [1, 2, 2, 1],
                    strides = [1, 2, 2, 1],
                    padding = 'VALID')
# flatten
conv_out = tf.layers.flatten(p5) # flatten to [batch size, f_size]
f_size = conv_out.get_shape().as_list()[1] # get_shape() could get tensor shape without session graph, but return dimension type. Use as_list to convert to list type
w14 = tf.Variable(tf.truncated_normal([f_size, 4096], stddev = 0.01), name = 'w14')
b14 = tf.Variable(tf.zeros(4096), 'b14')
fc_14 = tf.nn.relu(tf.nn.bias_add(tf.matmul(conv_out, w14), b14))


w15 = tf.Variable(tf.truncated_normal([4096, 4096], stddev = 0.01), 'w15')
b15 = tf.Variable(tf.zeros(4096), 'b15')
fc_15 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc_14, w15), b15))

w16 = tf.Variable(tf.truncated_normal([4096, num_class], stddev = 0.01), 'w16')
b16 = tf.Variable(tf.zeros(num_class), 'b16')
out = tf.nn.bias_add(tf.matmul(fc_15, w16), b16)









