
import numpy as np 
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math
import time
from tf_cnnvis import *
import saliency
import seaborn  as sns
inputLength = 784;
batchSize = 100
iterations = 100
dropOut = 1
numClasses = 10
learning_rate = 0.001

def conv2d(x, W, b, name,strides=1):
    x = tf.nn.conv2d(input = x, filter = W, strides=[1, strides, strides, 1], padding='VALID', name = name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x,name,k=2):
    return tf.nn.max_pool(value = x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID', name = name)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 3
    n_rows = math.ceil(6 / n_columns) + 1
    plt.figure()
    for i in range(9):
        ax3 = plt.subplot(n_rows, n_columns, i+1)
        ax3.axis('off')
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.savefig('conv1' + '.png')

def conv_net(x, weights, biases, dropout=dropOut):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'],"Convolution1")
    print conv1.shape
    conv1 = tf.layers.batch_normalization(conv1)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv1,"Pooling1",2)
    print conv2.shape
    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc2'], biases['bc2'],"Convolution2")
    print conv3.shape
    conv4 = conv2d(conv3, weights['wc3'], biases['bc2'],"Convolution3")
    print conv4.shape
    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv4,"Pooling2",2)
    print conv5.shape
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'],name = "FC")
    fc1 = tf.nn.relu(fc1,name ='relu10')
    print fc1.shape
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, 0.8,name = 'Dropout')

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'],name = "Output")
    return conv1,conv2, conv3,conv4, fc1, x, out
    

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():

            weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.05)),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.05)),

            'wc3': tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.05)),

            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.truncated_normal([2*2*64, 1024], stddev=0.05)),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.truncated_normal([1024, numClasses],stddev=0.05))
            }

            biases = {
            'bc1': tf.Variable(tf.truncated_normal([32],stddev=0.05)),
            'bc2': tf.Variable(tf.truncated_normal([64],stddev=0.05)),
            'bd1': tf.Variable(tf.truncated_normal([1024],stddev=0.05)),
            'out': tf.Variable(tf.truncated_normal([numClasses],stddev=0.05))
            }
            X = tf.placeholder(tf.float32, [None, inputLength])
            Y = tf.placeholder(tf.float32, [None, numClasses])
            conv1,conv2, conv3,conv4, fc1,x_image,logits = conv_net(X, weights, biases, dropOut)
            prediction = tf.nn.softmax(logits,name = "Final")
            # Define loss and optimizer
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss_op)
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            init = tf.global_variables_initializer()
            sess.run(init)

            for i in range(iterations):
                batch_x, batch_y = mnist.train.next_batch(batchSize)
                sess.run(train_op, feed_dict={X:batch_x,Y:batch_y})
                if i % 100 != 0 and i != 0:
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y,
                                                                            })
                    trainAccuracy = sess.run(accuracy, feed_dict={X:batch_x,Y:batch_y})
                    print("Step " + str(i) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

            testAccuracy = sess.run(accuracy, feed_dict={X:mnist.test.images,Y:mnist.test.labels})
            print("test accuracy %g"%(testAccuracy))
            imageToUse = mnist.test.images[1,:]
            image = np.reshape(imageToUse,[28,28])
            feed_dict1 = {X: [imageToUse]}
            neuron_selector = tf.placeholder(tf.int32)
            y = logits[0][neuron_selector]
            units = sess.run(conv2,feed_dict={X:np.reshape(imageToUse,[1,784],order='F')})
            layers = ['r', 'p', 'c']
            layer = 'relu1'
            feed_dict = {X:np.reshape(imageToUse,[1,784]), Y:np.reshape(mnist.test.labels[1,:],[1,10])}
            start = time.time()
            target = 0;
            classification = prediction.eval(feed_dict1)
            for i,data in enumerate(classification.tolist()[0]):
                if (data > 0.5):
                    print i,data
                    target = i
            start = time.time() - start
            occluding_size = 4 
            occluding_pixel = 0
            occluding_stride = 4
            height, width= image.shape
            output_height = int(math.ceil((height-occluding_size)/occluding_stride+1))
            output_width = int(math.ceil((width-occluding_size)/occluding_stride+1))
            heatmap = np.zeros((output_height, output_width))

            for h in range(output_height):
                for w in range(output_width):
                    h_start = h*occluding_stride
                    w_start = w*occluding_stride
                    h_end = min(height, h_start + occluding_size)
                    w_end = min(width, w_start + occluding_size)
                    input_image = np.array(image, copy=True) 
                    input_image[h_start:h_end,w_start:w_end] =  occluding_pixel
                    feed_dict2 = {X:np.reshape(input_image,[1,784])}
                    classification = prediction.eval(feed_dict2).tolist()[0]
                    heatmap[h,w] = classification[target] # the probability of the correct class
            print("Total Time = %f" % (start))
            ax = sns.heatmap(heatmap,xticklabels=False, yticklabels=False,cmap="YlGnBu",linewidths=.5)
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("Resulting probability of classifier after occlusion")
            plt.savefig('heatmap' + '.png')
            graph = tf.get_default_graph()
            tensor = graph.get_tensor_by_name("FC" + ":0")
            tensor1 = graph.get_tensor_by_name("Convolution2" + ":0")
            plt.figure()
            start = time.time()
            start = time.time() - start
            for i in range(9):    
                loss1 = tf.reduce_mean(tensor[:,i])
                synimage = 0.1 * np.random.uniform(size=[1,784]) + 0.45
                gradient = tf.gradients(ys = loss1, xs = X)
                for data in range(400):
                    img_reshaped = np.reshape(synimage,[1,784])
                    grad, loss_value = sess.run([gradient, loss1],feed_dict={X: img_reshaped})
                    grad = np.array(grad).squeeze()
                    step_size = 1.0 / (grad.std() + 1e-8)
                    synimage += step_size * grad
                    synimage = np.clip(synimage, 0.0, 1.0)
                    feed_dict2 = {X:np.reshape(synimage,[1,784])}
                classification = prediction.eval(feed_dict2).tolist()[0][target]
                print classification
                ax1 = plt.subplot(3, 3, i+1)
                ax1.axis('off')
                ax1.imshow(np.reshape(synimage,[28,28]), interpolation="nearest", cmap="gray")
            print("Total Time = %f" % (start))
            plt.savefig('synimage' + '.png')
            for i in range(9):    
                loss2 = tf.reduce_mean(tensor1[:,:,:,i])
                synimage1 = 0.1 * np.random.uniform(size=[1,784]) + 0.45
                gradient2 = tf.gradients(ys = loss2, xs = X)
                for data in range(400):
                    img_reshaped = np.reshape(synimage1,[1,784])
                    grad, loss_value = sess.run([gradient2, loss2],feed_dict={X: img_reshaped})
                    grad = np.array(grad).squeeze()
                    step_size = 1.0 / (grad.std() + 1e-8)
                    synimage += step_size * grad
                    synimage = np.clip(synimage1, 0.0, 1.0)
                    feed_dict2 = {X:np.reshape(synimage1,[1,784])}
                classification = prediction.eval(feed_dict2).tolist()[0][target]
                print classification
                ax1 = plt.subplot(3, 3, i+1)
                ax1.axis('off')
                ax1.imshow(np.reshape(synimage,[28,28]), interpolation="nearest", cmap="gray")
            print("Total Time = %f" % (start))
            plt.savefig('synimage1' + '.png')
            is_success = deconv_visualization(sess_graph_path = sess, value_feed_dict = feed_dict, input_tensor=x_image, layers=layers)
            activation_visualization(sess_graph_path = sess, value_feed_dict = feed_dict, input_tensor=x_image, layers=layers)
        start = time.time() - start
        print("Total Time = %f" % (start))
        plotNNFilter(units)
        plt.show()
    plt.figure()
    plt.imshow(np.reshape(imageToUse,[28,28]), interpolation="nearest", cmap="gray")
    plt.axis('off')
    plt.savefig('originalImage' + '.png')
    plt.show()
    sess.close()

if __name__ == '__main__':
    main()
