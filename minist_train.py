#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
# 加载mnist_inference.py中定义的常量和前向传播的函数
import mnist_inference

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.001
TRAINING_STEPS = 100000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"
x_list = []
y_list = []
z_list = []
 
def train(mnist):
    # 定义输入输出placeholder
    # 调整输入数据placeholder的格式，输入为一个四维矩阵
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,                             # 第一维表示一个batch中样例的个数
        mnist_inference.IMAGE_SIZE,             # 第二维和第三维表示图片的尺寸
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS],          # 第四维表示图片的深度，对于RBG格式的图片，深度为5
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
 
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用mnist_inference.py中定义的前向传播过程
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)
 
    #定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
 
    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            #类似地将输入的训练数据格式调整为一个四维矩阵，并将这个调整后的数据传入sess.run过程
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            #每1000轮保存一次模型。
            if (i+1) % 1000 == 0:
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。通过损失函数的大小可以大概了解训练的情况。
                # 在验证数据集上的正确率信息会有一个单独的程序来生成。
                print("After %d training step(s), loss on training batch is %f." % (step, loss_value))
                x_list.append(step)
                y_list.append(loss_value)
                # 保存当前的模型。注意这里隔出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                evaluate(mnist)



def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [
            mnist.validation.num_examples,           # 第一维表示样例的个数
            mnist_inference.IMAGE_SIZE,             # 第二维和第三维表示图片的尺寸
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS],          # 第四维表示图片的深度，对于RBG格式的图片，深度为5
                       name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
 
        validate_feed = {x: np.reshape(mnist.validation.images, (mnist.validation.num_examples, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS)),
                         y_: mnist.validation.labels}
        # 直接通过调用封装好的函数来计算前向传播的结果。
        # 因为测试时不关注正则损失的值，所以这里用于计算正则化损失的函数被设置为None。
        y = mnist_inference.inference(x, False, None)
 
        # 使用前向传播的结果计算正确率。
        # 如果需要对未知的样例进行分类，那么使用tf.argmax(y, 1)就可以得到输入样例的预测类别了。
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平局值了。
        # 这样就可以完全共用mnist_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
 
        #每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名 
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            # 加载模型
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 通过文件名得到模型保存时迭代的轮数
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
            print("After %s training step(s), validation accuracy = %f" % (global_step, accuracy_score))
            z_list.append(accuracy_score*100)


 
def main(argv=None):
    mnist = input_data.read_data_sets("dataset/", one_hot=True)
    train(mnist)
    plt.plot(x_list,y_list,color = 'm',label='Lost on training')
    plt.title("Learning Rate = %f" % LEARNING_RATE_BASE)
    plt.xlabel("Training step")
    plt.ylabel("Lost on training batch")
    plt.grid()
    plt.legend(loc='upper left')
    plt.twinx()
    plt.plot(x_list,z_list,color = 'c',label='Accuracy')
    plt.ylabel("Validation accuracy")
    plt.legend(loc='lower left')
    plt.show()

 
 
if __name__ == '__main__':
    tf.app.run()