from __future__ import division

import os
import math
import numpy as np
import tensorflow as tf

from ops import *
from datasets import *
from matplotlib import pyplot as plt


class GenNet(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.delta = config.delta
        self.sigma = config.sigma
        self.sample_steps = config.sample_steps
        self.z_dim = config.z_dim

        self.num_epochs = config.num_epochs
        self.data_path = os.path.join(config.data_path, config.category)
        self.log_step = config.log_step
        self.output_dir = os.path.join(config.output_dir, config.category)

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.sample_dir = os.path.join(self.output_dir, 'sample')
        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32)
        self.build_model()

    def generator(self, z, reuse=False, is_training=True):
        ####################################################
        # Define the structure of generator, you may use the
        # generator structure of DCGAN. ops.py defines some
        # layers that you may use.
        ####################################################
        with tf.variable_scope('gen') as scope:
            if reuse:
                scope.reuse_variables()

            # https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
            # support function to calculate the out size
            s_h, s_w = self.image_size, self.image_size
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            #s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)


            if is_training:
                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, 32*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

                h0 = tf.reshape(
                    self.z_, [-1, s_h16, s_w16, 32*8])
                h0 = leaky_relu(batch_norm(h0, name='g_bn0'))

                h1, h1_w, h1_b = deconv2d(
                    h0, [self.batch_size, s_h8, s_w8, 32*4], name='g_h1', with_w=True)
                h1 = leaky_relu(batch_norm(h1, name='g_bn1'))

                h2, h2_w, h2_b = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, 32*2], name='g_h2', with_w=True)
                h2 = leaky_relu(batch_norm(h2, name='g_bn2'))

                h3, h3_w, h3_b = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, 32*1], name='g_h3', with_w=True)
                h3 = leaky_relu(batch_norm(h3, name='g_bn3'))

                #h4, h4_w, h4_b = deconv2d(
                #    h3, [self.batch_size, s_h2, s_w2, 32*1], name='g_h4', with_w=True)
                #h4 = leaky_relu(batch_norm(h4, name='g_bn4'))

                h4, h4_w, h4_b = deconv2d(
                    h3, [self.batch_size, s_h, s_w, 3], name='g_h4', with_w=True)

                return tf.nn.tanh(h4)

            else:
                z_, h0_w, h0_b = linear(
                    z, 32*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

                h0 = tf.reshape(
                    z_, [-1, s_h16, s_w16, 32*8])
                h0 = leaky_relu(batch_norm(h0, train=False, name='g_bn0'))

                h1 = deconv2d(
                    h0, [self.batch_size, s_h8, s_w8, 32*4], name='g_h1')
                h1 = leaky_relu(batch_norm(h1, train=False, name='g_bn1'))

                h2 = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, 32*2], name='g_h2')
                h2 = leaky_relu(batch_norm(h2, train=False, name='g_bn2'))

                h3 = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, 32*1], name='g_h3')
                h3 = leaky_relu(batch_norm(h3, train=False, name='g_bn3'))

                #h4 = deconv2d(
                #    h3, [self.batch_size, s_h2, s_w2, 32*1], name='g_h4')
                #h4 = leaky_relu(batch_norm(h4, train=False, name='g_bn4'))

                h4 = deconv2d(
                    h3, [self.batch_size, s_h, s_w, 3], name='g_h4')

                return tf.nn.tanh(h4)



    def langevin_dynamics(self, z):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated z.
        ####################################################
        def cond(i, z):
            return tf.less(i, self.sample_steps)

        def body(i, z):
            noise = tf.random_normal(shape=tf.shape(z), name='noise')
            gen_res = self.generator(z, reuse=True)
            gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma * self.sigma) * tf.square(self.obs - gen_res), axis=0)
            grad = tf.gradients(gen_loss, z, name='grad_des')[0]
            z = z - 0.5 * self.delta * self.delta * (z + grad)
            z = z + self.delta * noise
            return tf.add(i, 1), z

        i = tf.constant(0)
        i, z = tf.while_loop(cond, body, [i, z])
        return z

    def build_model(self):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        self.g_res = self.generator(self.z)
        self.gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma * self.sigma) * tf.square(self.obs - self.g_res))
        tf.summary.scalar('gen_loss', self.gen_loss)
        self.merged_summary = tf.summary.merge_all()

        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        self.infer_op = self.langevin_dynamics(self.z)
        self.train_op = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1).minimize(self.gen_loss, var_list=self.g_vars)

    def train(self):
        # Prepare training data
        train_data = DataSet(self.data_path, image_size=self.image_size)
        train_data = train_data.to_range(-1, 1)

        num_batches = int(math.ceil(len(train_data) / self.batch_size))
        summary_op = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=50)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        #self.sess.graph.finalize()

        print('Start training ...')

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # reconstructed images and synthesized images in
        # self.sample_dir, loss in self.log_dir (using writer).
        ####################################################
        
        # Initialize (sample_z)
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        # Inference
        self.lossV = []
        for epoch in range(self.num_epochs):
            for idx in range(num_batches):
                batch_images = train_data[idx * self.batch_size:(idx + 1) * self.batch_size, :]

                batch_z = self.sess.run([self.infer_op], 
                    feed_dict={self.obs: batch_images,self.z:batch_z})
                batch_z = batch_z[0]

                _, g_res, gen_loss, summary = self.sess.run([self.train_op, self.g_res, self.gen_loss, self.merged_summary],
                    feed_dict={self.obs: batch_images,self.z: batch_z})
                
                print("Epoch: {:2d}/{:2d} {:4d}/{:4d} Loss:{:8f}"
                    .format(epoch, self.log_step, idx, num_batches, gen_loss))
                self.writer.add_summary(summary, epoch)

                if idx == num_batches - 1:
                    self.save(self.model_dir, epoch)

                if epoch == self.num_epochs - 1 and idx == num_batches - 1:
                    save_images(batch_images, os.path.join(self.sample_dir,"ori.png"))
                    save_images(g_res, os.path.join(self.sample_dir,"rec.png"))
                    print("... finished")
            self.lossV.append(gen_loss)
        
        losscurve   = plt.plot(np.arange(self.num_epochs),self.lossV)
        plt.title('Loss vs Epoch')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('LossvsEpoch.png')

    def test1(self):
        z_test_ran = np.random.uniform(-2, 2, [self.batch_size, self.z_dim]).astype(np.float32)
        
        ran = self.sess.run(self.generator(self.z, reuse=True, is_training=True),
            feed_dict={self.z:z_test_ran})
        print(os.path.join(self.sample_dir, "ran.png"))
        
        save_images(ran, os.path.join(self.sample_dir, "ran.png")) #"ran.png")#
        print("... finished")
        
    def test2(self):
        zline = 4*(np.column_stack((  np.repeat(np.arange(8),8)   ,  list(np.arange(8))*8  )  )/7-0.5)
        z2 =  np.array(zline[np.arange(11)])
        Y_hat2 = self.sess.run(self.generator(self.z, reuse=True, is_training=True),
                feed_dict={self.z:z2})
        imline = Y_hat2
        for i in range(4):
            z2 =  zline[(i+1)*11:(i+2)*11]
            Y_hat2 = self.sess.run(self.generator(self.z, reuse=True, is_training=True),
                feed_dict={self.z:z2})
            imline  = np.vstack((imline , Y_hat2 ))
        z2 =  zline[53:64]
        Y_hat2 = self.sess.run(self.generator(self.z, reuse=True, is_training=True),
                feed_dict={self.z:z2})[2:11]
        imline  = np.vstack((imline , Y_hat2 ))
        
#        idx = (np.arange(8) - np.mean(np.arange(8)))*2/3.5
#        z_lin = np.zeros((8,8, 2))
#        lin = np.zeros((8,8, self.image_size, self.image_size, 3))
#
#        for i in range(8):
#            for j in range(8):
#                z_lin[i,j,:] = [idx[i], idx[j]]
#
#        for i in range(8):
#            lin[i,:] = self.sess.run(self.generator(self.z, reuse=True, is_training=True),
#                feed_dict={self.z:z_lin[i,:]})

        #lin = lin.reshape((-1, self.image_size, self.image_size, 3))
        save_images(imline,os.path.join(self.sample_dir, "lin.png"))
        print("... finished")


    # property from DCGAN
    def save(self, checkpoint_dir, step):
        model_name = "GenNet.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
