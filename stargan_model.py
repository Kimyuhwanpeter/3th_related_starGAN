# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def generator(input_shape=(256, 256, 3), 
              label_shape=(54), 
              dim=64,
              D_samples=2,
              n_res=6,
              U_samples=2,
              final_output=3,
              batch_size=2):
    channel = dim
    def resblock(x_input, dim, use_bias=True):
        x = tf.pad(x_input, [[0,0],[1,1],[1,1],[0,0]])
        x = tf.keras.layers.Conv2D(dim, 3, 1, use_bias=use_bias, padding='same')(h)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]])
        x = tf.keras.layers.Conv2D(dim, 3, 1, use_bias=use_bias, padding='same')(h)
        x = InstanceNormalization()(x)
        return x + x_input

    h = inputs = tf.keras.Input(input_shape, batch_size=batch_size)
    labels = tf.keras.Input(label_shape, batch_size=batch_size)

    C = tf.cast(tf.reshape(labels, [-1, 1, 1, labels.shape[-1]]), tf.float32)
    C = tf.tile(C, [1, inputs.shape[1], inputs.shape[2], 1])
    h = tf.concat([inputs, C], -1)

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]])
    h = tf.keras.layers.Conv2D(channel, 7, 1, use_bias=False, padding='valid')(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    # Down-sampling
    for i in range(D_samples):
        h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]])
        h = tf.keras.layers.Conv2D(channel*2, 4, 2, use_bias=False, padding='valid')(h)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h)
        channel = channel * 2

    # Bottleneck
    for i in range(n_res):
        h = resblock(h, channel, use_bias=False)

    # Up-sampling
    for i in range(U_samples):
        h = tf.keras.layers.Conv2DTranspose(channel // 2, 4, 2, use_bias=False, padding='same')(h)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h)
        channel = channel // 2

    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]])
    h = tf.keras.layers.Conv2D(final_output, 7, 1, use_bias=False, padding='valid')(h)
    h = tf.keras.layers.Activation('tanh')(h)

    return tf.keras.Model(inputs=[inputs, labels], outputs=h)

def discriminator(input_shape=(256, 256, 3),
                  dim=64, 
                  n_dis=6,
                  c_dims=54,
                  batch_size=2):
    
    h = inputs = tf.keras.Input(input_shape, batch_size=batch_size)
    
    channel = dim

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]])
    h = tf.keras.layers.Conv2D(filters=channel, kernel_size=4, strides=2, use_bias=True, padding='valid')(h)
    h = tf.keras.layers.LeakyReLU(0.01)(h)

    for i in range(1, n_dis):
        h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]])
        h = tf.keras.layers.Conv2D(filters=channel * 2, kernel_size=4, strides=2, use_bias=True, padding='valid')(h)
        h = tf.keras.layers.LeakyReLU(0.01)(h)
        channel = channel * 2

    c_kernel = int(input_shape[1] / np.power(2, n_dis))

    h_buf = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]]) 

    logit = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, use_bias=False, padding='valid')(h_buf)
    x = tf.keras.layers.Conv2D(filters=c_dims, kernel_size=c_kernel, strides=1, use_bias=False)(h)
    x = tf.reshape(x, [-1, c_dims])

    #h = tf.reshape(h, [-1, c_dims])

    return tf.keras.Model(inputs=inputs, outputs=[logit, x])