# -*- coding: utf-8 -*-
# StarGAN: Unified Generative Adversarial Networks 
# for Multi-Domain Image-to-Image Translation
# https://github.com/taki0112/StarGAN-Tensorflow
from model.stargan_model import *
from absl import flags, app
from random import shuffle

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import sys
import random
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 128,
                           
                           "ch": 3,
                           
                           "batch_size": 16,
                           
                           "epochs": 100,
                           
                           "num_classes":48+2,
                           
                           "lr": 0.0001,
                           
                           "augment_flag": True,
                           
                           "adv_weight": 1,
                           
                           "rec_weight": 10.,
                           
                           "cls_weight": 10.,
                           
                           "A_txt_path": "/content/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_40_63_16_39/train/male_40_63_train.txt",
                           
                           "A_img_path": "/content/AFAD/All/male_40_63/",
                           
                           "A_n_images": 1751,
                           
                           "B_txt_path": "/content/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_40_63_16_39/train/female_16_39_train.txt",
                           
                           "B_img_path": "/content/Morph/All/female_16_39/",
                           
                           "B_n_images": 1751,
                           
                           "pre_checkpoint": False,
                           
                           "sample_images": "/content/drive/My Drive/3rd_paper/[1]Age_gender_race_dataset/starGAN/AFAD-M_Morph-F_40_63_16_39/sample_images",
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint_path": "/content/drive/My Drive/3rd_paper/[1]Age_gender_race_dataset/starGAN/AFAD-M_Morph-F_40_63_16_39/checkpoint",
                           
                           "train": True})

# Define optimization
generator_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
discriminator_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

#def augmentation(image, aug_size):
#    seed = random.randint(0, 2 ** 31 - 1)
#    ori_image_shape = tf.shape(image)
#    image = tf.image.random_flip_left_right(image, seed=seed)
#    image = tf.image.resize(image, [aug_size, aug_size])
#    image = tf.image.random_crop(image, ori_image_shape, seed=seed)
#    return image

#def load_train_images(filename):

#    img = cv2.imread(filename)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img = cv2.resize(img, (FLAGS.img_size,FLAGS.img_size), interpolation=cv2.INTER_CUBIC)
#    img = img / 127.5 - 1

#    if FLAGS.augment_flag == True:
#        augment_size = FLAGS.img_size + (30 if FLAGS.img_size == 256 else 15)
#        p = random.random()

#        if p > 0.5 :
#            img = augmentation(img, augment_size)

#    return img

def imread(data):
    image = cv2.imread(data)
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_RGB = image_RGB.astype(np.float)
    return image_RGB

def load_train_images(data_path, is_testing = False):
    
    image = imread(data_path)

    if not is_testing:
        resize_image = cv2.resize(image, (FLAGS.img_size + 30, FLAGS.img_size + 30), interpolation=cv2.INTER_CUBIC)
        h1 = int(np.ceil(np.random.uniform(1e-2, FLAGS.img_size + 30-FLAGS.img_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, FLAGS.img_size + 30-FLAGS.img_size)))
        resize_image = resize_image[h1:h1+FLAGS.img_size, w1:w1+FLAGS.img_size]

        if np.random.random() > 0.5:
            resize_image = np.fliplr(resize_image)
    else:
        resize_image = cv2.resize(resize_image, (FLAGS.img_size, FLAGS.img_size), interpolation=cv2.INTER_CUBIC)

    img = resize_image / 127.5 - 1

    return img

def load_train_labels(data):
    
    #data = int(data)
    #labels = np.eye(FLAGS.num_classes)[data]
    return data

@tf.function
def gen_model(model, images1, images2, training=True):
    generated_img = model([images1, images2], training=training)
    return generated_img

@tf.function
def dis_model(model, images, training=True):
    logits, cls = model(images, training=training)
    return logits, cls

#@tf.function
def train_step(fake_generator,
               recon_generator,
               fake_discriminator,
               real_discriminator,
               A_images,
               A_labels,
               B_labels):
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        # 이 비교실험은 현재 내가 새로 실험하는 방법은 인종 + 나이인데
        # 내 실험을 나이 변환에만 초점을 맞춰 비교해보는것도 괜찮은 것 같다.
        # 예를들면 MORPH 16 years -> MORPH 43 years
        x_fake = gen_model(fake_generator, A_images, B_labels)
        x_recon = gen_model(fake_generator, x_fake, A_labels)

        real_logit, real_cls = dis_model(fake_discriminator, A_images)
        fake_logit, fake_cls = dis_model(fake_discriminator, x_fake)

        g_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fake_logit), fake_logit))
        g_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(B_labels, fake_cls))
        g_rec_loss = tf.reduce_mean(tf.math.abs(A_images - x_recon))

        d_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(real_logit), real_logit)) \
                     + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fake_logit), fake_logit))
        d_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(A_labels, real_cls))

        g_loss = FLAGS.adv_weight * g_adv_loss + FLAGS.cls_weight * g_cls_loss + FLAGS.rec_weight * g_rec_loss
        d_loss = FLAGS.adv_weight * d_adv_loss + FLAGS.cls_weight * d_cls_loss

    generator_gradients = gen_tape.gradient(g_loss, fake_generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(d_loss, fake_discriminator.trainable_variables)

    generator_optim.apply_gradients(zip(generator_gradients, fake_generator.trainable_variables))
    
    discriminator_optim.apply_gradients(zip(discriminator_gradients, fake_discriminator.trainable_variables))

    return g_loss, d_loss

def main(argv=None):
    fake_generator = generator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch), 
                               label_shape=(FLAGS.num_classes), batch_size=FLAGS.batch_size)
    recon_generator = generator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch), 
                               label_shape=(FLAGS.num_classes), batch_size=FLAGS.batch_size)

    real_discriminator = discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch),
                                       c_dims=FLAGS.num_classes, batch_size=FLAGS.batch_size)
    fake_discriminator = discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.ch),
                                       c_dims=FLAGS.num_classes, batch_size=FLAGS.batch_size)

    fake_generator.summary()
    fake_discriminator.summary()        

    if FLAGS.pre_checkpoint is True:
    
        ckpt = tf.train.Checkpoint(fake_generator=fake_generator,
                                   fake_discriminator=fake_discriminator,
                                   generator_optim=generator_optim,
                                   discriminator_optim=discriminator_optim)

        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    if FLAGS.train == True:
        print('Start training.....')
        count = 0
        for epoch in range(FLAGS.epochs):
            A_name_buf = []
            A_label_buf = []
            B_label_buf = []
            A_data = open(FLAGS.A_txt_path, 'r')
            for i in range(FLAGS.A_n_images):
                line = A_data.readline()
                name = FLAGS.A_img_path + line.split(' ')[0]
                age = int((line.split(' ')[1]).split('\n')[0]) - 16
                age = tf.one_hot(age, 48)
                age = age.numpy()
                age = list(age)
                target_age = age
                race = [0.0]
                gender = [1.0]
                sum = race + gender + age
                shuffle(target_age)
                # [race, gender, age] asian-0, west-1, male-0, female-1

                A_name_buf.append(name)
                A_label_buf.append(sum)
                B_label_buf.append([1.0] + [0.0] + target_age)

                # 지금 가지고있는 데이터를 합쳐야함 그렇게하고 한꺼번에 돌려야한다

            #A = list(zip(A_name_buf, A_label_buf))
            #shuffle(A)
            #A_name_buf, A_label_buf = zip(*A)

            #B_name_buf = []
            #B_label_buf = []
            B_data = open(FLAGS.B_txt_path, 'r')
            for i in range(FLAGS.B_n_images):
                line = B_data.readline()
                name = FLAGS.B_img_path + line.split(' ')[0]
                age = int((line.split(' ')[1]).split('\n')[0]) - 16
                age = tf.one_hot(age, 48)
                age = age.numpy()
                age = list(age)
                target_age = age
                race = [1.0]
                gender = [0.0]
                sum = race + gender + age
                shuffle(target_age)
                
                A_name_buf.append(name)
                A_label_buf.append(sum)
                B_label_buf.append([0.0] + [1.0] + target_age)

            T = list(zip(A_name_buf, A_label_buf, B_label_buf))
            shuffle(T)
            A_name_buf, A_label_buf, B_label_buf = zip(*T)


            #B = list(B_label_buf)
            #shuffle(B)
            #B_label_buf = B

            batch_idx = (FLAGS.A_n_images + FLAGS.B_n_images) // FLAGS.batch_size
            for step in range(batch_idx):
                A_batch_images = list(A_name_buf[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size])
                A_batch_images = [load_train_images(A_batch_image) for A_batch_image in A_batch_images]
                A_batch_images = np.array(A_batch_images).astype(np.float32)
                #B_batch_images = list(B_name_buf[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size])
                #B_batch_images = [load_train_images(B_batch_image) for B_batch_image in B_batch_images]
                #B_batch_images = np.array(B_batch_images).astype(np.float32)

                A_batch_labels = list(A_label_buf[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size])
                A_batch_labels = [load_train_labels(A_batch_label) for A_batch_label in A_batch_labels]
                B_batch_labels = list(B_label_buf[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size])
                B_batch_labels = [load_train_labels(B_batch_label) for B_batch_label in B_batch_labels]
                A_batch_labels = np.array(A_batch_labels).astype(np.float32)
                B_batch_labels = np.array(B_batch_labels).astype(np.float32)
                
                G_loss, D_loss = train_step(fake_generator,
                                            recon_generator,
                                            fake_discriminator,
                                            real_discriminator,
                                            A_batch_images,
                                            A_batch_labels,
                                            B_batch_labels)
                

                if count % 100 == 0:
                    print("Epoch: {}, G_loss = {}, D_loss = {} [{}/{}]".format(epoch, G_loss, D_loss, step + 1, batch_idx))
                    x_fake = gen_model(fake_generator, A_batch_images, B_batch_labels, False)

                    plt.imsave(FLAGS.sample_images + "/" + "{}_1_fake.png".format(count), x_fake[0].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/" + "{}_2_fake.png".format(count), x_fake[1].numpy() * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/" + "{}_3_fake.png".format(count), x_fake[2].numpy() * 0.5 + 0.5)

                    # plt.imsave(FLAGS.sample_images + "/" + "{}_1_original.png".format(count), A_batch_images[0] * 0.5 + 0.5)
                    # plt.imsave(FLAGS.sample_images + "/" + "{}_2_original.png".format(count), A_batch_images[1] * 0.5 + 0.5)
                    # plt.imsave(FLAGS.sample_images + "/" + "{}_3_original.png".format(count), A_batch_images[2] * 0.5 + 0.5)

                if count % 1000 == 0:
                    model_dir = FLAGS.save_checkpoint_path
                    folder_name = int(count/1000)
                    folder_neme_str = '%s/%s' % (model_dir, folder_name)
                    if not os.path.isdir(folder_neme_str):
                        print("Make {} folder to save checkpoint".format(folder_name))
                        os.makedirs(folder_neme_str)
                    checkpoint = tf.train.Checkpoint(fake_generator=fake_generator,
                                               fake_discriminator=fake_discriminator,
                                               generator_optim=generator_optim,
                                               discriminator_optim=discriminator_optim)
                    checkpoint_dir = folder_neme_str + "/" + "starGAN_model_{}_steps.ckpt".format(count + 1)
                    checkpoint.save(checkpoint_dir)


                count += 1


if __name__ == '__main__':
    main()
