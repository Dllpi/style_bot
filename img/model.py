# Copyright (C) 2018  Artsiom Sanakoyeu and Dmytro Kotovenko
#
# This file is part of Adaptive Style Transfer
#
# Adaptive Style Transfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Adaptive Style Transfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import division
from __future__ import print_function

import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import multiprocessing
from definitions import ROOT_DIR, OUTPUT_IMAGES_DIR
from img.module import *
from img.utils import *
import img.prepare_dataset
import img.img_augm


class Artgan(object):
    def __init__(self, sess, model_name, batch_size, image_size):
        self.model_name = model_name
        self.root_dir = os.path.join(ROOT_DIR, 'models/')
        self.checkpoint_dir = os.path.join(self.root_dir, self.model_name, 'checkpoint')
        self.checkpoint_long_dir = os.path.join(self.root_dir, self.model_name, 'checkpoint_long')
        #self.sample_dir = os.path.join(self.root_dir, self.model_name, 'sample')
        #self.inference_dir = os.path.join(self.root_dir, self.model_name, 'inference')
        self.logs_dir = os.path.join(self.root_dir, self.model_name, 'logs')

        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size

        self.loss = sce_criterion

        self.initial_step = 0

        '''OPTIONS = namedtuple('OPTIONS',
                             'batch_size image_size \
                              total_steps save_freq lr\
                              gf_dim df_dim \
                              is_training \
                              path_to_content_dataset \
                              path_to_art_dataset \
                              discr_loss_weight transformer_loss_weight feature_loss_weight')
        self.options = OPTIONS._make((args.batch_size, args.image_size,
                                      args.total_steps, args.save_freq, args.lr,
                                      args.ngf, args.ndf,
                                      args.phase == 'train',
                                      args.path_to_content_dataset,
                                      args.path_to_art_dataset,
                                      args.discr_loss_weight, args.transformer_loss_weight, args.feature_loss_weight
                                      ))'''

        # Create all the folders for saving the model
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        if not os.path.exists(os.path.join(self.root_dir, self.model_name)):
            os.makedirs(os.path.join(self.root_dir, self.model_name))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_long_dir):
            os.makedirs(self.checkpoint_long_dir)

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=2)
        self.saver_long = tf.train.Saver(max_to_keep=None)

    def _build_model(self):

        # ==================== Define placeholders. ===================== #
        with tf.name_scope('placeholder'):
            self.input_photo = tf.placeholder(dtype=tf.float32,
                                                  shape=[self.batch_size, None, None, 3],
                                                  name='photo')

        # ===================== Wire the graph. ========================= #
        # Encode input images.
        self.input_photo_features = encoder(image=self.input_photo,
                                                options=None,
                                                reuse=False)

        # Decode obtained features.
        self.output_photo = decoder(features=self.input_photo_features,
                                        options=None,
                                        reuse=False)

    def inference(self, img_path, resize_to_original=True, ckpt_nmbr=None):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Start inference.")

        #if self.load(self.checkpoint_dir, ckpt_nmbr):
        #    print(" [*] Load SUCCESS")
        #else:
        if self.load(self.checkpoint_long_dir, ckpt_nmbr):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        print('IMG_PATH:', img_path)
        img = scipy.misc.imread(img_path, mode='RGB')
        img_shape = img.shape[:2]

        # Resize the smallest side of the image to the self.image_size
        alpha = float(self.image_size) / float(min(img_shape))
        img = scipy.misc.imresize(img, size=alpha)
        img = np.expand_dims(img, axis=0)

        print(img)

        img = self.sess.run(
                self.output_photo,
                feed_dict={
                    self.input_photo: normalize_arr_of_imgs(img),
        })

        print(img)

        img = img[0]
        img = denormalize_arr_of_imgs(img)
        if resize_to_original:
             img = scipy.misc.imresize(img, size=img_shape)
        else:
            pass
        img_name = os.path.basename(img_path)
        print(img_name)
        scipy.misc.imsave(os.path.join(OUTPUT_IMAGES_DIR, img_name[:-4] + "_stylized.jpg"), img)

    print("Inference is finished.")

    def save(self, step, is_long=False):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if is_long:
            self.saver_long.save(self.sess,
                                 os.path.join(self.checkpoint_long_dir, self.model_name+'_%d.ckpt' % step),
                                 global_step=step)
        else:
            self.saver.save(self.sess,
                            os.path.join(self.checkpoint_dir, self.model_name + '_%d.ckpt' % step),
                            global_step=step)

    def load(self, checkpoint_dir, ckpt_nmbr=None):
        if ckpt_nmbr:
            if len([x for x in os.listdir(checkpoint_dir) if ("ckpt-" + str(ckpt_nmbr)) in x]) > 0:
                print(" [*] Reading checkpoint %d from folder %s." % (ckpt_nmbr, checkpoint_dir))
                ckpt_name = [x for x in os.listdir(checkpoint_dir) if ("ckpt-" + str(ckpt_nmbr)) in x][0]
                ckpt_name = '.'.join(ckpt_name.split('.')[:-1])
                self.initial_step = ckpt_nmbr
                print("Load checkpoint %s. Initial step: %s." % (ckpt_name, self.initial_step))
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                return True
            else:
                return False
        else:
            print(" [*] Reading latest checkpoint from folder %s." % (checkpoint_dir))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.initial_step = int(ckpt_name.split("_")[-1].split(".")[0])
                print("Load checkpoint %s. Initial step: %s." % (ckpt_name, self.initial_step))
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                return True
            else:
                return False
