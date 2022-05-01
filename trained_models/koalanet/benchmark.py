from __future__ import print_function
from datetime import datetime
import time
from tensorflow.data.experimental import shuffle_and_repeat, unbatch

from utils import *
from ops import *

import numpy as np
import glob

from skimage.metrics import structural_similarity as ssim
import cv2

from tqdm import tqdm

class KOALAnet:
  def __init__(self, args):
    self.phase = args.phase
    self.factor = args.factor

    """ Training Settings """
    self.training_stage = args.training_stage
    self.tensorboard = args.tensorboard

    """ Testing Settings """
    self.eval = args.eval
    self.test_data_path = args.test_data_path
    self.test_label_path = args.test_label_path
    self.test_ckpt_path = args.test_ckpt_path
    self.test_patch = args.test_patch

    """ Model Settings """ 
    self.channels = args.channels
    self.bicubic_size = args.bicubic_size
    self.gaussian_size = args.gaussian_size
    self.down_kernel = args.down_kernel
    self.up_kernel = args.up_kernel
    self.anti_aliasing = args.anti_aliasing

    """ Hyperparameters """
    self.max_epoch = args.max_epoch
    self.batch_size = args.batch_size
    self.val_batch_size = args.val_batch_size
    self.patch_size = args.patch_size
    self.val_patch_size = args.val_patch_size
    self.lr = args.lr
    self.lr_type = args.lr_type
    self.lr_stair_decay_points = args.lr_stair_decay_points
    self.lr_stair_decay_factor = args.lr_stair_decay_factor
    self.lr_linear_decay_point = args.lr_linear_decay_point
    self.n_display = args.n_display

    if self.training_stage == 1:
      self.model_name = 'downsampling_network'
    elif self.training_stage == 2:
      self.model_name = 'upsampling_network_baseline'
    elif self.training_stage == 3:
      self.model_name = 'upsampling_network'

    """ Directories """
    self.ckpt_dir = os.path.join('ckpt', self.model_dir)
    self.result_dir = os.path.join('results')
    check_folder(self.ckpt_dir)
    check_folder(self.result_dir)

    """ Model Init """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)

    """ Print Model """
    print('Model arguments, [{:s}]'.format((str(datetime.now())[:-7])))
    for arg in vars(args):
      print('# {} : {}'.format(arg, getattr(args, arg)))
    print("\n")

  def upsampling_network_baseline(self, input_LR, factor, kernel, channels=3, reuse=False, scope='SISR_DUF'):
    with tf.variable_scope(scope, reuse=reuse):
      ch = 64
      n_res = 12
      net = conv2d(input_LR, ch, 3)
      for res in range(n_res):
        net = res_block(net, ch, 3, scope='Residual_block0_' + str(res + 1))
      net = tf.nn.relu(net)
      # upsampling kernel branch
      k2d = tf.nn.relu(conv2d(net, ch * 2, 3))
      k2d = conv2d(k2d, kernel * kernel * factor * factor, 3)
      # rgb residual image branch
      rgb = tf.nn.relu(conv2d(net, ch * 2, 3))
      rgb = tf.depth_to_space(rgb, 2)
      if factor == 4:
        rgb = tf.nn.relu(conv2d(rgb, ch, 3))
        rgb = tf.depth_to_space(rgb, 2)
      rgb = conv2d(rgb, channels, 3)
      # local filtering and upsampling
      output_k2d = local_conv_us(input_LR, k2d, factor, channels, kernel)
      output = output_k2d + rgb
    return output

  def upsampling_network(self, input_LR, k2d_ds, factor, kernel, channels=3, reuse=False, scope='SISR_DUF'):
    with tf.variable_scope(scope, reuse=reuse):
      ch = 64
      n_res = 12
      skip_idx = np.arange(0, 5, 1)
      # extract degradation kernel features
      k = cr_block(k2d_ds, 3, ch, 3, 'kernel_condition')
      net = conv2d(input_LR, ch, 3)
      filter_p_list = []
      for res in range(n_res):
        if res in skip_idx:
          net, filter_p = koala(net, k, ch, ch, conv_k_sz=3, lc_k_sz=7, scope_res='Residual_block0_' + str(res + 1), scope='KOALA_module/%d' % (res+1))
          filter_p_list.append(filter_p)
        else:
          net = res_block(net, ch, 3, scope='Residual_block0_' + str(res + 1))
      net = tf.nn.relu(net)
      # upsampling kernel branch
      k2d = tf.nn.relu(conv2d(net, ch * 2, 3))
      k2d = conv2d(k2d, kernel * kernel * factor * factor, 3)
      # rgb residual image branch
      rgb = tf.nn.relu(conv2d(net, ch * 2, 3))
      rgb = tf.depth_to_space(rgb, 2)
      if factor == 4:
        rgb = tf.nn.relu(conv2d(rgb, ch, 3))
        rgb = tf.depth_to_space(rgb, 2)
      rgb = conv2d(rgb, channels, 3)
      # local filtering and upsampling
      output_k2d = local_conv_us(input_LR, k2d, factor, channels, kernel)
      output = output_k2d + rgb
    return output, filter_p_list[-1]

  def downsampling_network(self, input_LR, kernel, reuse=False, scope='SISR'):
    with tf.variable_scope(scope, reuse=reuse):
      ch = 64
      skip = dict()
      # encoder
      n, skip[0] = enc_level_res(input_LR, ch, scope='enc_block_res/0')
      n, skip[1] = enc_level_res(n, ch*2, scope='enc_block_res/1')
      # bottleneck
      n = bottleneck_res(n, ch*4)
      # decoder
      n = dec_level_res(n, skip[1], ch*2, scope='dec_block_res/0')
      n = dec_level_res(n, skip[0], ch, scope='dec_block_res/1')
      # downsampling kernel branch
      n = tf.nn.relu(conv2d(n, ch, 3))
      k2d = conv2d(n, kernel * kernel, 3)
    return k2d

  def build_model(self, args):
    data = SISRData(args)
    if self.phase == 'train':
      """ Directories """
      self.log_dir = os.path.join('logs', self.model_dir)
      self.img_dir = os.path.join(self.result_dir, 'imgs_train', self.model_dir)
      check_folder(self.log_dir)
      check_folder(self.img_dir)

      self.updates_per_epoch = int(data.num_train / self.batch_size)
      print("Update per epoch : ", self.updates_per_epoch)

      """ Training Data Generation """
      train_folder_path = tf.data.Dataset.from_tensor_slices(data.list_train).apply(shuffle_and_repeat(len(data.list_train)))
      train_data = train_folder_path.map(data.image_processing, num_parallel_calls=4)
      train_data = train_data.apply(unbatch()).shuffle(data.Qsize*50).batch(data.batch_size).prefetch(1)
      train_data_iterator = train_data.make_one_shot_iterator()

      # self.train_hr : [B, H, W, C], self.gaussian_kernel : [B, gaussian_size, gaussian_size, 1], data.bicubic_kernel : [1, bicubic_size, bicubic_size, B]
      self.train_hr, self.gaussian_kernel = train_data_iterator.get_next()
      self.ds_kernel = get_ds_kernel(data.bicubic_kernel, self.gaussian_kernel)
      self.train_lr = get_ds_input(self.train_hr, self.ds_kernel, self.channels, self.batch_size, data.pad_left, data.pad_right, self.factor)
      self.train_lr = tf.math.round((self.train_lr+1.0)/2.0*255.0)
      self.train_lr = tf.cast(self.train_lr, tf.float32)/255.0 * 2.0 - 1.0
      print("#### Degraded train_lr is quantized.")

      # set placeholders for validation
      self.val_hr = tf.placeholder(tf.float32, (self.val_batch_size, self.val_patch_size * self.factor, self.val_patch_size * self.factor, self.channels))
      self.val_base_k = tf.placeholder(tf.float32, (1, self.bicubic_size, self.bicubic_size, self.val_batch_size))
      self.val_rand_k = tf.placeholder(tf.float32, (self.val_batch_size, self.gaussian_size, self.gaussian_size, 1))
      self.ds_kernel_val = get_ds_kernel(self.val_base_k, self.val_rand_k)
      self.val_lr = get_ds_input(self.val_hr, self.ds_kernel_val, self.channels, self.val_batch_size, data.pad_left, data.pad_right, self.factor)
      self.val_lr = tf.math.round((self.val_lr+1.0)/2.0*255.0)
      self.val_lr = tf.cast(self.val_lr, tf.float32)/255.0 * 2.0 - 1.0
      print("#### Degraded val_lr is quantized.")
      self.list_val = data.list_val
      print("Training patch size : ", self.train_lr.get_shape())
      
      """ Define Model """
      if self.training_stage == 1:
        self.k2d_ds = self.downsampling_network(self.train_lr, self.down_kernel, reuse=False, scope='SISR_DDF')
        self.k2d_ds_val = self.downsampling_network(self.val_lr, self.down_kernel, reuse=True, scope='SISR_DDF')
        # reconstructed LR images
        self.output_ds_hr = local_conv_ds(self.train_hr, self.k2d_ds, self.factor, self.channels, self.down_kernel)
        self.output_ds_hr_val = local_conv_ds(self.val_hr, self.k2d_ds_val, self.factor, self.channels, self.down_kernel)
      elif self.training_stage == 2:
        # reconstructed HR images
        self.output = self.upsampling_network_baseline(self.train_lr, self.factor, self.up_kernel, self.channels, reuse=False, scope='SISR_DUF')
        self.output_val = self.upsampling_network_baseline(self.val_lr, self.factor, self.up_kernel, self.channels, reuse=True, scope='SISR_DUF')
      elif self.training_stage == 3:
        self.k2d_ds = self.downsampling_network(self.train_lr, self.down_kernel, reuse=False, scope='SISR_DDF')
        self.k2d_ds_val = self.downsampling_network(self.val_lr, self.down_kernel, reuse=True, scope='SISR_DDF')
        # reconstructed LR images
        self.output_ds_hr = local_conv_ds(self.train_hr, self.k2d_ds, self.factor, self.channels, self.down_kernel)
        self.output_ds_hr_val = local_conv_ds(self.val_hr, self.k2d_ds_val, self.factor, self.channels, self.down_kernel)
        # reconstructed HR images
        self.output, self.filter_p = self.upsampling_network(self.train_lr, self.k2d_ds, self.factor, self.up_kernel, self.channels, reuse=False, scope='SISR_DUF')
        self.output_val, _ = self.upsampling_network(self.val_lr, self.k2d_ds_val, self.factor, self.up_kernel, self.channels, reuse=True, scope='SISR_DUF')

      """ Define Losses """
      if self.training_stage == 1:
        # training
        self.rec_loss_ds_hr = l1_loss(self.train_lr, self.output_ds_hr)
        self.k2d_ds = kernel_normalize(self.k2d_ds, self.down_kernel)
        k2d_mean = tf.reduce_mean(self.k2d_ds, axis=[1, 2], keepdims=True)
        self.kernel_loss = l1_loss(k2d_mean, get_1d_kernel(self.ds_kernel, self.batch_size))
        self.total_loss = self.rec_loss_ds_hr + self.kernel_loss
        # validation
        self.val_rec_loss_ds_hr = l1_loss(self.val_lr, self.output_ds_hr_val)
        self.k2d_ds_val = kernel_normalize(self.k2d_ds_val, self.down_kernel)
        k2d_mean_val = tf.reduce_mean(self.k2d_ds_val, axis=[1, 2], keepdims=True)
        self.val_kernel_loss = l1_loss(k2d_mean_val, get_1d_kernel(self.ds_kernel_val, self.val_batch_size))
        self.val_total_loss = self.val_rec_loss_ds_hr + self.val_kernel_loss
        self.val_PSNR = tf.reduce_mean(tf.image.psnr((self.val_lr + 1) / 2, (self.output_ds_hr_val + 1) / 2, max_val=1.0))

      elif self.training_stage == 2:
        # training
        self.rec_loss = l1_loss(self.train_hr, self.output)
        self.total_loss = self.rec_loss
        # validation
        self.val_rec_loss = l1_loss(self.val_hr, self.output_val)
        self.val_total_loss = self.val_rec_loss
        self.val_PSNR = tf.reduce_mean(tf.image.psnr((self.val_hr + 1) / 2, (self.output_val + 1) / 2, max_val=1.0))

      elif self.training_stage == 3:
        # training
        self.rec_loss = l1_loss(self.train_hr, self.output)
        self.rec_loss_ds_hr = l1_loss(self.train_lr, self.output_ds_hr)
        self.k2d_ds = kernel_normalize(self.k2d_ds, self.down_kernel)
        k2d_mean = tf.reduce_mean(self.k2d_ds, axis=[1, 2], keepdims=True)
        self.kernel_loss = l1_loss(k2d_mean, get_1d_kernel(self.ds_kernel, self.batch_size))
        self.total_loss = self.rec_loss + self.rec_loss_ds_hr + self.kernel_loss
        # validation
        self.val_rec_loss = l1_loss(self.val_hr, self.output_val)
        self.val_rec_loss_ds_hr = l1_loss(self.val_lr, self.output_ds_hr_val)
        self.k2d_ds_val = kernel_normalize(self.k2d_ds_val, self.down_kernel)
        k2d_mean_val = tf.reduce_mean(self.k2d_ds_val, axis=[1, 2], keepdims=True)
        self.val_kernel_loss = l1_loss(k2d_mean_val, get_1d_kernel(self.ds_kernel_val, self.val_batch_size))
        self.val_total_loss = self.val_rec_loss + self.val_rec_loss_ds_hr + self.val_kernel_loss
        self.val_PSNR = tf.reduce_mean(tf.image.psnr((self.val_hr + 1) / 2, (self.output_val + 1) / 2, max_val=1.0))

      """ Visualization """
      # visualization of GT degradation kernel
      self.ds_kernel_vis = tf.transpose(self.ds_kernel, (3, 1, 2, 0))  # [B, bicubic_size, bicubic_size, 1]
      kernel_min = tf.reduce_min(self.ds_kernel_vis, axis=(1, 2), keepdims=True)
      kernel_max = tf.reduce_max(self.ds_kernel_vis, axis=(1, 2), keepdims=True)
      self.scale_vis = (self.patch_size*self.factor)//self.bicubic_size
      self.ds_kernel_vis = local_conv_vis_ds(self.ds_kernel_vis, kernel_min, kernel_max, 3, self.scale_vis)

      # visualization of estimated degradation kernel
      if self.training_stage in [1, 3]:
        self.k2d_ds_vis = tf.reshape(k2d_mean, [self.batch_size, self.down_kernel, self.down_kernel, 1])  # [B, down_kernel, down_kernel, 1]
        self.k2d_ds_vis = local_conv_vis_ds(self.k2d_ds_vis, kernel_min, kernel_max, 3, self.scale_vis)

      # visualization of local filters in KOALA modules
      if self.training_stage == 3:
        self.filter_p = tf.reduce_mean(self.filter_p, axis=(1, 2))
        self.filter_p = tf.reshape(self.filter_p, [self.batch_size, 7, 7, 1])
        self.filter_p = local_conv_vis_ds(self.filter_p, None, None, 6, 10)

      """ Learning Rate Schedule """
      global_step = tf.Variable(initial_value=0, trainable=False)
      if self.lr_type == "stair_decay":
        self.lr_decay_boundary = [y * (self.updates_per_epoch) for y in self.lr_stair_decay_points]
        self.lr_decay_value = [self.lr * (self.lr_stair_decay_factor ** y) for y in range(len(self.lr_stair_decay_points) + 1)]
        self.reduced_lr = tf.train.piecewise_constant(global_step, self.lr_decay_boundary, self.lr_decay_value)
        print("lr_type: stair_decay")
      elif self.lr_type == "linear_decay":
        self.reduced_lr = tf.placeholder(tf.float32, name='learning_rate')
        print("lr_type: linear_decay")
      else:  # no decay
        self.reduced_lr = tf.convert_to_tensor(self.lr)
        print("lr_type: no decay")

      """ Optimizer """
      srnet_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="SISR")
      # print("\nTrainable Parameters:")
      # for param in srnet_params:
        # print(param.name)
      self.optimizer = tf.train.AdamOptimizer(self.reduced_lr).minimize(self.total_loss, global_step=global_step, var_list=srnet_params)

      """" TensorBoard Summary """
      if self.tensorboard:
        # loss summary
        total_loss_sum = tf.summary.scalar("val_total_loss", self.val_total_loss)
        train_PSNR_sum = tf.summary.scalar("val_PSNR", self.val_PSNR)
        self.total_summary_loss = tf.summary.merge([total_loss_sum, train_PSNR_sum])
        # image summary
        lr_sum = tf.summary.image("LR", self.val_lr, max_outputs=self.val_batch_size)
        hr_sum = tf.summary.image("HR", self.val_hr, max_outputs=self.val_batch_size)
        # kernel summary
        self.ds_kernel_val_vis = tf.transpose(self.ds_kernel_val, [3, 1, 2, 0])  # [B, bicubic_size, bicubic_size, 1]
        self.ds_kernel_val_vis = local_conv_vis_ds(self.ds_kernel_val_vis, None, None, 3, self.scale_vis)
        ds_kernel_sum = tf.summary.image("Degradation Kernel (GT)", self.ds_kernel_val_vis, max_outputs=self.val_batch_size)
        self.total_summary_img = tf.summary.merge([ds_kernel_sum, lr_sum, hr_sum])
        # result summary
        if self.training_stage in [1, 3]:
          self.k2d_ds_val_vis = tf.reshape(k2d_mean_val, [self.val_batch_size, self.down_kernel, self.down_kernel, 1])
          self.k2d_ds_val_vis = local_conv_vis_ds(self.k2d_ds_val_vis, None, None, 3, self.scale_vis)
          k2d_ds_sum = tf.summary.image("Degradation Kernel (Predicted)", self.k2d_ds_val_vis, max_outputs=self.val_batch_size)
          output_sum_ds_hr = tf.summary.image("LR (Predicted)", self.output_ds_hr_val, max_outputs=self.val_batch_size)
          self.total_summary_img = tf.summary.merge([self.total_summary_img, k2d_ds_sum, output_sum_ds_hr])
        if self.training_stage in [2, 3]:
          output_sum = tf.summary.image("SR (Predicted)", self.output_val, max_outputs=self.val_batch_size)
          self.total_summary_img = tf.summary.merge([self.total_summary_img, output_sum])
    
    elif self.phase == 'test':
      assert self.training_stage == 3, "training_stage should be 3"

      """ Directories """
      self.test_img_dir = os.path.join(self.result_dir, 'imgs_test', self.model_dir)
      check_folder(self.test_img_dir)

      """ Set Data Paths """
      self.list_test_lr = data.list_test_lr  # test_data_path (LR)
      if self.eval:
        self.list_test_hr = data.list_test_hr  # test_label_path (HR)

      """ Set Placeholders """
      self.test_lr = tf.placeholder(tf.float32, (1, None, None, self.channels))
      self.test_hr = tf.placeholder(tf.float32, (1, None, None, self.channels))

      """ Define Model """
      self.k2d_ds_test = self.downsampling_network(self.test_lr, self.down_kernel, reuse=False, scope='SISR_DDF')
      self.output_test, _ = self.upsampling_network(self.test_lr, self.k2d_ds_test, self.factor, self.up_kernel, self.channels, reuse=False, scope='SISR_DUF')

    self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

  def test(self):
    assert self.training_stage == 3, "training_stage should be 3"
    # saver to save model
    self.saver = tf.train.Saver()
    # restore checkpoint
    ckpt = tf.train.get_checkpoint_state(os.path.join(self.test_ckpt_path, "new_trained"))
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      print("!!!!!!!!!!!!!! Restored from {}".format(ckpt.model_checkpoint_path))

    datasets = glob.glob("/content/drive/MyDrive/koalanet/test/*")

    # Produce benchmark
    for dataset in datasets:
      dataset = dataset.replace('\\', '/')
      dataset_name = dataset.split('/')[-1]
      lr_paths = glob.glob(dataset + "/LR/X4/imgs/*")
      hr_paths = glob.glob(dataset + "/HR/*")
      lr_paths.sort()
      hr_paths.sort()
      eval_psnr = 0
      eval_ssim = 0
      """" Test """
      avg_inf_time = 0.0
      avg_test_PSNR = 0.0
      patch_boundary = 0
      for lr_path, hr_path in tqdm(zip(lr_paths, hr_paths)):
        test_lr = read_img_trim(lr_path, factor=4*self.test_patch[0])
        test_lr = check_gray(test_lr)
        test_hr = read_img_trim(hr_path, factor=self.factor*4*self.test_patch[0])
        test_hr = check_gray(test_hr)
        _, h, w, c = test_lr.shape
        output_test = np.zeros((1, h*self.factor, w*self.factor, c))
        inf_time = 0.0
        # test image divided into test_patch[0]*test_patch[1] to fit memory (default: 1x1)
        for p in range(self.test_patch[0] * self.test_patch[1]):
          pH = p // self.test_patch[1]
          pW = p % self.test_patch[1]
          sH = h // self.test_patch[0]
          sW = w // self.test_patch[1]
          # process data considering patch boundary
          H_low_ind, H_high_ind, W_low_ind, W_high_ind = get_HW_boundary(patch_boundary, h, w, pH, sH, pW, sW)
          test_lr_p = test_lr[:, H_low_ind: H_high_ind, W_low_ind: W_high_ind, :]
          st = time.time()
          output_test_p = self.sess.run([self.output_test], feed_dict={self.test_lr: test_lr_p})
          inf_time_p = time.time() - st
          inf_time += inf_time_p
          output_test_p = trim_patch_boundary(output_test_p, patch_boundary, h, w, pH, sH, pW, sW, self.factor)
          output_test[:, pH * sH * self.factor: (pH + 1) * sH * self.factor, pW * sW * self.factor: (pW + 1) * sW * self.factor, :] = output_test_p
        avg_inf_time += inf_time
        # compute PSNR and print results
        test_PSNR = compute_y_psnr(output_test, test_hr)
        avg_test_PSNR += test_PSNR
        print(" <Test> time: %4.4f(seconds), test_PSNR: %2.2f[dB]  "
            % (inf_time, test_PSNR))
        # save predicted SR images
        # save_path = os.path.join(self.test_img_dir, os.path.basename(self.list_test_lr[test_cnt]))
        # save_img(output_test, save_path)
       
        # Metrics
        sr = np.squeeze(output_test)
        sr = np.clip((sr + 1.) / 2. * 255., 0, 255).astype(np.float32)
        hr = np.squeeze(test_hr)
        hr = np.clip((hr + 1.) / 2. * 255., 0, 255).astype(np.float32)

        eval_psnr += cv2.PSNR(hr,sr)
        eval_ssim += ssim(hr, sr, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)


      avg_test_PSNR /= float(len(lr_paths))
      print("######### Average Test PSNR: %.8f[dB]  #########" % avg_test_PSNR)
      avg_inf_time /= float(len(lr_paths))
      print("######### Average Inference Time: %.8f[s]  #########" % avg_inf_time)
      print(f'{dataset_name}: PSNR: {eval_psnr / len(lr_paths)}, SSIM: {eval_ssim / len(lr_paths)}')

  @property
  def model_dir(self):
    return "{}_x{}".format(self.model_name, self.factor)

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="SISR")

	parser.add_argument('--phase', type=str, default='test', choices=['train', 'test'])
	parser.add_argument('--factor', type=int, default=4, help='scale factor')

	""" Training Settings """
	parser.add_argument('--training_stage', type=int, default=3, choices=[1, 2, 3], help='Set stage for the 3-stage training strategy.')
	parser.add_argument('--tensorboard', type=bool, default=True, help='If set to True, tensorboard summaries are created')
	parser.add_argument('--training_data_path', type=str, default='./dataset/DIV2K/train/DIV2K_train_HR', help='training_dataset path')
	parser.add_argument('--validation_data_path', type=str, default='./dataset/DIV2K/val/DIV2K_valid_HR', help='validation_dataset path')

	""" Testing Settings """
	parser.add_argument('--eval', type=bool, default=True, help='If set to True, evaluation is performed with HR images during the testing phase')
	parser.add_argument('--test_data_path', type=str, default='./testset/Set5/LR/X4/imgs', help='test dataset path')
	parser.add_argument('--test_label_path', type=str, default='./testset/Set5/HR', help='test dataset label path for eval')
	parser.add_argument('--test_ckpt_path', type=str, default='./pretrained_ckpt', help='checkpoint path with trained weights')
	parser.add_argument('--test_patch', type=int, nargs='+', default=[2, 2], help='input image can be divide into an nxn grid of smaller patches in the test phase to fit memory')

	""" Model Settings """ 
	parser.add_argument('--channels', type=int, default=3, help='img channels')
	parser.add_argument('--bicubic_size', type=int, default=20, help='size of bicubic kernel - should be an even number; we recommend at least 4*factor; only 4 centered values are meaningful and other (bicubic_size-4) values are all zeros.')
	parser.add_argument('--gaussian_size', type=int, default=15, help='size of anisotropic gaussian kernel - should be an odd number')
	parser.add_argument('--down_kernel', type=int, default=20, help='downsampling kernel size in the downsampling network')
	parser.add_argument('--up_kernel', type=int, default=5, help='upsampling kernel size in the upsampling network')
	parser.add_argument('--anti_aliasing', type=bool, default=False, help='Matlab anti-aliasing')

	""" Hyperparameters """
	parser.add_argument('--max_epoch', type=int, default=2000, help='number of total epochs')
	parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
	parser.add_argument('--val_batch_size', type=int, default=4, help='batch size for validation')
	parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
	parser.add_argument('--val_patch_size', type=int, default=100, help='validation patch size')
	parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
	parser.add_argument('--lr_type', type=str, default='stair_decay', choices=['stair_decay', 'linear_decay', 'no_decay'])
	parser.add_argument('--lr_stair_decay_points', type=int, nargs='+', help='stair_decay - Epochs where lr is decayed', default=[1600, 1800])
	parser.add_argument('--lr_stair_decay_factor', type=float, default=0.1, help='stair_decay - lr decreasing factor')
	parser.add_argument('--lr_linear_decay_point', type=int, default=100, help='linear decay - Epoch to start lr decay')
	parser.add_argument('--Qsize', type=int, default=50, help='number of random crop patches from a image')
	parser.add_argument('--n_display', type=int, default=4, help='number images to display - Should be less than or equal to batch_size')
	return parser.parse_args()


def main():
  args = parse_args()
  # set model class
  model = KOALAnet(args)
  # build model
  model.build_model(args)

  print("Testing phase starts!!!")
  model.test()

if __name__ == "__main__":
  main()
