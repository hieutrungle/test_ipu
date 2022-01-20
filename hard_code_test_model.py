import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import sys
from utils import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import configargparse
from typing import *
from layers import *
from layers_iaf import *
import distribution
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'swish': tf.keras.layers.Activation(tf.keras.activations.swish)})


CHANNEL_MULT = 2

class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, args, model_arch, in_shape,
                name="Variational_AutoEncoder", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.model_name = name
        self.model_arch = model_arch
        self.dataset = args.dataset
        self.crop_output = self.dataset in {'mnist', 'omniglot', 'stacked_mnist'}
        self.use_se = args.use_se

        self.num_scales = args.num_scales
        self.num_groups_per_scale = args.num_groups_per_scale
        self.num_channels_of_latent = args.num_channels_of_latent

        # Adjust number of groups per scale in the top-down fashion
        self.groups_per_scale = utils.groups_per_scale(self.num_scales, 
                                    self.num_groups_per_scale, 
                                    args.is_adaptive,
                                    minimum_groups=args.min_groups_per_scale)

        self.vanilla_vae = self.num_scales == 1 and self.num_groups_per_scale == 1

        # Pre-process and post-process parameter
        self.num_initial_channel = args.num_initial_channel

        # encoder parameteres
        self.num_process_blocks = args.num_process_blocks  # block is defined as series of Normal followed by Down
        self.num_preprocess_cells = args.num_preprocess_cells   # number of cells per block
        self.num_cell_per_group_enc = args.num_cell_per_group_enc  # number of cell for each group encoder

        # decoder parameters
        self.num_postprocess_cells = args.num_postprocess_cells
        self.num_cell_per_group_dec = args.num_cell_per_group_dec  # number of cell for each group decoder

        # general cell parameters
        self.in_shape = in_shape
        self.input_channel = self.in_shape[0]

        # used for generative block
        channel_scaling = CHANNEL_MULT ** (self.num_process_blocks + self.num_scales - 1)
        final_channel = self.input_channel // channel_scaling
        self.prior_shape = (final_channel, final_channel, int(channel_scaling * self.num_initial_channel))

        self.sampler_qs, self.sampler_ps, self.log_qs, self.log_ps = [], [], [], []

        self.pre_prior = tf.Variable(tf.random.normal(shape=self.prior_shape), trainable=True)
        self.pre_prior_layer = PrePriorLayer(self.prior_shape)
        self.prior = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.prior_shape),
            ConvWNElu(self.prior_shape[-1], kernel_size=1, padding="same", name="prior_0"),
            ConvWN(self.prior_shape[-1], kernel_size=1, padding="same", name="prior_1")
        ], name="prior")

        self.init_stem = ConvWN(self.num_initial_channel, kernel_size=1, strides=1, name="init_stem")
        
        cell_type='normal_pre'
        cell_archs = self.model_arch[cell_type]
        self.normal_pre_0_0 = Cell(self.num_initial_channel, cell_type, cell_archs, use_se=self.use_se, name='normal_pre_0_0')

        cell_type='down_sampling_pre'
        cell_archs = self.model_arch[cell_type]
        self.down_sampling_pre_0_1 = Cell(self.num_initial_channel*2, cell_type, cell_archs, use_se=self.use_se, name="down_sampling_pre_0_1")

        cell_type = 'normal_enc'
        cell_archs = self.model_arch[cell_type]
        self.normal_enc_0_0_0 = Cell(self.num_initial_channel*2, cell_type, cell_archs, use_se=self.use_se, name="normal_enc_0_0_0")

        cell_type = 'down_sampling_enc'
        cell_archs = self.model_arch[cell_type]
        self.down_sampling_enc_0 = Cell(self.num_initial_channel*4, cell_type, cell_archs, use_se=self.use_se, name="down_sampling_enc_0")

        cell_type = 'normal_enc'
        cell_archs = self.model_arch[cell_type]
        self.normal_enc_1_0_0 = Cell(self.num_initial_channel*4, cell_type, cell_archs, use_se=self.use_se, name="normal_enc_1_0_0")

        self.encoder0 = ConvWNElu(self.num_initial_channel*4, name="encoder0")

        self.enc_mu_log_sig_0 = ConvWN(2*self.num_channels_of_latent, kernel_size=3, padding="same", name="enc_mu_log_sig_0")

        self.sampler_q_0 = distribution.NormalSampler(name="sampler_q_0")

        cell_type = 'ar_nn'
        cell_archs = self.model_arch[cell_type]
        self.nf_0 = AutoregresiveCell(self.num_channels_of_latent, cell_type=cell_type, cell_archs=cell_archs, name="NF_0")

        self.sampler_p_0 = distribution.NormalSampler(name="sampler_p_0")

        cell_type = 'combiner_dec'
        self.combiner_dec_0_0 = DecCombinerCell(self.num_initial_channel*4, cell_type=cell_type, name="combiner_dec_0_0")

        cell_type = 'up_sampling_dec'
        cell_archs = self.model_arch[cell_type]
        self.up_sampling_dec_0 = Cell(self.num_initial_channel*2, cell_type=cell_type, cell_archs=cell_archs, use_se=self.use_se, name="up_sampling_dec_0")

        cell_type = 'normal_dec'
        cell_archs = self.model_arch[cell_type]
        self.normal_dec_1_0_0 = Cell(self.num_initial_channel*2, cell_type=cell_type, cell_archs=cell_archs, use_se=self.use_se, name="normal_dec_1_0_0")

        cell_type = 'combiner_enc'
        self.combiner_enc_0_0 = EncCombinerCell(self.num_initial_channel*2, cell_type=cell_type, name="combiner_enc_0_0")

        self.dec_mu_log_sig_0 = ConvWN(2*self.num_channels_of_latent, kernel_size=3, padding="same", name="dec_mu_log_sig_0")
        self.enc_mu_log_sig_1 = ConvWN(2*self.num_channels_of_latent, kernel_size=3, padding="same", name="enc_mu_log_sig_1")

        self.sampler_q_1 = distribution.NormalSampler(name="sampler_q_1")
        self.sampler_p_1 = distribution.NormalSampler(name="sampler_p_1")

        cell_type = 'ar_nn'
        cell_archs = self.model_arch[cell_type]
        self.nf_1 = AutoregresiveCell(self.num_channels_of_latent, cell_type=cell_type, cell_archs=cell_archs, name="NF_1")

        cell_type = 'combiner_dec'
        self.combiner_dec_1_0 = DecCombinerCell(self.num_initial_channel*2, cell_type=cell_type, name="combiner_dec_1_0")

        cell_type = 'up_sampling_post'
        cell_archs = self.model_arch[cell_type]
        self.up_sampling_post_0_0 = Cell(self.num_initial_channel, cell_type=cell_type, cell_archs=cell_archs, use_se=self.use_se, name="up_sampling_post_0_0")

        cell_type = 'normal_post'
        cell_archs = self.model_arch[cell_type]
        self.normal_post_0_1 = Cell(self.num_initial_channel, cell_type=cell_type, cell_archs=cell_archs, use_se=self.use_se, name="normal_post_0_0")

        self.decoder_output = ConvWN(channel=1, name="decoder_output")

        # self.sampler_qs = [self.sampler_q_0, self.sampler_q_1]
        # self.sampler_ps = [self.sampler_p_0, self.sampler_p_1]
        self.sampler_qs = []
        self.sampler_ps = []

    def call(self, x):
        x = self.init_stem(x)
        x = self.normal_pre_0_0(x)
        x = self.down_sampling_pre_0_1(x)
        x = self.normal_enc_0_0_0(x)
        enc_combiner_0 = x
        x = self.down_sampling_enc_0(x)
        x = self.normal_enc_1_0_0(x)

        z_samples = []
        ftrs = []
        ftr = self.encoder0(x)

        mu_and_log_sigma_q = self.enc_mu_log_sig_0(ftr)
        mu_q, log_sigma_q = tf.split(mu_and_log_sigma_q, num_or_size_splits=2, axis=3) # B x H x W x C
        
        sampler_q_0 = distribution.Normal(mu_q, log_sigma_q)
        z = sampler_q_0.sample()
        
        # z, _ = self.sampler_q_0(mu_q, log_sigma_q)
        z, log_det = self.nf_0(z, ftr)

        # _, _ = self.sampler_p_0(mu=tf.zeros(shape=tf.shape(z)), log_sigma=tf.zeros(shape=tf.shape(z)))
        sampler_p_0 = distribution.Normal(mu_q, log_sigma_q)
        

        x = 0
        x = self.pre_prior_layer(self.pre_prior, z)
        x = self.prior(x)

        x = self.combiner_dec_0_0(x, z)
        x = self.up_sampling_dec_0(x)
        x = self.normal_dec_1_0_0(x)

        ftr = self.combiner_enc_0_0(enc_combiner_0, x)
        
        mu_and_log_sigma_q = self.enc_mu_log_sig_1(ftr)
        mu_q, log_sigma_q = tf.split(mu_and_log_sigma_q, num_or_size_splits=2, axis=3)

        mu_and_log_sigma_p = self.dec_mu_log_sig_0(x)
        mu_p, log_sigma_p = tf.split(mu_and_log_sigma_p, num_or_size_splits=2, axis=3)

        sampler_q_1 = distribution.Normal(mu=tf.math.multiply(tf.add(mu_p, mu_q), 0.5), 
                                log_sigma=tf.math.multiply(tf.add(log_sigma_p, log_sigma_q), 0.5))
        z = sampler_q_1.sample()

        # z, _ = self.sampler_q_1(mu=tf.math.multiply(tf.add(mu_p, mu_q), 0.5), 
        #                         log_sigma=tf.math.multiply(tf.add(log_sigma_p, log_sigma_q), 0.5))
        z, log_det = self.nf_1(z, ftr)

        # _, _ = self.sampler_p_1(mu_p, log_sigma_p)
        sampler_p_1 = distribution.Normal(mu_q, log_sigma_q)

        x = self.combiner_dec_1_0(x, z)
        x = self.up_sampling_post_0_0(x)
        x = self.normal_post_0_1(x)
        x = self.decoder_output(x)

        self.sampler_qs = [sampler_q_0, sampler_q_1]
        self.sampler_ps = [sampler_p_0, sampler_p_1]

        return x

    def cal_kl_components(self):
        kl_all, kl_diag = [], []
        total_log_p, total_log_q = 0.0, 0.0
        for sampler_q, sampler_p in zip(self.sampler_qs, self.sampler_ps):
            kl_per_var = sampler_q.kl(sampler_p)
            kl_diag.append(tf.reduce_mean(tf.reduce_sum(kl_per_var, axis=[1, 2]), axis=0))
            kl_all.append(tf.reduce_sum(kl_per_var, axis=[1, 2, 3]))
        return total_log_q, total_log_p, kl_all, kl_diag

    def model(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.model_name)