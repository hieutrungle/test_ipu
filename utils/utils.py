import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import math
from typing import *

"""Data Preprocessing Helper Functions"""

def get_data_info(data:np.ndarray):
    print(f"data shape: {data.shape}")
    print(f"maximum value: {data.max()}")
    print(f"minimum value: {data.min()}\n")
    
def draw_hist(data:np.ndarray, title:str=""):
    plt.hist(data, bins=40)
    plt.title(title +"\n")
    plt.ylabel('Count')
    plt.xlabel('Value')
    plt.show()

# Get all file names in directory
def get_filename(parent_dir:str, file_extension:str)->str:
    filenames = []
    for root, dirs, files in os.walk(parent_dir):
        for filename in files:
            if (file_extension in filename):
                filenames.append(os.path.join(parent_dir, filename))
    return filenames

# Convert the input data into a vector of numbers;
def load_data(filename:str)->tf.Tensor:
    with open(filename, "rb") as f:
        data = f.read()
        print(f"Name of the training dataset: {filename}")
        print(f"Length of file: {len(data)} bytes\n")

        if filename[-4:] == ".txt":
            data = data.decode().split('\n')
        # If the input file is a standard file, 
        # there is a chance that the last line could simply be an empty line;
        # if this is the case, then remove the empty line
        if (data[len(data) - 1] == ""):
            data.remove("")
            data = np.array(data)
            data = data.astype('float32')
        else:
        # The input file is a binary file
            data = tf.io.decode_raw(data, tf.float32)
    return data

# Create new directory if not exist
def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Create folders to store training information
def mkdir_storage(model_dir, resume={}):
    if os.path.exists(os.path.join(model_dir, 'summaries')):
        # val = input("The model directory %s exists. Overwrite? (y/n) " % model_dir)
        val = 'n' if len(resume)==0 else 'y'
        print()
        if val == 'y':
            if os.path.exists(os.path.join(model_dir, 'summaries')):
                shutil.rmtree(os.path.join(model_dir, 'summaries'))
            if os.path.exists(os.path.join(model_dir, 'checkpoints')):
                shutil.rmtree(os.path.join(model_dir, 'checkpoints'))
    
    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    mkdir_if_not_exist(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    mkdir_if_not_exist(checkpoints_dir)
    return summaries_dir, checkpoints_dir

# Use tensorflow to split data into train + val + test, compatible with tf.data API
def split_train_val_test_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, 
                        shuffle=True, shuffle_size=10000, seed=0):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=seed)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

# Use tensorflow to split data into train + val, compatible with tf.data API
def split_train_val_tf(ds, ds_size, train_size=0.9, shuffle=True, shuffle_size=10000, seed=0):
    assert train_size <= 1, 'Split proportion must be in [0, 1]'
    assert train_size >= 0, 'Split proportion must be in [0, 1]'
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=seed)
    
    train_size = int(train_size * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)
    
    return train_ds, val_ds

# Use pandas to split data into train + val + test, compatible with pandas DataFrame
def split_train_val_test_pd(df, train_split=0.8, val_split=0.1, test_split=0.1, random_state=0):
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    assert val_split == test_split 

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=random_state)
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split - test_split) * len(df))]
    
    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds, val_ds, test_ds

# Use pandas to split data into train + val, compatible with pandas DataFrame
def split_train_val_pd(df, train_size=0.9, random_state=0):
    assert train_size <= 1, 'Split proportion must be in [0, 1]'
    assert train_size >= 0, 'Split proportion must be in [0, 1]'

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=random_state)
    indices_or_sections = [int(train_size * len(df)), len(df)]
    
    train_ds, val_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds, val_ds

def split_train_test_np(df, train_size=0.9, random_state=0):
    assert train_size <= 1, 'Split proportion must be in [0, 1]'
    assert train_size >= 0, 'Split proportion must be in [0, 1]'

    split = np.random.choice(range(df.shape[0]), int(train_size*df.shape[0]))
    train_ds = df[split]
    test_ds =  df[~split]
    
    print(f"train_ds.shape : {train_ds.shape}")
    print(f"test_ds.shape : {test_ds.shape}")
    return train_ds, test_ds

# Find number of row and column to slice the original image into many smaller
# block of given block_size
def find_dim_sblock(data:np.ndarray, block_size:int, verbose:bool=True):
    # Calculate number of blocks in a row (j) and in a column (i)
    num_block_row = int(data.shape[0] / block_size)
    num_block_col = int(data.shape[1] / block_size)

    # Add 1 more batch if mod( data.shape[0], block_size) != 0
    num_block_row = num_block_row if data.shape[0] % block_size == 0 else num_block_row+1
    num_block_col = num_block_col if data.shape[1] % block_size == 0 else num_block_col+1

    if verbose==True:
        print("Number of rows in the data that can be split into smaller blocks:", num_block_row)
        print("Number of columns in the data that can be split into smaller blocks:", num_block_col)

    return num_block_row, num_block_col

def split_image(image3, tile_size):
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size, image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    return tf.reshape(serial_tiles, [-1, tile_size, tile_size, image_shape[2]])

def unsplit_image(tiles4, image_shape):
    tile_width = tf.shape(tiles4)[1]
    serialized_tiles = tf.reshape(tiles4, [-1, image_shape[0], tile_width, image_shape[2]])
    rowwise_tiles = tf.transpose(serialized_tiles, [1, 0, 2, 3])
    return tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])

def get_folder_size(start_path:str='.')->int:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def predict(model, data):
    decoded_data = list()
    z_samples = [[] for _ in range(len(model.z_samples))]
    ftrs = [[] for _ in range(len(model.z_samples))]

    for _, x in enumerate(data):
        x_pred, z_sample, ftr = model.predict(x)
        # print(f"x_pred.shape: {x_pred.shape}")
        decoded_data.append(x_pred)
        for i in range(len(z_sample)):
            z_samples[i].append(z_sample[i])
            ftrs[i].append(ftr[i])

    decoded_data = np.concatenate(decoded_data, axis=0)
    for i in range(len(z_samples)):
        z_samples[i] = np.concatenate(z_samples[i], axis=0)
        ftrs[i] = np.concatenate(ftrs[i], axis=0)
    return decoded_data, z_samples, ftrs

def groups_per_scale(num_scales, num_groups_per_scale, is_adaptive, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g

def get_model_arch(arch_type):
    if arch_type == 'res_bnswish':
        model_arch = dict()
        model_arch['normal_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_dec'] = ['res_bnswish', 'res_bnswish']
        model_arch['up_sampling_dec'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_post'] = ['res_bnswish', 'res_bnswish']
        model_arch['up_sampling_post'] = ['res_bnswish', 'res_bnswish']
        model_arch['ar_nn'] = ['res_bnswish']
    elif arch_type == 'res_mbconv':
        model_arch = dict()
        model_arch['normal_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_dec'] = ['mconv_e6k5g0']
        model_arch['up_sampling_dec'] = ['mconv_e6k5g0']
        model_arch['normal_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_post'] = ['mconv_e3k5g0']
        model_arch['up_sampling_post'] = ['mconv_e3k5g0']
        model_arch['ar_nn'] = ['mconv_e6k5g0']
    elif arch_type == 'res_wnelu':
        model_arch = dict()
        model_arch['normal_enc'] = ['res_wnelu', 'res_elu']
        model_arch['down_sampling_enc'] = ['res_wnelu', 'res_elu']
        model_arch['normal_dec'] = ['mconv_e3k5g0']
        model_arch['up_sampling_dec'] = ['mconv_e3k5g0']
        model_arch['normal_pre'] = ['res_wnelu', 'res_elu']
        model_arch['down_sampling_pre'] = ['res_wnelu', 'res_elu']
        model_arch['normal_post'] = ['mconv_e3k5g0']
        model_arch['up_sampling_post'] = ['mconv_e3k5g0']
        model_arch['ar_nn'] = ['mconv_e3k5g0']
    return model_arch

# @keras_export("keras.optimizers.schedules.CosineDecayRestarts",
#              "keras.experimental.CosineDecayRestarts")
class CosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that uses a cosine decay schedule with restarts.
  See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
  SGDR: Stochastic Gradient Descent with Warm Restarts.
  When training a model, it is often useful to lower the learning rate as
  the training progresses. This schedule applies a cosine decay function with
  restarts to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.
  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  The learning rate multiplier first decays
  from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
  restart is performed. Each new warm restart runs for `t_mul` times more
  steps and with `m_mul` times initial learning rate as the new learning rate.
  Example usage:
  ```python
  first_decay_steps = 1000
  lr_decayed_fn = (
    tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps))
  ```
  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.
  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      first_decay_steps,
      t_mul=2.0,
      m_mul=1.0,
      alpha=0.0,
      name=None):
    """Applies cosine decay with restarts to the learning rate.
    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
        number. Number of steps to decay over.
      t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the number of iterations in the i-th period.
      m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the initial learning rate of the i-th period.
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of the initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
    """
    super(CosineDecayRestarts, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.first_decay_steps = first_decay_steps
    self._t_mul = t_mul
    self._m_mul = m_mul
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "SGDRDecay") as name:
      initial_learning_rate = tf.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      first_decay_steps = tf.cast(self.first_decay_steps, dtype)
      alpha = tf.cast(self.alpha, dtype)
      t_mul = tf.cast(self._t_mul, dtype)
      m_mul = tf.cast(self._m_mul, dtype)

      global_step_recomp = tf.cast(step, dtype)
      completed_fraction = global_step_recomp / first_decay_steps

      def compute_step(completed_fraction, geometric=False):
        """Helper for `cond` operation."""
        if geometric:
          i_restart = tf.floor(
              tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
              tf.math.log(t_mul))

          sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
          completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

        else:
          i_restart = tf.floor(completed_fraction)
          completed_fraction -= i_restart

        return i_restart, completed_fraction

      i_restart, completed_fraction = tf.cond(
          tf.equal(t_mul, 1.0),
          lambda: compute_step(completed_fraction, geometric=False),
          lambda: compute_step(completed_fraction, geometric=True))

      m_fac = m_mul**i_restart
      cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(
          tf.constant(math.pi, dtype=dtype) * completed_fraction))
      decayed = (1 - alpha) * cosine_decayed + alpha

      return tf.multiply(initial_learning_rate, decayed, name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "first_decay_steps": self.first_decay_steps,
        "t_mul": self._t_mul,
        "m_mul": self._m_mul,
        "alpha": self.alpha,
        "name": self.name
    }  
