'''
Implement a generic training loop
'''

from tqdm.autonotebook import tqdm
import timeit
import numpy as np
import os
import sys
from utils import utils
from generate import generate
import json
import tensorflow as tf
import pickle
import gc


def compute_kl_loss(model):
    total_log_q, total_log_p, kl_all, kl_diag = model.cal_kl_components()
    kl_loss = tf.reduce_sum(kl_all)
    return kl_loss

def compute_recon_loss(x_pred, x_orig):
    recon_loss = tf.reduce_sum(tf.square(tf.subtract(x_pred,x_orig)))
    return recon_loss

def compute_loss(model, x_orig, batch_size, kl_weight=1.0, training=False):
    x_pred, kl_loss = model(x_orig, training=training)
    recon_loss = compute_recon_loss(x_pred, x_orig) / batch_size
    # kl_loss = compute_kl_loss(model) / batch_size
    # kl_loss = kl_loss / batch_size
    total_loss = recon_loss + kl_weight*kl_loss
    # print(f"\nrecon_loss: {recon_loss.numpy():0.6f} \tkl_loss: {kl_loss.numpy():0.6f}" +
    #         f"\t total_loss: {total_loss.numpy():0.6f}")
    return (total_loss, recon_loss, kl_loss)

@tf.function
def train_step(model, x_orig, optimizer, batch_size, kl_weight=1.0):
    with tf.GradientTape() as tape:
        (total_loss, recon_loss, kl_loss) = compute_loss(model, x_orig, batch_size, 
                                                        kl_weight, training=True)
        total_loss += sum(model.losses)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return (total_loss, recon_loss, kl_loss)

# @tf.function
def test_step(model, x_orig, batch_size, kl_weight=1.0):
    (total_loss, recon_loss, kl_loss) = compute_loss(model, x_orig, batch_size, 
                                                        kl_weight, training=False)
    total_loss += sum(model.losses)
    return (total_loss, recon_loss, kl_loss)

def train_loop(model, train_data, val_data, train_losses, val_losses,
                optimizer, batch_size, kl_weight=1.0):

    (train_loss, recon_train_loss, kl_train_loss) = train_losses
    # with tqdm(total=train_data.cardinality().numpy()) as pbar:
    # pbar = tqdm(total=train_data.cardinality().numpy())
    for step, x_train in enumerate(train_data):
        losses = train_step(model, tf.convert_to_tensor(x_train, dtype=tf.float32), 
                            optimizer, batch_size, 
                            kl_weight=kl_weight)
        train_loss.update_state(losses[0])
        recon_train_loss.update_state(losses[1])
        kl_train_loss.update_state(losses[2])
        # pbar.update(1)
    # pbar.close()
        
    (val_loss, recon_val_loss, kl_val_loss) = val_losses
    for _, x_val in enumerate(val_data):
        losses = test_step(model, tf.convert_to_tensor(x_val), 
                            batch_size, kl_weight=kl_weight)
        val_loss.update_state(losses[0])
        recon_val_loss.update_state(losses[1])
        kl_val_loss.update_state(losses[2])

    tf.keras.backend.clear_session()
    gc.collect()

def train(model, data, epochs, optimizer, train_portion, 
        model_dir, batch_size, kl_anneal_portion,
        epochs_til_ckpt=10, steps_til_summary=10, resume_checkpoint={}):
    
    summaries_dir, checkpoints_dir = utils.mkdir_storage(model_dir, resume_checkpoint)

    # Save training parameters if we need to resume training in the future
    start_epoch = 1
    if 'resume_epoch' in resume_checkpoint:
        start_epoch = resume_checkpoint['resume_epoch'] + 1

    train_loss_results, val_loss_results = [], []
    total_training_time = 0
    training_results = (train_loss_results, val_loss_results, total_training_time)

    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    kl_train_loss = tf.keras.metrics.Mean()
    recon_train_loss = tf.keras.metrics.Mean()
    kl_val_loss = tf.keras.metrics.Mean()
    recon_val_loss = tf.keras.metrics.Mean()
    train_losses = [train_loss, recon_train_loss, kl_train_loss]
    val_losses = [val_loss, recon_val_loss, kl_val_loss]

    min_kl_weight = 1e-3
    kl_weight = min(max((start_epoch-1)/(kl_anneal_portion*epochs), min_kl_weight), 1)
    
    # Start training
    for epoch in range(start_epoch, epochs+1):
        train_data, val_data = None, None
        train_data, val_data = utils.split_train_val_tf(data, data.cardinality().numpy(), 
                                        train_size=train_portion, shuffle=True, 
                                        shuffle_size=data.cardinality().numpy(), seed=None)
        
        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr.numpy()

        for loss in train_losses + val_losses:
            loss.reset_state()

        start_time = timeit.default_timer()
        
        # training loop
        tqdm.write(f"Epoch {epoch} ")
        train_loop(model, train_data, val_data, train_losses, val_losses,
                optimizer, batch_size, kl_weight=kl_weight)
        
        train_loss_results.append(train_loss.result().numpy())
        val_loss_results.append(val_loss.result().numpy())

        training_time = timeit.default_timer()-start_time
        total_training_time += training_time
        
        # after each epoch
        tqdm.write(f"training time: {training_time:0.5f}, " + 
                    f"LR: {current_lr:0.5f}, kl_weight: {kl_weight:0.5f}, \n\t" +
                    f"kl_train_loss: {kl_train_loss.result():0.5f}, " +
                    f"recon_train_loss: {recon_train_loss.result():0.5f}, " +
                    f"train_loss: {train_loss.result():0.5f}, \n\t" + 
                    f"kl_val_loss: {kl_val_loss.result():0.5f}, " +
                    f"recon_val_loss: {recon_val_loss.result():0.5f}, " + 
                    f"val_loss: {val_loss.result():0.5f}")
        kl_weight = min(max((epoch)/(kl_anneal_portion*epochs), min_kl_weight), 1)

        training_results = (train_loss_results, val_loss_results, total_training_time)

        # save model when epochs_til_ckpt requirement is met
        if (not epoch % epochs_til_ckpt) and epoch:
            save_training_parameters(checkpoints_dir, epoch, model, training_results)

        tf.keras.backend.clear_session()
        gc.collect()

    test_sample = [sample for sample in data.shuffle(data.cardinality().numpy()).take(1)][0]
    generate(model, test_sample, model_dir, "gen_img_from_training.png")

    # save model at end of training
    save_training_parameters(checkpoints_dir, epochs, model, training_results)

def save_training_parameters(checkpoints_dir, epochs, model, training_results):
    (train_loss_results, val_loss_results, total_training_time) = training_results
    model.save_weights(os.path.join(checkpoints_dir, f'model_{epochs:06d}'))
    np.savetxt(os.path.join(checkpoints_dir, f'train_losses_{epochs:06d}.txt'),
                np.array(train_loss_results))
    np.savetxt(os.path.join(checkpoints_dir, f'val_losses_{epochs:06d}.txt'),
                np.array(val_loss_results))
    np.savetxt(os.path.join(checkpoints_dir, f'training_time_{epochs:06d}.txt'),
                np.array([total_training_time]))