import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import configargparse
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from utils import utils
import data_io
from utils.normalizer import DataNormalizer
from utils.padder import Padder
# from hard_code_model import VariationalAutoencoder
from model import VariationalAutoencoder
import numpy as np
import tensorflow as tf
import pprint, pickle

def main(args):
    
    dataio = data_io.Data(args.batch_size)
    if args.dataset == "mnist":
        normalizer = DataNormalizer(min_scale=0.0, max_scale=1.0)
        data = dataio.load_process_mnist(normalizer)
    elif args.dataset == "cloud":
        print(f"Data: cloud")
        tile_size = 64
        normalizer = DataNormalizer(min_scale=1.0, max_scale=10.0)
        padder = Padder(tile_size)
        data = dataio.load_process_cloud_without_split(normalizer, padder)
    elif args.dataset == "isabel":
        print(f"Data: Huricane ISABEL")
        sys.exit()

    # Model Initialization
    in_shape = list(data.element_spec.shape[1:])
    # in_shape = list(inputs.shape[1:])
    model_arch = utils.get_model_arch(args.model_arch)
    print(f"model_arch: {args.model_arch}")

    vae = VariationalAutoencoder(args, model_arch, in_shape)
    print(f"is using se: {vae.use_se}\n")
    vae.build(input_shape=([None]+ in_shape))
    vae.model().summary()

    # Set up for training, evaluation, or generation
    # Default model_path
    folder_name = "vae_bidirectional_" + str(args.model_arch)
    model_path = os.path.join("model_output", folder_name)
    if args.model_path:
        if not os.path.exists(args.model_path):
            val = input(f"The model directory {args.model_path} does not exist. Create? (y/n) ")
            if val == 'y':
                utils.mkdir_if_not_exist(args.model_path)
                model_path = args.model_path
            else:
                print(f"Folder was not created. Logging information to default path: " + \
                    f"{model_path}")
        else:
            model_path = args.model_path
                
    print(f"\nlogging information to: {model_path}\n")
    
    resume_checkpoint={}
    if args.resume or args.generate:
        resume_checkpoint['resume_epoch'] = args.iter
        weight_path = model_path + '/checkpoints/' + f'model_{args.iter:06d}'
        vae.load_weights(weight_path)
        pre_prior = np.loadtxt(model_path + '/checkpoints/' + f'pre_prior_{args.iter:06d}.txt')
        pre_prior_shape = np.loadtxt(model_path + '/checkpoints/' + f'pre_prior_shape_{args.iter:06d}.txt', 
                                    dtype=np.int32)
        print(f"Before Loadding vae.pre_prior: {vae.pre_prior}")
        vae.pre_prior = tf.Variable(pre_prior.reshape(pre_prior_shape), trainable=True)
        print(f"After Loadding vae.pre_prior: {vae.pre_prior}")
        print(f"Model weights successfully loaded.")
    print(resume_checkpoint)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cloud', 'isabel'],
                        help='which dataset to use, default="mnist')
    parser.add_argument('--data', type=str, default='/data',
                        help='location of the data corpus')
    # Genral training options
    parser.add_argument('--eval', action='store_true', default=False,
                        help="run evaluation on testing dataset")
    parser.add_argument('--generate', action='store_true', default=False,
                        help="run generation")
    # parser.add_argument('--resume', type=int, default=None,
    #            help='resume training, must specify the iteration of ckpt.')
    # parser.add_argument('--weight_path', default=None,
    #                     help="Path to weight directory")
    parser.add_argument('--model_path', default=None,
                        help="Path to model folder")
    parser.add_argument('--path_img_output', default=None,
                        help="Path to image output folder when generating new images")
    # logging options
    # parser.add_argument('--experiment_name', type=str, required=True,
    #                help='path to directory where checkpoints & tensorboard events will be saved.')
    parser.add_argument('--epochs_til_ckpt', type=int, default=10,
                        help="Epochs until checkpoint is saved")
    parser.add_argument('--steps_til_summary', type=int, default=50,
                        help="Number of iterations until tensorboard summary is saved")
    parser.add_argument('--logging_root', type=str, default='./logs',
                        help="root for logging")
    # optimization
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="batch size. default=32")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=5e-5,
                        help='min learning rate')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=30,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--model_arch', type=str, default='res_wnelu',
                        help='which model architecture to use')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.5,
                        help='The portions epochs that KL is annealed')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=1,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    # latent variables
    parser.add_argument('--num_channels_of_latent', type=int, default=1,
                        help='number of channels of latent variables')
    # Initial channel
    parser.add_argument('--num_initial_channel', type=int, default=16,
                        help='number of channels in pre-enc and post-dec')
    # Share parameter of preprocess and post-process blocks
    parser.add_argument('--num_process_blocks', type=int, default=1,
                        help='number of preprocessing and post-processing blocks')
    # Preprocess cell
    parser.add_argument('--num_preprocess_cells', type=int, default=2,
                        help='number of cells per proprocess block')
    # Encoder and Decoder Tower
    parser.add_argument('--num_scales', type=int, default=2,
                        help='the number of scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=1,
                        help='number of groups per scale')
    parser.add_argument('--is_adaptive', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_cell_per_group_enc', type=int, default=1,
                        help='number of cells per group in encoder')
    # decoder parameters
    parser.add_argument('--num_cell_per_group_dec', type=int, default=1,
                        help='number of cell per group in decoder')
    # Post-process cell
    parser.add_argument('--num_postprocess_cells', type=int, default=2,
                        help='number of cells per post-process block')
    # Squeeze-and-Excitation
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    # Resume
    parser.add_argument('--resume', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    parser.add_argument('--iter', type=int, default=0,
                        help='resume iteration')
    args = parser.parse_args()

    if (args.generate and (args.model_path is None or args.path_img_output is None or args.iter is None)):
        parser.error('The --generate argument requires the --model_path and --path_img_output')

    if (args.resume and args.iter is None):
        parser.error('The --resume argument requires the --iter')

    # for k, v in args.__dict__.items():
    #     print(f"{k}: {v}")
    # print()
    
    # devices = tf.config.list_physical_devices()
    # print(devices)
    # print(f"Tennsorflow version: {tf.__version__}\n")

    # main(args=args)

    # # a = tf.Variable(tf.random.normal(shape=(3,4,4)))
    # # print(f"a:{a}")
    # # b = tf.random.normal(shape=(3,4,4))
    # # print(f"b:{b}")
    # a = {}
    # a = {"z"+str(len(a)): tf.random.normal(shape=(3,4,4))}
    # print(f"len a: {len(a)}")
    # print(f"a: {a}")
    

    pkl_file = open('D:\\research\\Autoencoder\\model_output\\vae_bidirectional_res_wnelu\\z_samples_000001.pkl', 'rb')

    data1 = pickle.load(pkl_file)
    pprint.pprint(data1)

    pkl_file.close()