import argparse
import os
import warnings
import time

import cv2
import numpy as np
import torch
from put.PUT.image_synthesis.modeling.build import build_model
from put.PUT.image_synthesis.data.build import build_dataloader
from put.PUT.image_synthesis.utils.misc import seed_everything, merge_opts_to_config, modify_config_for_debug
from put.PUT.image_synthesis.utils.io import load_yaml_config
from put.PUT.image_synthesis.engine.logger import Logger
from put.PUT.image_synthesis.engine.solver import Solver
from put.PUT.image_synthesis.distributed.launch import launch

# environment variables
# NODE_RANK = os.environ['AZ_BATCHAI_TASK_INDEX'] if 'AZ_BATCHAI_TASK_INDEX' in os.environ else 0
NODE_RANK = os.environ['RANK'] if 'RANK' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = os.environ['AZ_BATCH_MASTER_NODE'].split(':') if 'AZ_BATCH_MASTER_NODE' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training script')
    parser.add_argument('--config_file', type=str, default='',
                        help='path of config file')
    parser.add_argument('--name', type=str, default='',
                        help='the name of this experiment, if not provided, set to'
                             'the name of config file')
    parser.add_argument('--output', type=str, default='OUTPUT',
                        help='directory to save the results')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='print frequency (default: 10)')
    parser.add_argument('--load_path', type=str, default=None,
                        help='path to model that need to be loaded, '
                             'used for loading pretrained model')
    parser.add_argument('--resume_name', type=str, default=None,
                        help='resume one experiment with the given name')
    parser.add_argument('--auto_resume', action='store_true',
                        help='automatically resume the training')

    # args for ddp
    parser.add_argument('--backend', type=str, default='NCCL',
                        choices=['nccl', 'mpi', 'gloo'],
                        help='which type of bakend for ddp')
    parser.add_argument('--num_node', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=NODE_RANK,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default=DIST_URL,
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    parser.add_argument('--sync_bn', action='store_true',
                        help='use sync BN layer')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')
    parser.add_argument('--timestamp', action='store_true', # default=True,
                        help='use tensorboard for logging')
    # args for random
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', action='store_true',
                        help='set cudnn.deterministic True')

    parser.add_argument('--amp', action='store_true', # default=True,
                        help='automatic mixture of precesion')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='set as debug mode')
    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    if args.resume_name is not None:
        args.name = args.resume_name
        args.config_file = os.path.join(args.output, args.resume_name, 'configs', 'config.yaml')
        args.auto_resume = True
    else:
        if args.name == '':
            args.name = os.path.basename(args.config_file).replace('.yaml', '')
        if args.timestamp:
            assert not args.auto_resume, "for timstamp, auto resume is hard to find the save directory"
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            args.name = time_str + '-' + args.name

    # modify args for debugging
    if args.debug:
        args.name = 'debug'
        if args.gpu is None:
            args.gpu = 0

    args.save_dir = os.path.join(args.output, args.name)

    #自己加的
    args.distributed=False
    return args


def main():
    config = load_yaml_config("/data/zhujinxian/code/put/PUT/configs/put_cvpr2022/sintel/p_vqvae_sintel.yaml")
    args = get_args()

    # get logger
    logger = Logger(args)
    logger.save_config(config)

    dataloader_info = build_dataloader(config, args)
    for itr, batch in enumerate(dataloader_info['train_loader']):
        #print(itr)
        #print(batch)
        file_path="/data/zhujinxian/code/put/PUT/data/sintel/sintel_tensor/"+str(itr)+".png"
        tensor = batch['image']
        numpy_array = tensor.cpu().numpy()
        #np.save(file_path, numpy_array)
        numpy_array = np.transpose(numpy_array, (2, 3, 1, 0)).squeeze()
        cv2.imwrite(file_path, numpy_array)


if __name__ == '__main__':
    main()