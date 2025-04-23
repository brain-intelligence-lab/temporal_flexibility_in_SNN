import argparse
import torch
import os

#### parse
def start_parse() :
    parser = argparse.ArgumentParser(description = '')
    ### Paths
    parser.add_argument('--path_dataset', type = str, default = "", help = 'The path of dataset')
    parser.add_argument('--record_path', type = str, default = '/checkpoint.pt', help = 'Path of the records file')
    parser.add_argument('--log_path', type = str, default = '/log.txt', help = 'Path of the log file')
    parser.add_argument('--path_prefix', type = str, default = './test', help = '')
    ### Data
    parser.add_argument('--use_cifar10', action = "store_true", default = False, help = 'Use CIFAR10, default: use CIFAR100')
    parser.add_argument('--use_nc101', action = "store_true", default = False, help = 'Use N-Caltech101, default: use cifar10-dvs')
    parser.add_argument('--disable_autoaug', action = "store_true", default = False, help = '')
    ### DP Params
    parser.add_argument('--data_parallel', action = "store_true", default = False, help = 'Data Parallel')
    ### Training Params
    parser.add_argument('--manual_seed', type = int, default = 1000, help = 'manual random seed, -1 when off')
    parser.add_argument('--mixed_precision', action = "store_true", default = False, help = 'Using mixed precision(FP16) for training')
    parser.add_argument('--gpu', type = str, default = "3,4,5", help = 'GPUs used')
    parser.add_argument('--save_best_model', action = "store_true", default = False, help = 'Save the best model or not')
    parser.add_argument('--saving_period', type = int, default = -1, help = 'Saving period (-1 when off)')# 40
    parser.add_argument('--epochs', type = int, default = 300, help = 'The number of epochs')
    parser.add_argument('--lr', type = float, default = 0.1, help = 'Initial Learning Rate')
    parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'The number of epochs')
    parser.add_argument('--batch_size', type = int, default = 256, help = 'Batch Size')
    parser.add_argument('--optimizer', type = str, default = 'SGD', help = 'Optimizer')
    ### Network Params
    parser.add_argument('--category_num', type = int, default = 10, help = 'The number of categories')
    parser.add_argument('--model', type = str, default = "TFSNN_ResNet19", help = 'Which model to use')
    parser.add_argument('--time_res', nargs = '+', type = int, default = [6, 6, 6, 6, 6, 6, 6, 6], help = 'Time resolution settings')
    parser.add_argument('--Vth', type = float, default = 1, help = 'Membrane threshold')
    parser.add_argument('--early_stop_epoch', type = int, default = -1, help = 'Early stop, -1 if banned')
    parser.add_argument('--decay', type = float, default = 0.5, help = 'Decay')
    parser.add_argument('--time_trans_mode', type = int, default = 0, help = 'Mode 0: evenly distributed 1: sorted 2: sorted(reverse) 3: random')
    ### NAS Params
    parser.add_argument('--sample_num', type = int, default = 3, help = '...')
    parser.add_argument('--group_block_num', type = int, default = 1, help = 'Number of the blocks within a group')
    parser.add_argument('--max_T', type = int, default = 6, help = 'Max timestep')
    parser.add_argument('--min_T', type = int, default = 1, help = 'Min timestep')
    parser.add_argument('--cal_iter', type = int, default = 10, help = 'Calibration iteration times')
    args = parser.parse_args()
    assert args.path_dataset != ""
    assert args.optimizer in ['SGD', 'AdamW']
    assert args.model in ['TFSNN_ResNet19', 'TFSNN_ResNet18', 'TFSNN_VGG14', 'TFSNN_ResNet18_DVS']
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args

args = start_parse()