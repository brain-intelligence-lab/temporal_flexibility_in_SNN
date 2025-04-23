import argparse
import os
import torch
from models.snn_speck import dvs_gesture_snn_32x32, dvs_gesture_snn, dvs_gesture_tfsnn, dvs_gesture_tfsnn_wobn, VGG9_tfsnn_wobn, VGG9_tfsnn_WS, VGG9_tfsnn_WSAF, VGG9_tfsnn_bn, VGG9_tfsnn_bn_meminit
from sinabs.activation import MembraneSubtract, MultiSpike, SingleSpike, MembraneReset
from data import DVS_Gesture_loader, CIFAR10DVS_loader

    # Gesture MTT args
    # ### Dataset
    # parser.add_argument("--path_dataset", type=str, default="/data_nv/dataset")
    # parser.add_argument("--dataset", type=str, default="dvs_gesture")
    # ### Model Load&Save
    # parser.add_argument("--load_path", type=str, default="")
    # parser.add_argument("--save_path", type=str, default="./MTT_Gesture.pt")
    # ### Training Phase
    # parser.add_argument("--phase", type=str, default="normal")
    # ### Training Time
    # parser.add_argument("--T", type=int, default=20)
    # parser.add_argument("--MTT", type=bool, default=True)
    # parser.add_argument("--T_L_radius", type=int, default=19)
    # parser.add_argument("--T_R_radius", type=int, default=0)
    # parser.add_argument("--sample_num", type=int, default=3)
    # parser.add_argument("--cal_iter", type=int, default=0)
    # ### Training
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--fine_tune_epochs", type=int, default=60)
    # parser.add_argument("--optimizer", type=str, default='AdamW')
    # parser.add_argument("--scheduler", type=str, default='None')
    # parser.add_argument("--lr", type=float, default=0.0005)
    # parser.add_argument("--weight_decay", type=float, default=0.02)
    # parser.add_argument("--seed", type=int, default=1000)
    # parser.add_argument("--gpu", type=str, default="3")
    # parser.add_argument("--model_name", type=str, default="dvs_gesture_tfsnn_wobn")



def parse_args(is_notebook_mode=False) :
    parser = argparse.ArgumentParser()
    ### Dataset
    parser.add_argument("--path_dataset", type=str, default="/data_smr/dataset")
    parser.add_argument("--dataset", type=str, default="cifar10dvs")
    ### Model Load&Save
    parser.add_argument("--load_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="./SDT_CIFAR10DVS_BN_COSLR_VROLL20_MYVERTFLIP_SDTFINETUNE.pt")
    ### Model Attribute
    parser.add_argument("--spike_fn", type=str, default="MultiSpike")
    parser.add_argument("--reset_fn", type=str, default="MembraneSubstract")
    ### Training Phase
    parser.add_argument("--phase", type=str, default="normal")
    ### Training MTT
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--MTT", type=bool, default=False)
    parser.add_argument("--T_L_radius", type=int, default=9)
    parser.add_argument("--T_R_radius", type=int, default=0)
    parser.add_argument("--sample_num", type=int, default=3)
    parser.add_argument("--cal_iter", type=int, default=0)
    ### Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--fine_tune_epochs", type=int, default=60)
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument("--scheduler", type=str, default='CosLR')
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model_name", type=str, default="VGG9_tfsnn_bn")

    args = parser.parse_args(args=[]) if is_notebook_mode else parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert(args.phase in ['normal', 'bias_removal'])
    assert(args.reset_fn in ['MembraneSubstract', 'MembraneReset'])
    assert(args.spike_fn in ['MultiSpike', 'SingleSpike'])
    print(args)
    return args

def reset_fn_parse(args) :
    if args.reset_fn == 'MembraneReset' :
        return MembraneReset()
    elif args.reset_fn == 'MembraneSubstract' :
        return MembraneSubtract()
    assert(0, 'Invalid reset function')

def spike_fn_parse(args) :
    if args.spike_fn == 'SingleSpike' :
        return SingleSpike
    elif args.spike_fn == 'MultiSpike' :
        return MultiSpike
    assert(0, 'Invalid spike function')

def model_parse(args) :
    spike_fn = spike_fn_parse(args)
    reset_fn = reset_fn_parse(args)
    if args.model_name == 'dvs_gesture_tfsnn' :
        return dvs_gesture_tfsnn(batch_size=args.batch_size, width=1, reset_fn=reset_fn, spike_fn=spike_fn)
    elif args.model_name == 'dvs_gesture_snn' :
        return dvs_gesture_snn(batch_size=args.batch_size, width=1, reset_fn=reset_fn, spike_fn=spike_fn)
    elif args.model_name == 'dvs_gesture_tfsnn_wobn' :
        return dvs_gesture_tfsnn_wobn(batch_size=args.batch_size, width=1, reset_fn=reset_fn, spike_fn=spike_fn)
    elif args.model_name == 'VGG9_tfsnn_wobn' :
        return VGG9_tfsnn_wobn(input_size=128, category_num=10, batch_size=args.batch_size, reset_fn=reset_fn, spike_fn=spike_fn)
    elif args.model_name == 'VGG9_tfsnn_bn' :
        return VGG9_tfsnn_bn(input_size=128, category_num=10, batch_size=args.batch_size, reset_fn=reset_fn, spike_fn=spike_fn)
    elif args.model_name == 'VGG9_tfsnn_bn_meminit' :
        return VGG9_tfsnn_bn_meminit(input_size=128, category_num=10, batch_size=args.batch_size, reset_fn=reset_fn, spike_fn=spike_fn)
    elif args.model_name == 'VGG9_tfsnn_WS' :
        return VGG9_tfsnn_WS(input_size=128, category_num=10, batch_size=args.batch_size, reset_fn=reset_fn, spike_fn=spike_fn)
    elif args.model_name == 'VGG9_tfsnn_WSAF' :
        return VGG9_tfsnn_WSAF(input_size=128, category_num=10, batch_size=args.batch_size, reset_fn=reset_fn, spike_fn=spike_fn)
    assert(0, 'Invalid model name')

def optimizer_parse(parameters, args) :
    optimizer = None
    if args.optimizer == 'Adam' :
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    elif args.optimizer == 'AdamW' :
        optimizer = torch.optim.AdamW(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD' :
        optimizer = torch.optim.SGD(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
    else : assert(0, 'Invalid optimizer name')
    return optimizer

def scheduler_parse(optimizer, args) :
    scheduler = None
    if args.scheduler == 'CosLR' :
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-7)
    elif args.scheduler != 'None' : 
        assert(0, 'Invalid scheduler name')
    return scheduler

def dataset_parse(args) :
    if args.dataset == 'dvs_gesture' :
        train_loader, val_loader = DVS_Gesture_loader(root_dir=args.path_dataset, test_only=False, run_batch=True, batch_size=args.batch_size, T=args.T, data_aug=False)
    elif args.dataset == 'cifar10dvs' :
        train_loader, val_loader = CIFAR10DVS_loader(root_dir=args.path_dataset, test_only=False, run_batch=True, batch_size=args.batch_size, T=args.T, data_aug=True)
    else : assert(0, 'Invalid dataset name')
    return train_loader, val_loader

