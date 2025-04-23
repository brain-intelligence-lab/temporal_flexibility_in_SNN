import torch
import torch.nn as nn
from my_parser import parse_args, model_parse, optimizer_parse, scheduler_parse, dataset_parse
from tqdm import tqdm
from utils import fine_tune, fine_tune_mixed_timestep, SNN_Loss_Wrapper, set_time_config, UniformTimeSampler, lock_random_seed, remove_v_mem, bias_remove, param_filter, bn_merge_biasless, load_snn
import os

args = parse_args()
lock_random_seed(args.seed)

model = model_parse(args).cuda()
if len(args.load_path) != 0 :
    load_snn(model, args.load_path)
    print(f"Successfully load model: {args.load_path}")

set_time_config(model, args.T)

if args.MTT :
    TSampler = UniformTimeSampler(model, args.T, args.T_L_radius, args.T_R_radius)

is_removing_bias = (args.phase == 'bias_removal')
if is_removing_bias :
    param_set = param_filter(model.named_parameters(), ex_subs=[['bias']])
    bias_remove(model)
    bn_merge_biasless(model.net)
else :
    param_set = model.parameters()
optimizer = optimizer_parse(param_set, args)
scheduler = scheduler_parse(optimizer, args)

loss_func = SNN_Loss_Wrapper(torch.nn.CrossEntropyLoss())
train_loader, val_loader = dataset_parse(args)

real_epochs = args.fine_tune_epochs if is_removing_bias else args.epochs
if not args.MTT :
    fine_tune(model, train_loader, val_loader, loss_func, optimizer, scheduler, real_epochs, lock_BN=is_removing_bias, save_file_name=args.save_path)
else :
    fine_tune_mixed_timestep(model, train_loader, val_loader, loss_func, optimizer, TSampler, args.sample_num, args.cal_iter, scheduler, real_epochs, lock_BN=is_removing_bias, save_file_name=args.save_path)

if is_removing_bias :
    bn_merge_biasless(model.net)
torch.save(model.state_dict(), args.save_path)



        