import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm
import copy
import os

### utils

def DP_to_Normal(state_dict) :
    nsd = dict()
    for name, i in state_dict.items() :
        nsd[name[7:]] = i
    return nsd

def Normal_to_DP(state_dict) :
    nsd = dict()
    for name, i in state_dict.items() :
        nsd['module.' + name] = i
    return nsd

### reproductablity

def lock_random_seed(seed) : 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

### model tools
def bn_calibration_init(m):
    if getattr(m, 'track_running_stats', False):
        m.reset_running_stats()
        m.training = True
        m.momentum = None

### model training
def MTT_train_epoch(model_wrapper, loader, optimizer, scheduler, scaler, loss_func, category_num, min_T, max_T, sample_num = 1, mixed_precision = True, DVS = False, group_block_num = 1) :
    model_wrapper.train()
    model = model_wrapper
    if type(model_wrapper) == torch.nn.DataParallel :
        model = model_wrapper.module
    train_dataset_size = len(loader)
    epoch_loss = []
    with tqdm(total=train_dataset_size, leave=True, unit='batch', unit_scale=True) as pbar:
        for batch_index, (input, labels) in enumerate(loader) :
            optimizer.zero_grad()
            if DVS == False :
                input = input.cuda().unsqueeze(1)
            else :
                input = input.cuda()
            labels = labels.cuda()
            labels = torch.zeros((labels.shape[0], category_num)).cuda().scatter(1, labels.view(-1, 1), 1)
            
            time_res_pool = []
            def T_sampler(mn, mx, num, group_block_num) :
                if mn > 0 :
                    T_series = [random.randint(mn, mx) for _ in range(num)]
                else :
                    fac = -mn
                    _mx = (mx + fac - 1) // fac
                    T_series = [random.randint(1, _mx) * fac for _ in range(num)]
                return [T_series[i // group_block_num] for i in range(num * group_block_num)]

            num = (len(model.feature_ext) + group_block_num - 1) // group_block_num ## ceil
            for _ in range(sample_num) :
                time_res_pool.append(T_sampler(min_T, max_T, num, group_block_num))
            
            iter_loss = []
            for time_res in time_res_pool :
                in_T = time_res[0]

                if type(model_wrapper) == torch.nn.DataParallel : 
                    model_wrapper.module.reset_time_res(time_res)
                else : 
                    model_wrapper.reset_time_res(time_res)

                if DVS == False :
                    x = input.repeat(1, in_T, 1, 1, 1)
                else :
                    x = input
                
                with amp.autocast(enabled = mixed_precision) :
                    output, = model_wrapper(x, remain_dim_T = True)
                    loss = loss_func(output, labels)
                    scaler.scale(loss).backward()
                iter_loss.append(loss.clone().detach().item())

            scaler.step(optimizer)
            scaler.update()
            epoch_loss.append(torch.tensor(iter_loss))
            pbar.update(1)
    scheduler.step()
    epoch_loss = torch.stack(epoch_loss, dim = 0)
    mean_loss = epoch_loss.mean(dim = 0)
    return mean_loss
    
### loss function
class SNN_Loss(torch.nn.Module) :
    def __init__(self, criterion) :
        super().__init__()
        self.criterion = criterion
    
    def forward(self, input, target) :
        return self.criterion(input.mean(dim = 1), target)

### model evaluate

def bn_calibrate(model_wrapper, loader, cal_iter = None, DVS = False) :
    model_wrapper.train()
    model_wrapper.apply(bn_calibration_init)
    with torch.no_grad() :
        for batch_idx, (input, labels) in enumerate(loader) :
            if cal_iter != None and batch_idx >= cal_iter : 
                break
            in_T = model_wrapper.module.input_time_res \
            if type(model_wrapper) == torch.nn.DataParallel else \
            model_wrapper.input_time_res
            if DVS == False :
                input = input.cuda().unsqueeze(1).repeat(1, in_T, 1, 1, 1)
            else :
                input = input.cuda()
            model_wrapper(input)
    model_wrapper.eval()

def test_acc(model_wrapper, loader, DVS = False) :
    model_wrapper.eval()
    model = model_wrapper
    if type(model_wrapper) == torch.nn.DataParallel :
        model = model_wrapper.module
    with torch.no_grad() :
        correct = 0
        total = 0
        for batch_index, (input, labels) in enumerate(loader) :
            if DVS == False :
                input = input.cuda().unsqueeze(1).repeat(1, model.input_time_res, 1, 1, 1)
            else :
                input = input.cuda()
            labels = labels.cuda()

            output, = model_wrapper(input)
            _, predict_idx = torch.max(output, dim = 1)
            correct += (predict_idx == labels).sum().item()
            total += labels.numel()
    return correct / total

### model related

def table_spwn(in_T : int, out_T : int) : ## trans_table[t_out] = [matched t_in]
    trans_table = []
    eps = 1e-5
    fewer, more, expand = in_T, out_T, True
    if(fewer > more) :
        fewer, more = more, fewer
        expand = False
    last = 0
    for i in range(fewer) :
        nearest = round((i + 1) * more / fewer - eps)
        if(expand != True) :
            trans_table.append([])
            for j in range(last, nearest) :
                trans_table[i].append(j)
        else :
            for j in range(last, nearest) :
                trans_table.append([i,])
        last = nearest
    return trans_table