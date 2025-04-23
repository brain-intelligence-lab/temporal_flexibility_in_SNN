import torch
import torch.nn as nn
import numpy as np
import random
import os
import sinabs.layers as sl
import sinabs
import copy
from tqdm import tqdm
from models.snn_speck import TimeTrans

### Network Edit

def bias_remove(model) :
    named_mod = model.named_modules()
    # BN running mean remove
    for name, m in named_mod :
        if type(m) == nn.BatchNorm2d :
            m.running_mean = m.running_mean * 0.0
    # bias remove
    for name, p in model.named_parameters() :
        if 'bias' in name :
            nn.init.constant_(p, 0)

def bn_merge_biasless(seq, running_var_sqrt_eps=1e-5) :
    assert reg_bias(seq) < 1e-18
    with torch.no_grad() :
        for i in range(1, len(seq)) :
            if type(seq[i]) == nn.BatchNorm2d :
                assert type(seq[i - 1]) == nn.Conv2d
                channel_affine = (seq[i].weight / torch.sqrt(seq[i].running_var + running_var_sqrt_eps)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                
                seq[i - 1].weight *= channel_affine
                nn.init.constant_(seq[i].weight, 1.0)
                nn.init.constant_(seq[i].running_var, 1.0)
            

def detach_states_activations(model):
    # detach the neuron states and activations from current computation graph(necessary)
    for layer in model.modules():
        if isinstance(layer, sl.StatefulLayer):
            for name, buffer in layer.named_buffers():
                buffer.detach_()

def model_stage_num(model) :
    ret = 1
    if hasattr(model, "module") :
        model = model.module
    for m in model.net :
        if type(m).__name__[:9] == "TimeTrans" :
            ret += 1
    return ret

def set_time_config(model, time_config) :
    now_m_pos = 0
    if hasattr(model, "module") :
        model = model.module
    if type(time_config) is not list :
        time_config = [time_config] * model_stage_num(model)
    for m in model.net :
        if type(m).__name__[:9] == "TimeTrans" :
            m.update_T(time_config[now_m_pos], time_config[now_m_pos + 1])
            now_m_pos += 1

### Network Loading
def is_tfsnn_model(model : nn.Module) :
    for m in model.modules() :
        if type(m) is TimeTrans :
            return True
    return False

def is_tfsnn_state_dict(state_dict) :
    for m in state_dict.keys() :
        if 'TimeTrans' in m :
            return True
    return False

def load_snn(model, load_path):
    state_dict = remove_v_mem(torch.load(load_path)) ## remove neuron states
    state_dict_for_each_child = dict()
    for name, param in state_dict.items() :
        child_name = '.'.join(name.split('.')[:2])
        child_param_name = '.'.join(name.split('.')[2:])
        if state_dict_for_each_child.get(child_name) == None :
            state_dict_for_each_child[child_name] = dict()
        state_dict_for_each_child[child_name][child_param_name] = param
    
    state_dict_for_each_child = list(state_dict_for_each_child.values())

    pos = 0
    for i in range(len(model.net)) :
        if len(model.net[i].state_dict()) <= 0 :
            continue
        if not issubclass(type(model.net[i]), sl.IAF) :
            model.net[i].load_state_dict(state_dict_for_each_child[pos])
        pos += 1

    # model.load_state_dict(state_dict, strict=False)

### Regularizer

def reg_bias(model) :
    ret = 0
    for name, p in model.named_parameters() :
        if 'bias' in name :
            ret += (p ** 2).sum()
    return ret

### Loss functions

class CrossEntropyLoss_with_Temperature(nn.Module) :
    def __init__(self, T=1) :
        super().__init__()
        self.T = T
    
    def forward(self, x, y) :
        return torch.nn.functional.cross_entropy(x / self.T, y / self.T)

class SNN_Loss_Wrapper(nn.Module) :
    def __init__(self, criterion) :
        super().__init__()
        self.criterion = criterion
    
    def forward(self, input, target) :
        return self.criterion(input.mean(dim = 1), target)

class Proj_Time_Squeeze(nn.Module) :
    def __init__(self, criterion) :
        super().__init__()
        self.criterion = criterion
    
    def forward(self, input, target) :
        return self.criterion(input.mean(dim = 1), target.mean(dim = 1))


def remove_v_mem(state_dict) :
    new_dict = dict()
    for k, v in state_dict.items() :
        if 'v_mem' not in k :
            new_dict[k] = v
    return new_dict

### subs: [condition1, condition2, ...]
### condition: [substr1, substr2, ...]: strings included in name
def param_filter(named_params, subs:list=None, ex_subs:list=None) :
    ret = []
    for name, i in named_params :
        contains_all = True
        ex_any = False
        if subs is not None :
            for cond in subs :
                if type(cond) == str :
                    cond = [cond]
                for s in cond :
                    if s not in name :
                        contains_all = False
                        break
        if ex_subs is not None :
            for cond in ex_subs :
                if type(cond) == str :
                    cond = [cond]
                for s in cond :
                    if s in name :
                        ex_any = True
                        break

        if contains_all and not ex_any :
            ret.append(i)

    return ret

def lock_random_seed(seed) : 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

### Regularization

# class Regularizer_


### Time Config Sampler

class UniformTimeSampler :
    def __init__(self, model, T, T_L_radius, T_R_radius) :
        self.T = T
        self.T_min, self.T_max = T - T_L_radius, T + T_R_radius
        self.stage_num = model_stage_num(model)
        print(self.stage_num)
    
    def get_T(self) :
        return self.T
    
    def sample(self) :
        ret = [self.T]
        for i in range(self.stage_num - 1) :
            ret.append(random.randint(self.T_min, self.T_max))
        return ret


### Training

def fine_tune_mixed_timestep(model, train_loader, val_loader, loss_func, optimizer, time_config_sampler, forward_s, cal_iter=10, scheduler=None, epochs=10, lock_BN=False, reg=None, save_last=False, save_file_name='default_name.pt') :
    mx = 0
    save_state_dict = None

    for num_epochs in range(1, epochs + 1) :

        if lock_BN :
            model.eval()
        else :
            model.train()

        training_acc_record = []
        for x, label in tqdm(train_loader) :
            optimizer.zero_grad()
            x, label = x.cuda(), label.cuda()

            # loss = 0
            for _ in range(forward_s) :
                time_config = time_config_sampler.sample()
                set_time_config(model, time_config)
                o = model(x)

                loss = loss_func(o, label)
                if reg is not None :
                    loss += reg(model)
                detach_states_activations(model)
                loss.backward()   ### accumulate gradient

                training_acc_record.append((o.mean(dim=1).argmax(dim=1) == label).sum().item() / label.numel())
            
            optimizer.step()

        if scheduler is not None :
            scheduler.step()

        model.eval()
        total = 0
        correct = 0
        set_time_config(model, time_config_sampler.get_T())
        if cal_iter > 0 : ### Or momentum will be erased
            bn_calibrate(model, train_loader, cal_iter)

        # sinabs.reset_states(model)
        with torch.no_grad() :
            for x, label in tqdm(val_loader) :
                x, label = x.cuda(), label.cuda()
                o = model(x).sum(dim=1)
                total += label.numel()
                correct += (o.argmax(dim=1) == label).sum().item()
        
        training_acc = round(sum(training_acc_record) / len(training_acc_record) * 100, 2)
        val_acc = round(correct / total * 100, 2)

        if not save_last and val_acc > mx :
            mx = val_acc
            save_state_dict = copy.deepcopy(model.state_dict())

        print(f"Epoch: {num_epochs} Training Acc : {training_acc}  Validation Acc: {val_acc}")
        print(f"reg: {reg_bias(model)}")

    if save_last :
        save_state_dict = copy.deepcopy(model.state_dict())
    
    print(f"Max Test Acc: {mx}%")
    model.load_state_dict(save_state_dict)
    torch.save(save_state_dict, save_file_name)

    sinabs.reset_states(model)


def fine_tune(model, train_loader, val_loader, loss_func, optimizer, scheduler=None, epochs=10, lock_BN=False, reg=None, save_last=False, save_file_name='default_name.pt') :
    mx = 0
    save_state_dict = None

    for num_epochs in range(1, epochs + 1) :

        if lock_BN :
            model.eval()
        else :
            model.train()

        training_acc_record = []
        for x, label in tqdm(train_loader) :
            optimizer.zero_grad()
            x, label = x.cuda(), label.cuda()
            o = model(x)
            loss = loss_func(o, label)
            if reg is not None :
                loss += reg(model)

            detach_states_activations(model)
            # sinabs.reset_states(model)
            loss.backward()
            training_acc_record.append((o.mean(dim=1).argmax(dim=1) == label).sum().item() / label.numel())
            optimizer.step()

        if scheduler is not None :
            scheduler.step()

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad() :
            for x, label in tqdm(val_loader) :
                x, label = x.cuda(), label.cuda()
                o = model(x).sum(dim=1)
                total += label.numel()
                correct += (o.argmax(dim=1) == label).sum().item()
        
        training_acc = round(sum(training_acc_record) / len(training_acc_record) * 100, 2)
        val_acc = round(correct / total * 100, 2)

        if not save_last and val_acc > mx :
            mx = val_acc
            save_state_dict = copy.deepcopy(model.state_dict())

        print(f"Epoch: {num_epochs} Training Acc : {training_acc}  Validation Acc: {val_acc}")
        print(f"reg: {reg_bias(model)}")

    if save_last :
        save_state_dict = copy.deepcopy(model.state_dict())
    
    print(f"Max Test Acc: {mx}%")
    model.load_state_dict(save_state_dict)
    torch.save(save_state_dict, save_file_name)

    sinabs.reset_states(model)

def fine_tune_distiller(model, train_loader, val_loader, loss_func, optimizer, scheduler=None, epochs=10, lock_BN=False, reg=None, distill_func=None) :
    if distill_func == None :
        distill_func = loss_func
    
    for num_epochs in range(1, epochs + 1) :

        if lock_BN :
            model.student.eval()
        else :
            model.student.train()

        for x, label in tqdm(train_loader) :
            optimizer.zero_grad()
            x, label = x.cuda(), label.cuda()
            o, loss = model(x, distill_func)
            loss += loss_func(o, label)
            if reg is not None :
                loss += reg(model)
                
            detach_states_activations(model.student)
            loss.backward()
            optimizer.step()

        if scheduler is not None :
            scheduler.step()

        model.student.eval()
        total = 0
        correct = 0
        with torch.no_grad() :
            for x, label in tqdm(val_loader) :
                x, label = x.cuda(), label.cuda()
                o = model.student(x).mean(dim=1)
                total += label.numel()
                correct += (o.argmax(dim=1) == label).sum().item()
        
        print(f"Epoch: {num_epochs}  Validation Acc: {round(correct / total * 100, 2)}")
    sinabs.reset_states(model)

def bn_calibration_init(m):
    if getattr(m, 'track_running_stats', False):
        m.reset_running_stats()
        m.training = True
        m.momentum = None

def bn_calibrate(model, loader, cal_iter=0) :
    if cal_iter <= 0 :
        return
    model.train()
    model.apply(bn_calibration_init)
    with torch.no_grad() :
        for batch_idx, (input, labels) in enumerate(loader) :
            input, labels = input.cuda(), labels.cuda()
            if cal_iter != None and batch_idx >= cal_iter : 
                break
            model(input)
    model.eval()

### Test

def test_acc(model, val_loader) :
    sinabs.reset_states(model)
    with torch.no_grad() :
        model.eval()
        total = 0
        correct = 0
        for x, label in tqdm(val_loader) :
            x, label = x.cuda(), label.cuda()
            o = model(x).sum(dim=1)
            total += label.numel()
            correct += (o.argmax(dim=1) == label).sum().item()
            # break #############################################3
    return round(correct / total * 100, 2)