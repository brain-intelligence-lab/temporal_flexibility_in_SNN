import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import utils

class Act(torch.autograd.Function) :

    @staticmethod
    def forward(ctx, input) :
        ctx.save_for_backward(input)
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, backGrad) :
        input, = ctx.saved_tensors
        return backGrad * torch.clamp(1.0 - abs(input), min = 0)

act_fun = Act.apply;

def update(x, mem, Vth, decay) :
    mem = mem * decay + x
    spike = act_fun(mem - Vth)
    mem = (1.0 - spike) * mem
    return mem, spike

class SpikeLayer(nn.Module) :
    def __init__(self, Vth, decay) :
        super().__init__()
        self.Vth = Vth
        self.decay = decay

    def forward(self, x, now_T) :
        o = []
        u = 0 
        for i in range(now_T) :
            u, s = update(x[:, i, ...], u, self.Vth, self.decay)
            o.append(s)
        o = torch.stack(o, dim = 1)
        return o

class ConvertToTimeMod(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1: self.module = args[0]
        else: self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class TimeTrans(nn.Module) :
    def __init__(self, mode = False) :
        super().__init__()
        self.out_T = None
        self.trans_table = None
        self.mode = mode

    def forward(self, x, out_T) : ####
        in_T = x.shape[1]
        if(in_T == out_T) : 
            return x
        o = []
        if self.mode == 0 :
            self.trans_table = utils.table_spwn(in_T, out_T)

        for i in range(out_T) :
            o.append(x[:, self.trans_table[i][0], ...].clone())
            for j in range(1, len(self.trans_table[i])) :
                o[i] += x[:, self.trans_table[i][j], ...]
        o = torch.stack(o, dim = 1)
        return o

class SResBlock(nn.Module) :
    def __init__(self, in_channels, out_channels, max_T, Vth, decay, down_sample = False, 
                 time_trans = True, time_trans_mode : int = 0) :
        super().__init__()
        self.max_T = max_T
        self.now_T = max_T
        if(time_trans) :
            self.time_trans = TimeTrans(time_trans_mode)
        else: self.time_trans = None
        s1 = 2 if down_sample else 1
        self.shortcut = (in_channels != out_channels | down_sample)
        if(self.shortcut) :
            self.shortcut = ConvertToTimeMod(nn.Conv2d(in_channels, out_channels, 1, s1, 0, bias = False),
                                             nn.BatchNorm2d(out_channels))
        self.conv1 = ConvertToTimeMod(nn.Conv2d(in_channels, out_channels, 3, s1, 1, bias = False),
                                      nn.BatchNorm2d(out_channels))
        self.spike1 = SpikeLayer(Vth, decay)
        self.conv2 = ConvertToTimeMod(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
                                      nn.BatchNorm2d(out_channels))
        self.spike2 = SpikeLayer(Vth, decay)
    
    def forward(self, x) :
        if(self.time_trans != None) :
            x = self.time_trans(x, self.now_T)
        _x = x.clone() 
        if(self.shortcut != False) :
            _x = self.shortcut(x)
        x = self.conv1(x)
        x = self.spike1(x, self.now_T)
        x = self.conv2(x)
        x = x + _x
        x = self.spike2(x, self.now_T)
        return x

class TFSNN_ResNet18(nn.Module) :
    def __init__(self, category_num, time_res : list, Vth, decay, time_trans_mode = 0) :
        super().__init__()
        self.info = ['ResNet18 | TFSNN_ResNet18', 
                    f'TimeRes={str(time_res)}',
                    f'category_num={category_num}']
        
        self.conv = ConvertToTimeMod(nn.Conv2d(3, 64, 3, 1, 1, bias = False), nn.BatchNorm2d(64))
        conv_info = [ # [inC, outC, downSample, timeTransform]
            [64, 64, False, False], [64, 64, False, True],
            [64, 128, True, True], [128, 128, False, True],
            [128, 256, True, True], [256, 256, False, True],
            [256, 512, True, True], [512, 512, False, True],
        ]
        modules = []
        module_num = len(conv_info)
        for i in range(module_num) :
            modules.append(SResBlock(conv_info[i][0], conv_info[i][1], time_res[i], Vth, decay, conv_info[i][2], conv_info[i][3], time_trans_mode))
        self.spike = SpikeLayer(Vth, decay)
        self.feature_ext = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ConvertToTimeMod(nn.Linear(512, category_num))

    @property
    def input_time_res(self) :
        return self.feature_ext[0].now_T

    @property
    def time_res(self) :
        return [i.now_T for i in self.feature_ext]

    def reset_time_res(self, time_res : list) :
        for i in range(len(self.feature_ext)) :
            self.feature_ext[i].now_T = time_res[i]

    def forward(self, x, remain_dim_T = False) :
        x = self.conv(x)
        x = self.spike(x, self.input_time_res)
        x = self.feature_ext(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        x = x if remain_dim_T else x.mean(dim = 1)
        return x,

class TFSNN_ResNet34(nn.Module) :
    def __init__(self, category_num, time_res : list, Vth, decay, time_trans_mode = 0) :
        super().__init__()
        self.info = ['ResNet34 | TFSNN_ResNet34', 
                    f'TimeRes={str(time_res)}',
                    f'category_num={category_num}']
        
        self.conv = ConvertToTimeMod(nn.Conv2d(3, 64, 3, 1, 1, bias = False), nn.BatchNorm2d(64))
        conv_info = [ # [inC, outC, downSample, timeTransform]
            [64, 64, False, False], [64, 64, False, True], [64, 64, False, True],
            [64, 128, True, True], [128, 128, False, True], [128, 128, False, True], [128, 128, False, True],
            [128, 256, True, True], [256, 256, False, True], [256, 256, False, True], [256, 256, False, True], [256, 256, False, True], [256, 256, False, True],
            [256, 512, True, True], [512, 512, False, True], [512, 512, False, True]
        ]
        modules = []
        module_num = len(conv_info)
        for i in range(module_num) :
            modules.append(SResBlock(conv_info[i][0], conv_info[i][1], time_res[i], Vth, decay, conv_info[i][2], conv_info[i][3], time_trans_mode))
        self.spike = SpikeLayer(Vth, decay)
        self.feature_ext = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ConvertToTimeMod(nn.Linear(512, category_num))

    @property
    def input_time_res(self) :
        return self.feature_ext[0].now_T

    @property
    def time_res(self) :
        return [i.now_T for i in self.feature_ext]

    def reset_time_res(self, time_res : list) :
        for i in range(len(self.feature_ext)) :
            self.feature_ext[i].now_T = time_res[i]

    def forward(self, x, remain_dim_T = False) :
        x = self.conv(x)
        x = self.spike(x, self.input_time_res)
        x = self.feature_ext(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        x = x if remain_dim_T else x.mean(dim = 1)
        return x,

## ResNet19
class TFSNN_ResNet19(nn.Module) :
    def __init__(self, category_num, time_res : list, Vth, decay, time_trans_mode = 0) :
        super().__init__()
        self.info = ['ResNet19 | TFSNN_ResNet19', 
                    f'TimeRes={str(time_res)}',
                    f'category_num={category_num}']
        
        self.conv = ConvertToTimeMod(nn.Conv2d(3, 128, 3, 1, 1, bias = False), nn.BatchNorm2d(128))
        conv_info = [ # [inC, outC, downSample, timeTransform]
            [128, 128, False, False], [128, 128, False, True], [128, 128, False, True],
            [128, 256, True, True], [256, 256, False, True], [256, 256, False, True],
            [256, 512, True, True], [512, 512, False, True],
        ]
        modules = []
        module_num = len(conv_info)
        for i in range(module_num) :
            modules.append(SResBlock(conv_info[i][0], conv_info[i][1], time_res[i], Vth, decay, conv_info[i][2], conv_info[i][3], time_trans_mode))
        self.spike = SpikeLayer(Vth, decay)
        self.feature_ext = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ConvertToTimeMod(nn.Linear(512, category_num))

    @property
    def input_time_res(self) :
        return self.feature_ext[0].now_T

    @property
    def time_res(self) :
        return [i.now_T for i in self.feature_ext]

    def reset_time_res(self, time_res : list) :
        for i in range(len(self.feature_ext)) :
            self.feature_ext[i].now_T = time_res[i]

    def forward(self, x, remain_dim_T = False) :
        x = self.conv(x)
        x = self.spike(x, self.input_time_res)
        x = self.feature_ext(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        x = x if remain_dim_T else x.mean(dim = 1)
        return x,


class TFSNN_ResNet18_DVS(nn.Module) :
    def __init__(self, category_num, time_res : list, Vth, decay, time_trans_mode = 0) :
        super().__init__()
        self.info = ['ResNet18 | TFSNN_ResNet18_DVS', 
                    f'TimeRes={str(time_res)}',
                    f'category_num={category_num}']
        
        self.conv = ConvertToTimeMod(nn.Conv2d(2, 64, 3, 1, 1, bias = False), nn.BatchNorm2d(64))
        conv_info = [ # [inC, outC, downSample, timeTransform]
            [64, 64, False, True], [64, 64, False, True],
            [64, 128, True, True], [128, 128, False, True],
            [128, 256, True, True], [256, 256, False, True],
            [256, 512, True, True], [512, 512, False, True],
        ]
        modules = []
        module_num = len(conv_info)
        for i in range(module_num) :
            modules.append(SResBlock(conv_info[i][0], conv_info[i][1], time_res[i], Vth, decay, conv_info[i][2], conv_info[i][3], time_trans_mode))
        self.spike = SpikeLayer(Vth, decay)
        self.feature_ext = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ConvertToTimeMod(nn.Linear(512, category_num))

    @property
    def input_time_res(self) :
        return self.feature_ext[0].now_T

    @property
    def time_res(self) :
        return [i.now_T for i in self.feature_ext]

    def reset_time_res(self, time_res : list) :
        for i in range(len(self.feature_ext)) :
            self.feature_ext[i].now_T = time_res[i]

    def forward(self, x, remain_dim_T = False) :
        x = self.conv(x)
        x = self.spike(x, self.input_time_res)
        x = self.feature_ext(x)
        #print(self.feature_ext[7].conv1.module[1].running_mean.mean())
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        x = x if remain_dim_T else x.mean(dim = 1)
        return x,

class SConv(nn.Module) :
    def __init__(self, in_channels, out_channels, max_T, Vth, decay, pool,
                 time_trans = True, time_trans_mode : int = 0) :
        super().__init__()
        self.max_T = max_T
        self.now_T = max_T
        if(time_trans) :
            self.time_trans = TimeTrans(time_trans_mode)
        else: self.time_trans = None
        self.conv = ConvertToTimeMod(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
                                      nn.BatchNorm2d(out_channels))
        self.spike = SpikeLayer(Vth, decay)
        self.pool = None
        if pool == True :
            self.pool = ConvertToTimeMod(nn.AvgPool2d(2))
    
    def forward(self, x) :
        if(self.time_trans != None) :
            x = self.time_trans(x, self.now_T)
        x = self.conv(x)
        x = self.spike(x, self.now_T)
        if(self.pool != None) :
            x = self.pool(x)
        return x

class TFSNN_VGG14(nn.Module) :
    def __init__(self, category_num, time_res : list, Vth, decay, time_trans_mode) :
        super().__init__()
        self.info = ['VGG14 | TFSNN_VGG14', 
                    f'TimeRes={str(time_res)}',
                    f'category_num={category_num}']

        conv_info = [ # [inC, outC, pool, timeTransform]
            [3, 64, False, False], [64, 64, True, True],
            [64, 128, False, True], [128, 128, True, True],
            [128, 256, False, True], [256, 256, False, True], [256, 256, True, True],
            [256, 512, False, True], [512, 512, False, True], [512, 512, True, True],
            [512, 512, False, True], [512, 512, False, True], [512, 512, True, True],
        ]
        modules = []
        module_num = len(conv_info)
        for i in range(module_num) :
            modules.append(SConv(conv_info[i][0], conv_info[i][1], time_res[i], Vth, decay, conv_info[i][2], conv_info[i][3], time_trans_mode))
        self.spike = SpikeLayer(Vth, decay)
        self.feature_ext = nn.Sequential(*modules)
        self.fc3 = ConvertToTimeMod(nn.Linear(512, category_num))


    @property
    def input_time_res(self) :
        return self.feature_ext[0].now_T

    @property
    def time_res(self) :
        return [i.now_T for i in self.feature_ext]

    def reset_time_res(self, time_res : list) :
        L = len(self.feature_ext)
        for i in range(L) :
            self.feature_ext[i].now_T = time_res[i]

    def forward(self, x, remain_dim_T = False) :
        x = self.feature_ext(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc3(x)
        x = x if remain_dim_T else x.mean(dim = 1)
        return x,