import torch
import torch.nn as nn
import sinabs.layers as sl
import random
import torch.nn.functional as F
from math import floor
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from sinabs.activation import MembraneSubtract, MultiSpike, MembraneReset, SingleSpike
from sinabs.from_torch import from_model

def overlap(L1, R1, L2, R2) :
    return max(min(R1, R2) - max(L1, L2), 0)

def SoftTTransMatrix(in_T : int, out_T : int) : 
    mat = torch.zeros((in_T, out_T))
    out_T_size = in_T / out_T
    for i in range(in_T) : ## enumerate in_T
        out_Li = floor(i / out_T_size)
        out_Ri = floor((i + 1) / out_T_size)
        out_L = out_Li * out_T_size
        out_R = out_Ri * out_T_size
        for j in range(out_Li, min(out_Ri + 1, out_T)) :
            mat[i][j] = overlap(j * out_T_size, (j + 1) * out_T_size, i, i + 1) # + 0.4 * torch.randn(1).item()
    return mat

def HardTTransTable(in_T : int, out_T : int) : ## trans_table[t_out] = [matched t_in]
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

class TimeTrans_Soft(nn.Module) :
    def __init__(self, in_T=3, out_T=3) :
        super().__init__()
        self.in_T = in_T
        self.out_T = out_T
        self.mat = SoftTTransMatrix(in_T, out_T).cuda()

    def update_T(self, in_T, out_T) :
        self.in_T, self.out_T = in_T, out_T
        if hasattr(self.mat, "device") :
            self.mat = SoftTTransMatrix(in_T, out_T).to(self.mat.device)
        return 

    def forward(self, x) : ####
        assert (x.shape[0] % self.in_T == 0)
        x = x.reshape(x.shape[0] // self.in_T, self.in_T, x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 3, 4, 1)
        x = x @ self.mat
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        return x

class TimeTrans_Hard(nn.Module) :
    def __init__(self, in_T=1, out_T=1) :
        super().__init__()
        self.in_T = in_T
        self.out_T = out_T
        self.trans_table = None

    def update_T(self, in_T, out_T) :
        self.in_T, self.out_T = in_T, out_T

    def forward(self, x) : ####
        if(self.in_T == self.out_T) : 
            return x
        o = []
        self.trans_table = HardTTransTable(self.in_T, self.out_T)
        # print(f"in: {self.in_T} out: {self.out_T}")

        assert (x.shape[0] % self.in_T == 0)
        x = x.reshape(x.shape[0] // self.in_T, self.in_T, x.shape[1], x.shape[2], x.shape[3])

        for i in range(self.out_T) :
            o.append(x[:, self.trans_table[i][0], ...].clone())
            for j in range(1, len(self.trans_table[i])) :
                o[i] += x[:, self.trans_table[i][j], ...]
        x = torch.stack(o, dim = 1)

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        return x

class TimeTrans_HardTrim(nn.Module) :
    def __init__(self, in_T=1, out_T=1) :
        super().__init__()
        self.in_T = in_T
        self.out_T = out_T
        self.trans_table = None

    def update_T(self, in_T, out_T) :
        self.in_T, self.out_T = in_T, out_T

    def forward(self, x) : ####
        if(self.in_T == self.out_T) : 
            return x

        o = []
        self.trans_table = HardTTransTable(self.in_T, self.out_T)

        assert (x.shape[0] % self.in_T == 0)
        x = x.reshape(x.shape[0] // self.in_T, self.in_T, x.shape[1], x.shape[2], x.shape[3])

        if(self.in_T > self.out_T) :
            x = x[:, :self.out_T, ...]
        else :
            for i in range(self.out_T) :
                o.append(x[:, self.trans_table[i][0], ...].clone())
                for j in range(1, len(self.trans_table[i])) :
                    o[i] += x[:, self.trans_table[i][j], ...]
            x = torch.stack(o, dim = 1)

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        return x

TimeTrans = TimeTrans_Hard ## using hard trans


#############################
###
### My Neurons
###
class Act(torch.autograd.Function) :

    @staticmethod
    def forward(ctx, input) :
        ctx.save_for_backward(input)
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, backGrad) :
        input, = ctx.saved_tensors
        return backGrad * torch.clamp(1.0 - abs(input), min = 0)

act_fun = Act.apply

def update(x, mem, Vth, decay) :
    mem = mem * decay + x
    spike = act_fun(mem - Vth)
    mem = (1.0 - spike) * mem
    return mem, spike

class IF(nn.Module) :
    def __init__(self, batch_size, Vth=1, decay=1) :
        super().__init__()
        self.Vth = Vth
        self.decay = decay
        self.batch_size = batch_size

    def forward(self, x) :
        T = x.shape[0] // self.batch_size
        x = x.reshape(self.batch_size, T, x.shape[1], x.shape[2], x.shape[3])

        o = []
        u = 0 
        for i in range(T) :
            u, s = update(x[:, i, ...], u, self.Vth, self.decay)
            o.append(s)
        o = torch.stack(o, dim = 1)

        o = o.flatten(0, 1)
        return o




#############################
###
### From WS 
### link: https://arxiv.org/abs/1903.10520
###
class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



#############################
###
### From WS with Affine
### link: https://arxiv.org/abs/1903.10520
###
class WSAFConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.aff_k = nn.Parameter(torch.ones(out_channels))
        self.aff_b = nn.Parameter(torch.zeros(out_channels))
        # nn.init.constant()

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        weight = self.aff_k[:, None, None, None] * weight + self.aff_b[:, None, None, None]
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

#############################


class dvs_gesture_snn_small(nn.Module):
    def __init__(self, category_num=11, batch_size=25):
        super().__init__()
        self.net = nn.Sequential(
            
            nn.Conv2d(2, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  
            # nn.BatchNorm2d(8),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=MembraneReset(), spike_fn = SingleSpike),
            nn.AvgPool2d(kernel_size=(2, 2)), 

            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), 
            # nn.BatchNorm2d(16),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=MembraneReset(), spike_fn = SingleSpike),
            nn.AvgPool2d(kernel_size=(2, 2)),  


            nn.Flatten(),
            # nn.Dropout2d(0.5),
            nn.Linear(16 * 8 * 8, 64, bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=MembraneReset(), spike_fn = SingleSpike),
            # nn.Dropout2d(0.5),
            nn.Linear(64, category_num, bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=MembraneReset(), spike_fn = SingleSpike),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, t_len, -1)

        return out

class dvs_gesture_tfsnn_wobn(nn.Module):
    def __init__(self, category_num=11, batch_size=25, width=1, reset_fn=MembraneSubtract(), spike_fn=MultiSpike):
        super().__init__()
        self.net = nn.Sequential(
            TimeTrans(),

            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(2, 8 * width, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(8 * width, 16 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),

            nn.AvgPool2d(kernel_size=(2, 2)),  
            nn.Conv2d(16 * width, 32 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.AvgPool2d(kernel_size=(2, 2)),  
            nn.Conv2d(32 * width, 32 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            # nn.BatchNorm2d(32 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),

            nn.Flatten(),
            # nn.Dropout2d(0.5),
            nn.Linear(32 * 4 * 4 * width, 64, bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            # nn.Dropout2d(0.5),
            nn.Linear(64, category_num, bias=False),
            # sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, -1)

        return out

class dvs_gesture_tfsnn(nn.Module):
    def __init__(self, category_num=11, batch_size=25, width=1, reset_fn=MembraneSubtract(), spike_fn=MultiSpike):
        super().__init__()
        self.net = nn.Sequential(
            TimeTrans_HardTrim(),

            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(2, 8 * width, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  
            nn.BatchNorm2d(8 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(8 * width, 16 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            nn.BatchNorm2d(16 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),

            nn.AvgPool2d(kernel_size=(2, 2)),  
            nn.Conv2d(16 * width, 32 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            nn.BatchNorm2d(32 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.AvgPool2d(kernel_size=(2, 2)),  
            nn.Conv2d(32 * width, 32 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            nn.BatchNorm2d(32 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),

            nn.Flatten(),
            # nn.Dropout2d(0.5),
            nn.Linear(32 * 4 * 4 * width, 64, bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            # nn.Dropout2d(0.5),
            nn.Linear(64, category_num, bias=False),
            # sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, -1)

        return out


class dvs_gesture_snn(nn.Module):
    def __init__(self, category_num=11, batch_size=25, width=1, reset_fn=MembraneSubtract(), spike_fn=MultiSpike):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(2, 8 * width, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),  
            nn.BatchNorm2d(8 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(8 * width, 16 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            nn.BatchNorm2d(16 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.AvgPool2d(kernel_size=(2, 2)),  
            nn.Conv2d(16 * width, 32 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            nn.BatchNorm2d(32 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.AvgPool2d(kernel_size=(2, 2)),  
            nn.Conv2d(32 * width, 32 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            nn.BatchNorm2d(32 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.Flatten(),
            # nn.Dropout2d(0.5),
            nn.Linear(32 * 4 * 4 * width, 64, bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            # nn.Dropout2d(0.5),
            nn.Linear(64, category_num, bias=False),
            # sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, t_len, -1)

        return out

class dvs_gesture_snn_32x32(nn.Module):
    def __init__(self, category_num=11, batch_size=25, width=1, reset_fn=MembraneSubtract(), spike_fn=MultiSpike):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4, 4)),
            nn.Conv2d(2, 32 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  
            nn.BatchNorm2d(32 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.Conv2d(32 * width, 32 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            nn.BatchNorm2d(32 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.Conv2d(32 * width, 32 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            nn.BatchNorm2d(32 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32 * width, 64 * width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
            nn.BatchNorm2d(64 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.AvgPool2d(kernel_size=(4, 4)),
            nn.Conv2d(64 * width, 64 * width, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False), 
            nn.BatchNorm2d(64 * width),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            nn.Flatten(),
            # nn.Dropout2d(0.5),
            nn.Linear(64 * 4 * 4 * width, 64, bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            # nn.Dropout2d(0.5),
            nn.Linear(64, category_num, bias=False),
            # sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, t_len, -1)

        return out

class VGG9(nn.Module):
    def __init__(self, input_size, category_num=11, batch_size=25, reset_fn=MembraneSubtract(), spike_fn=MultiSpike):
        super().__init__()
        self.net = nn.Sequential(
            
            nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.AvgPool2d(kernel_size=(2, 2)), 

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.AvgPool2d(kernel_size=(2, 2)), 

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.AvgPool2d(kernel_size=(2, 2)), 

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.AvgPool2d(kernel_size=(2, 2)), 

            nn.Flatten(),

            nn.Linear(input_size * input_size // 16, category_num, bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, t_len, -1)

        return out

class VGG9_tfsnn_bn_meminit(nn.Module):
    def __init__(self, input_size, category_num=11, batch_size=25, reset_fn=MembraneSubtract(), spike_fn=MultiSpike):
        super().__init__()
        W = input_size // 64
        self.net = nn.Sequential(

            nn.AvgPool2d(kernel_size=(4, 4)), # (128 * 128 -> 32 * 32)
            TimeTrans(),
            nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  
            nn.BatchNorm2d(64),
            IF(batch_size=batch_size),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            IF(batch_size=batch_size),

            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            IF(batch_size=batch_size),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            IF(batch_size=batch_size),

            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            IF(batch_size=batch_size),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            IF(batch_size=batch_size),
            
            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            IF(batch_size=batch_size),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            IF(batch_size=batch_size),
            nn.AvgPool2d(kernel_size=(2, 2)), 

            nn.Flatten(),

            nn.Linear(W * W * 512, category_num, bias=False),
            # IF(batch_size=batch_size),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, -1)

        return out

class VGG9_tfsnn_bn(nn.Module):
    def __init__(self, input_size, category_num=11, batch_size=25, reset_fn=MembraneSubtract(), spike_fn=MultiSpike):
        super().__init__()
        W = input_size // 64
        self.net = nn.Sequential(

            nn.AvgPool2d(kernel_size=(4, 4)), # (128 * 128 -> 32 * 32)
            TimeTrans(),
            nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  
            nn.BatchNorm2d(64),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            
            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.AvgPool2d(kernel_size=(2, 2)), 

            nn.Flatten(),

            nn.Linear(W * W * 512, category_num, bias=False),
            # sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, -1)

        return out

class VGG9_tfsnn_wobn(nn.Module):
    def __init__(self, input_size, category_num=11, batch_size=25, reset_fn=MembraneSubtract(), spike_fn=MultiSpike):
        super().__init__()
        W = input_size // 64
        self.net = nn.Sequential(

            nn.AvgPool2d(kernel_size=(4, 4)), # (128 * 128 -> 32 * 32)
            TimeTrans(),
            nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            
            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.AvgPool2d(kernel_size=(2, 2)), 

            nn.Flatten(),

            nn.Linear(W * W * 512, category_num, bias=False),
            # sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, -1)

        return out

class VGG9_tfsnn_WS(nn.Module):
    def __init__(self, input_size, category_num=11, batch_size=25, reset_fn=MembraneSubtract(), spike_fn=MultiSpike):
        super().__init__()
        W = input_size // 64
        self.net = nn.Sequential(

            nn.AvgPool2d(kernel_size=(4, 4)), # (128 * 128 -> 32 * 32)
            TimeTrans(),
            WSConv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            WSConv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            WSConv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            WSConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            WSConv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            WSConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            
            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            WSConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            WSConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.AvgPool2d(kernel_size=(2, 2)), 

            nn.Flatten(),

            nn.Linear(W * W * 512, category_num, bias=False),
            # sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, -1)

        return out

class VGG9_tfsnn_WSAF(nn.Module):
    def __init__(self, input_size, category_num=11, batch_size=25, reset_fn=MembraneSubtract(), spike_fn=MultiSpike):
        super().__init__()
        W = input_size // 64
        self.net = nn.Sequential(

            nn.AvgPool2d(kernel_size=(4, 4)), # (128 * 128 -> 32 * 32)
            TimeTrans(),
            WSAFConv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            WSAFConv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            WSAFConv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            WSAFConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),

            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            WSAFConv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            WSAFConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            
            TimeTrans(),
            nn.AvgPool2d(kernel_size=(2, 2)), 
            WSAFConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            WSAFConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
            nn.AvgPool2d(kernel_size=(2, 2)), 

            nn.Flatten(),

            nn.Linear(W * W * 512, category_num, bias=False),
            # sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, reset_fn=reset_fn, spike_fn = spike_fn),
        )
        # self.vote_layer = nn.AvgPool1d(10, 10)
    
    
    def forward(self, x):
        (batch_size, t_len, channel,  height, width) = x.shape
        x = x.reshape((batch_size * t_len, channel, height, width))
        out = self.net(x)
        # out = self.vote_layer(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, -1)

        return out