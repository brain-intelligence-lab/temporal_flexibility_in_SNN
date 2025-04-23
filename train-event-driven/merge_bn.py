import torch
import torch.nn as nn
import utils
from sinabs import reset_states
from my_parser import parse_args
from tqdm import tqdm
from data import DVS_Gesture_loader
from utils import fine_tune, SNN_Loss_Wrapper, param_filter, remove_v_mem, bias_remove, bn_merge_biasless, test_acc
from models.snn import TA_GestureNet
from models.snn_speck import dvs_gesture_snn_32x32, dvs_gesture_snn
import os

args = parse_args()
utils.lock_random_seed(args.seed)


# model = TA_GestureNet(T=args.T).cuda()
model = dvs_gesture_snn(batch_size=args.batch_size, width=1).cuda()
model.load_state_dict(remove_v_mem(torch.load('./model_bn_removal.pt')), strict=False)
bias_remove(model)

train_loader, val_loader = DVS_Gesture_loader(root_dir=args.path_dataset, test_only=False, \
                run_batch=True, batch_size=args.batch_size, T=args.T, data_aug=False)

reset_states(model)
print(test_acc(model, val_loader))
bn_merge_biasless(model.net)
reset_states(model)
print(test_acc(model, val_loader))
torch.save(model.state_dict(), './_model_bn_removal.pt')



