import torch
import torch.cuda.amp as amp
import time
import logging
import copy
import utils
import os
from my_parse import args
from models import TFSNN_ResNet18_DVS
from data.dvs_cifar10 import build_data as dvs_cifar10_build_data
from data.nc101 import build_data as nc101_build_data

if args.manual_seed != -1 :
    utils.lock_random_seed(args.manual_seed)

#### Mixed Precision
scaler = amp.GradScaler(enabled = args.mixed_precision)
if not os.path.exists(args.path_prefix) :
    os.makedirs(args.path_prefix)

#### data
if args.use_nc101 :
    train_loader, test_loader = nc101_build_data(args.batch_size, args.path_dataset)
else :
    train_loader, test_loader = dvs_cifar10_build_data(args.batch_size, args.path_dataset)

#### model
assert args.model == 'TFSNN_ResNet18_DVS'

model = TFSNN_ResNet18_DVS(args.category_num, args.time_res, args.Vth, args.decay, args.time_trans_mode).cuda()
if args.data_parallel : 
    model_wrapper = torch.nn.DataParallel(model)
    model = model_wrapper.module
else : model_wrapper = model

#### optimizer

if args.optimizer == 'SGD' : optimizer = torch.optim.SGD(params = model_wrapper.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = 0.9)
elif args.optimizer == 'AdamW' : optimizer = torch.optim.AdamW(params = model_wrapper.parameters(), lr = args.lr, weight_decay = args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = args.epochs)
loss_func = torch.nn.CrossEntropyLoss()

loss_func = utils.SNN_Loss(loss_func)

#### logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sHandler = logging.StreamHandler()
fHandler = logging.FileHandler(args.path_prefix + args.log_path, mode = 'w')
formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s", datefmt="%a %b %d %H:%M:%S %Y")
sHandler.setFormatter(formatter)
fHandler.setFormatter(formatter)
logger.addHandler(fHandler)
logger.addHandler(sHandler)

#### records

class record_manager() :

    def __init__(self, path, save_best_model = True, saving_period = -1) :
        self.acc_record = []
        self.loss_record = []
        self.lr_record = []
        self.max_acc = -1
        self.max_acc_epoch = -1
        self.best_model_state_dict = None
        self.save_best_model = save_best_model
        self.saving_period = saving_period  #if not -1, conduct periodic save
        self.state_dict_pool = None
        if self.saving_period > 0 :
            self.state_dict_pool = dict()
        self.path = path

    def update_record(self, model, now_epoch, acc, loss) :
        self.acc_record.append(acc)
        self.loss_record.append(loss)
        self.lr_record.append(scheduler.get_last_lr())
        if self.saving_period > 0 : ##周期性保存
            if now_epoch % self.saving_period == 0 :
                self.state_dict_pool[now_epoch] = copy.deepcopy(model.state_dict())
        if (acc > self.max_acc or acc < 0) : ##acc == -1 即刻保存
            self.max_acc = acc
            self.max_acc_epoch = now_epoch
            if (self.save_best_model) :
                self.best_model_state_dict = copy.deepcopy(model.state_dict())

    def save(self) :
        torch.save({
            'acc_record' : self.acc_record,
            'loss_record' : self.loss_record,
            'lr_record' : self.lr_record, 
            'max_acc' : self.max_acc,
            'max_acc_epoch' : self.max_acc_epoch,
            'best_model_state_dict' : self.best_model_state_dict,
            'state_dict_pool' : self.state_dict_pool,
        }, self.path)

records = record_manager(args.path_prefix + args.record_path, save_best_model = args.save_best_model, saving_period = args.saving_period)

if __name__ == "__main__" :
    logger.info('----- training info -----')
    logger.info(args)
    logger.info('----- model info -----')
    for i in model.info :
        logger.info(i)

    logger.info('----- start training -----')
    for now_epoch in range(1, args.epochs + 1) :
        epoch_start_time = time.time()

        #### training

        model_wrapper.train()
        mean_loss = utils.MTT_train_epoch(model_wrapper, train_loader, optimizer, scheduler, scaler, 
                                          loss_func, args.category_num, args.min_T, args.max_T, 
                                          args.sample_num, args.mixed_precision, DVS = True, group_block_num = args.group_block_num)


        #### testing

        model_wrapper.eval()
        model.reset_time_res([args.max_T] * len(model.feature_ext))
        utils.bn_calibrate(model_wrapper, train_loader, args.cal_iter, DVS = True)
        acc = utils.test_acc(model_wrapper, test_loader, DVS = True)

        records.update_record(model_wrapper, now_epoch, acc, mean_loss)

        epoch_end_time = time.time()
        logger.info(f"Epoch: {now_epoch} | Time elapsed: {epoch_end_time - epoch_start_time}s | Accuracy : {acc * 100}% | Loss : {mean_loss}")
    
    records.save()
    print(f"### Accuracy: {records.max_acc}")