## Training with MTT

Replace [path_of_dataset] with the path of the corresponding dataset

In the paper, seeds for each experiment are 1000, 2000, 3000.



##### CIFAR100 ResNet19

```
python train.py --path_dataset [path_of_dataset] --gpu 0,1,2 --data_parallel --saving_period -1 --category_num 100 --model TFSNN_ResNet19 --time_trans_mode 0 --time_res 6 6 6 6 6 6 6 6 --max_T 6 --min_T 1 --manual_seed 2000 --save_best_model --mixed_precision --sample_num 3 --path_prefix ./CIFAR100-ResNet19-sample3-300epochs-seed2000 --epochs 300
```



##### CIFAR10 ResNet19

```
python train.py --path_dataset [path_of_dataset] --gpu 0,1,2 --data_parallel --saving_period -1 --use_cifar10 --category_num 10 --model TFSNN_ResNet19 --time_trans_mode 0 --time_res 6 6 6 6 6 6 6 6 --max_T 6 --min_T 1 --manual_seed 2000 --save_best_model --mixed_precision --sample_num 3 --path_prefix ./CIFAR10-ResNet19-sample3-300epochs-seed2000 --epochs 300
```



##### CIFAR100 VGG14

```
python train.py --path_dataset [path_of_dataset] --gpu 2,3 --data_parallel --saving_period -1 --category_num 100 --model TFSNN_VGG14 --time_trans_mode 0 --time_res 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 --max_T 5 --min_T 1 --manual_seed 2000 --save_best_model --mixed_precision --sample_num 3 --path_prefix ./CIFAR100-VGG14-sample3-300epochs-seed2000 --epochs 300
```



##### CIFAR10-DVS ResNet18

```
python train-dvs.py --path_dataset [path_of_dataset] --batch_size 50 --gpu 0,1 --data_parallel --category_num 10 --model TFSNN_ResNet18_DVS --group_block_num 1 --sample_num 3 --max_T 10 --min_T 1 --time_res 10 10 10 10 10 10 10 10 --manual_seed 2000 --save_best_model --saving_period -1 --mixed_precision --path_prefix ./CIFAR10-DVS-ResNet18-sample3-300epochs-seed2000 --epochs 300
```



##### N-Caltech101 ResNet18

```
python train-dvs.py --path_dataset [path_of_dataset] --batch_size 50 --use_nc101 --category_num 101 --gpu 0,1 --model TFSNN_ResNet18_DVS --group_block_num 1 --sample_num 3 --max_T 10 --min_T 1 --time_res 10 10 10 10 10 10 10 10 --manual_seed 2000 --save_best_model --saving_period -1 --mixed_precision --path_prefix ./NC101-ResNet18-sample3-300epochs-seed2000 --epochs 300
```
