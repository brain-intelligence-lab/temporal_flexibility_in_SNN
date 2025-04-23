
dvs-gesture
```
//Train:
python train_student.py --dataset dvs_gesture --save_path DVSGESTURE_BASE_T40.pt --epochs 100 --T 40 --model_name dvs_gesture_tfsnn

//Bias-removal:
//MTT:
python train_student.py --phase bias_removal --dataset dvs_gesture --load_path DVSGESTURE_BASE_T40.pt --save_path MTT_DVSGESTURE_BASET40_woBN_AdamW20.pt --MTT True --T 40 --T_L_radius 39 --model_name dvs_gesture_tfsnn --fine_tune_epochs 20 --gpu 5

//SDT:
python train_student.py --phase bias_removal --dataset dvs_gesture --load_path DVSGESTURE_BASE_T40.pt --save_path SDT_DVSGESTURE_BASET40_woBN_AdamW20.pt --T 40 --model_name dvs_gesture_tfsnn --fine_tune_epochs 20 --gpu 4
```



cifar10-dvs

```
//Train:
python train_student.py --optimizer AdamW --lr 0.001 --weight_decay 0.02 --save_path SDT_CIFAR10DVS_BN_COSLR_VROLL20_MYVERTFLIP.pt --model_name VGG9_tfsnn_bn --dataset cifar10dvs --gpu 3

//Bias-removal:
//MTT:
python train_student.py --phase bias_removal --optimizer SGD --lr 0.001 --weight_decay 0.0005 --load_path ./SDT_CIFAR10DVS_BN_COSLR_VROLL20_MYVERTFLIP.pt --save_path MTT_FINETUNE_SGD_60.pt --gpu 6 --MTT True --model_name VGG9_tfsnn_bn

//SDT:
python train_student.py --phase bias_removal --optimizer SGD --lr 0.001 --weight_decay 0.0005 --load_path ./SDT_CIFAR10DVS_BN_COSLR_VROLL20_MYVERTFLIP.pt --save_path SDT_FINETUNE_SGD_60.pt --gpu 3 --model_name VGG9_tfsnn_bn
```

