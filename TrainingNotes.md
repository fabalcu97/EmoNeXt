### 21-02-25 
Training with standard parameters, just changed the `lr` and scheduler `max_lr` to `1e-5`

[Kaggle execution](https://www.kaggle.com/code/fabalcu97/notebook7a361dbffe/log?scriptVersionId=223739654) - 
[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/2lgpzu6t)

### 22-02-25
Picked up where the last training ended with ACC of `64.0290%`

[Kaggle execution](https://www.kaggle.com/code/fabalcu97/notebook7a361dbffe/log?scriptVersionId=223834882) - 
[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/yp5ibufd)

```
Run history:
         Epoch ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
Train Accuracy ▁▂▁▃▂▂▃▂▃▂▃▃▄▄▄▅▃▅▄▅▅▅▆▅▆▅▇▆▆▇▆▇▆█▆█▇▇
    Train Loss █▇▇▆▇▇▆▇▆▆▅▆▅▅▅▄▅▄▅▄▄▄▃▃▃▄▂▃▂▂▃▂▃▁▂▁▁▁
  Val Accuracy ▄▄▃▄▅▄▄▄▂▅▅▄▄▃▅▆▆▆▅▆▁▄▄▅▄█▆▆▆▃▆▆▇▆▅▅█▆
      Val Loss ▃▂▃▃▃▂▄▃▅▃▂▃▃▅▃▃▃▂▁▁█▃▄▃▃▁▁▃▂▆▄▃▂▄▃▂▂▃

Run summary:
         Epoch 38
Train Accuracy 72.99131
    Train Loss 1.21648
  Val Accuracy 64.92059
      Val Loss 1.37431
```

### 23-02-25
Training with standard parameters, picked up where the last training ended with accuracy of `64.92059%`.
Modified:
* `early_stopping_patience=20`


[Kaggle execution](https://www.kaggle.com/code/fabalcu97/notebook7a361dbffe?scriptVersionId=224148651) -
[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/rd964rs7)


### 24-02-25
Training in VastAI. Standard parameters. Changed `lr` to `1e-5`.

[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/bik59ns6)

Completed with `64.17%` accuracy. Continued training with `max_lr` in the scheduler of `1e-6`

[Second WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/pyyxay0n)

Completed with `65.51%` accuracy. Continued training with increased patience (30) in the early stopping.

```bash
python train.py --dataset-path='FER2013' --batch-size=64 --lr=0.0001 --epochs=300 --amp --in_22k --num-workers=8 --model-size='tiny' --checkpoint /workspace/EmoNeXt/out/latest/second_checkpoint.pt --patience=30
```

[Third WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/psw83gix)

```
Run history:
         Epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇████
Train Accuracy ▁▁▂▁▁▂▂▃▂▂▃▃▃▄▃▄▃▄▄▄▄▅▄▅▄▅▆▆▆▆▇▆▆▇▇▇▇▇▇█
    Train Loss ████▇▇▇▆▇▆▆▆▅▆▅▅▅▅▄▅▄▄▄▃▅▃▃▃▂▂▃▂▂▂▂▂▁▁▂▁
  Val Accuracy ▂▂▂▄▂▃▄▅▂▂▅▂▃▁▄▇▂▃▅▆▅▁█▃▅▄▄▆▇▄▆▆▇▆▆▃▆▆▆▇
      Val Loss ▁▂▁▁▂▃▂▂▂▂▂▄▂▂▁▃▃▃▂▃▄▃▃█▄▂▅▃▃▄▄▄▆▄▅▄▅▅▆▅

Run summary:
         Epoch 73
Train Accuracy 79.46629
    Train Loss 1.12147
  Val Accuracy 66.03511
      Val Loss 1.40177
```
### 25-02-25
New training with standard parameters.

```bash
python train.py --dataset-path='FER2013' --batch-size=64 --lr=0.0001 --epochs=300 --amp --in_22k --num-workers=8 --model-size='tiny' --checkpoint /workspace/EmoNeXt/out/latest/second_checkpoint.pt --patience=50
```

[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/3wi3del2)

```bash
python train.py --dataset-path='FER2013' --batch-size=64 --lr=0.00001 --scheduler-max-lr=0.00001 --epochs=150 --amp --in_22k --num-workers=8 --model-size='tiny' --patience=30
```
[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/ysj776ra)

### 26-02-25
EmoNeXt training without pre-trained model. Standard parameters.
```bash
python train.py --dataset-path='FER2013' --batch-size=64 --lr=0.0001 --scheduler-max-lr=0.00001 --epochs=150 --amp --num-workers=8 --model-size='tiny' --patience=30 --checkpoint /workspace/EmoNeXt/out/latest/first_checkpoint_63_81.pt
```

### 26-02-25
EmoNeXt training without pre-trained model. Standard parameters.

```bash
python train.py --dataset-path='FER2013' --batch-size=64 --lr=0.0001 --scheduler-max-lr=0.00001 --epochs=150 --amp --num-workers=8 --model-size='tiny' --patience=30 --checkpoint=/workspace/EmoNeXt/out/latest/second_checkpoint_66_73.pt

python train.py --dataset-path='FER2013' --batch-size=64 --lr=0.0001 --scheduler-max-lr=0.00001 --epochs=150 --amp --num-workers=8 --model-size='tiny' --patience=30 --checkpoint=/workspace/EmoNeXt/out/latest/first_checkpoint_63_81.pt

python train.py --dataset-path='FER2013' --batch-size=64 --lr=0.0001 --scheduler-max-lr=0.00001 --epochs=150 --amp --num-workers=8 --model-size='tiny' --patience=30 --checkpoint=/workspace/EmoNeXt/out/latest/second_checkpoint_66_73.pt
```

### 28-03-25
EmoNeXt training with new dataset.
```bash
python train.py --dataset-path='/workspace/givemefive-dataset/parsed-dataset' --batch-size=64 --lr=0.0001 --scheduler-max-lr=0.00001 --epochs=150 --amp --num-workers=8 --model-size='tiny' --patience=30
```

Results: https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/ccjeevk1

### 29-03-25
EmoNeXt training with new dataset and pre-trained model

```bash
python train.py --dataset-path='/workspace/givemefive-dataset/parsed-dataset' --batch-size=64 --lr=0.0001 --epochs=150 --amp --in_22k --num-workers=8 --model-size='tiny' --patience=30
```
Results: https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/z2outr71
