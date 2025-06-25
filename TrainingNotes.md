## Setup
VastAI
  1x RTX 4090
  AMD Epic 7282 16-Core
  100GB RAM


### 21-02-25 
Training with standard parameters, just changed the `lr` and scheduler `max_lr` to `1e-5`

[Kaggle execution](https://www.kaggle.com/code/fabalcu97/notebook7a361dbffe/log?scriptVersionId=223739654) - 
[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/2lgpzu6t)

### 22-02-25
Picked up where the last training ended with ACC of `64.0290%`

[Kaggle execution](https://www.kaggle.com/code/fabalcu97/notebook7a361dbffe/log?scriptVersionId=223834882) - 
[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/yp5ibufd)


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
```

[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/0w0bz95l)
Accuracy: 68.0970 %

### 28-03-25
EmoNeXt training with new dataset.
```bash
python train.py --dataset-path='/workspace/givemefive-dataset/parsed-dataset' --batch-size=64 --lr=0.0001 --scheduler-max-lr=0.00001 --epochs=150 --amp --num-workers=8 --model-size='tiny' --patience=30
```

[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/ccjeevk1)

### 29-03-25
EmoNeXt training with new dataset and pre-trained model. No neutral emotion

```bash
python train.py --dataset-path='/workspace/givemefive-dataset/parsed-dataset' --batch-size=64 --lr=0.0001 --epochs=150 --amp --in_22k --num-workers=8 --model-size='tiny' --patience=30 --experiment-name='givemefive-dataset'
```
[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/z2outr71)

### 03-04-25
EmoxNet training with original dataset and pre-trained model. No neutral emotion.

```bash
python train.py --dataset-path='FER2013' --batch-size=64 --lr=0.0001 --epochs=150 --amp --in_22k --num-workers=8 --model-size='tiny' --patience=30 --experiment-name='FER2013_no_neutral'
```
[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/mdk4k69z)

ACC: 75.25%

### 03-04-25
EmoNeXt training with gimefive dataset and pre-trained model. No neutral emotion

```bash
python train.py --dataset-path='/workspace/givemefive-dataset/parsed-dataset' --batch-size=64 --lr=0.0001 --epochs=150 --amp --in_22k --num-workers=8 --model-size='tiny' --patience=30 --experiment-name='givemefive-dataset'
```
Results: https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/k4b93wds

ACC: 81.5%

# Experiment: `cbam_custom_implementation`
### Details
- All trainings are without neutral emotion
- All trainings are with pre-trained model
- All trainings are with the custom `cbam` block

### Runs
- [Original dataset](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/ya4txb7g) - Accuracy: 75.5316%
- [GimeFive dataset](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/o9mkmp79) - Accuracy: 79.7792%

# Experiment: `cbam_external_implementation`
### Details
- All trainings are without neutral emotion
- All trainings are with pre-trained model
- All trainings are with the [external `cbam` block](https://github.com/xmu-xiaoma666/External-Attention-pytorch)

### Runs
- [ConvNeXt - Tiny] - Accuracy: 71.99%
- [Original dataset w/neutral emotion 0-150epochs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/wytzhslv) - Accuracy: 72.3043%
- [Original dataset w/neutral emotion 150-300epochs]() - Accuracy: 
- [Original dataset w/o neutral emotion](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/hdk4q7zg) - Accuracy: 76.4765%
- [GimeFive dataset](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/7h9tt9o9) - Accuracy: 80.2352%



# New Runs

- [x] `python trainer.py --batch-size=128 --lr=0.0001 --epochs=150 --patience=15 --amp --in_22k --num-workers=8 --model-size=base --scheduler-max-lr=0.00001 --dataset=fer2013 --experiment-name=baseline_fer2013`
- [x] `python trainer.py --batch-size=128 --lr=0.0001 --epochs=150 --patience=15 --amp --in_22k --num-workers=8 --model-size=base --scheduler-max-lr=0.00001 --dataset=fer2013 --use-cbam --drop-path-rate=0.2 --experiment-name=cbam_fer2013`
- [x] `python trainer.py --batch-size=128 --lr=0.0001 --epochs=150 --patience=15 --amp --in_22k --num-workers=8 --model-size=base --scheduler-max-lr=0.00001 --dataset=ckplus --experiment-name=baseline_ckplus`
- [x] `python trainer.py --batch-size=128 --lr=0.0001 --epochs=150 --patience=15 --amp --in_22k --num-workers=8 --model-size=base --scheduler-max-lr=0.00001 --dataset=ckplus --use-cbam --drop-path-rate=0.2 --experiment-name=cbam_ckplus`
- [x] `python trainer.py --batch-size=128 --lr=0.0001 --epochs=150 --patience=15 --amp --in_22k --num-workers=8 --model-size=base --scheduler-max-lr=0.00001 --dataset=rafdb --experiment-name=baseline_rafdb`
- [x] `python trainer.py --batch-size=128 --lr=0.0001 --epochs=150 --patience=15 --amp --in_22k --num-workers=8 --model-size=base --scheduler-max-lr=0.00001 --dataset=rafdb --use-cbam --drop-path-rate=0.2 --experiment-name=cbam_rafdb`
