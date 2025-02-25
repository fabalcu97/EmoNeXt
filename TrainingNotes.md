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
Trining in VastAI. Standard parameters just changed the `lr` to `1e-5`.

[WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/bik59ns6)

Completed with `64.17%` accuracy. Continued training with `max_lr` in the scheduler of `1e-6`

[Second WandB logs](https://wandb.ai/fabalcu97-personal/EmoNeXt/runs/pyyxay0n)

Completed with `65.51%` accuracy. Continued training with increased patience (30) in the early stopping.
