import os
import torch
import wandb
import random
import numpy as np

from torchinfo import summary
from dotenv import load_dotenv

from models import get_model
from train.train import Trainer
from train.configuration import parse_arguments
from train.utils import get_data_loaders, get_datasets

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

load_dotenv()

seed = 2001
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    configuration = parse_arguments()

    train_dataset, val_dataset, test_dataset = get_datasets(configuration)
    train_loader, val_loader, test_loader = get_data_loaders(
        configuration, train_dataset, val_dataset, test_dataset)

    exec_name = f"{configuration.model_size}_{configuration.experiment_name}"
    net = get_model(len(train_dataset.classes), configuration=configuration)

    if configuration.show_summary:
        print(train_dataset.classes)
        summary(net)
        os._exit(0)

    run = wandb.init(
        resume='auto',
        name=exec_name,
        anonymous='never',
        config=configuration,
        project="emonext-cbam",
    )
    run.watch(net, log="all")

    try:
        Trainer(
            model=net,
            lr=configuration.lr,
            wandb_run=run,
            amp=configuration.amp,
            execution_name=exec_name,
            classes=train_dataset.classes,
            testing_dataloader=test_loader,
            max_epochs=configuration.epochs,
            training_dataloader=train_loader,
            validation_dataloader=val_loader,
            ema_decay=configuration.ema_decay,
            output_dir=configuration.output_dir,
            checkpoint_path=configuration.checkpoint,
            early_stopping_patience=configuration.patience,
            ema_update_every=configuration.ema_update_every,
            scheduler_max_rate=configuration.scheduler_max_lr,
            gradient_accumulation_steps=configuration.gradient_accumulation_steps,
        ).run()
    except Exception as e:
        raise e
    finally:
        run.finish()
