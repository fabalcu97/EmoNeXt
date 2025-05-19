import torch
import wandb
# import torchvision
import numpy as np

import torchvision
from pathlib import Path
from ema_pytorch import EMA
from torch.optim import AdamW
from wandb.wandb_run import Run
from torchvision import transforms
from tqdm.contrib import tenumerate
from torch.utils.data import DataLoader


from .utils import get_device
from .notifications.main import Notifier
from .scheduler import CosineAnnealingWithWarmRestartsLR


class Trainer:
    def __init__(
        self,
        classes: list,
        wandb_run: Run,
        output_dir: str,
        model: torch.nn.Module,
        testing_dataloader: DataLoader,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        lr: float = 1e-4,
        amp: bool = False,
        execution_name=None,
        ema_decay: float = 0.99,
        max_epochs: int = 10000,
        ema_update_every: int = 16,
        checkpoint_path: str = None,
        scheduler_max_rate: int = 1e-4,
        notifier: Notifier = Notifier(),
        early_stopping_patience: int = 12,
        gradient_accumulation_steps: int = 1,
    ):
        self.wandb_run = wandb_run
        self.notifier = notifier

        self.epochs = max_epochs
        self.testing_dataloader = testing_dataloader
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader

        self.classes = classes
        self.num_classes = len(classes)

        self.device = get_device()

        self.wandb_run.summary['classes'] = str(self.classes)
        self.wandb_run.summary['num_classes'] = self.num_classes
        self.wandb_run.summary['device'] = str(self.device)

        self.amp = amp
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model = model.to(self.device)

        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler(self.device, enabled=self.amp)
        self.scheduler = CosineAnnealingWithWarmRestartsLR(
            self.optimizer, warmup_steps=128, cycle_steps=1024, max_lr=scheduler_max_rate
        )
        self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every).to(
            self.device
        )

        self.early_stopping_patience = early_stopping_patience

        self.output_directory = Path(output_dir)
        self.output_directory.mkdir(exist_ok=True)

        self.best_val_accuracy = 0

        self.execution_name = "model" if execution_name is None else execution_name

        if checkpoint_path:
            self.load(checkpoint_path)

    def run(self):
        self.notifier.report_experiment_start(self.execution_name)

        self.visualize_stn(before_training=True)

        patience_counter = 0
        for epoch in range(self.epochs):
            print("[Epoch: %d/%d]" % (epoch + 1, self.epochs))

            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.val_epoch()

            self.wandb_run.log(
                {
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Train Accuracy": train_accuracy,
                    "Val Accuracy": val_accuracy,
                },
                step=epoch + 1,
            )
            self.notifier.report_epoch_results(
                epoch + 1, self.epochs, train_loss, train_accuracy, val_loss,
                val_accuracy, patience_counter, self.early_stopping_patience
            )

            if val_accuracy > self.best_val_accuracy:
                self.save()
                patience_counter = 0
                self.best_val_accuracy = val_accuracy
                self.wandb_run.summary['best_val_accuracy'] = val_accuracy
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{self.early_stopping_patience}")
                if patience_counter >= self.early_stopping_patience:
                    print(
                        "Validation loss did not improve for %d epochs. Stopping training."
                        % self.early_stopping_patience
                    )
                    break

        self.test_model()
        self.visualize_stn(before_training=False)

        self.notifier.report_experiment_end(self.execution_name)

    def train_epoch(self):
        self.model.train()

        avg_accuracy = []
        avg_loss = []

        for batch_idx, data in tenumerate(self.training_dataloader, unit="batch", desc="Training: "):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                predictions, _, loss = self.model(inputs, labels)

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.ema.update()
                self.scheduler.step()

            batch_accuracy = (predictions == labels).sum().item() / labels.size(0)

            avg_loss.append(loss.item())
            avg_accuracy.append(batch_accuracy)

        return np.mean(avg_loss), np.mean(avg_accuracy) * 100.0

    def val_epoch(self):
        self.model.eval()

        avg_loss = []
        predicted_labels = []
        true_labels = []

        for batch_idx, (inputs, labels) in tenumerate(self.validation_dataloader, unit="batch", desc="Validation: "):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                predictions, _, loss = self.model(inputs, labels)

            avg_loss.append(loss.item())
            predicted_labels.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels))
            .float()
            .mean()
            .item()
        )
        # self.wandb_run.log(
        #     {
        #         "confusion_matrix": self.wandb_run.plot.confusion_matrix(
        #             probs=None,
        #             y_true=true_labels,
        #             preds=predicted_labels,
        #             class_names=self.classes,
        #         )
        #     }
        # )

        print(
            "Eval loss: %.4f, Eval Accuracy: %.4f %%"
            % (np.mean(avg_loss) * 1.0, accuracy * 100.0)
        )
        return np.mean(avg_loss), accuracy * 100.0

    def test_model(self):
        self.ema.eval()

        predicted_labels = []
        true_labels = []

        for batch_idx, (inputs, labels) in tenumerate(self.testing_dataloader, unit="batch", desc="Testing: "):
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                _, logits = self.ema(inputs)
            outputs_avg = logits.view(bs, ncrops, -1).mean(1)
            predictions = torch.argmax(outputs_avg, dim=1)

            predicted_labels.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels))
            .float()
            .mean()
            .item()
        )
        print("Test Accuracy: %.4f %%" % (accuracy * 100.0))
        self.notifier.report_test_results(accuracy * 100.0)

        self.wandb_run.summary['test_accuracy'] = accuracy * 100.0
        self.wandb_run.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=true_labels,
                    preds=predicted_labels,
                    class_names=self.classes,
                )
            }
        )

    def visualize_stn(self, before_training=False):
        self.model.eval()

        batch = torch.utils.data.Subset(self.training_dataloader.dataset, range(32))

        batch = torch.stack([batch[i][0] for i in range(len(batch))]).to(self.device)
        with torch.autocast(self.device.type, enabled=self.amp):
            stn_batch = self.model.stn(batch)

        to_pil = transforms.ToPILImage()

        grid = to_pil(torchvision.utils.make_grid(batch, nrow=16, padding=4))
        stn_batch = to_pil(torchvision.utils.make_grid(stn_batch, nrow=16, padding=4))

        if before_training:
            self.wandb_run.log({
                "before_training_pre_stn": wandb.Image(grid),
                "before_training_post_stn": wandb.Image(stn_batch)
            })
        else:
            self.wandb_run.log({
                "after_training_pre_stn": wandb.Image(grid),
                "after_training_post_stn": wandb.Image(stn_batch)
            })

    def save(self):
        data = {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_acc": self.best_val_accuracy,
        }

        torch.save(data, str(self.output_directory / f"{self.execution_name}.pt"))

    def load(self, path):
        data = torch.load(path, map_location=self.device)

        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])
        self.scaler.load_state_dict(data["scaler"])
        self.scheduler.load_state_dict(data["scheduler"])
        self.best_val_accuracy = data["best_acc"]
