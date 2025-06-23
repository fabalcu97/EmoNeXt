from .telegram import TelegramNotifier


class Notifier:
    def __init__(self):
        self.reporter = TelegramNotifier()

    def report_epoch_results(
            self, epoch, epochs, train_loss, train_accuracy, val_loss,
            val_accuracy, patience_counter, early_stopping_patience
    ):
        message = (
            f"Epoch {epoch}/{epochs}: Train Loss: {train_loss: .4f}, Val Loss: {val_loss: .4f},"
            + f"Train Accuracy: {train_accuracy: .2f} % , Val Accuracy: {val_accuracy: .2f} %, "
            + f"Patience: {patience_counter}/{early_stopping_patience}"
        )
        self.reporter.send_message(message)

    def report_test_results(self, test_accuracy):
        message = (
            f"Test Accuracy: {test_accuracy:.2f}%"
        )
        self.reporter.send_message(message)

    def report_experiment_start(self, experiment_name):
        message = f"Starting experiment: {experiment_name}"
        self.reporter.send_message(message)

    def report_experiment_end(self, experiment_name):
        message = f"Experiment {experiment_name} has ended."
        self.reporter.send_message(message)
