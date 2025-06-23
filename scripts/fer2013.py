from dataset import Dataset


class Fer2013(Dataset):
    emotion_labels = {
        "0": "Angry",
        "1": "Disgust",
        "2": "Fear",
        "3": "Happy",
        "4": "Sad",
        "5": "Surprise",
        "6": "Neutral",
    }

    def __init__(self):
        super().__init__("fer2013", "deadskull7/fer2013")
        self.csv_location = "1/fer2013.csv"
