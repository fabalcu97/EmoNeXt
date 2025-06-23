from dataset import Dataset


class CkPlus(Dataset):
    emotion_labels = {
        "0": "Angry",
        "1": "Disgust",
        "2": "Fear",
        "3": "Happy",
        "4": "Sad",
        "5": "Surprise",
        "6": "Neutral",
        # "7": "Contempt",
    }

    def __init__(self):
        super().__init__("ckplus", "davilsena/ckdataset")
        self.csv_location = "2/ckextended.csv"
