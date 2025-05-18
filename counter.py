# AI: I want to count the number of images in each subfolder of the fer2013/test directory
import os

emotions = ['Angry', 'Disgust', 'Fear', 'Happy',
            # 'Neutral',
            'Sad', 'Surprise']
total_images = 0

for emotion in emotions:
    # Get the path to the subfolder
    path = os.path.join('./fer2013/train', emotion)

    # List all files in the subfolder
    lst = os.listdir(path)

    # Count the number of files
    total_images += len(lst)

# Print the count
print(f"Number of {emotion} images: {total_images}")
