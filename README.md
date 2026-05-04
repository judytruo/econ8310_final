link to working docs on onedrive: https://uofnebraska-my.sharepoint.com/:f:/r/personal/00838983_nebraska_edu/Documents/ECON8310?csf=1&web=1&e=pAdADg
# Baseball Detection Order 
Faster R-CNN with a ResNet50 backbone, fine-tuned to detect baseballs in training video frames.
## Setup
Make sure your environment has the required packages installed (PyTorch, torchvision, OpenCV, etc.) and that your raw video files and CVAT annotations are in the expected input directories before starting.
## Run Order
Run the scripts in the following order. 
### 1. `main.py`
Entry point that handles frame extraction. Reads each `.mov` file, pulls out individual frames, and saves them as `.jpeg` images at 95% quality into a structured folder hierarchy.
```bash
python main.py
```
### 2. `prepare_dataset.py`
Pairs the extracted frames with their CVAT annotations and organizes everything into the train, validation, and test splits using a 70/15/15 split by video.
```bash
python prepare_dataset.py
```
### 3. `baseball_dataset.py`
Defines the custom PyTorch Dataset class that loads images and bounding box annotations in the format Faster R-CNN expects. This file is imported by `train.py` and does not need to be run on its own, but verify it runs without errors before training.
```bash
python baseball_dataset.py
```
### 4. `train.py`
Fine-tunes the Faster R-CNN ResNet50 model on the prepared dataset. Trains for 30 epochs, evaluates on the validation set after each epoch, and saves the best-performing checkpoint.
```bash
python train.py
```
