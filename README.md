# Real-Time Custom In-Screen Object Detection Using YOLOv5

Below are real-time demonstrations of the object detection system in action:

![Demo 1](https://github.com/uoRetr0/Zombie-Detector/blob/main/Demo%201.gif) 

![Demo 2](https://github.com/uoRetr0/Zombie-Detector/blob/main/Demo%202.gif)

The above GIFs illustrate the system's ability to accurately and quickly detect zombies in a live gaming environment, showcasing the practical application and performance of the custom-trained YOLOv5 model.

## Project Description
This project implements a custom object detection system that operates in real-time on a computer screen. It leverages the robust and efficient YOLOv5 (You Only Look Once version 5) deep learning model for object detection tasks. YOLOv5 is known for its speed and accuracy, making it ideal for real-time applications. This custom system is designed to detect objects directly within the screen space, allowing for a wide range of applications, including surveillance, accessibility features, and live content analysis.

## Features
- Real-time detection of zombies within the gaming environment
- Custom trained YOLOv5 model specifically for zombie detection
- Easy-to-use interface for live in-game object detection

## Training Details

The object detection model provided with this project has been trained on a dataset of 228 images specifically curated to detect zombies within a gaming environment. While the current model achieves a certain level of accuracy with this dataset, the performance of object detection models generally improves with larger and more diverse training datasets.

Included in this repository is the `training` folder, which contains the images and annotations used to train the model. This serves as a starting point for those interested in understanding the training process or looking to further improve the model.

It is worth noting that the more images you use to train your model, and the more varied the images are in terms of angles, lighting conditions, and backgrounds, the better your AI will perform. Expanding the dataset can lead to significant improvements in the model's ability to generalize and detect objects across different scenarios.

For those interested in training their own models, it is highly recommended to collect as many labeled images as possible to achieve the best results.


## Modifications for Windows Compatibility
The `yolov5-master` folder included in this repository has been slightly modified to work with Windows, based on a solution from [this Stack Overflow thread](https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath). For Linux-based systems, these modifications are not needed and the original version of the YOLOv5 should be used. If you are running a Linux system, please ensure to revert these changes or clone the original repository from [YOLOv5 GitHub](https://github.com/ultralytics/yolov5).

## Getting Started

### Prerequisites
- Python 3.8 or newer
- pip and virtualenv

### Installation

1. **Clone the repository**
    ```sh
    git clone [your-repo-link]
    cd [your-repo-directory]
    ```

2. **Set up a Python virtual environment**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment**
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```
    ```sh
    pip install -r requirements_torch.txt
    ```

### Usage

To run the object detection, simply execute the main script:

```sh
python LiveDetect.py
```
When running the script, a window will display the live screen content with detected objects highlighted by bounding boxes.

Note on Performance: It is highly recommended to run this script on a system with a CUDA-enabled GPU. Using CUDA will significantly increase the framerate and overall performance of the object detection model. If you have a compatible Nvidia GPU, make sure that the latest drivers and CUDA toolkit are installed for optimal performance.

# For Developers: How to Train Your Own Model

If you're looking to train the object detection model to recognize custom objects, here's a step-by-step guide on how you can collect data, label it, and train your own version of YOLOv5.

## Data Collection
Begin by gathering a diverse set of images of the objects you wish to detect. The larger and more varied your dataset, the better the model will perform across different scenarios.

## Data Labeling
Once you have your images, you will need to label them. Labeling involves drawing bounding boxes around the objects of interest in each image and assigning a class label to each box. Tools like [CVAT](https://github.com/openvinotoolkit/cvat) can be used for this purpose, providing an interface for manual labeling.

## Preparing the Dataset
After labeling, you need to convert your dataset into a format suitable for training. [Roboflow](https://roboflow.com/) is an excellent tool for this. It can help you to:
- Convert labels to YOLO format.
- Split the dataset into training, validation, and test sets.
- Apply preprocessing and augmentations to increase dataset diversity.

## Training the Model
With your dataset prepared, you can proceed to train the model. Google Colab offers free GPU resources that can be utilized for training. You can follow this [custom training notebook](https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb) provided by Roboflow, which is a great starting point.

## Applying Your Weights
After training, download the `bestv5.pt` file which contains the weights of your trained model. Replace the existing `bestv5.pt` file in the `AI` directory of this project with your new weights file.

Remember, the effectiveness of your model heavily depends on the quality and quantity of the data used for training, so invest a good amount of time in creating a robust dataset.

## Tips
- Regularly evaluate your model on the validation set to monitor its performance.
- Experiment with different hyperparameters and training strategies to optimize your model's performance.

For detailed instructions on each step, refer to the documentation of the respective tools and platforms. Good luck with your training!
