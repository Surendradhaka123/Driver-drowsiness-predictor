# Driver Drowsiness Detection using Transfer Learning with EfficientNetB5

![Drowsiness Detection](https://example.com/path/to/drowsiness-detection-image.png)

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Driver drowsiness detection using transfer learning with EfficientNetB5 is a deep learning-based application that aims to detect the drowsiness level of drivers from real-time video streams. The model utilizes transfer learning with the EfficientNetB5 architecture, pre-trained on a large image dataset, to achieve high accuracy in real-world scenarios.

This README provides an overview of the project, instructions for setting up the environment, guidelines for model training, and information on how to contribute.

## Dataset

The dataset used for training and evaluation can be found at [https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset]

This dataset have two directories one contains 2000 photos with open eyes and another contains 2000 photos with closed eyes.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

2. Set up a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the trained drowsiness detection model on real-time video streams, follow these steps:

1. Ensure you have installed the required dependencies as mentioned in the [Installation](#installation) section.

2. Load the pre-trained EfficientNetB5 model:

```python

# Load the pre-trained EfficientNetB5 model
from tensorflow.keras.models import load_model
model =  load_model('Driver_drowsiness_efficientnet.h5)

```

3. Use the model for drowsiness detection on video frames captured from a camera:

```python
import cv2

# Assuming 'camera' is the camera object capturing video frames
while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Preprocess the frame if necessary (resize, normalize, etc.)
    # ...

    # Perform inference on the preprocessed frame
    predictions = model.predict(frame)

    # Process predictions to determine drowsiness level
    # ...

    # Display the frame with drowsiness level indication
    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
```

## Model Training

As mentioned earlier, the dataset used for training is not publicly available. Therefore, you can use your own dataset and train the model using transfer learning with EfficientNetB5.

1. Prepare your dataset with labeled video samples of drivers and their corresponding drowsiness levels.

2. Update the `config.yaml` file to set the hyperparameters and other configurations for training.

3. Execute the training script:

```bash
python train.py
```

4. After training, the script will save the trained model.

## Evaluation

It is essential to evaluate the trained model's performance on a separate test dataset to assess its accuracy and generalization.

To evaluate the model, you can use the test dataset and the `evaluate.py` script:

```bash
python evaluate.py
```

## Contributing

Contributions to this project are welcome. If you find any issues or want to propose enhancements, please open an issue or submit a pull request. Before making significant changes, it's best to discuss your ideas through the issues page.

## License

This project is licensed under the [MIT License](LICENSE).

---

We hope this README helps you understand the Driver Drowsiness Detection using Transfer Learning with EfficientNetB5 project. If you have any questions or need further assistance, please don't hesitate to reach out.

Thank you for using our project!
