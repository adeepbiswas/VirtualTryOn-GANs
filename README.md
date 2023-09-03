# VirtualTryOn-GANs
This repository contains the implementation of a virtual clothing try-on system using Generative Adversarial Networks (GANs). The project aims to generate realistic images of a person wearing different clothing items, given an image of the person and an image of the clothing item.

# Main Components
The main components of the project are:

Final_VLRProject_PolyGAN.ipynb: This Jupyter Notebook contains the final implementation of the project.

data_differenceMask.py: This script is used to create difference masks for training data.

datasets/dataloader.py: This script is responsible for loading and preprocessing data for the GANs.

inception_score.py: This script implements the Inception Score, a metric for evaluating the quality of generated images.

models/models.py, models/models_og.py, models/minibatch_discrim.py: These scripts contain the implementation of the neural network models used in the system.

test.py: This script is used for testing the GANs model.

train.py: This script is the main training file for the project.

utils/utils.py: This script includes utility functions used throughout the project.

# Frameworks and Libraries
The project uses several Python libraries and frameworks, including:

PyTorch: A popular open-source machine learning library for Python, used for creating and training the GANs.
NumPy: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
OpenCV: A library of programming functions mainly aimed at real-time computer vision.
Matplotlib: A plotting library for the Python programming language and its numerical mathematics extension NumPy.
The complete list of dependencies can be found in the requirements.txt file.

# How to Run the Code
Clone the repository to your local machine.

Install the required Python packages. You can do this by running pip install -r requirements.txt in your terminal.

To train the model, navigate to the project directory and run python train.py.

To test the model, run python test.py.

Please note that you might need to adjust the paths in the scripts according to your local setup.

# Additional Information
For more information about the project, please refer to the index.html file. This webpage provides detailed information about the project's motivation, prior work, idea, methodology, results, conclusion, future directions, and references.

# License
This project is licensed under the MIT License.
