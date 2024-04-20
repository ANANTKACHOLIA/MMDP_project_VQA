# MMDP_project_VQA

# Visual Question Answering (VQA) Implementation

This project aims to implement a Visual Question Answering (VQA) system, as proposed in the paper "VQA: Visual Question Answering" by Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence Zitnick, Dhruv Batra, and Devi Parikh. The task of VQA involves providing accurate natural language answers to questions asked about images, making it a challenging problem that requires a detailed understanding of the image and complex reasoning.

### Project Overview
Given an image and a natural language question about the image, the task is to provide an accurate natural language answer. Mirroring real-world scenarios, such as helping the visually impaired, both the questions and answers are open-ended. Visual questions selectively target different areas of an image, including background details and underlying context. As a result, a system that succeeds at VQA typically needs a more detailed understanding of the image and complex reasoning than a system producing generic image captions. Moreover, VQA is amenable to automatic evaluation, since many open-ended answers contain only a few words or a closed set of answers that can be provided in a multiple-choice format. We provide a dataset containing ∼0.25M images, ∼0.76M questions, and ∼10M answers (www.visualqa.org), and discuss the information it provides. Numerous baselines and methods for VQA are provided and compared with human performance. Our VQA demo is available on CloudCV (http://cloudcv.org/vqa).

### Dataset Details
- The repository contains a smaller dataset with 1500 images and 12000 questions, divided into training and validation sets (10000+2500).
- The dataset includes two CSV files: `train.csv` and `test.csv`, containing the list of questions mapped to images for training and testing purposes.
- Additionally, there are `train-images.txt` and `test-images.txt` files containing the image lists for training and validation.
- The `qa-full.txt` file contains all the questions for reference.

### Repository Structure
The repository consists of the following files:
1. **Python Scripts**:
   - `main.py`: Main script to run the VQA model.
   - `model.py`: Contains the implementation of the VQA model architecture.
   - `train.py`: Script for training the VQA model.
   - `utils.py`: Utility functions used in the project.
   - `datasets.py`: Script for loading and preprocessing the dataset.

2. **Jupyter Notebook**:
   - `VQA_Implementation.ipynb`: Complete implementation of all the Python scripts in a Jupyter notebook format.

3. **Dataset Files**:
   - `train.csv` and `test.csv`: CSV files containing training and testing questions mapped to images.
   - `train-images.txt` and `test-images.txt`: Text files containing the image lists for training and validation.
   - `qa-full.txt`: Text file containing all the questions in the dataset.
### Note
The dataset used for this implementation can be obtained from [Kaggle](https://www.kaggle.com/) or an alternative source. Due to the large size of the images (over 400MB), they are not included in this repository.

### Usage Instructions
To use the implementation:
1. Download the notebook and the MMDP-VQA dataset from Kaggle.
2. Update the paths and directory names in the notebook accordingly.
3. Install the required dependencies.
4. Run the notebook (GPU needed for faster processing).

Alternatively, users can upload the notebook to Kaggle, import the dataset, turn on the GPU accelerator, and run the notebook directly.
Alternatively:
   - Assemble the Python files (`main.py`, `model.py`, `train.py`, `utils.py`, `datasets.py`) into a single directory.
   - Change the directory accordingly to where the Python files and dataset are located.
   - Download the MMDP-VQA dataset from Kaggle or the provided source.
   - Install the required dependencies and packages using the provided `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
   - Ensure that GPU acceleration is available and enabled for faster processing. If using a GPU, make sure the appropriate drivers and libraries are installed.
   - Execute the assembled Python files by running the main script (`main.py` or `VQA_Implementation.ipynb`).
## Code Desription



## main.py

This script is the main entry point for training the Visual Question Answering (VQA) model. It handles dataset loading, model initialization, and training.

### Contents:
- **Import Statements**: Import necessary libraries and modules.
- **Device Configuration**: Determine the device for computation (CPU or GPU).
- **Paths and Configurations**: Define input paths, data files, and image root directory.
- **Dataset Loading and Preprocessing**: Load the dataset, split it into training and testing sets, and preprocess questions and answers.
- **Transformations**: Define image preprocessing transformations.
- **Dataset Initialization**: Initialize the dataset and data loaders.
- **Vocabulary and Answer Space Initialization**: Construct vocabulary and answer space dictionaries.
- **Save Dictionaries**: Save constructed dictionaries to text files.
- **Model Initialization**: Initialize the VQA model.
- **Model Training**: Train the initialized model.

### Usage:
```bash
python main.py
```

Ensure GPU support for faster training. Customize parameters as needed.

### Description of `VQAModel.py`

The `VQAModel.py` file contains the implementation of the Visual Question Answering (VQA) model. Below is a detailed description of the contents of this file:

- **Imports**: The file imports necessary libraries and modules including `torch`, `nltk`, `numpy`, `PIL`, `pandas`, `os`, `re`, `json`, `torchvision`, and `torch.nn`. It also imports specific functions and classes from these libraries such as `word_tokenize` from `nltk.tokenize`, `load_dataset` and `set_caching_enabled` from `datasets`, and various modules from `torchvision.models`.

- **VQAModel Class**: 
  - `__init__`: The constructor initializes the VQA model. It sets up the image embedding using VGG16 pretrained on ImageNet and removes the last fully connected layer. It defines the architecture of the model including text embedding, LSTM layers, and fully connected layers.
  - `forward`: This method defines the forward pass of the model. It takes images and prompts as inputs and returns the predicted features.

This file encapsulates the architecture of the VQA model, allowing for its easy integration and usage within the broader VQA system.

![image](https://github.com/ANANTKACHOLIA/MMDP_project_VQA/assets/95161741/94396b58-31f6-4113-bd4e-21be4d39bd2a)


## Citation
If you use this implementation or dataset in your research, please consider citing the original paper:

Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence Zitnick, Dhruv Batra, Devi Parikh. "VQA: Visual Question Answering." [www.visualqa.org](www.visualqa.org).

