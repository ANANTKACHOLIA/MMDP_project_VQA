# MMDP_project_VQA
Absolutely, it's important to acknowledge the source of inspiration and reference. Here's how you can include that information in the README:

### Project Overview
We propose the task of free-form and open-ended Visual Question Answering (VQA), inspired by the work presented in the paper "VQA: Visual Question Answering" by Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence Zitnick, Dhruv Batra, and Devi Parikh. The paper can be found at www.visualqa.org.

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
The images in the dataset are too large to be uploaded to the repository (more than 400MB). Hence, users need to download the images separately.

### Usage Instructions
To use the implementation:
1. Download the notebook and the MMDP-VQA dataset from Kaggle.
2. Update the paths and directory names in the notebook accordingly.
3. Install the required dependencies.
4. Run the notebook (GPU needed for faster processing).

Alternatively, users can upload the notebook to Kaggle, import the dataset, turn on the GPU accelerator, and run the notebook directly.

