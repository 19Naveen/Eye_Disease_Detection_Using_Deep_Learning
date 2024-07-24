# Eye Disease Detection

## Overview

This project leverages machine learning to identify Glaucoma from a set of eye images. The model is built using TensorFlow 2.0 and transfer learning techniques. It utilizes a dataset from a Kaggle competition, consisting of images labeled into different categories.

## Table of Contents

1. [Project Description](#project-description)
2. [Requirements](#requirements)
3. [Data Preparation](#data-preparation)
4. [Running the Code](#running-the-code)
5. [Training and Evaluation](#training-and-evaluation)
6. [Making Predictions](#making-predictions)
7. [Results Visualization](#results-visualization)
8. [Contributing](#contributing)

## Project Description

The objective of this project is to classify images of eyes to detect the presence of Glaucoma. The workflow involves:

1. **Data Preparation**: Downloading and preprocessing the data.
2. **Model Selection**: Using a pre-trained model from TensorFlow Hub.
3. **Training**: Training the model with the processed data.
4. **Evaluation**: Evaluating the model performance on the test set.
5. **Prediction**: Making predictions on new images.

## Requirements

- Python 3.x
- TensorFlow 2.x
- TensorFlow Hub
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- ipywidgets

You can install the required packages using:

```bash
pip install tensorflow tensorflow-hub pandas numpy matplotlib scikit-learn ipywidgets
```

## Data Preparation

To run this project, you need to have your dataset organized in a specific structure. The dataset should be organized into training and testing directories, with each directory containing subdirectories for each class (CNV, DME, DRUSEN, NORMAL).

### Dataset Structure:

```
Data/
  ├── train/
  │   ├── CNV/
  │   ├── DME/
  │   ├── DRUSEN/
  │   └── NORMAL/
  └── test/
      ├── CNV/
      ├── DME/
      ├── DRUSEN/
      └── NORMAL/
```

### Download and Unzip Data

1. Download the dataset from the Kaggle competition.
2. Unzip the dataset and place it in the `Data` directory.

### Code Modifications

In the provided code, ensure the paths to the dataset are correctly set. If your data is stored in a different location or has a different structure, modify the paths accordingly.

Example code to unzip and check data:

```python
!unzip "drive/MyDrive/Data_Science_Project/Glaucoma_Detection/archive.zip" -d "drive/MyDrive/Data_Science_Project/Glaucoma_Detection"
print("Completed")
```

## Running the Code

1. Ensure you have all the necessary libraries installed.
2. Modify the data paths as needed.
3. Run the notebook or Python script to start training the model.

## Training and Evaluation

The model is trained using the images from the training dataset. You can monitor the training process using TensorBoard. The evaluation of the model is done using the validation dataset.

Example command to start TensorBoard:

```python
%tensorboard --logdir logs/
```

## Making Predictions

Once the model is trained, you can use it to make predictions on new images. Ensure the new images are preprocessed in the same way as the training images.

Example code to load and predict on new images:

```python
test_path = "Test_Images"
test_data = ["Test_Images/" + filename for filename in os.listdir(test_path)]
predict_on_image = create_batch_data(test_data, Test_Data=True)
result_on_predict = loaded_model.predict(predict_on_image)
```

## Results Visualization

The results of the predictions can be visualized using Matplotlib. This helps in understanding how well the model is performing.

Example code to visualize predictions:

```python
plt.figure(figsize=(50, 10))
for i, image in enumerate(result_images):
    plt.subplot(1, len(result_on_predict), i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(result_label[i])
    plt.imshow(image)
```

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

### Main Contributor
- Naveenkumar D

If you encounter any issues or have suggestions for improvements, please open an issue in the project repository.

---

*Feel free to modify and extend this project according to your needs. If you encounter any issues, Contact me😊.*
