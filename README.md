# COVID-19 Prediction App

This application predicts COVID-19 using lung radiography images. It classifies the images into three categories: HEALTHY, COVID-19, or PNEUMONIA. The app leverages a Convolutional Neural Network (CNN) model to provide the classification and confidence score.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use the app, follow these steps:

1. Place the lung radiography images in the `input_images` directory.
2. Run the prediction script:

```bash
python predict.py
```

3. The results will be saved in the `output` directory with the classification labels and confidence scores.

## Model Architecture

The CNN model used in this application consists of multiple convolutional layers followed by max-pooling layers, and fully connected layers. The architecture is designed to extract features from the radiography images and classify them accurately.

## Dataset

The model is trained on a small dataset consisting of labeled lung radiography images. The dataset includes images categorized as HEALTHY, COVID-19, and PNEUMONIA. Ensure that the dataset is preprocessed and split into training, validation, and test sets.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.