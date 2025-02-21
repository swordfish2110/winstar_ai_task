## Overview
This project implements a pipeline that verifies whether a given text description matches the content of an image. The pipeline utilizes:
- A Named Entity Recognition (NER) model to extract animal names from text.
- An image classification model to identify sea ​​animals in images.
- A comparison mechanism to check if the extracted entity matches the detected object in the image.

## Project Structure
```
project_root/
│── ner_model/
│   │── train_ner.py
│   │── infer_ner.py
│── image_classifier/
│   │── train_cnn.py
│   │── infer_cnn.py
│── pipeline.py
│── requirements.txt
│── README.md
```

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
Run the pipeline with a text description and an image file:
```bash
python pipeline.py --text "There is a cat in the picture." --image path/to/image.jpg
```

## Dependencies
The project requires the following dependencies:
- Python 3.8+
- TensorFlow
- PyTorch
- Transformers
- Datasets
- Pandas
- NumPy
- Pillow
- KaggleHub

## Model Details
- The NER model is based on BERT and fine-tuned for animal recognition.
- The image classifier is a CNN model trained on an [animal dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste).