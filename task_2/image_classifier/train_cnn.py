import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path
import kagglehub

BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
DATASET_NAME = "vencerlanz09/sea-animals-image-dataste"
EPOCHS = 20

class ImageClassifier:
    def __init__(self, dataset_name, batch_size, target_size, epochs):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.target_size = target_size
        self.epochs = epochs
        self.model = None
        self.train_images = None
        self.validation_images = None
        self.test_images = None
        self.class_indices = None

    def load_data(self):
        path = kagglehub.dataset_download(self.dataset_name)
        print("Path to dataset files:", path)
        image_dir = Path(path)
        filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.PNG')) # Collecting images file paths
        labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')
        image_df = pd.concat([filepaths, labels], axis=1) # Creating DataFrame containing file paths and class labels

        train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)
        return train_df, test_df

    def preprocess_data(self, train_df, test_df):
        train_generator = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            validation_split=0.2
        )

        test_generator = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input #Applying tf.keras.applications.efficientnet.preprocess_input for normalization
        )
        
        self.train_images = train_generator.flow_from_dataframe( #creating train_images
            dataframe=train_df,
            x_col='Filepath',
            y_col='Label',
            target_size=self.target_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            subset='training'
        )

        self.validation_images = train_generator.flow_from_dataframe( #creating validation_images
            dataframe=train_df,
            x_col='Filepath',
            y_col='Label',
            target_size=self.target_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            subset='validation'
        )

        self.test_images = test_generator.flow_from_dataframe( #creating test_images
            dataframe=test_df,
            x_col='Filepath',
            y_col='Label',
            target_size=self.target_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False
        )

        self.class_indices = self.train_images.class_indices

    def build_model(self, num_classes):
        augment = tf.keras.Sequential([
            layers.Resizing(224, 224),
            layers.Rescaling(1./255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1)
        ])

        model = models.Sequential([
            augment,
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)), #creating convultional layer
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(), #flattens the data
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']) #compiles the model with 'Adam' optimizer and categorical_crossentropy loss function
        self.model = model

    def train_model(self):
        history = self.model.fit(
            self.train_images,
            epochs=self.epochs,
            validation_data=self.validation_images,
            verbose=1
        ) #training the model using train_images and validation_images for a given number of epochs
        return history

    def save_model(self, save_path):
        self.model.save(save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    classifier = ImageClassifier(DATASET_NAME, BATCH_SIZE, TARGET_SIZE, EPOCHS)
    train_df, test_df = classifier.load_data()
    classifier.preprocess_data(train_df, test_df)
    classifier.build_model(num_classes=len(classifier.class_indices))
    classifier.train_model()
    classifier.save_model("image_classifier.h5")