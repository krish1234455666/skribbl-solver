import nltk
from nltk.corpus import words
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt

# Download word list
nltk.download('words')
word_list = words.words()

# Load the pre-trained Quick, Draw! model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")


# Load the 345 class labels
labels_url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
labels_path = tf.keras.utils.get_file("categories.txt", labels_url)

with open(labels_path, "r") as f:
    labels = f.read().splitlines()

print(f"Loaded {len(labels)} categories!")


def filter_words(word_list, known_pattern):
    """
    Filters words based on the pattern provided (_ for unknown letters).
    """
    length = len(known_pattern)
    filtered = [
        word for word in word_list
        if len(word) == length and all(
            ch == "_" or ch == word[i] for i, ch in enumerate(known_pattern)
        )
    ]
    return filtered


def preprocess_image(image_path):
    """
    Prepares an input image for the Quick, Draw! model.
    - Converts the image to grayscale.
    - Resizes it to 28x28 pixels.
    - Normalizes pixel values to [0, 1].
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))              # Resize to 28x28
    image = np.array(image) / 255.0             # Normalize to [0, 1]
    image = image.astype(np.float32)            # Ensure float32 type
    return image


def predict_doodle(image, model, labels):
    """
    Predicts the category of a doodle using the Quick, Draw! model.
    - `image`: Preprocessed 28x28 grayscale image.
    - `model`: TensorFlow Hub Quick, Draw! model.
    - `labels`: List of class labels.
    """
    # Add batch dimension
    input_tensor = tf.convert_to_tensor([image], dtype=tf.float32)

    # Run inference
    predictions = model(input_tensor)["probabilities"].numpy()

    # Get the top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_labels = [(labels[i], predictions[0][i]) for i in top_5_indices]

    return top_5_labels


def filter_predictions_by_length(predictions, word_length):
    """
    Filters predictions based on the word length.
    - `predictions`: List of (label, confidence) tuples.
    - `word_length`: Target word length.
    """
    return [(label, confidence) for label, confidence in predictions if len(label) == word_length]


# Main script workflow
if __name__ == "__main__":
    print("Welcome to Skribbl Solver!")

    while True:
        # Step 1: Get the word pattern from the user
        pattern = input("Enter the word pattern (use _ for unknown letters, or type 'exit' to quit): ")
        if pattern.lower() == "exit":
            break

        # Filter word list based on the pattern
        suggestions = filter_words(word_list, pattern)
        print(f"Word suggestions based on pattern: {suggestions[:10]}")  # Show the first 10 matches

        # Step 2: Get the drawing input
        print("Upload your drawing (image path):")
        image_path = input("> ")
        try:
            processed_img = preprocess_image(image_path)

            # Step 3: Predict categories using the model
            predictions = predict_doodle(processed_img, model, labels)
            print("Model Predictions:")
            for label, confidence in predictions:
                print(f"{label}: {confidence:.2f}")

            # Step 4: Filter predictions by word length
            filtered = filter_predictions_by_length(predictions, len(pattern))
            print("Filtered Predictions:")
            for label, confidence in filtered:
                print(f"{label}: {confidence:.2f}")
        except Exception as e:
            print(f"Error processing the image: {e}")
