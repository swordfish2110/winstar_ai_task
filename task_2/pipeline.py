import argparse
from ner_model.infer_ner import NERWrap
from image_classifier.infer_cnn import ImageCNNWrap
 
def process_pipeline(text, image_path):
    # Load NER model and extract entities
    id_to_label = {0: "O", 1: "ANIMAL"}
    ner = NERWrap()
    result = ner.run(text, id_to_label)
    
    animal_from_text = None
    for token, label in result:
        if label == "ANIMAL":
            animal_from_text = token.lower()
            break
    
    if not animal_from_text:
        print("No animal mentioned in text.")
        return False
    
    # Load image classifier and predict
    MODEL_PATH = "image_classifier.h5"
    TARGET_SIZE = (224, 224)
    img_classification = ImageCNNWrap(MODEL_PATH, TARGET_SIZE)
    predicted_class, confidence = img_classification.predict(image_path)
    
    # Compare results
    result = animal_from_text in predicted_class.lower()
    print(f"Text mentions: {animal_from_text}, Image contains: {predicted_class} with confidence {confidence} -> Result: {result}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Input text describing the image")
    parser.add_argument("--image", type=str, required=True, help="Input path to the image file")
    args = parser.parse_args()
    
    process_pipeline(args.text, args.image)
