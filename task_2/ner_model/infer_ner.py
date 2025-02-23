import torch
from transformers import BertTokenizerFast, BertForTokenClassification

class NERWrap:
    def __init__(self, model_path="./ner_model"): # loading pretrained NER model and tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.model.eval() # setting the model to evaluation mode
    
    def run(self, text, id_to_label):
        tokens = text.split()
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt")# returns tensors in Pytorch format
        
        with torch.no_grad(): # Disabling gradient tracking to speed up inference and save memory
            outputs = self.model(**inputs).logits # the model processes inputs and returns logits(raw scores for each token classification)
        
        predictions = torch.argmax(outputs, dim=-1).squeeze().tolist() # finds the most likely class for each token. '.squeeze().tolist()' converts the tensor into python list
        
        word_ids = inputs.word_ids() # maps tokens back to their original words
        prev_word_idx = None
        token_predictions = []
        for word_idx, pred in zip(word_ids, predictions):
            if word_idx is not None and word_idx != prev_word_idx:
                token_predictions.append((tokens[word_idx], id_to_label.get(pred, "O"))) # defaults to 'O' for words without a recognized entity
            prev_word_idx = word_idx
        
        return token_predictions
