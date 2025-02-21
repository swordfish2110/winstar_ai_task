import torch
from transformers import BertTokenizerFast, BertForTokenClassification

class NERWrap:
    def __init__(self, model_path="./ner_model"):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.model.eval()
    
    def run(self, text, id_to_label):
        tokens = text.split()
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        
        predictions = torch.argmax(outputs, dim=-1).squeeze().tolist()
        
        word_ids = inputs.word_ids()
        prev_word_idx = None
        token_predictions = []
        for word_idx, pred in zip(word_ids, predictions):
            if word_idx is not None and word_idx != prev_word_idx:
                token_predictions.append((tokens[word_idx], id_to_label.get(pred, "O")))
            prev_word_idx = word_idx
        
        return token_predictions
