import nltk
import random
import torch
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
from nltk.tokenize import word_tokenize
import numpy as np

# https://rotational.io/blog/building-an-ai-text-detector/

class AIOrHumanScorer():
    """
    Score text that may have been produced by a generative model.
    """
    #alt model: "google-bert/bert-base-german-cased"
    def __init__(self, mask_filler="bert-base-german-cased"):
        self.model = AutoModelForMaskedLM.from_pretrained("dbmdz/bert-base-german-cased")
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
        self.mask_fill = pipeline("fill-mask", model=mask_filler)
        self.labels = ["human", "auto"]
        nltk.download("punkt")
        nltk.download("stopwords")
        self.stop_words = set(nltk.corpus.stopwords.words("german"))

    def _mask_fill(self, text: str, mask_ratio=0.15, max_tokens=512, random_state=42) -> tuple:
        """
        This function computes a mask fill score for a text sample. This score is
        computed by randomly masking words in the text and checking how well a mask
        fill model can predict the masked words. Returns a tuple of
        (true_tokens, pred_tokens).
        """

        # Truncate to ensure the text is within the token limit
        tokens = word_tokenize(text, language="german")[:max_tokens]
        # Randomly select words to mask, ignoring stopwords
        random.seed(random_state)
        candidates = [(i, t) for i, t in enumerate(tokens) if t.lower() not in self.stop_words and t.isalnum() and len(t.strip()) > 1 and not t.isnumeric()]
        if len(candidates) == 0:
            raise ValueError("No valid tokens after stopword removal.")

        n_mask = int(len(candidates) * mask_ratio)
        print(len(candidates))
        if n_mask == 0:
            n_mask = 1
        # Mask the target words
        targets = sorted(random.sample(candidates, n_mask), key=lambda x: x[0])
        #targetsNC = sorted(np.random.choice())

        print(targets)
        masked_tokens = [t[1] for t in targets]
        masked_text = ''
        for i, token in enumerate(tokens):
            if len(targets) > 0 and token == targets[0][1]:
                masked_text += '[MASK] '
                targets.pop(0)
            else:
                masked_text += token + ' '
        print(masked_text)
        # Get the mask fill predictions
        #fill_preds = [f['token_str'] for f in self.mask_fill(masked_text, tokenizer_kwargs={'truncation': True})[0]]
        predictions = []
       # res = [f['token_str'] for f in self.mask_fill(masked_text, tokenizer_kwargs={'truncation': True})[0]]
        #print(res)
        #predictions.append(res)
        fill_preds = self.predict_seqs_dict(text=masked_text)
        return masked_tokens, fill_preds, n_mask

    def score(self, text: str, mask_fill_threshold=0.4) -> float:
        """
        Return a dict of scores that represents how likely the text was produced by a
        generative model.
        """

        # Compute the mask fill score
        
        true_tokens, pred_tokens, n_mask = self._mask_fill(text)
        print(true_tokens)
        print(pred_tokens)
        print(tuple(zip(true_tokens, pred_tokens)))
        score = sum([1 for t, p in zip(true_tokens, pred_tokens) if t in p]) / len(true_tokens)
        return score, n_mask

    def predict_seqs_dict(self, text, top_k=5, order='right-to-left'):
        ids_main = self.tokenizer.encode(text,
                                return_tensors="pt",
                                add_special_tokens=False)

        ids_ = ids_main.detach().clone()
        position = torch.where(ids_main == self.tokenizer.mask_token_id)
        positions_list = position[1].numpy().tolist()
        predictions = []
        if order =='left-to-right':
            positions_list.reverse()

        elif order=='random':
            random.shuffle(positions_list)

        print(positions_list)
        for i in range(len(positions_list)):
            token_pred = []
            model_logits = self.model(ids_main)['logits'][0][positions_list[i]]
            top_k_tokens = torch.topk(model_logits, top_k, dim=0).indices.tolist()
            for j in range(len(top_k_tokens)):
                token_pred.append(self.tokenizer.decode(top_k_tokens[j]))
            predictions.append(token_pred)
        return predictions
    