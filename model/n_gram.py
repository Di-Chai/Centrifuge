import os
import json
import nltk
import pickle
import numpy as np

from transformers import AutoTokenizer


class NGramModel:

    def __init__(self, n_gram_path, tokenizer_path="TinyLlama/TinyLlama_v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        with open(os.path.join(n_gram_path, "word1_num.json"), 'r') as json_file:
            self.fdist_1 = json.load(json_file)
        with open(os.path.join(n_gram_path, "word2_num.json"), 'r') as json_file:
            self.fdist_2 = json.load(json_file)   
        with open(os.path.join(n_gram_path, "word3_num.json"), 'r') as json_file:
            self.fdist_3 = json.load(json_file)  

    def _compute_bigram_probability(self, tokens):
        tokens = [e.lower() for e in tokens]
        setence_bigrams = list(nltk.bigrams(tokens))
        bigram_probs = list()
        for bigram in setence_bigrams:
            bigram_count = self.fdist_2.get(str(bigram), 0)
            word_count = self.fdist_1.get(str(bigram[0]), 0)
            assert bigram_count <= word_count
            if bigram_count == 0:
                bigram_prob = 1e-20
            else:
                bigram_prob = bigram_count / word_count
            bigram_probs.append(-np.log(bigram_prob))
        return bigram_probs

    def _compute_trigram_probability(self, tokens):
        tokens = [e.lower() for e in tokens] # required by the pre-processing
        trigram_probs = list()
        setence_trigrams = list(nltk.trigrams(tokens))
        for trigram in setence_trigrams:
            trigram_count = self.fdist_3.get(str(trigram), 0)
            word_count = self.fdist_2.get(str(trigram[0:2]), 0)
            assert trigram_count <= word_count
            if trigram_count == 0:
                trigram_prob = 1e-20
            else:
                trigram_prob = trigram_count / word_count
            trigram_probs.append(-np.log(trigram_prob))
        return trigram_probs   
    
    def build_ngram_model(model, target_tokenizer_name, tokenizer_args, datasets):
        # ToDo (Pengbo)
        ngram_model = None
        return ngram_model

    def obtain_ngram_loss(self, model, messages, return_pickle=True):
        """
        model : bigram or trigram model
        messages: input_ids
        """
        messages = [self.tokenizer.convert_ids_to_tokens(e) for e in messages]
        loss_list = []
        for msg in messages:
            if model == 'bigram':
                loss_list.append(self._compute_bigram_probability(msg))
            elif model == 'trigram':
                loss_list.append(self._compute_trigram_probability(msg))
            else:
                raise NotImplementedError(f"Model {model} not implemented")
        
        # Serialize the loss object to a byte stream
        if return_pickle:
            response = pickle.dumps({
                "loss": loss_list
            })
            return response
        else:
            return loss_list
        