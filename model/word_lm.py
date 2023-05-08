import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from fastDamerauLevenshtein import damerauLevenshtein
import copy

class SpellCorrectionModel(nn.Module):
    def __init__(self, NCBI_BERT = "./bert/ncbi_bert_base", config_file = None, max_candidates=50):
        super().__init__()
        model_file = NCBI_BERT + "/pytorch_model.bin"
        self.config = None
        if config_file is not None:
            config_file = NCBI_BERT + config_file
            self.config = AutoConfig.from_pretrained(config_file)
        self.bert = AutoModelForMaskedLM.from_pretrained(model_file, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(NCBI_BERT, config=self.config)
        self.max_candidates = max_candidates
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.linear = torch.nn.Linear(768, 3) # correct incorrect and uncertain
        self.softmax = torch.nn.Softmax(dim=-1)
        self.scores = None
        self.labels = None
        self.loss = None 
    
    def forward(self, input_ids, attention_mask, misspelling, correct_spelling):
        
        wones = torch.ones(input_ids.shape[0], dtype= torch.int64).unsqueeze(1)
        cls_t = wones * self.tokenizer.cls_token_id
        sep_t = wones * self.tokenizer.sep_token_id
        input_ids = torch.cat([cls_t, input_ids, sep_t], dim=1).to(self.device)
        attention_mask = torch.cat([wones, attention_mask, wones], dim=1).to(self.device)

        masked_token_locations = [ torch.where(tokens == self.tokenizer.mask_token_id)[0] for tokens in input_ids]
        labels  = copy.deepcopy(input_ids)
        labels[input_ids != self.tokenizer.mask_token_id] = -100
        target_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(correct_spelling))
        labels[input_ids == self.tokenizer.mask_token_id] = target_ids.to(self.device)
        self.labels = labels

        candidate_words = {}
        with torch.no_grad():
            loss, outputs = self.bert(input_ids, attention_mask=attention_mask, labels = self.labels, return_dict=False)

            topk_candidates = torch.topk(outputs, self.max_candidates, dim=-1).indices
            topk_candidates = topk_candidates.permute(0, 2, 1)
           
            for i, (input_id, candidate_indices) in enumerate(zip(input_ids, topk_candidates)):
                candidate_words[i] = []
                for j, candidate_index in enumerate(candidate_indices):                    
                    candidate_words[i].append(self.tokenizer.decode(candidate_index[masked_token_locations[i]]))
                    
        correction_final = []
        for index, word in enumerate(misspelling):
            correct_word = None
            score = 99999999
            for possible_candidate in candidate_words[index]:
                distance = damerauLevenshtein(word, possible_candidate, similarity=False)
                if score > distance and possible_candidate is not word: # hacking here we know it is wrong to ignore correct spelling
                    correct_word = possible_candidate
                    score = distance
            correction_final.append(correct_word)
        
        self.loss = loss
        self.loss.requires_grad = True
        self.scores = self.softmax(outputs)
        return self.scores, correction_final
