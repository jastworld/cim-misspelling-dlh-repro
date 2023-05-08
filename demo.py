from transformers import pipeline
from fastDamerauLevenshtein import damerauLevenshtein

mask_filler = pipeline(
    "fill-mask", model="./bluebert-finetuned-mimic-v1", top_k= 20
)
while True:
    sentence = input("Note: ")
    misspelling = input("error: ")
    index = sentence.split().index("[MASK]") 

    preds = mask_filler(sentence)
    score = 100000
    word = None
    for pred in preds:
        replacement = pred['sequence'].split()[index]
        distance = damerauLevenshtein(misspelling, replacement, similarity=False)
        if score > distance:
            score = distance
            word = replacement
        
    print(sentence.replace("[MASK]", word))


