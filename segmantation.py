# IMPORTS
import pandas as pd
# import preprocessing
import spacy
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# %%
data = pd.read_csv("essays.csv")

# %%
from typing import List, Tuple

# Load spaCy's English model
nlp = spacy.load("en_core_web_lg")

def segment_documents(docs: List[str]) -> List[Tuple[str, List[str], List[str]]]:
    result = []
    for doc in tqdm(docs):
        parsed_doc = nlp(doc)
        
        # Get sentences
        sentences = [sent.text.strip() for sent in parsed_doc.sents]
        
        # Get words (excluding punctuation and spaces)
        words = [token.text for token in parsed_doc if not token.is_punct and not token.is_space]
        
        result.append((doc, sentences, words))
    return result

# %%
seg = segment_documents(data)

# %%
o = []
c = []
e = []
a = []
n = []

for i in range(0, len(labels)):
    o.append(labels[i][0])
    c.append(labels[i][1])
    e.append(labels[i][2])
    a.append(labels[i][3])
    n.append(labels[i][4])
# %%
labels[0][0]    