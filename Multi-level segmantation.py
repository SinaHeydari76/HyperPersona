# %% ==============================================================
# === IMPORTS AND SETUP ===
# ==============================================================

import pandas as pd
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# --- Load spaCy English model ---
# "en_core_web_lg" is a large model (500k+ word vectors)
nlp = spacy.load("en_core_web_lg")

# ==============================================================
# === TEXT SEGMENTATION FUNCTION ===
# ==============================================================
def segment(docs: List[str]) -> List[Tuple[str, List[str], List[str]]]:
    """
    Segments each document into sentences and words using spaCy.

    Parameters
    ----------
    docs : List[str]
        A list of text documents (essays).

    Returns
    -------
    List[Tuple[str, List[str], List[str]]]
        A list where each element is a tuple:
            (original_text, list_of_sentences, list_of_words)
    """
    result = []

    # tqdm adds a progress bar for long document lists
    for doc in tqdm(docs, desc="Segmenting documents"):
        parsed_doc = nlp(doc)

        # Extract sentences
        sentences = [sent.text.strip() for sent in parsed_doc.sents]

        # Extract word tokens (excluding punctuation and spaces)
        words = [token.text for token in parsed_doc if not token.is_punct and not token.is_space]

        result.append((doc, sentences, words))

    return result
