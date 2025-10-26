# %%
import pandas as pd
import preprocessing
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the fine-tuned model directory
model_path = "essays/bert-base-personality"

# Load pretrained tokenizer and model from the specified directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode (disables dropout, etc.)

print(device)


# %%
def string_components_embeddings(
        text: str,
        layers: Tuple[int, int] = (9, 12)
        ) -> List[torch.Tensor]:
    """
    Generates embeddings for individual word-level components (merged subwords)
    in a given text using a specified range of BERT layers.

    The function:
        1. Tokenizes the input text.
        2. Extracts hidden states from the given layers (e.g., 9–12).
        3. Merges subword tokens (##word pieces) into full words by averaging their embeddings.
        4. Returns a list of word embeddings (one tensor per word).

    Parameters
    ----------
    text : str
        Input text string to embed.
    layers : Tuple[int, int], optional
        Range of layers (inclusive) from which to average hidden states.
        Defaults to (9, 12), meaning layers 9 through 12.

    Returns
    -------
    List[torch.Tensor]
        List of embeddings, each corresponding to one merged word.
        Each tensor has shape (hidden_dim,).
    """
    
    # Tokenize the input text and move tensors to the appropriate device
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Disable gradient computation for efficiency (no training)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Tuple of all layer outputs

    # Select the specified range of layers and average across them
    selected_layers = hidden_states[layers[0]:layers[1] + 1]
    stacked_layers = torch.stack(selected_layers)  # (num_layers, batch, seq_len, hidden_dim)
    embeddings = torch.mean(stacked_layers, dim=0)[0]  # (seq_len, hidden_dim)

    # List to store merged word embeddings
    embeddings_list = []
    current_word = ""
    current_subword_embs = []

    # Iterate over tokens and merge subwords (BERT tokens starting with "##")
    for token, embedding in zip(tokens, embeddings):
        # Skip special tokens like [CLS], [SEP], [PAD]
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue

        if token.startswith("##"):
            # Append subword to the current word and collect its embedding
            current_word += token[2:]
            current_subword_embs.append(embedding)
        else:
            # If a previous word exists, average its subword embeddings
            if current_word:
                avg_embedding = torch.mean(torch.stack(current_subword_embs), dim=0)
                embeddings_list.append(avg_embedding.cpu())

            # Start a new word
            current_word = token
            current_subword_embs = [embedding]

    # Handle the final word (if any subwords remain unmerged)
    if current_word and current_subword_embs:
        avg_embedding = torch.mean(torch.stack(current_subword_embs), dim=0)
        embeddings_list.append(avg_embedding.cpu())

    return embeddings_list


# %% 
def string_embedding(
        text: str,
        layers: Tuple[int, int] = (9, 12)
        ) -> torch.Tensor:
    """
    Generates a single, aggregated sentence/document-level embedding
    by averaging token embeddings from selected transformer layers.

    The function:
        1. Tokenizes the input text and computes hidden states from BERT.
        2. Selects a given range of layers and averages them.
        3. Applies the attention mask to ignore padding tokens.
        4. Returns a mean-pooled embedding vector representing the full text.

    Parameters
    ----------
    text : str
        Input text to embed.
    layers : Tuple[int, int], optional
        Range of layers (inclusive) to average across. Defaults to (9, 12).

    Returns
    -------
    torch.Tensor
        Tensor of shape (hidden_dim,) representing the mean embedding
        of the entire input text.
    """

    # Tokenize and move tensors to the correct device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of hidden states for all layers

    # Select and average embeddings from the specified layers
    selected = hidden_states[layers[0]:layers[1] + 1]
    stacked_layers = torch.stack(selected)  # Shape: (num_layers, batch, seq_len, hidden_dim)
    token_embeddings = stacked_layers.mean(dim=0)[0]  # Mean across layers → (seq_len, hidden_dim)

    # Apply attention mask to zero out padding tokens
    attention_mask = inputs['attention_mask'].squeeze(0).unsqueeze(-1)  # (seq_len, 1)
    masked_embeddings = token_embeddings * attention_mask  # Masked tokens = 0

    # Compute mean pooling over non-padding tokens
    sum_embeddings = masked_embeddings.sum(dim=0)  # Sum across sequence
    valid_tokens = attention_mask.sum(dim=0).clamp(min=1e-9)  # Avoid division by zero
    mean_embedding = sum_embeddings / valid_tokens

    return mean_embedding.cpu()


# %%
