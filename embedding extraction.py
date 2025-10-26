# %%
import pandas as pd
import preprocessing
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple
import pandas as pd

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to saved model directory
model_path = "essays/bert-base-personality"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()
# %%
segments = data['segmented_text_list']
labels = data['labels']
print(device)

# %%
def string_components_embeddings(
        text: str,
        layers: Tuple[int, int] = (9, 12)
        ) -> List[torch.Tensor]:
  
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple all of the layers

    # Average specified layers (e.g., 4–6)
    selected_layers = hidden_states[layers[0]:layers[1] + 1]
    stacked_layers = torch.stack(selected_layers)  # (num_layers, batch, seq_len, hidden_dim)
    embeddings = torch.mean(stacked_layers, dim=0)[0]  # (seq_len, hidden_dim)

    # words_list = []
    embeddings_list = []

    current_word = ""
    current_subword_embs = []

    for token, embedding in zip(tokens, embeddings):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue

        if token.startswith("##"):
            current_word += token[2:]
            current_subword_embs.append(embedding)
        else:
            if current_word:
                avg_embedding = torch.mean(torch.stack(current_subword_embs), dim=0)
                # words_list.append(current_word)
                embeddings_list.append(avg_embedding.cpu())

            current_word = token
            current_subword_embs = [embedding]

    if current_word and current_subword_embs:
        avg_embedding = torch.mean(torch.stack(current_subword_embs), dim=0)
        # words_list.append(current_word)
        embeddings_list.append(avg_embedding.cpu())

    return embeddings_list

# %% 
def string_embedding(
        text: str,
        layers: Tuple[int, int] = (9, 12)
        ) -> torch.Tensor:

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of all layers

    # Stack and average selected layers
    selected = hidden_states[layers[0]:layers[1]+1]  # e.g., layers 10,11,12
    stacked_layers = torch.stack(selected)  # Shape: (num_layers, batch, seq_len, hidden_dim)
    token_embeddings = stacked_layers.mean(dim=0)[0]  # Mean across layers → (seq_len, hidden_dim)

    # Apply attention mask to exclude padding
    attention_mask = inputs['attention_mask'].squeeze(0).unsqueeze(-1)  # (seq_len, 1)
    masked_embeddings = token_embeddings * attention_mask  # Masked tokens zeroed out
    sum_embeddings = masked_embeddings.sum(dim=0)  # Sum across seq_len
    valid_tokens = attention_mask.sum(dim=0).clamp(min=1e-9)  # Avoid division by zero
    mean_embedding = sum_embeddings / valid_tokens  # Final mean
    embedding = mean_embedding.cpu()

    return embedding

# %%
docs = []
for j in tqdm(range(0, len(segments))):
    sentences_sub_words = []
    doc = segments[j][1]
    for i in range(0,len(doc)):
        embeddings = string_embedding(doc[i]), string_components_embeddings(doc[i])
        sentences_sub_words.append(embeddings)
    docs.append(sentences_sub_words)

# %%
hirarchical_embeddings = []
for i in tqdm(range(0, len(segments))):
    d = string_embedding(segments[i][0]), docs[i]
    hirarchical_embeddings.append(d)

# %%
