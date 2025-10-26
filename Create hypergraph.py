import hypernetx as hnx
import networkx as nx
import torch
import numpy as np
from typing import List, Tuple, Dict

def create_hypergraph(
    segmented_doc: Tuple[str, List[str], List[str]],
    word_embeddings: Dict[str, torch.Tensor],
    sentence_embeddings: List[torch.Tensor],
    document_embedding: torch.Tensor
) -> hnx.Hypergraph:
    """
    Create a hypergraph from a segmented document with embeddings.

    Parameters
    ----------
    segmented_doc : tuple
        (document_text, sentences, words) as returned by `segment_documents`.
    word_embeddings : dict
        Mapping {word: embedding_tensor}.
    sentence_embeddings : list of torch.Tensor
        Embeddings corresponding to each sentence.
    document_embedding : torch.Tensor
        Embedding vector for the whole document.

    Returns
    -------
    H : hnx.Hypergraph
        Hypergraph with word, sentence, and document-level embeddings stored as attributes.
    """
    doc_text, sentences, words = segmented_doc

    # Build hyperedges: each sentence connects its words
    hyperedges = {}
    for i, sent in enumerate(sentences):
        sent_words = [w for w in sent.split() if w in word_embeddings]
        hyperedges[f"sentence_{i}"] = set(sent_words)

    H = hnx.Hypergraph(hyperedges)

    # Attach word embeddings
    for w in word_embeddings:
        if w in H.nodes:
            H.nodes[w]['feature'] = word_embeddings[w]

    # Attach sentence embeddings
    for i, emb in enumerate(sentence_embeddings):
        node_id = f"sentence_{i}"
        if node_id in H.nodes:
            H.nodes[node_id]['feature'] = emb

    # Add document node with its embedding
    H.add_node("document_0")
    H.nodes["document_0"]['feature'] = document_embedding

    return H
