# %%
import hypernetx as hnx
import networkx as nx
from typing import List, Tuple

def create_hypergraph(segmented_doc: Tuple[str, List[str], List[str]]) -> hnx.Hypergraph:
    """
    Parameters
    ----------
    segmented_doc : tuple
        A tuple of (document_text, sentences, words) as returned by `segment_documents`.

    Returns
    -------
    hnx.Hypergraph
        A hypergraph where each sentence is a hyperedge connecting its words.
    """
    doc_text, sentences, words = segmented_doc

    # Build a dictionary of hyperedges: sentence_id → list of words
    hyperedges = {}
    for i, sent in enumerate(sentences):
        sent_words = [w for w in sent.split() if w in words]
        hyperedges[f"sentence_{i}"] = set(sent_words)

    # Create hypergraph
    H = hnx.Hypergraph(hyperedges)
    return H
