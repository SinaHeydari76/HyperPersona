# Hyper Persona

Hyper Persona is a research framework that models the multi-level structure of written-text (document, sentence, and word) using hypergraph and hierarchical graph representations.
It integrates transformer-based graph learning to predict personality traits from text, capturing both semantic and syntactic relationships in a unified model.

### Multi-Level Segmentation

Splits text into document → sentence → word hierarchy usesing NLP preprocessing (spaCy) for syntactic and lexical segmentation.

# Multi-Level Vectorization

Generates embeddings for texual components using BERT to for graph construction.

# Hypergraph Construction

Builds hypergraph structures capturing multi-level dependencies among linguistic units. Nodes represent words; hyperedges represent document/sentences to model shared contextual or semantic relationships.

# Hypergraph to Hierarchical Graph Conversion

Converts the hypergraph into a hierarchical graph compatible with PyTorch Geometric (PyG).

# Model

Defines and trains a transformer-based graph neural network (GNN). Supports multi-trait personality prediction (e.g., Big Five: OCEAN). Includes its own data loader, training loop, validation split, and evaluation metrics.

# Citation:
````
@article{heydari2025hyperpersona,
  title={Hyper Persona: A Multi-Level Hypergraph Framework for Text-Based Personality Prediction},
  author={Sina Heydari and Majid Ramezani},
  year={2025},
  journal={Preprint}
}
````
