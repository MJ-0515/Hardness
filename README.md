# Hardness

This repository implements the Proxy-dominant Robust Multimodal Fusion (P-RMF) architecture for sentiment analysis. Key modules:

- `basic_layers.py`: Core transformer-based building blocks including cross-modal attention, gradient reversal, and encoder/decoder stacks used throughout the model.
- `bert.py`: Wrapper around Hugging Face BERT/Roberta models to produce contextualized text embeddings with optional fine-tuning.
- `generate_proxy_modality.py`: Variational autoencoder pipeline that produces a proxy modality and modality weights by reconstructing inputs and encouraging agreement between modalities.
- `P_RMF.py`: Top-level model composition that extracts unimodal features, builds a proxy modality, performs cross-modal fusion, reconstructs inputs when available, and outputs sentiment predictions.

Typical workflow:
1. Text tokens are encoded by `BertTextEncoder`, while audio/video features are projected and passed through lightweight transformers.
2. `Generate_Proxy_Modality` blends latent representations via modality-aware weights to create a proxy representation.
3. A gradient reversal layer decorrelates the proxy from input biases before `CrossmodalEncoder` fuses it with unimodal streams.
4. The fused representation is pooled for sentiment prediction; when complete modalities are present, reconstruction losses help stabilize training and compute KL regularization.

The code is organized so the `build_model` function in `P_RMF.py` constructs the full network from a configuration dictionary. Adjust transformer depths, hidden sizes, and dropout settings through that config to experiment with model capacity and regularization.
