"""
embeddings.py
Purpose: Generate dense clinical text embeddings using transformer models
         (ClinicalBERT, BioBERT, or general BERT) for readmission prediction.

Handles long clinical notes by chunking (BERT max = 512 tokens) and
averaging chunk embeddings. Supports GPU acceleration when available.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model options
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "bert": "bert-base-uncased",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}


# ---------------------------------------------------------------------------
# 1. Load model and tokenizer
# ---------------------------------------------------------------------------

def load_embedding_model(
    model_name: str = "clinicalbert",
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    """
    Load a pretrained transformer model and tokenizer.

    Parameters
    ----------
    model_name : str
        Short name ('clinicalbert', 'biobert', 'bert', 'pubmedbert')
        or a HuggingFace model ID string.
    device : str, optional
        'cuda', 'cpu', or None (auto-detect).

    Returns
    -------
    tuple
        (tokenizer, model, device)
    """
    model_id = MODEL_REGISTRY.get(model_name, model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    logger.info("Loading embedding model: %s → %s (device=%s)", model_name, model_id, device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded — %.1fM parameters, hidden_size=%d", n_params, model.config.hidden_size)

    return tokenizer, model, device


# ---------------------------------------------------------------------------
# 2. Embed a single text (with chunking for long notes)
# ---------------------------------------------------------------------------

def _embed_single_text(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int = 512,
    stride: int = 256,
    pooling: str = "mean",
) -> np.ndarray:
    """
    Embed a single text, handling documents longer than max_length by
    sliding-window chunking and averaging chunk embeddings.

    Parameters
    ----------
    text : str
        Input text string.
    tokenizer : AutoTokenizer
    model : AutoModel
    device : torch.device
    max_length : int
        Max tokens per chunk (BERT limit = 512).
    stride : int
        Overlap between chunks. 256 means 50% overlap with max_length=512.
    pooling : str
        'cls' — use [CLS] token embedding.
        'mean' — mean-pool all non-padding token embeddings (recommended).

    Returns
    -------
    np.ndarray
        Embedding vector of shape (hidden_size,).
    """
    # Tokenize the full text without truncation
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    all_input_ids = encoded["input_ids"]

    # If text fits in a single chunk, process directly
    usable_length = max_length - 2  # reserve for [CLS] and [SEP]

    if len(all_input_ids) <= usable_length:
        chunks = [all_input_ids]
    else:
        # Sliding window chunks
        chunks = []
        for start in range(0, len(all_input_ids), stride):
            chunk = all_input_ids[start : start + usable_length]
            chunks.append(chunk)
            if start + usable_length >= len(all_input_ids):
                break

    cls_id = tokenizer.cls_token_id or tokenizer.convert_tokens_to_ids("[CLS]")
    sep_id = tokenizer.sep_token_id or tokenizer.convert_tokens_to_ids("[SEP]")
    pad_id = tokenizer.pad_token_id or 0

    chunk_embeddings = []

    for chunk_ids in chunks:
        # Add [CLS] ... [SEP] and pad to max_length
        input_ids = [cls_id] + chunk_ids + [sep_id]
        attention_mask = [1] * len(input_ids)

        # Pad
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len

        input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask_t = torch.tensor([attention_mask], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t)

        last_hidden = outputs.last_hidden_state[0]  # (seq_len, hidden_size)

        if pooling == "cls":
            emb = last_hidden[0].cpu().numpy()  # [CLS] token
        else:
            # Mean pool over non-padding tokens
            mask = attention_mask_t[0].unsqueeze(-1).float()  # (seq_len, 1)
            masked = last_hidden * mask
            emb = (masked.sum(dim=0) / mask.sum()).cpu().numpy()

        chunk_embeddings.append(emb)

    # Average chunk embeddings
    return np.mean(chunk_embeddings, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Batch embed multiple texts
# ---------------------------------------------------------------------------

def embed_texts(
    texts: List[str],
    model_name: str = "clinicalbert",
    device: Optional[str] = None,
    max_length: int = 512,
    stride: int = 256,
    pooling: str = "mean",
    batch_size: int = 16,
    show_progress: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Generate embeddings for a list of clinical texts.

    Parameters
    ----------
    texts : list of str
        Clinical note texts (cleaned or raw).
    model_name : str
        Model name or HuggingFace ID.
    device : str, optional
        'cuda', 'cpu', or None (auto-detect).
    max_length : int
        Max tokens per chunk.
    stride : int
        Overlap between chunks for long documents.
    pooling : str
        'cls' or 'mean'.
    batch_size : int
        Not used for per-document chunking, reserved for future batched mode.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    tuple
        (embeddings_matrix of shape (n_docs, hidden_size), metadata dict)
    """
    tokenizer, model, dev = load_embedding_model(model_name, device=device)
    hidden_size = model.config.hidden_size

    embeddings = np.zeros((len(texts), hidden_size), dtype=np.float32)
    n_chunked = 0

    iterator = tqdm(texts, desc="Embedding texts", disable=not show_progress)
    for i, text in enumerate(iterator):
        if not text or not text.strip():
            # Zero vector for empty texts
            continue

        encoded = tokenizer(text, add_special_tokens=False)
        n_tokens = len(encoded["input_ids"])
        if n_tokens > max_length - 2:
            n_chunked += 1

        embeddings[i] = _embed_single_text(
            text, tokenizer, model, dev,
            max_length=max_length, stride=stride, pooling=pooling,
        )

    metadata = {
        "model_name": model_name,
        "model_id": MODEL_REGISTRY.get(model_name, model_name),
        "hidden_size": hidden_size,
        "n_documents": len(texts),
        "n_chunked": n_chunked,
        "pooling": pooling,
        "max_length": max_length,
        "stride": stride,
        "device": str(dev),
    }

    logger.info(
        "Embeddings complete — shape: (%d, %d), chunked: %d/%d docs, device: %s",
        len(texts), hidden_size, n_chunked, len(texts), dev,
    )

    return embeddings, metadata


# ---------------------------------------------------------------------------
# 4. Dimensionality reduction (optional, for visualization or smaller models)
# ---------------------------------------------------------------------------

def reduce_embeddings(
    embeddings: np.ndarray,
    n_components: int = 50,
    method: str = "pca",
) -> Tuple[np.ndarray, object]:
    """
    Optionally reduce embedding dimensionality for use with simpler classifiers.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_docs, hidden_size), e.g. (n, 768).
    n_components : int
        Target dimensions.
    method : str
        'pca' or 'umap'.

    Returns
    -------
    tuple
        (reduced_embeddings, fitted reducer)
    """
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        explained = reducer.explained_variance_ratio_.sum()
        logger.info(
            "PCA reduction: %d → %d dims (%.1f%% variance retained)",
            embeddings.shape[1], n_components, explained * 100,
        )
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        logger.info("UMAP reduction: %d → %d dims", embeddings.shape[1], n_components)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return reduced.astype(np.float32), reducer
