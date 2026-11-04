from sentence_transformers import SentenceTransformer
import torch
import my_solution.config


embedding_model = SentenceTransformer(
    my_solution.config.EMBEDDING_MODEL_NAME,
    revision=my_solution.config.EMBEDDING_MODEL_REVISION,
    device="cpu",
    model_kwargs={"dtype": torch.float32},
)


def embed_texts(text: str) -> list[list[float]]:
    """Embed a single string into a vector."""
    return embedding_model.encode(
        [text], 
        convert_to_numpy=True, 
        normalize_embeddings=True).tolist()