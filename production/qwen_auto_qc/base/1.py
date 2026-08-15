import numpy as np


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"query embedding_dim {query.shape[-1]} != key embedding_dim {key.shape[-1]}"
        )
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            f"key num_keys {key.shape[-2]} != value num_keys {value.shape[-2]}"
        )

    embedding_dim = query.shape[-1]
    scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(embedding_dim)

    if mask is not None:
        if mask.shape != scores.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match scores shape {scores.shape}"
            )
        scores = np.where(mask, scores, -np.inf)

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = np.matmul(attention_weights, value)
    return output, attention_weights


def main():
    query = np.array([[[1.0, 0.0]]])

    key = np.array(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ]
    )

    value = np.array(
        [
            [
                [10.0, 0.0],
                [0.0, 20.0],
            ]
        ]
    )
    mask = None

    output, attention_weights = scaled_dot_product_attention(query, key, value, mask)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)


if __name__ == "__main__":
    main()
