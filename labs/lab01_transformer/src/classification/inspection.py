"""Educational forward-pass inspection, separate from the training model."""
import math
import pandas as pd
import torch
from ..common.text import PAD


@torch.no_grad()
def inspect_classifier_pipeline(model, input_ids):
    """Expose actual intermediate shapes and verify agreement with model.forward."""
    was_training = model.training
    model.eval()
    try:
        input_ids = input_ids.to(next(model.parameters()).device)
        padding_mask = input_ids.eq(PAD)
        if padding_mask.all(1).any():
            raise ValueError("Each sequence must contain a non-PAD token")
        embeddings = model.embedding(input_ids) * math.sqrt(model.config["d_model"])
        positional = model.position(embeddings)
        encoded = model.encoder(positional, src_key_padding_mask=padding_mask)
        pooled = model.masked_mean(encoded, padding_mask)
        logits = model.classifier(model.dropout(pooled))
        torch.testing.assert_close(logits, model(input_ids))
        stages = dict(input_ids=input_ids, padding_mask=padding_mask, embedding=embeddings,
                      positional_encoding=positional, encoder_output=encoded, pooled=pooled, logits=logits)
        return pd.DataFrame([{"stage": name, "shape": tuple(value.shape), "dtype": str(value.dtype)}
                             for name, value in stages.items()]).set_index("stage")
    finally:
        model.train(was_training)
