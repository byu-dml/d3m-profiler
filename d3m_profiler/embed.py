import os
import numpy as np
import sent2vec
from sentence_transformers import SentenceTransformer


SENT2VEC_MODEL = os.getenv('MODEL_WEIGHTS_DIR', '') + 'torontobooks_unigrams.bin'
SENTENCE_TRANSFORMER_MODEL = 'distilbert-base-nli-stsb-mean-tokens'


def get_encoding_method(encoder: str):
    if encoder == 'sent2vec':
        model = sent2vec.Sent2vecModel()
        model.load_model(SENT2VEC_MODEL)
        return model.embed_sentences
    elif encoder == 'sentence_transformer':
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        return model.encode
    else:
        raise NotImplementedError

    
"""
Embeds textual data in a DataFrame based on model weights and retains 
group level of dataset name along with the response variable.

Returns
-------
embedded_data: pandas.DataFrame
    The embedded DataFrame along with the group level dataset name and response
    variable.
"""
def embed(X, encoder: str):
    encoding_method = get_encoding_method(encoder)
    embeddings = []
    for i in range(len(X[0])):
        embeddings.append(encoding_method(np.char.lower(X[:, i])))
    return np.hstack(tuple(embeddings))
