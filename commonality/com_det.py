import numpy as np
import pickle
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))


def load_vectors(fname, embed_dim, embed_format=None, out_file=None):
    embeddings_dict = {}

    print("Loading the input embedding file ....")

    with open(fname, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if embed_format == "word2vec" and idx == 0:
                num_words, vector_dim = line.split()
                print(
                    f"The input file is in Word2Vec formate ; Num Words : {num_words}, Vectors Dim {vector_dim}"
                )
                continue

            split_line = line.split()
            word = " ".join(split_line[0 : len(split_line) - embed_dim])
            embedding = np.array([float(val) for val in split_line[-embed_dim:]])
            embeddings_dict[word] = embedding

        print("Done.\n" + str(len(embeddings_dict)) + " words loaded!")

    if out_file:
        with open(out_file, "wb") as pkl_file:
            pickle.dump(embeddings_dict, pkl_file)

        print(f"Embedding Pickle is Saved at : {out_file}")

    return embeddings_dict


# For Fasttext
file_name = "/scratch/c.scmag3/static_embeddings/fasttext/crawl-300d-2M-subword.vec"
embed_dim = 300
embed_format = "word2vec"
out_file = "/scratch/c.scmag3/static_embeddings/fasttext_crawl_300d_2M_subword.vec_embedding_dict.pkl"


load_vectors(
    fname=file_name, embed_dim=embed_dim, embed_format=embed_format, out_file=out_file
)
