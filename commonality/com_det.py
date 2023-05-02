import numpy as np
import pickle
import sys
import os
from pathlib import Path
from gensim.models import KeyedVectors

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))


def read_data(file_path):
    with open(file_path, "r") as in_file:
        lines = in_file.read().splitlines()

    return lines


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


with open(
    "/scratch/c.scmag3/static_embeddings/fasttext_crawl_300d_2M_subword.vec_embedding_dict.pkl",
    "rb",
) as pkl_inp:
    emb_dict = pickle.load(pkl_inp)


# For Fasttext

# file_name = "/scratch/c.scmag3/static_embeddings/fasttext/crawl-300d-2M-subword.vec"
# embed_dim = 300
# embed_format = "word2vec"
# out_file = "/scratch/c.scmag3/static_embeddings/fasttext_crawl_300d_2M_subword.vec_embedding_dict.pkl"


# load_vectors(
#     fname=file_name, embed_dim=embed_dim, embed_format=embed_format, out_file=out_file
# )

# For Numberbatch
# file_name = "/scratch/c.scmag3/static_embeddings/numberbatch/numberbatch-en-19.08.txt"
# embed_dim = 300
# embed_format = "word2vec"
# out_file = "/scratch/c.scmag3/static_embeddings/numberbatch-en-19.08_embedding_dict.pkl"


# load_vectors(
#     fname=file_name, embed_dim=embed_dim, embed_format=embed_format, out_file=out_file
# )


def create_vector_model(fname):
    vector_model = KeyedVectors.load_word2vec_format(fname, binary=False)

    return vector_model


def get_similar_words(embedding_fname, concept_1_list, sim_thresh):
    vector_model = KeyedVectors.load_word2vec_format(embedding_fname, binary=False)
    vocab = np.array(vector_model.key_to_index.keys(), dtype=str)

    def get_similarity_score(con):
        sim_scores = vector_model.most_similar(con, topn=None)

        index_thresh = np.argwhere(sim_scores > sim_thresh).flatten()

        sim_words = vocab[index_thresh]
        sim_scores = sim_scores[index_thresh]

        print(f"Concept : {con}", flush=True)
        print(f"sim_words: {sim_words}", flush=True)
        print(f"sim_scores : {sim_scores}", flush=True)
        print(f"index_thresh : {index_thresh}", flush=True)

    words_in_vocab = []
    for con in concept_1_list:
        if con in vocab:
            get_similarity_score(con=con)
        else:
            print(f"Concept not in Vocab : {con}")


embedding_file = (
    "/scratch/c.scmag3/static_embeddings/numberbatch/numberbatch-en-19.08.txt"
)
concept_1_file = "datasets/ufet_clean_types.txt"

vector_model = create_vector_model(fname=embedding_file)

concept_1_list = read_data(file_path=concept_1_file)
print(f"Num concepts : {len(concept_1_list)}")


get_similar_words(
    embedding_fname=embedding_file, concept_1_list=concept_1_list, sim_thresh=0.90
)
