import numpy as np
import pickle
import sys
import os
from pathlib import Path
from gensim.models import KeyedVectors
import csv

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

    vocab = np.array(list(vector_model.key_to_index.keys()), dtype=str)

    print(f"Vocab Len : {vocab.shape}", flush=True)
    print(vocab, flush=True)

    def get_similarity_score(con):
        all_sim_scores = vector_model.most_similar(con, topn=None)

        index_thresh = np.argwhere(all_sim_scores > sim_thresh).flatten()

        sim_words = vocab[index_thresh]
        sim_scores = all_sim_scores[index_thresh]

        index_sim_dict = {k: v for k, v in zip(index_thresh, sim_scores)}

        sorted_index_sim_dict = sorted(
            index_sim_dict.items(), key=lambda x: x[1], reverse=True
        )

        sorted_sim_words = [
            (con, vocab[idx], score) for idx, score in sorted_index_sim_dict
        ]

        print(f"Concept : {con}", flush=True)
        print(f"sorted_sim_words: {sorted_sim_words}", flush=True)

        print(flush=True)

        return sorted_sim_words

    c_word, c_hyphen_word, c_underscore_word, c_word_not_found = 0, 0, 0, 0
    vocab_word, hyphen_word, underscore_word, word_not_found = [], [], [], []

    all_con_similar_data = []

    for con in concept_1_list:
        con_split = con.strip().split()
        con_len = len(con_split)

        if con_len >= 2:
            hyphen_con = "-".join(con_split)
            underscore_con = "_".join(con_split)
        else:
            hyphen_con = None
            underscore_con = None

        if con in vocab:
            con_sim_word_score = get_similarity_score(con=con)
            all_con_similar_data.extend(con_sim_word_score)

            c_word = +1
            vocab_word.append(con)

        elif con_len >= 2 and hyphen_con in vocab:
            hyphen_con_sim_word_score = get_similarity_score(con=hyphen_con)
            all_con_similar_data.extend(hyphen_con_sim_word_score)

            c_hyphen_word += 1
            hyphen_word.append(hyphen_con)

        elif con_len >= 2 and underscore_con in vocab:
            underscore_con_sim_word_score = get_similarity_score(con=underscore_con)
            all_con_similar_data.extend(underscore_con_sim_word_score)

            c_underscore_word += 1
            underscore_word.append(underscore_con)

        else:
            c_word_not_found += 1
            word_not_found.append(con)

            print(f"Concept not in Vocab : {con}", flush=True)
            print(flush=True)

    print(f"hyphen_word : {c_hyphen_word}, {hyphen_word}", flush=True)
    print(f"underscore_word : {c_underscore_word}, {underscore_word}", flush=True)
    print(f"con_not_in_vocab : {c_word_not_found}, {word_not_found}", flush=True)

    with open("numberbatch_con_similar.txt", "w") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerows(all_con_similar_data)


embedding_file = (
    "/scratch/c.scmag3/static_embeddings/numberbatch/numberbatch-en-19.08.txt"
)
concept_1_file = "datasets/ufet_clean_types.txt"

vector_model = create_vector_model(fname=embedding_file)

concept_1_list = read_data(file_path=concept_1_file)
print(f"Num concepts : {len(concept_1_list)}", flush=True)


get_similar_words(
    embedding_fname=embedding_file, concept_1_list=concept_1_list, sim_thresh=0.90
)
