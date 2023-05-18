import csv
import os
import pickle
import sys
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import gensim.downloader as api

from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))


def get_word_vectors(
    embedding_model,
    word_in_vocab,
    multi_words=None,
):
    print(f"in get_word_vectors function", flush=True)
    vecs_word_in_vocab = embedding_model[word_in_vocab]

    print(f"word_in_vocab : {len(word_in_vocab)}", flush=True)
    print(f"vecs_word_in_vocab : {vecs_word_in_vocab.shape}", flush=True)

    if multi_words:
        vecs_multi_words = []
        for word in multi_words:
            print(f"multiword_get_vector : {word}", flush=True)
            multiword_mean_vec = np.mean(
                np.vstack([embedding_model[w] for w in word.split()]), axis=0
            )

            vecs_multi_words.append(multiword_mean_vec)

        vecs_multi_words = np.vstack(vecs_multi_words)

        print(f"multi_words : {len(multi_words)}", flush=True)
        print(f"vecs_multi_words : {vecs_multi_words.shape}", flush=True)

        all_words = np.array(word_in_vocab + multi_words)
        vecs_all_words = np.vstack((vecs_word_in_vocab, vecs_multi_words))

    else:
        all_words = np.array(word_in_vocab)
        vecs_all_words = vecs_word_in_vocab

    print(f"all_words : {len(all_words)}", flush=True)
    print(f"vecs_all_words : {vecs_all_words.shape}", flush=True, end="\n")

    return (all_words, vecs_all_words)


def get_words_in_vocab_and_vecs(concept_list, embedding_model):
    if os.path.isfile(embedding_model):
        print(
            f"Loading the Embedding Model from the File : {embedding_model}", flush=True
        )
        vector_model = KeyedVectors.load_word2vec_format(embedding_model, binary=False)
    else:
        print(f"Loading Embedding Model from Gensim...", flush=True)
        vector_model = api.load(embedding_model, return_path=False)

    vocab = np.array(list(vector_model.key_to_index.keys()), dtype=str)  # type: ignore
    print(f"vocab_len: {vocab.shape}", flush=True)

    c_word, c_hyphen_word, c_underscore_word, c_word_not_found = 0, 0, 0, 0
    c_multi_word = 0
    vocab_word, hyphen_word, underscore_word, word_not_found = [], [], [], []
    multi_word_list = []

    for con in concept_list:
        con_split = con.strip().split()
        con_len = len(con_split)

        if con_len >= 2:
            hyphen_con = "-".join(con_split)
            underscore_con = "_".join(con_split)
        else:
            hyphen_con = None
            underscore_con = None

        if con in vocab:
            c_word += 1
            vocab_word.append(con)

        elif hyphen_con in vocab:
            c_hyphen_word += 1
            hyphen_word.append(hyphen_con)

        elif underscore_con in vocab:
            c_underscore_word += 1
            underscore_word.append(underscore_con)

        elif con_len >= 2:
            multi_word = [word for word in con_split if word in vocab]

            if " ".join(multi_word).strip() == " ".join(con_split).strip():
                print(f"multiword_concept_found : {con}", flush=True)

                c_multi_word += 1
                multi_word_list.append(" ".join(multi_word))

            else:
                c_word_not_found += 1
                word_not_found.append(con)
                print(f"multiword_concept_not_in_vocab : {con}", flush=True)

        else:
            c_word_not_found += 1
            word_not_found.append(con)

            print(f"single_concept_not_in_vocab : {con}", flush=True, end="\n")

    print(f"individual_c_word : {c_word}", flush=True, end="\n")
    print(f"hyphen_word : {c_hyphen_word}, {hyphen_word}", flush=True, end="\n")
    print(
        f"underscore_word : {c_underscore_word}, {underscore_word}",
        flush=True,
        end="\n",
    )
    print(f"multi_word_list : {c_multi_word}, {multi_word_list}", flush=True, end="\n")
    print(
        f"concept_not_in_vocab : {c_word_not_found}, {word_not_found}",
        flush=True,
        end="\n",
    )

    all_words_in_vocab = vocab_word + hyphen_word + underscore_word

    print(f"all_words_in_vocab : {len(all_words_in_vocab)}", flush=True)
    print(f"multi_words : {len(multi_word_list)}", flush=True, end="\n")

    # Getting the Embedding for words
    all_words, vecs_all_words = get_word_vectors(
        embedding_model=vector_model,
        word_in_vocab=all_words_in_vocab,
        multi_words=multi_word_list,
    )

    return all_words, vecs_all_words


def get_cosine_similar_words(
    concept_1_in_vocab,
    vecs_concept_1,
    concept_2_in_vocab,
    vecs_concept_2,
    outfile,
    sim_thresh=0.5,
):
    print(f"in get_cosine_similar_words function", flush=True, end="\n")

    sim_array = cosine_similarity(vecs_concept_2, vecs_concept_1)

    print(f"vecs_concept_1_shape : {vecs_concept_1.shape}", flush=True)
    print(f"vecs_concept_2_shape : {vecs_concept_2.shape}", flush=True)
    print(f"sim_array_shape : {sim_array.shape}", flush=True, end="\n")

    with open(outfile, "w") as f:
        for idx, sim_scores in enumerate(sim_array):
            index_thresh = np.argwhere(sim_scores > sim_thresh).flatten()
            concept1_similar_to_concept2 = concept_1_in_vocab[index_thresh]

            print(f"concept_2 : {concept_2_in_vocab[idx]}", flush=True)
            print(
                f"concept1_similar_to_concept2 : {concept1_similar_to_concept2}\n",
                flush=True,
            )

            for concept1 in concept1_similar_to_concept2:
                f.write(f"{concept1}\t{concept_2_in_vocab[idx]}\n")


def read_relbert_filetered_file(inpfile):
    print(f"in read_relbert_filetered_file function", flush=True)
    df = pd.read_csv(inpfile, sep="\t", names=["concept_1", "concept_2", "rel_score"])

    concept_1 = df["concept_1"].unique()[0:500]
    concept_2 = df["concept_2"].unique()[0:700]

    print(f"num_concept_1: {concept_1.shape[0]} concept_1: {concept_1}", flush=True)
    print(f"num_concept_2: {concept_2.shape[0]} concept_2: {concept_2}", flush=True)

    return concept_1, concept_2


if __name__ == "__main__":
    relbert_filtered_files = [
        "output_files/relbert_filtered/word2vec_relbert_filetered.txt",
        "output_files/relbert_filtered/numberbatch_relbert_filetered.txt",
        "output_files/relbert_filtered/fasttext_relbert_filetered.txt",
    ]

    embedding_model_files = [
        "word2vec-google-news-300",
        "/scratch/c.scmag3/static_embeddings/numberbatch/numberbatch-en-19.08.txt",
        "/scratch/c.scmag3/static_embeddings/fasttext/crawl-300d-2M-subword.vec",
    ]

    out_complementary_clusters_file = [
        f"output_files/complementary_clusters_files/word2vec_complementary_clusters.txt",
        f"output_files/complementary_clusters_files/numberbatch_complementary_clusters.txt",
        f"output_files/complementary_clusters_files/fasttext_complementary_clusters.txt",
    ]

    for inpfile, embed_model_file, outfile in zip(
        relbert_filtered_files, embedding_model_files, out_complementary_clusters_file
    ):
        print(f"*" * 50, flush=True)
        print(f"new_run", flush=True)
        print(f"relbert_filtered_file : {inpfile}", flush=True)
        print(f"embedding_model_file : {embed_model_file}", flush=True)
        print(f"out_complementary_clusters_file : {outfile}", flush=True)
        print(f"*" * 50, flush=True, end="\n")

        concept_1, concept_2 = read_relbert_filetered_file(inpfile=inpfile)

        concept_1_in_vocab, vecs_concept_1 = get_words_in_vocab_and_vecs(
            concept_list=concept_1, embedding_model=embed_model_file
        )
        concept_2_in_vocab, vecs_concept_2 = get_words_in_vocab_and_vecs(
            concept_list=concept_2, embedding_model=embed_model_file
        )

        get_cosine_similar_words(
            concept_1_in_vocab=concept_1_in_vocab,
            vecs_concept_1=vecs_concept_1,
            concept_2_in_vocab=concept_2_in_vocab,
            vecs_concept_2=vecs_concept_2,
            outfile=outfile,
            sim_thresh=0.50,
        )
