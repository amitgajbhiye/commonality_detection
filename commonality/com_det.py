import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import gensim.downloader as api

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))


def clean_text(word):
    t = word.strip().replace("_", " ").split().lower()

    return " ".join(t)


def read_data(file_path):
    with open(file_path, "r") as in_file:
        lines = in_file.read().splitlines()
        lines = [clean_text(line) for line in lines if line != ""]

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


def create_clusters(concept_similar_list, output_file_name):
    print(f"Clustering data ....", flush=True)
    df = pd.DataFrame(
        concept_similar_list, columns=["concept_1", "concept_2", "sim_score"]
    )

    print(f"input_shape: {df.shape}", flush=True)

    con12_df = df[["concept_1", "concept_2"]]
    df["concept_2_counts"] = con12_df.groupby(by=["concept_2"]).transform("count")

    print(f"df_shape: {df.shape}", flush=True)

    df = df.sort_values(by=["concept_2"], inplace=False)

    print(f"clustered_data_shape: {df.shape}", flush=True)

    df["concept_1"] = df["concept_1"].apply(clean_text)
    df["concept_2"] = df["concept_2"].apply(clean_text)

    df.to_csv(output_file_name, header=True, index=False, sep="\t")


def get_similar_words(concept_1_list, sim_thresh, out_fname, embedding_model):
    """
    Get cosine similar to concepts
    """

    if os.path.isfile(embedding_model):
        print(
            f"Loading the Embedding Model from the File : {embedding_model}", flush=True
        )
        vector_model = KeyedVectors.load_word2vec_format(embedding_model, binary=False)
    else:
        print(f"Loading Embedding Model from Gensim...", flush=True)
        vector_model = api.load("word2vec-google-news-300", return_path=False)

    vocab = np.array(list(vector_model.key_to_index.keys()), dtype=str)

    print(f"Vocab Len : {vocab.shape}", flush=True)

    def get_similarity_score(con=None, multiword=None):
        if con:
            assert multiword is None, "multiword not None"
            all_sim_scores = vector_model.most_similar(con, topn=None)

        elif multiword:
            assert con is None, "con not None"
            multiword_mean_vec = np.mean(
                np.vstack([vector_model[w] for w in multiword]), axis=0
            )

            print(f"multiword_mean_vec_shape : {multiword_mean_vec.shape}", flush=True)

            all_sim_scores = vector_model.most_similar(multiword_mean_vec, topn=None)

        index_thresh = np.argwhere(all_sim_scores > sim_thresh).flatten()

        sim_scores = all_sim_scores[index_thresh]

        index_sim_dict = {
            idx: sim_score for idx, sim_score in zip(index_thresh, sim_scores)
        }

        sorted_index_sim_dict = sorted(
            index_sim_dict.items(), key=lambda x: x[1], reverse=True
        )

        if con:
            sorted_sim_words = [
                (con, vocab[idx], score) for idx, score in sorted_index_sim_dict
            ]

            print(f"con : {con}", flush=True)
            print(f"sorted_sim_words: {sorted_sim_words}", flush=True)
            print(flush=True)

        elif multiword:
            multi_word_con = " ".join(multiword)
            sorted_sim_words = [
                (multi_word_con, vocab[idx], score)
                for idx, score in sorted_index_sim_dict
            ]

            print(f"multi_word_con : {multi_word_con}", flush=True)
            print(f"sorted_sim_words: {sorted_sim_words}", flush=True)
            print(flush=True)

        return sorted_sim_words

    ########################

    ########################
    c_word, c_hyphen_word, c_underscore_word, c_word_not_found = 0, 0, 0, 0
    c_multi_word = 0
    vocab_word, hyphen_word, underscore_word, word_not_found = [], [], [], []
    multi_word_list = []

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

            c_word += 1
            vocab_word.append(con)

        elif hyphen_con in vocab:
            hyphen_con_sim_word_score = get_similarity_score(con=hyphen_con)
            all_con_similar_data.extend(hyphen_con_sim_word_score)

            c_hyphen_word += 1
            hyphen_word.append(hyphen_con)

        elif underscore_con in vocab:
            underscore_con_sim_word_score = get_similarity_score(con=underscore_con)
            all_con_similar_data.extend(underscore_con_sim_word_score)

            c_underscore_word += 1
            underscore_word.append(underscore_con)

        elif con_len >= 2:
            multi_word = [word for word in con_split if word in vocab]

            if " ".join(multi_word) == con:
                print(f"multiword_concept_found : {con}", flush=True)

                multi_con_sim_word_score = get_similarity_score(
                    con=None, multiword=multi_word
                )
                all_con_similar_data.extend(multi_con_sim_word_score)

                c_multi_word += 1
                multi_word_list.append(" ".join(multi_word))

            else:
                c_word_not_found += 1
                word_not_found.append(con)
                print(f"single_concept_not_in_vocab : {con}", flush=True)
                print(flush=True)

        else:
            c_word_not_found += 1
            word_not_found.append(con)

            print(f"single_concept_not_in_vocab : {con}", flush=True)
            print(flush=True)

    print(f"individual_c_word : {c_word}", flush=True)
    print(flush=True)
    print(f"hyphen_word : {c_hyphen_word}, {hyphen_word}", flush=True)
    print(flush=True)
    print(f"underscore_word : {c_underscore_word}, {underscore_word}", flush=True)
    print(flush=True)
    print(f"multi_word_list : {c_multi_word}, {multi_word_list}", flush=True)
    print(flush=True)
    print(f"concept_not_in_vocab : {c_word_not_found}, {word_not_found}", flush=True)
    print(flush=True)

    with open(out_fname, "w") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerows(all_con_similar_data)

    clustered_fname = f"{os.path.splitext(out_fname)[0]}_concept2counts.tsv"

    create_clusters(
        concept_similar_list=all_con_similar_data, output_file_name=clustered_fname
    )


if __name__ == "__main__":
    ############# For UFET Experiments #############

    # concept_1_file = "datasets/ufet_clean_types.txt"
    # concept_1_list = read_data(file_path=concept_1_file)
    # print(f"Num concepts : {len(concept_1_list)}", flush=True)

    # similarity_thresh = 0.50

    # embedding_files = [
    #     "word2vec-google-news-300",
    #     "/scratch/c.scmag3/static_embeddings/numberbatch/numberbatch-en-19.08.txt",
    #     "/scratch/c.scmag3/static_embeddings/fasttext/crawl-300d-2M-subword.vec",
    # ]
    # out_fnames = [
    #     f"output_files/word2vec_ueft_label_similar_{similarity_thresh}thresh.txt",
    #     f"output_files/numberbatch_ueft_label_similar_{similarity_thresh}thresh.txt",
    #     f"output_files/fasttext_ueft_label_similar_{similarity_thresh}thresh.txt",
    # ]

    # for emb_file, out_file in zip(embedding_files, out_fnames):
    #     print("*" * 60, flush=True)
    #     print("new_run", flush=True)
    #     print(f"embedding_file : {emb_file}", flush=True)
    #     print(f"output_file :{out_file}", flush=True)
    #     print("*" * 60)

    #     get_similar_words(
    #         concept_1_list=concept_1_list,
    #         sim_thresh=similarity_thresh,
    #         embedding_model=emb_file,
    #         out_fname=out_file,
    #     )

    ############# For Classification Vocab Experiments #############
    exp_name = str(sys.argv[1])  # classi_vocab, ontology_comp
    dataset = str(sys.argv[2])  # babelnet, wordnet, xmcrae, sumo, all_except_sumo

    if exp_name == "classi_vocab":
        if dataset == "babelnet":
            concept1_file_list = [
                (
                    "datasets/classification_vocabs/BabelnetDomain.txt",
                    "output_files/classification_vocabs/bablenet_domain",
                )
            ]
        elif dataset == "wordnet":
            concept1_file_list = [
                (
                    "datasets/classification_vocabs/WordNet.txt",
                    "output_files/classification_vocabs/wordnet",
                )
            ]
        elif dataset == "xmcrae":
            concept1_file_list = [
                (
                    "datasets/classification_vocabs/X-McRae.txt",
                    "output_files/classification_vocabs/xmcrae",
                )
            ]
        else:
            raise Exception(f"Specify dataset from: babelnet, wordnet, xmcrae")

        for concept_1_file, output_dir in concept1_file_list:
            concept_1_list = read_data(file_path=concept_1_file)
            print(f"concept_1_file : {concept_1_file}", flush=True)
            print(f"Num concepts : {len(concept_1_list)}", flush=True)
            print(f"concept_1_list : {concept_1_list}", flush=True)

            similarity_thresh = 0.50

            embedding_files = [
                "word2vec-google-news-300",
                "/scratch/c.scmag3/static_embeddings/numberbatch/numberbatch-en-19.08.txt",
                "/scratch/c.scmag3/static_embeddings/fasttext/crawl-300d-2M-subword.vec",
            ]
            out_fnames = [
                os.path.join(
                    output_dir, f"word2vec_similarthresh{similarity_thresh}.tsv"
                ),
                os.path.join(
                    output_dir, f"numberbatch_similarthresh{similarity_thresh}.tsv"
                ),
                os.path.join(
                    output_dir, f"fasttext_similarthresh{similarity_thresh}.tsv"
                ),
            ]

            for emb_file, out_file in zip(embedding_files, out_fnames):
                print("*" * 60, flush=True)
                print("new_run", flush=True)
                print("*" * 60)

                print(f"concept_1_list : {concept_1_file}", flush=True)
                print(f"output_dir : {output_dir}", flush=True)
                print(f"embedding_file : {emb_file}", flush=True)
                print(f"output_file :{out_file}", flush=True, end="\n")

                get_similar_words(
                    concept_1_list=concept_1_list,
                    sim_thresh=similarity_thresh,
                    embedding_model=emb_file,
                    out_fname=out_file,
                )
    elif exp_name == "ontology_comp":
        if dataset == "sumo":
            concept1_file_list = [
                (
                    "datasets/ontology_completion/sumo-1_split.txt",
                    "output_files/ontology_completion/sumo",
                )
            ]
        elif dataset == "all_except_sumo":
            concept1_file_list = [
                (
                    "datasets/ontology_completion/economy-1_split.txt",
                    "output_files/ontology_completion/economy",
                ),
                (
                    "datasets/ontology_completion/olympics-1_split.txt",
                    "output_files/ontology_completion/olympics",
                ),
                (
                    "datasets/ontology_completion/transport-1_split.txt",
                    "output_files/ontology_completion/transport",
                ),
                (
                    "datasets/ontology_completion/wine-1_split.txt",
                    "output_files/ontology_completion/wine",
                ),
            ]
        else:
            raise Exception(f"Specify dataset from: sumo, all_except_sumo")

        for concept_1_file, output_dir in concept1_file_list:
            concept_1_list = read_data(file_path=concept_1_file)
            print(f"concept_1_file : {concept_1_file}", flush=True)
            print(f"Num concepts : {len(concept_1_list)}", flush=True)
            print(f"concept_1_list : {concept_1_list}", flush=True)

            similarity_thresh = 0.50

            embedding_files = [
                "word2vec-google-news-300",
                "/scratch/c.scmag3/static_embeddings/numberbatch/numberbatch-en-19.08.txt",
                "/scratch/c.scmag3/static_embeddings/fasttext/crawl-300d-2M-subword.vec",
            ]
            out_fnames = [
                os.path.join(
                    output_dir, f"word2vec_similarthresh{similarity_thresh}.tsv"
                ),
                os.path.join(
                    output_dir, f"numberbatch_similarthresh{similarity_thresh}.tsv"
                ),
                os.path.join(
                    output_dir, f"fasttext_similarthresh{similarity_thresh}.tsv"
                ),
            ]

            for emb_file, out_file in zip(embedding_files, out_fnames):
                print("*" * 60, flush=True)
                print("new_run", flush=True)
                print("*" * 60)

                print(f"concept_1_list : {concept_1_file}", flush=True)
                print(f"output_dir : {output_dir}", flush=True)
                print(f"embedding_file : {emb_file}", flush=True)
                print(f"output_file :{out_file}", flush=True, end="\n")

                get_similar_words(
                    concept_1_list=concept_1_list,
                    sim_thresh=similarity_thresh,
                    embedding_model=emb_file,
                    out_fname=out_file,
                )
