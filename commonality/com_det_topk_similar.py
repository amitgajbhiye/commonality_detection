import os
import sys
from pathlib import Path

import gensim.downloader as api
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))


def clean_text(word):
    t = word.strip().replace("_", " ").lower().split()

    return " ".join(t)


def read_data(file_path):
    with open(file_path, "r") as in_file:
        lines = in_file.read().splitlines()
        lines = [clean_text(line) for line in lines if line != ""]

    return lines


def get_similar_words(
    concept_1_list, top_k_similar_concepts, out_fname, embedding_model
):
    """
    Get cosine similar top_k_similar_concepts concepts
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

    all_con_and_top_k_similar_words = []

    def get_top_k_similar_word(
        con=None, multiword=None, top_k_similar_concepts=top_k_similar_concepts
    ):
        if con:
            print(flush=True)
            print(f"concept_in_vocab : {con}", flush=True)
            assert multiword is None, "multiword not None"
            similar_words_and_scores = vector_model.most_similar(
                con, topn=top_k_similar_concepts
            )
            con_and_similar_words = [
                (con, sim_word, sim_score)
                for sim_word, sim_score in similar_words_and_scores
            ]

            print(
                f"con_and_similar_words: {con_and_similar_words}", flush=True, end="\n"
            )

            all_con_and_top_k_similar_words.extend(con_and_similar_words)

        elif multiword:
            assert con is None, "con not None"
            print(flush=True)
            print(f"multiword_concept_in_vocab : {multiword}", flush=True)

            multiword_mean_vec = np.mean(
                np.vstack([vector_model[w] for w in multiword]), axis=0
            )
            print(f"multiword_mean_vec_shape : {multiword_mean_vec.shape}", flush=True)

            multiword_similar_words_and_scores = vector_model.most_similar(
                multiword_mean_vec, topn=top_k_similar_concepts
            )
            multiword_con_and_similar_words = [
                (" ".join(multiword), sim_word, sim_score)
                for sim_word, sim_score in multiword_similar_words_and_scores
            ]
            print(
                f"multiword_con_and_similar_words: {multiword_con_and_similar_words}",
                flush=True,
                end="\n",
            )

            all_con_and_top_k_similar_words.extend(multiword_con_and_similar_words)

    ########################

    ########################
    c_word, c_hyphen_word, c_underscore_word, c_word_not_found = 0, 0, 0, 0
    c_multi_word = 0
    vocab_word, hyphen_word, underscore_word, word_not_found = [], [], [], []
    multi_word_list = []

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
            get_top_k_similar_word(con=con)

            c_word += 1
            vocab_word.append(con)

        elif hyphen_con in vocab:
            get_top_k_similar_word(con=hyphen_con)

            c_hyphen_word += 1
            hyphen_word.append(hyphen_con)

        elif underscore_con in vocab:
            get_top_k_similar_word(con=underscore_con)

            c_underscore_word += 1
            underscore_word.append(underscore_con)

        elif con_len >= 2:
            multi_word = [word for word in con_split if word in vocab]

            if " ".join(multi_word) == con:
                print(flush=True)
                print(f"multiword_concept_found : {con}", flush=True)

                get_top_k_similar_word(con=None, multiword=multi_word)

                c_multi_word += 1
                multi_word_list.append(" ".join(multi_word))

            else:
                c_word_not_found += 1
                word_not_found.append(con)
                print(flush=True)
                print(f"multiword_concept_not_in_vocab : {con}", flush=True)

        else:
            c_word_not_found += 1
            word_not_found.append(con)

            print(flush=True)
            print(f"concept_not_in_vocab : {con}", flush=True)

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

    print("Finished getting similar data.")
    print(f"Now Clustering data ....", flush=True)
    df = pd.DataFrame(
        all_con_and_top_k_similar_words, columns=["concept_1", "concept_2", "sim_score"]
    )

    print(f"all_con_and_top_k_similar_words_df.shape: {df.shape}", flush=True)

    df = df.sort_values(by=["concept_2"], inplace=False)

    df["concept_1"] = df["concept_1"].apply(clean_text)
    df["concept_2"] = df["concept_2"].apply(clean_text)

    df.to_csv(out_fname, header=True, index=False, sep="\t")


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

    top_k_similar_concepts = 50

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

            embedding_files = [
                "word2vec-google-news-300",
                "/scratch/c.scmag3/static_embeddings/numberbatch/numberbatch-en-19.08.txt",
                "/scratch/c.scmag3/static_embeddings/fasttext/crawl-300d-2M-subword.vec",
            ]
            out_fnames = [
                os.path.join(
                    output_dir,
                    f"word2vec_top{top_k_similar_concepts}_similar_words.tsv",
                ),
                os.path.join(
                    output_dir,
                    f"numberbatch_top{top_k_similar_concepts}_similar_words.tsv",
                ),
                os.path.join(
                    output_dir,
                    f"fasttext_top{top_k_similar_concepts}_similar_words.tsv",
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
                    top_k_similar_concepts=top_k_similar_concepts,
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

            embedding_files = [
                "word2vec-google-news-300",
                "/scratch/c.scmag3/static_embeddings/numberbatch/numberbatch-en-19.08.txt",
                "/scratch/c.scmag3/static_embeddings/fasttext/crawl-300d-2M-subword.vec",
            ]
            out_fnames = [
                os.path.join(
                    output_dir,
                    f"word2vec_top{top_k_similar_concepts}_similar_words.tsv",
                ),
                os.path.join(
                    output_dir,
                    f"numberbatch_top{top_k_similar_concepts}_similar_words.tsv",
                ),
                os.path.join(
                    output_dir,
                    f"fasttext_top{top_k_similar_concepts}_similar_words.tsv",
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
                    top_k_similar_concepts=top_k_similar_concepts,
                    embedding_model=emb_file,
                    out_fname=out_file,
                )
