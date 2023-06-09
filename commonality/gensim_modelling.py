import torch
import os
import numpy as np
import logging
import pickle
import hdbscan
import pandas as pd
import copy
import sys
from pathlib import Path

from gensim.models import KeyedVectors

from sklearn.neighbors import NearestNeighbors
from nltk.stem import WordNetLemmatizer

from relbert import RelBERT

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))

log = logging.getLogger(__name__)


class GloveVectorsGensim:
    def __init__(self, wv_format_glove_file):
        self.glove_model = KeyedVectors.load_word2vec_format(
            wv_format_glove_file, binary=False
        )

    def read_data(self, file_path):
        with open(file_path, "r") as in_file:
            lines = in_file.read().splitlines()

        return lines

    def read_wiki_data(self, file_path, take_top_k_words=None):
        wiki_df = pd.read_csv(file_path, sep="\t", names=["word", "frequency"])

        print(f"Wiki words file read in a dataframe : {wiki_df.shape}")
        sorted_wiki_df = wiki_df.sort_values(
            by=["frequency"], inplace=False, ascending=False
        )

        if take_top_k_words:
            print(f"Taking Top {take_top_k_words} words")

            wiki_word_list = sorted_wiki_df["word"].values[0:take_top_k_words].tolist()

        else:
            print("Taking all the wiki words")
            wiki_word_list = sorted_wiki_df["word"].tolist()

        return wiki_word_list

    def get_vocab(self):
        vocab = self.glove_model.key_to_index.keys()
        print(f"Vocab Size : {len(vocab)}")

        return vocab

    def get_glove_vectors(self, word_list):
        c_word_vocab = 0
        c_undescore_word_vocab = 0
        c_hyphen_word_vocab = 0
        c_multi_word = 0
        c_multi_word_2 = 0

        word_vocab = []
        undescore_word_vocab = []
        hyphen_word_vocab = []
        multi_word = []

        vocab = self.glove_model.key_to_index.keys()

        for word in word_list:
            if word in vocab:
                word_vocab.append(word)
                c_word_vocab += 1

            else:
                underscore_word = "_".join(str(word).split())
                if underscore_word in vocab:
                    undescore_word_vocab.append(underscore_word)
                    c_undescore_word_vocab += 1

                else:
                    hyphen_word = "-".join(str(word).split())

                    if hyphen_word in vocab:
                        hyphen_word_vocab.append(hyphen_word)
                        c_hyphen_word_vocab += 1

                    else:
                        print(f"Word Not Found in Vocab : {word}", flush=True)
                        multi_word.append(word)
                        c_multi_word += 1

        words = word_vocab + undescore_word_vocab + hyphen_word_vocab
        gv_words = self.glove_model[words]

        print(f"c_word_vocab : {c_word_vocab}")
        print(f"c_undescore_word_vocab : {c_undescore_word_vocab}")
        print(f"c_hyphen_word_vocab : {c_hyphen_word_vocab}")
        print(f"c_multi_word : {c_multi_word}")

        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()

        v_multi_word = np.empty((0, 300), dtype=float)
        multi_word_vocab, word_not_found = [], []
        c_word_not_found = 0

        for mword in multi_word:
            splitted_word = mword.split()

            try:
                mw_vector = np.mean(self.glove_model[splitted_word], axis=0)
                v_multi_word = np.append(
                    v_multi_word, np.expand_dims(mw_vector, axis=0), axis=0
                )
                multi_word_vocab.append(mword)
                c_multi_word_2 += 1

            except KeyError:
                try:
                    lemmas = [lemmatizer.lemmatize(word) for word in splitted_word]

                    lemma_vector = np.mean(self.glove_model[lemmas], axis=0)
                    v_multi_word = np.append(
                        v_multi_word, np.expand_dims(lemma_vector, axis=0), axis=0
                    )

                    multi_word_vocab.append(mword)
                    c_multi_word_2 += 1

                except KeyError:
                    print(f"Word Not Found : {mword}")
                    word_not_found.append(mword)
                    c_word_not_found += 1

                    continue

        all_words = words + multi_word_vocab
        all_vectors = np.concatenate((gv_words, v_multi_word))

        print(f"all_words_count: {len(all_words)}", flush=True)
        print(f"all_vectors_count: {all_vectors.shape[0]}", flush=True)

        assert len(all_words) == all_vectors.shape[0]

        # print(f"word_vocab : {word_vocab}", flush=True)
        print(f"undescore_word_vocab : {undescore_word_vocab}", flush=True)
        print(f"hyphen_word_vocab : {hyphen_word_vocab}", flush=True)
        print(f"multi_word_vocab : {multi_word_vocab}", flush=True)
        print(f"word_not_found : {c_word_not_found}, {word_not_found}", flush=True)

        return all_words, all_vectors


def find_most_similar():
    pass


def get_nearest_neighbours(
    num_nearest_neighbours,
    concept_list,
    concept_embeddings,
    property_list,
    property_embeddings,
    output_file,
):
    con_similar_properties = NearestNeighbors(
        n_neighbors=num_nearest_neighbours, algorithm="brute", metric="cosine"
    ).fit(np.array(property_embeddings))

    con_distances, con_indices = con_similar_properties.kneighbors(
        np.array(concept_embeddings)
    )

    log.info(f"con_distances shape : {con_distances.shape}")
    log.info(f"con_indices shape : {con_indices.shape}")

    con_similar_prop_dict = {}

    # file_name = f"datasets/concept_{num_nearest_neighbours}similar_wiki_words.txt"

    with open(output_file, "w") as file:
        for con_idx, prop_idx in enumerate(con_indices):
            concept = concept_list[con_idx]
            similar_properties = [property_list[idx] for idx in prop_idx]

            log.info(f"{concept} : {similar_properties}")

            con_similar_prop_dict[concept] = similar_properties

            print(f"{concept}\t{similar_properties}\n")

            for prop in similar_properties:
                line = concept + "\t" + prop + "\n"
                file.write(line)

    con_sim_wiki_word_df = pd.read_csv(
        output_file, sep="\t", names=["concept", "wiki_word"]
    )

    con_sim_wiki_word_df["counts"] = con_sim_wiki_word_df.groupby(
        ["wiki_word"]
    ).transform("count")

    con_sim_wiki_word_df.sort_values(by=["wiki_word"], inplace=True, ascending=False)

    sorted_fn = os.path.splitext(output_file)[0] + "wiki_words_clustered.txt"

    con_sim_wiki_word_df.to_csv(sorted_fn, sep="\t", header=False, index=False)

    log.info(f"Finished getting similar properties")

    return sorted_fn


class RelBertEmbeddings:
    def __init__(self):
        self.relbert_model = RelBERT()

    def read_data(self, file_path):
        with open(file_path, "r") as in_file:
            lines = in_file.read().splitlines()

        lines = [l.split("\t") for l in lines]

        return lines

    def get_relbert_embeds(self, con_prop_list, batch_size):
        return self.relbert_model.get_embedding(con_prop_list, batch_size=batch_size)


def main():
    #########################
    # Hawk Paths

    # wv_format_glove_file = "/scratch/c.scmag3/glove/glove.840B.300d.word2vec.format.txt"

    # 42B
    w2v_format_glove_file = "/scratch/c.scmag3/glove/glove.42B.300d.word2vec.format.txt"

    concept_file = (
        "/scratch/c.scmag3/commonality_detection/datasets/ufet_clean_types.txt"
    )
    wiki_word_frequency_file = "datasets/glove_words_wiki_count_pytorch_tok.txt"

    #########################

    gv = GloveVectorsGensim(wv_format_glove_file=w2v_format_glove_file)

    concept_list = gv.read_data(file_path=concept_file)
    con_in_vocab, gvs_concept = gv.get_glove_vectors(concept_list)

    print(f"concept_list : {len(concept_list)}")
    print(f"con_in_vocab : {len(con_in_vocab)}")

    all_glove_word_list_1 = gv.get_vocab()

    print(f"all_glove_word_list_1: {len(all_glove_word_list_1)}")

    all_glove_word_list_2, gvs_all_glove = gv.get_glove_vectors(all_glove_word_list_1)

    print(f"all_glove_word_list_2 : {len(all_glove_word_list_2)}")

    num_nearest_neighbours = [30, 50]

    for num_nn in num_nearest_neighbours:
        wiki_word_list = gv.read_wiki_data(
            file_path=wiki_word_frequency_file, take_top_k_words=None
        )

        wiki_word_in_vocab, gvs_wiki_word = gv.get_glove_vectors(wiki_word_list)

        print(f"gvs_concept.shape : {gvs_concept.shape}", flush=True)
        print(f"gvs_wiki_word.shape : {gvs_wiki_word.shape}", flush=True)

        print(f"con_in_vocab : {len(con_in_vocab)}", flush=True)
        print(
            f"wiki_word_in_vocab : {len(wiki_word_in_vocab)}",
            flush=True,
        )

        out_file = f"output_files/concept_{num_nn}similar_all_wiki_words.txt"

        # concept_list = np.array(concept_list, dtype=str)
        # wiki_word_list = np.array(wiki_word_list, dtype=str)

        con_similar_prop_file = get_nearest_neighbours(
            num_nearest_neighbours=num_nn,
            concept_list=con_in_vocab,
            concept_embeddings=gvs_concept,
            property_list=wiki_word_in_vocab,
            property_embeddings=gvs_wiki_word,
            output_file=out_file,
        )

    # relbert = RelBertEmbeddings()
    # con_prop_list = relbert.read_data(con_similar_prop_file)
    # relbert_embeds = relbert.get_relbert_embeds(con_prop_list, batch_size=32)

    # print(f"relbert_embeds.shape : {torch.tensor(relbert_embeds).shape}", flush=True)

    # con_prop_rel_embeds = []

    # for con_prop, rel_embed in zip(con_prop_list, relbert_embeds):
    #     con_prop = "#".join(con_prop)
    #     con_prop_rel_embeds.append([con_prop, rel_embed])

    # with open("con_prop_relbert_embeddings.pkl", "wb") as emb_pkl:
    #     pickle.dump(con_prop_rel_embeds, emb_pkl)

    # def hdbscan_clusters(embeds):
    #     clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    #     clusterer.fit(np.array(embeds))

    #     return (clusterer.labels_, clusterer.probabilities_)

    # print("Starting Clustering ...", flush=True)
    # labels, probs = hdbscan_clusters(relbert_embeds)
    # print("Finished Clustering ...", flush=True)

    # df = pd.DataFrame(con_prop_list, columns=["concept", "property"])
    # df["cluster_label"] = labels
    # df["cluster_probs"] = probs

    # df.sort_values("cluster_label", axis=0, inplace=True)

    # print(f"Df Shape : {df.shape}")
    # print(df.head(n=10))

    # df.to_csv("clustered_con_prop.txt", sep="\t", header=True, index=False)


if __name__ == "__main__":
    main()
