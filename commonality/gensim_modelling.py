import torch
import os
import numpy as np
import logging
import pickle
import hdbscan
import pandas as pd
import copy
import sys


from gensim.models import KeyedVectors

from sklearn.neighbors import NearestNeighbors
from nltk.stem import WordNetLemmatizer

from relbert import RelBERT

sys.path.insert(0, os.getcwd())

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

    def get_vocab(self):
        vocab = self.glove_model.key_to_index.keys()
        log.info(f"Vocab Size : {len(vocab)}")

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
                underscore_word = "_".join(word.split())
                if underscore_word in vocab:
                    undescore_word_vocab.append(underscore_word)
                    c_undescore_word_vocab += 1

                else:
                    hyphen_word = "-".join(word.split())

                    if hyphen_word in vocab:
                        hyphen_word_vocab.append(hyphen_word)
                        c_hyphen_word_vocab += 1

                    else:
                        # print(f"Word Not Found in Vocab : {word}", flush=True)
                        multi_word.append(word)
                        c_multi_word += 1

        print(f"undescore_word_vocab : undescore_word_vocab")
        v_word_vocab = self.glove_model[word_vocab]
        v_undescore_word_vocab = self.glove_model[undescore_word_vocab]
        v_hyphen_word_vocab = self.glove_model[hyphen_word_vocab]

        print(flush=True)
        print(c_word_vocab, flush=True)
        print(c_undescore_word_vocab, flush=True)
        print(c_hyphen_word_vocab, flush=True)
        print(c_multi_word, flush=True)
        print()
        print(v_word_vocab.shape, flush=True)
        print(v_undescore_word_vocab.shape, flush=True)
        print(v_hyphen_word_vocab.shape, flush=True)

        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()

        v_multi_word = np.empty((0, 300), dtype=float)
        multi_word_vocab = []

        for mword in multi_word:
            splitted_word = mword.split()

            try:
                mw_vector = np.mean(self.glove_model[splitted_word], axis=0)
                v_multi_word = np.append(
                    v_multi_word, np.expand_dims(mw_vector, axis=0), axis=0
                )
                multi_word_vocab.append(mword)

            except KeyError:
                try:
                    lemmas = [lemmatizer.lemmatize(word) for word in splitted_word]

                    lemma_vector = np.mean(self.glove_model[lemmas], axis=0)
                    v_multi_word = np.append(
                        v_multi_word, np.expand_dims(lemma_vector, axis=0), axis=0
                    )

                    multi_word_vocab.append(mword)

                except KeyError:
                    print(f"Multiword Not Found : {mword}")
                    continue

            c_multi_word_2 += 1

        print(f"c_multi_word_2 : {c_multi_word_2}")
        print(f"v_multi_word.shape : {v_multi_word.shape}")

        all_vectors = np.concatenate(
            (v_word_vocab, v_undescore_word_vocab, v_hyphen_word_vocab, v_multi_word)
        )

        #     print (type(word_vocab))
        #     print (type(undescore_word_vocab))
        #     print (type(hyphen_word))
        #     print (type(multi_word))

        all_words = (
            word_vocab + undescore_word_vocab + hyphen_word_vocab + multi_word_vocab
        )

        print(len(all_words), flush=True)
        print(all_vectors.shape[0], flush=True)

        assert len(all_words) == all_vectors.shape[0]

        return all_words, all_vectors


def match_multi_words(word1, word2):
    lemmatizer = WordNetLemmatizer()

    word1 = " ".join([lemmatizer.lemmatize(word) for word in word1.split()])
    word2 = " ".join([lemmatizer.lemmatize(word) for word in word2.split()])

    return word1 == word2


def get_nearest_neighbours(
    num_nearest_neighbours,
    concept_list,
    concept_embeddings,
    property_list,
    property_embeddings,
):
    con_similar_properties = NearestNeighbors(
        n_neighbors=num_nearest_neighbours, algorithm="brute"
    ).fit(np.array(property_embeddings))

    con_distances, con_indices = con_similar_properties.kneighbors(
        np.array(concept_embeddings)
    )

    log.info(f"con_distances shape : {con_distances.shape}")
    log.info(f"con_indices shape : {con_indices.shape}")

    con_similar_prop_dict = {}
    # file_name = os.path.join(save_dir, dataset_params["dataset_name"]) + ".tsv"

    file_name = f"concept_similar_wiki_words.txt"

    with open(file_name, "w") as file:
        for con_idx, prop_idx in enumerate(con_indices):
            concept = concept_list[con_idx]
            similar_properties = [property_list[idx] for idx in prop_idx]

            log.info(f"{concept} : {similar_properties}")

            # similar_properties = [
            #     prop
            #     for prop in similar_properties
            #     if not match_multi_words(concept, prop)
            # ]

            # con_similar_prop_dict[concept] = similar_properties

            # print(f"{concept}\t{similar_properties}\n")

            for prop in similar_properties:
                line = concept + "\t" + prop + "\n"
                file.write(line)

    log.info(f"Finished getting similar properties")

    return file_name


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


#########################
# Paths

# Local Paths
# wv_format_glove_file = "/home/amitgajbhiye/Downloads/embeddings_con_prop/glove.42B.300d.word2vec.format.txt"

# concept_file = (
#     "/home/amitgajbhiye/cardiff_work/property_augmentation/data/ufet/clean_types.txt"
# )
# property_file = "/home/amitgajbhiye/cardiff_work/property_augmentation/data/prop_vocab/prop_vocab_cnetp_clean.txt"

#########################
# Hawk Paths

# wv_format_glove_file = "/scratch/c.scmag3/glove/glove.840B.300d.word2vec.format.txt"

# 42B
w2v_format_glove_file = "/scratch/c.scmag3/glove/glove.42B.300d.word2vec.format.txt"

concept_file = "/scratch/c.scmag3/commonality_detection/datasets/ufet_clean_types.txt"
wiki_word_file = "/scratch/c.scmag3/commonality_detection/datasets/stopword_filtered_all_wikipedia.txt"

num_nearest_neighbours = 50

#########################

gv = GloveVectorsGensim(wv_format_glove_file=w2v_format_glove_file)

concept_list = gv.read_data(file_path=concept_file)
wiki_word_list = gv.read_data(file_path=wiki_word_file)

con_in_vocab, gvs_concept = gv.get_glove_vectors(concept_list)
wiki_word_in_vocab, gvs_wiki_word = gv.get_glove_vectors(wiki_word_list)


print(f"gvs_concept.shape : {gvs_concept.shape}", flush=True)
print(f"gvs_wiki_word.shape : {gvs_wiki_word.shape}", flush=True)

print(f"con_in_vocab : {len(con_in_vocab)}, {con_in_vocab}", flush=True)
print(f"wiki_word_in_vocab : {len(wiki_word_in_vocab)}, {gvs_wiki_word}", flush=True)


con_similar_prop_file = get_nearest_neighbours(
    num_nearest_neighbours=num_nearest_neighbours,
    concept_list=concept_list,
    concept_embeddings=gvs_concept,
    property_list=wiki_word_list,
    property_embeddings=gvs_wiki_word,
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
