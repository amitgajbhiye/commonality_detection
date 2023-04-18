import torch
import os
import numpy as np
import logging


from gensim.models import KeyedVectors

from sklearn.neighbors import NearestNeighbors
from nltk.stem import WordNetLemmatizer

from relbert import RelBERT


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

    def get_glove_vectors(self, data):
        vocab = self.get_vocab()
        word_vectors, words_in_vocab, words_not_in_vocab = [], [], []

        for word in data:
            if word in vocab:
                word_vectors.append(self.glove_model[word])
                words_in_vocab.append(word)
            else:
                words_not_in_vocab.append(word)

        return (np.array(word_vectors, dtype=float), words_in_vocab, words_not_in_vocab)


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
    num_nearest_neighbours = 10

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

    file_name = f"concept_similar_properties.txt"

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

    def get_relbert_embeds(self, con_prop_list):
        return self.relbert_model.get_embedding(con_prop_list)


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

wv_format_glove_file = "/scratch/c.scmag3/glove/glove.840B.300d.word2vec.format.txt"
concept_file = "/scratch/c.scmag3/property_augmentation/data/ufet/clean_types.txt"
property_file = (
    "/scratch/c.scmag3/property_augmentation/data/prop_vocab/prop_vocab_cnetp_clean.txt"
)

#########################

gv = GloveVectorsGensim(wv_format_glove_file=wv_format_glove_file)

concept_list = gv.read_data(file_path=concept_file)
property_list = gv.read_data(file_path=property_file)

gvs_concept, con_in_vocab, con_not_in_vocab = gv.get_glove_vectors(concept_list)
gvs_property, prop_in_vocab, prop_not_in_vocab = gv.get_glove_vectors(property_list)


print(f"gvs_concept.shape : {gvs_concept.shape}")
print(f"gvs_property.shape : {gvs_property.shape}")

print(f"con_in_vocab : {con_in_vocab}")
print(f"prop_in_vocab : {prop_in_vocab}")

print(f"con_not_in_vocab : {con_not_in_vocab}")
print(f"prop_not_in_vocab : {prop_not_in_vocab}")


con_similar_prop_file = get_nearest_neighbours(
    num_nearest_neighbours=10,
    concept_list=concept_list,
    concept_embeddings=gvs_concept,
    property_list=property_list,
    property_embeddings=gvs_property,
)


relbert = RelBertEmbeddings()
con_prop_list = relbert.read_data(con_similar_prop_file)
relbert_embeds = relbert.get_relbert_embeds(con_prop_list)

print(f"relbert_embeds.shape : {torch.tensor(relbert_embeds).shape}")
