import torch
import os
import numpy as np
import logging
import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from sklearn.neighbors import NearestNeighbors
from nltk.stem import WordNetLemmatizer


log = logging.getLogger(__name__)


class GloveVectors:
    def __init__(self):
        self.glove_model = GloVe(
            name="42B",
            dim=300,
            cache="/home/amitgajbhiye/Downloads/embeddings_con_prop/",
        )

    def read_data(self, file_path):
        with open(file_path, "r") as in_file:
            lines = in_file.read().splitlines()

        return lines

    def get_glove_vectors(self, data):
        # +++++++++++++++ check for non existing glove vectors +++++++++++++++++++

        return self.glove_model.get_vecs_by_tokens(data)


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


def main():
    gv = GloveVectors()

    concept_lists = gv.read_data(
        file_path="/home/amitgajbhiye/cardiff_work/property_augmentation/data/ufet/clean_types.txt"
    )

    property_list = gv.read_data(
        file_path="/home/amitgajbhiye/cardiff_work/property_augmentation/data/prop_vocab/prop_vocab_cnetp_clean.txt"
    )

    gvs_concept = gv.get_glove_vectors(concept_lists)
    gvs_property = gv.get_glove_vectors(property_list)

    get_nearest_neighbours(
        num_nearest_neighbours=10,
        concept_list=concept_lists,
        concept_embeddings=gvs_concept,
        property_list=property_list,
        property_embeddings=gvs_property,
    )


if __name__ == "__main__":
    main()
