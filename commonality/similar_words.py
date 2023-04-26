import numpy as np
from scipy import spatial


def load_glove_model(glove_file):
    print("Loading Glove Model...")

    embeddings_dict = {}
    vector_size = 300

    with open(glove_file, "r") as f:
        for line in f:
            split_line = line.split()
            print(f"split_line : {split_line}")
            word = " ".join(split_line[0 : len(split_line) - vector_size])
            print(f"word : {word}")
            embedding = np.array([float(val) for val in split_line[-vector_size:]])
            print(f"embedding : {embedding}")
            embeddings_dict[word] = embedding

        print("Done.\n" + str(len(embeddings_dict)) + " words loaded!")

    return embeddings_dict


embeddings_dict = load_glove_model("/scratch/c.scmag3/glove/glove.840B.300d.txt")

print("Loaded %s word vectors." % len(embeddings_dict))

# define (euclidean) distance function


def find_closest_embeddings(embedding):
    return sorted(
        embeddings_dict.keys(),
        key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding),
    )


print(find_closest_embeddings(embeddings_dict["king"])[1:30], flush=True)
print(flush=True)
print(find_closest_embeddings(embeddings_dict["person"])[1:30], flush=True)
print(flush=True)
print(find_closest_embeddings(embeddings_dict["organization"])[1:30], flush=True)
print(flush=True)
