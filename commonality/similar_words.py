import numpy as np
from scipy import spatial

embeddings_dict = {}
with open("/scratch/c.scmag3/glove/glove.840B.300d.txt", "r") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vectors
f.close()

print("Loaded %s word vectors." % len(embeddings_dict))


# define (euclidean) distance function


def find_closest_embeddings(embedding):
    return sorted(
        embeddings_dict.keys(),
        key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding),
    )


def read_data(file_path):
    with open(file_path, "r") as in_file:
        lines = in_file.read().splitlines()

    return lines


print(find_closest_embeddings(embeddings_dict["king"])[1:30], flush=True)
print(flush=True)
print(find_closest_embeddings(embeddings_dict["person"])[1:30], flush=True)
print(flush=True)
print(find_closest_embeddings(embeddings_dict["organization"])[1:30], flush=True)
print(flush=True)
