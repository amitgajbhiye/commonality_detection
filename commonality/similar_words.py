import numpy as np
from scipy import spatial


embeddings_dict = {}

print(f"Loading Glove ... ")

with open("/scratch/c.scmag3/glove/glove.840B.300d.txt", "r", encoding="utf-8") as f:
    for line in f:
        ######

        print(line, flush=True)

        values = line.split()
        word = values[0]

        print(values[0], flush=True)
        print(values[1], flush=True)
        print(flush=True)

        vectors = np.asarray(values[1:], dtype=np.float32)

        embeddings_dict[word] = vectors

f.close()

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
