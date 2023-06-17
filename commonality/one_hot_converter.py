import os
import sys
import numpy

from os import listdir

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot_encoder(inp_dir_path, out_dir_path):
    # inp_files =  listdir(dir_path)

    inp_files = [
        fname
        for fname in listdir(inp_dir_path)
        if fname.startswith("main_cluster_fasttext_filter")
    ]

    print(flush=True)
    print(f"inp_files: {inp_files}", flush=True)

    for inp_file in inp_files:
        abs_path = os.path.join(inp_dir_path, inp_file)

        print(f"processing inp_file : {inp_file}", flush=True)
        print(f"input_file_absolute_path : {abs_path}", flush=True)

        df = pd.read_csv(abs_path, sep="\t")
        df_new = df.iloc[:, [0, 1]]
        df_new.columns = ["concept", "property"]

        print(flush=True)
        print(f"df_new")
        print(df_new)

        enc = OneHotEncoder()
        enc.fit_transform(df_new[["property"]])

        out_file_name = os.path.join(out_dir_path, inp_file)
        print(f"out_file_name : {out_file_name}", flush=True)

        with open(out_file_name, "w") as f:
            for concept, prop in df_new.values:
                encoding = enc.transform([[prop]]).toarray().astype(int).flatten()
                print(concept, prop, encoding.shape)
                print(encoding)

                embedding_str = " ".join(["{:d}".format(item) for item in encoding])
                f.write(f"{concept}\t{prop}\t{embedding_str}\n")


if __name__ == "__main__":
    task_name = "ontology_completion"  # classification_vocabs

    print(f"task_name : {task_name}", flush=True)

    if task_name == "ontology_completion":
        inp_dirs = [
            "output_files/ontology_completion_similar_thresh_50/sumo",
            "output_files/ontology_completion_similar_thresh_50/economy",
            "output_files/ontology_completion_similar_thresh_50/olympics",
            "output_files/ontology_completion_similar_thresh_50/transport",
            "output_files/ontology_completion_similar_thresh_50/wine",
        ]

        out_dirs = [
            "output_files/ontology_completion_similar_thresh_50/sumo/onehot_encodings",
            "output_files/ontology_completion_similar_thresh_50/economy/onehot_encodings",
            "output_files/ontology_completion_similar_thresh_50/olympics/onehot_encodings",
            "output_files/ontology_completion_similar_thresh_50/transport/onehot_encodings",
            "output_files/ontology_completion_similar_thresh_50/wine/onehot_encodings",
        ]

        for inp_dir_path, out_dir_path in zip(inp_dirs, out_dirs):
            print(f"ontology_completion", flush=True)
            print(f"inp_dir_path : {inp_dir_path}", flush=True)
            print(f"out_dir_path: {out_dir_path}", flush=True)
            print(flush=True)

            one_hot_encoder(inp_dir_path=inp_dir_path, out_dir_path=out_dir_path)

    else:
        inp_dir_path = str(sys.argv[1])
        out_dir_path = str(sys.argv[2])

        print(flush=True)
        print(f"sys.argv : {sys.argv}", flush=True)
        print(f"inp_dir_path : {inp_dir_path}", flush=True)
        print(f"out_dir_path: {out_dir_path}", flush=True)
        print(flush=True)

        one_hot_encoder(inp_dir_path=inp_dir_path, out_dir_path=out_dir_path)
