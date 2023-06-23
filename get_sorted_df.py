import pandas as pd

numberbatch_relbert_scored_all = (
    "output_files/clustered_relbert_scored_file/numberbatch_relation_probs.txt"
)

df = pd.read_csv(
    numberbatch_relbert_scored_all, sep="|", names=["con", "prop", "thresh"]
)

print(df, flush=True)

all_df = []
for c in df.con:
    con_df = df[df["con"] == c]
    con_df = con_df.sort_values("thresh", ascending=False)
    print(con_df.head(n=20), flush=True)
    all_df.append(con_df)


final_df = pd.concat(all_df)

sorted_out_file = "output_files/clustered_relbert_scored_file/con_grouped_thresh_sored_numberbatch_relation_probs.txt"
final_df.to_csv(sorted_out_file, sep="\t", index=False)
