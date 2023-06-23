import pandas as pd

numberbatch_relbert_scored_all = (
    "output_files/clustered_relbert_scored_file/numberbatch_relation_probs.txt"
)

df = pd.read_csv(
    numberbatch_relbert_scored_all, sep="|", names=["con", "prop", "thresh"]
)

print(df, flush=True)

all_df = []

print(f"concepts : {df.con.values[0:1000]}", flush=True)

for c in df.con.values[0:3000]:
    con_df = df[df["con"] == c]
    con_df = con_df.sort_values("thresh", ascending=False)

    print(flush=True)
    print(con_df, flush=True)

    all_df.append(con_df)


final_df = pd.concat(all_df)

sorted_out_file = "output_files/clustered_relbert_scored_file/con3000_grouped_thresh_sored_numberbatch_relation_probs.txt"
final_df.to_csv(sorted_out_file, sep="\t", index=False)
