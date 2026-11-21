"""Build South Asia metadata CSVs from the official GEOID-Flood chip CSV.

Usage: python pakflood/make_sa_split.py data/geoid-flood

data_tiles_sa_holdout.csv : SA events -> test; other events keep train/val; original non-SA test rows dropped.
                            (train a model that has never seen South Asia; test it on South Asia)
data_tiles_sa_fewshot.csv : train = 10% of SA 1024px tiles (by label_id, seed 0) ONLY;
                            val   = official non-SA val (for the training loop);
                            test  = the remaining 90% of SA tiles.
                            (few-shot adaptation of the hold-out model; test the hold-out model on the same 90% for before/after)
data_tiles_pk_test.csv    : official train/val unchanged; test = official test chips of Pakistani events only
                            (how the global models do on Pakistan in-distribution)
"""
import sys

import numpy as np
import pandas as pd

SA_EVENTS = ["EMSR205", "EMSR459", "EMSR629", "EMSR631", "EMSR838", "EMSR851"]
PK_EVENTS = ["EMSR629", "EMSR631", "EMSR838"]


def report(name: str, d: pd.DataFrame) -> None:
    s = d[(d.modality == "s1grd") & (d.image_time == "post")]
    print(f"{name:12s} paired chips per split: {s.groupby('split').size().to_dict()}")


def main(root: str) -> None:
    df = pd.read_csv(f"{root}/data_tiles_s256_st128.csv")
    is_sa = df.event_id.isin(SA_EVENTS)

    # 1) hold-out
    ho = df[is_sa | (df.split != "test")].copy()
    ho.loc[ho.event_id.isin(SA_EVENTS), "split"] = "test"
    ho.to_csv(f"{root}/data_tiles_sa_holdout.csv", index=False)
    report("holdout", ho)

    # 2) few-shot: 10% of SA tiles -> train, 90% -> test, non-SA val kept
    rng = np.random.default_rng(0)
    sa_tiles = sorted(df.loc[is_sa, "label_id"].unique())
    k = max(1, int(round(0.10 * len(sa_tiles))))
    train_tiles = set(rng.choice(sa_tiles, size=k, replace=False))
    sa = df[is_sa].copy()
    sa["split"] = np.where(sa.label_id.isin(train_tiles), "train", "test")
    nonsa_val = df[(~is_sa) & (df.split == "val")]
    fs = pd.concat([sa, nonsa_val], ignore_index=True)
    fs.to_csv(f"{root}/data_tiles_sa_fewshot.csv", index=False)
    report("fewshot", fs)
    print(f"SA tiles: {len(sa_tiles)} | few-shot train tiles: {k} | test tiles: {len(sa_tiles) - k}")
    print("few-shot train tiles by event:", sa[sa.split == "train"].drop_duplicates("label_id").event_id.value_counts().to_dict())

    # 3) Pakistan-only official test
    pk = df[(df.split != "test") | df.event_id.isin(PK_EVENTS)].copy()
    pk.to_csv(f"{root}/data_tiles_pk_test.csv", index=False)
    report("pk_test", pk)


if __name__ == "__main__":
    main(sys.argv[1])
