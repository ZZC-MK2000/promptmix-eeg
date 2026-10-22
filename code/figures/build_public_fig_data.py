"""Build the small public data files used by the new webpage figures.

Every number is read from the result JSON that produced it; assertions pin the values
so a figure cannot silently drift from the experiment.

Run: python code/figures/build_public_fig_data.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
R = ROOT / "results"
OUT = R / "public"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    song_q = json.load(open(R / "descriptions" / "desc_selection_qwen_medoid.json", encoding="utf-8"))
    song_t = json.load(open(R / "calib_head_pilot" / "desc_selection_nmedt.json", encoding="utf-8"))
    genre = json.load(open(R / "calib_head_pilot" / "genre_readout.json", encoding="utf-8"))
    gclap = json.load(open(R / "descriptions" / "genre_selection_qwen.json", encoding="utf-8"))
    bridge = json.load(open(R / "bandpower_clap_k_shot" / "bridge.json", encoding="utf-8"))
    pilot = json.load(open(R / "descriptions" / "pilot_qwen_two_paths.json", encoding="utf-8"))

    ks = [0, 10, 100, 300]
    tag = [song_t["aggregate_10cand"][f"k{k}"]["mean"] for k in ks]
    qwen = [song_q["aggregate"]["caption"][str(k)]["mean"] for k in ks]
    floor = song_q["majority_floor"]["mean"]
    assert abs(tag[2] - 0.350) < 1e-3, tag
    assert abs(qwen[2] - 0.201) < 1e-3, qwen

    g = genre["aggregate"]
    head = [g["head_zero"]["kappa"]["mean"]] + [g[f"head_calib_{k}"]["kappa"]["mean"] for k in (10, 100, 300)]
    clip = [g["clip_zero"]["kappa"]["mean"]] + [g[f"clip_calib_{k}"]["kappa"]["mean"] for k in (10, 100, 300)]
    mainline_acc = g["mainline"]["acc"]["mean"]
    mainline_kappa = g["mainline"]["kappa"]["mean"]
    assert abs(mainline_acc - 0.742) < 1e-3 and abs(mainline_kappa - 0.399) < 1e-3

    clap_acc = [gclap["aggregate"][f"qwen_k{k}"]["acc"] for k in ks]
    clap_kappa = [gclap["aggregate"][f"qwen_k{k}"]["kappa"] for k in ks]
    clap_majority = gclap["majority_floor"]
    assert abs(clap_acc[2] - 0.056) < 1e-3, clap_acc

    folds = bridge["folds"]
    ret = {f"{space}_{k}": float(np.mean([folds[f]["curve"][str(k)][space] for f in folds]))
           for space in ("bp", "clap") for k in (10, 100)}
    assert abs(ret["bp_10"] - 0.413) < 2e-3 and abs(ret["bp_100"] - 0.504) < 2e-3
    assert abs(ret["clap_10"] - 0.339) < 2e-3 and abs(ret["clap_100"] - 0.383) < 2e-3

    rec = pilot["records"]
    def arm_margin(arm):
        vals = {}
        for r in rec:
            if r["arm"] == arm:
                vals.setdefault(r["row"], []).append(r["song_cen_margin"])
        return np.array([np.mean(v) for v in vals.values()])
    so, sc, ss = arm_margin("song_oracle"), arm_margin("song_calib"), arm_margin("song_samegenre")
    d = so - ss
    song_gen = {"oracle": float(so.mean()), "calib": float(sc.mean()), "samegenre": float(ss.mean()),
                "oracle_minus_samegenre": float(d.mean()),
                "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))}
    assert abs(song_gen["oracle_minus_samegenre"] - 0.034) < 2e-3, song_gen

    sources = {"song_kshot": "results/calib_head_pilot/desc_selection_nmedt.json + "
                             "results/descriptions/desc_selection_qwen_medoid.json",
               "genre_kshot": "results/calib_head_pilot/genre_readout.json",
               "genre_clap": "results/descriptions/genre_selection_qwen.json (library_b/oracle_all)",
               "retrieval": "results/bandpower_clap_k_shot/bridge.json",
               "song_generation": "results/descriptions/pilot_qwen_two_paths.json (song arms)"}
    kshot = {"k": ks, "song": {"tag": tag, "qwen": qwen, "chance": 0.10, "majority": floor,
                               "n_candidates": 10},
             "genre": {"k": ks, "head_calib_kappa": head, "clip_calib_kappa": clip,
                       "mainline_acc": mainline_acc, "mainline_kappa": mainline_kappa,
                       "clap_prompt_acc": clap_acc, "clap_prompt_kappa": clap_kappa,
                       "clap_majority": clap_majority},
             "sources": sources}
    three = {"retrieval": ret, "retrieval_chance": 0.10,
             "genre": {"mainline_acc": mainline_acc, "mainline_kappa": mainline_kappa,
                       "kshot_kappa": head[2], "clap_prompt_acc": clap_acc[2],
                       "clap_majority": clap_majority},
             "song": {"tag_k100": tag[2], "qwen_k100": qwen[2], "floor": floor,
                      "generation": song_gen},
             "sources": sources}
    json.dump(kshot, open(OUT / "fig_kshot_two_levels.json", "w", encoding="utf-8"), indent=2)
    json.dump(three, open(OUT / "fig_three_paths.json", "w", encoding="utf-8"), indent=2)
    print("wrote", OUT / "fig_kshot_two_levels.json")
    print("wrote", OUT / "fig_three_paths.json")
    print("song tag", [round(x, 3) for x in tag], "qwen", [round(x, 3) for x in qwen])
    print("genre mainline", round(mainline_acc, 3), round(mainline_kappa, 3),
          "head kappa", [round(x, 3) for x in head], "clap acc", [round(x, 3) for x in clap_acc])
    print("retrieval", {k: round(v, 3) for k, v in ret.items()})
    print("song generation", {k: round(v, 3) if isinstance(v, float) else v for k, v in song_gen.items()})


if __name__ == "__main__":
    main()
