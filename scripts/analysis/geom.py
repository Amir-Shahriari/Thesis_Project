"""Pull the geometry-study numbers needed for the thesis write-up."""
import json
import pathlib

RUNS = pathlib.Path(r"C:\Users\amirh\Desktop\qwarm-gnn-rl\runs")
d = json.load(open(RUNS / "geometry_study_aggregate.json"))

for scale, v in d["scales"].items():
    a = v["A_geometry"]
    print(f"\n===== {scale}  (A: geometry, n={a['n_checkpoints']} ckpts, "
          f"dim={a['dim']}) =====")
    print(f"  IsoScore                {a['isoscore_mean']:.3f}")
    print(f"  effective rank          {a['effective_rank_mean']:.1f} / {a['dim']}")
    print(f"  mean pairwise cosine    {a['mean_pair_cosine']:.3f}"
          f"   (centred {a['mean_pair_cosine_centred']:.3f})")
    print(f"  locality rho encoder    {a['locality_rho_encoder']:.3f}")
    print(f"  locality rho raw feats  {a['locality_rho_input_features']:.3f}")
    wc = a.get("warm_vs_cold", {})
    for k, s in wc.items():
        print(f"  warm vs cold {k:16s} warm {s['warm']:.3f} cold {s['cold']:.3f}"
              f"  {s['n_pairs_favouring_warm']}/{s['n_pairs']} seeds"
              f"  p={s['wilcoxon_p']}")

    b = v["B_retrieval_gate"]
    print(f"  -- retrieval gate (coverage {b['library_coverage_fraction']:.3f},"
          f" best hop {b['mean_best_hop']:.2f}, random hop"
          f" {b['mean_random_hop']:.2f})")
    for name, t in b["treatments"].items():
        print(f"     {name:10s} sim_max {t['sim_max_mean']:.3f}"
              f"  sat>0.95 {t['saturation_frac_above_0.95']:.3f}"
              f"  AUC {t['gate_auc']:.3f}"
              f"  quality {t['retrieval_quality']:.4f}"
              + (f"  (sat p={t['saturation_vs_raw_p']},"
                 f" quality p={t['quality_vs_raw_p']})"
                 if "saturation_vs_raw_p" in t else ""))
    print(f"     raw quality vs random: p={b['raw_quality_vs_random']['p']:.2g}")
    print(f"     raw AUC vs chance:     p={b['raw_auc_vs_chance']['p']:.2g}")

    c = v["C_metric_probe"]
    print(f"  -- metric probe (rho with graph distance)")
    print(f"     cosine {c['cosine']:.3f}   ridge {c['ridge']:.3f}   "
          f"mlp {c['mlp']:.3f}   best learned {c['best_learned_rho']:.3f}")
    print(f"     raw input feats: cosine {c['input_cosine']:.3f}  "
          f"ridge {c['input_ridge']:.3f}")
    r = c["ridge_vs_cosine"]
    print(f"     ridge vs cosine: ratio {r['ratio']:.2f}, p={r['wilcoxon_p']}")

print("\n\n===== headline block =====")
print(json.dumps(d.get("headline", {}), indent=1)[:2500])
