import json
from prov import OUT_DIR

p = OUT_DIR / "item_e_exact_diameters.json"
d = json.loads(p.read_text(encoding="utf-8"))
d["sub_item_status"] = {
    "as_used 25x25 (e=2), 5 thesis seeds": ["REPRODUCED", "exact diameter 5 at all five seeds"],
    "as_used 50x50 (e=3), 5 thesis seeds": ["DIFFERENT", "exact diameter 5 at four seeds, 6 at seed 1173222464 (2 nodes attain eccentricity 6)"],
    "as_used 100x100 (e=4), 5 thesis seeds": ["DIFFERENT", "exact diameter 6 at all five seeds (126-166 nodes have eccentricity 6); the sampled 50/60-source value is 5"],
    "pure grids 0 chords": ["DIFFERENT", "exact 48 / 98 / 198 = 2(W-1), against thesis 47 / 96 / 194 (sampled lower bounds)"],
    "tab:density 60 chords": ["DIFFERENT", "exact 16, thesis 15 (sampled)"],
    "tab:density 1,250 chords": ["DIFFERENT", "exact 6, thesis 5 (sampled)"],
    "tab:density 125/250/400/625 chords": ["REPRODUCED", "exact 13/10/8/7 equal the sampled values"],
    "tab:density 0 chords": ["DIFFERENT", "exact 48, thesis 47"],
}
d["status"] = "DIFFERENT"
d["notes"] = ("The thesis diameters are the largest eccentricity over 50 (step23_calibrate.py) "
              "or 60 (hopdist.py) sampled BFS sources, a lower bound; the sampled method "
              "reproduces the thesis values exactly (47/15/13/10/8/7/5). Exact all-source "
              "diameters are 48/16/13/10/8/7/6 on the density axis, 48/98/198 on the pure "
              "grids, and 5 (25x25), 5-6 (50x50) and 6 (100x100) on the as-used graphs at the "
              "five thesis grid seeds. 'Diameter five at every scale' is therefore exact only "
              "at 25x25. The diameter_sampled_50_sources_rng0 field for the 100x100 pure grid "
              "(188) uses this script's 50-source draw; hopdist.py's 60-source draw gives 194.")
p.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
print("ok")
