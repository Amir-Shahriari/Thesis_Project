import pathlib

SRC = pathlib.Path(r"C:\Users\amirh\Desktop\Thesis_Project\scripts\analysis")
DST = pathlib.Path(__file__).parent
OLD = r'r"C:\Users\amirh\Desktop\qwarm-gnn-rl\src"'
NEW = r'r"C:\Users\amirh\Desktop\Thesis_Project\src"'
for f in "blockers revisit_rules locality_control hopdist step1_returns step1b_nonleak step23_calibrate".split():
    t = (SRC / f"{f}.py").read_text(encoding="utf-8")
    assert t.count(OLD) == 1, f
    (DST / f"{f}.py").write_text(t.replace(OLD, NEW), encoding="utf-8")
    print(f, "ok")
