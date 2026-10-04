import pathlib

SRC = pathlib.Path(r"C:\Users\amirh\Desktop\Demo\scripts\analyze_invalid_action_returns.py")
DST = pathlib.Path(__file__).parent / "analyze_invalid_action_returns.py"
t = SRC.read_text(encoding="utf-8")
anchor = "import heapq\n"
assert t.count(anchor) == 1
t = t.replace(anchor, anchor + 'import sys\nsys.path.insert(0, r"C:\\Users\\amirh\\Desktop\\Thesis_Project\\src")\n')
DST.write_text(t, encoding="utf-8")
print("ok")
