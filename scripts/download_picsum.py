"""scripts.download_picsum

Plain-English purpose: Command-line runners for datasets, benchmarks, reports, and exports.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

import os, sys, time, random, concurrent.futures, urllib.request
out_train = r"data/photos/train"
out_val   = r"data/photos/val"
N = int(sys.argv[1]) if len(sys.argv)>1 else 1000
os.makedirs(out_train, exist_ok=True); os.makedirs(out_val, exist_ok=True)

def grab(idx, split):
    url = f"https://picsum.photos/1024/1024?random={idx}"
    out = os.path.join(out_train if split=="train" else out_val, f"img_{idx:06d}.jpg")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=20) as r, open(out, "wb") as f:
                f.write(r.read())
            return True
        except Exception:
            time.sleep(0.5*(attempt+1))
    return False

idxs = list(range(1, N+1))
random.shuffle(idxs)
cut = int(0.9*N)
pairs = [(i, "train") for i in idxs[:cut]] + [(i, "val") for i in idxs[cut:]]
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
    ok = sum(ex.map(lambda p: grab(*p), pairs))
print(f"downloaded {ok}/{len(pairs)}")
