"""Fails the build if the real-network test scene wasn't un-tied properly."""
import json
import sys

line = [l for l in open(sys.argv[1]) if l.startswith("SELFTEST_JSON ")]
if not line:
    raise SystemExit("selftest did not finish")
r = json.loads(line[-1][len("SELFTEST_JSON "):])
print("result:", r)
assert r["null_db"] < -100, "stems do not add back to the original"
assert all(g > 3 for g in r["gain"]), "a character's stem is not clearly better than the mix"
assert sum(r["gain"]) / len(r["gain"]) > 8, "separation quality too low"
if r["found"] != r["expected"]:
    print("::warning::automatic character count found %d of %d" % (r["found"], r["expected"]))
print("SELFTEST PASSED")
