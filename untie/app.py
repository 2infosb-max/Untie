"""Un-tie local app: the window's page talks to this small server running on the Mac."""
import base64
import os
import threading
import time
import traceback

import numpy as np
from flask import Flask, jsonify, request, send_file

from . import __version__, engine
from .models import TorchModels

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(os.path.expanduser("~"), "Library", "Caches", "Untie")

# skip the slow reverse-DNS lookup the built-in web server does when it starts
import http.server as _hs  # noqa: E402
import socketserver as _ss  # noqa: E402


def _fast_server_bind(self):
    _ss.TCPServer.server_bind(self)
    host, port = self.server_address[:2]
    self.server_name = host
    self.server_port = port


_hs.HTTPServer.server_bind = _fast_server_bind

app = Flask(__name__, static_folder=None)
MODELS_FACTORY = [TorchModels]
P = engine.Project(lambda: MODELS_FACTORY[0](), WORK)
JOB = {"running": False, "stage": "", "pct": 0, "eta": None, "error": None, "kind": None, "cancel": False}
JLOCK = threading.Lock()


def _clean_old_cache():
    try:
        import shutil
        for d in os.listdir(WORK):
            full = os.path.join(WORK, d)
            if os.path.isdir(full) and time.time() - os.path.getmtime(full) > 3 * 86400:
                shutil.rmtree(full, ignore_errors=True)
    except OSError:
        pass


_clean_old_cache()


def run_job(kind, fn):
    with JLOCK:
        if JOB["running"]:
            return jsonify(error="Un-tie is busy - wait for the current step to finish."), 409
        JOB.update(running=True, stage="Starting…", pct=0, eta=None, error=None, kind=kind, cancel=False)

    def prog(pct, stage, eta):
        JOB.update(pct=float(pct), stage=stage, eta=None if eta is None else float(eta))

    def go():
        try:
            fn(prog)
            JOB.update(stage="Done", pct=100, eta=0)
        except engine.Cancelled:
            JOB.update(error="Stopped.")
        except Exception as e:  # shown to the user
            traceback.print_exc()
            JOB.update(error=str(e) or e.__class__.__name__)
        finally:
            JOB["running"] = False
    threading.Thread(target=go, daemon=True).start()
    return jsonify(ok=True)


def cancelled():
    return JOB["cancel"]


def chars_json():
    return [{"id": c["id"], "name": c["name"], "color": c["color"], "sample": c["sample"]} for c in P.chars]


@app.get("/")
def index():
    return send_file(os.path.join(HERE, "index.html"))


@app.get("/api/state")
def state():
    return jsonify(version=__version__, job={k: v for k, v in JOB.items() if k != "cancel"},
                   file=P.info(), analysed=P.analysed, analysed_voices=P.analysed_voices,
                   chars=chars_json(), settings=P.settings, rendered=P.rendered,
                   render_version=P.render_version, edits=P.edits,
                   null_db=P.null_db, null_db_24=P.null_db_24,
                   overlaps=P.overlaps(), stems=P.stem_names() if P.rendered else [])


@app.post("/api/open")
def open_file():
    path = (request.get_json(silent=True) or {}).get("path")
    if not path or not os.path.exists(path):
        return jsonify(error="File not found."), 400

    def fn(prog):
        prog(20, "Reading the file", None)
        P.load(path)
    return run_job("open", fn)


@app.get("/api/wave")
def wave():
    if P.x is None:
        return jsonify(error="no file"), 400
    mn, mx = P.peaks()
    return jsonify(per_second=100, min=base64.b64encode(mn.tobytes()).decode(),
                   max=base64.b64encode(mx.tobytes()).decode())


@app.get("/api/lanes")
def lanes():
    q = P.lanes() if P.rendered else []
    return jsonify(per_second=100, version=P.render_version,
                   lanes=[base64.b64encode(np.ascontiguousarray(r).tobytes()).decode() for r in q])


@app.post("/api/settings")
def settings():
    d = request.get_json(silent=True) or {}
    if "voices" in d and int(d["voices"]) == 2:
        P.settings["voices"] = int(d["voices"])
    if d.get("bg") in ("natural", "clean"):
        if P.settings["bg"] != d["bg"]:
            P.settings["bg"] = d["bg"]
            P.rendered = False
    return jsonify(P.settings)


@app.post("/api/analyse")
def analyse():
    d = request.get_json(silent=True) or {}
    k = d.get("count")

    def fn(prog):
        prog(1, "Loading the voice networks", None)
        P.models  # noqa: B018  (load now so the time shows as its own step)
        P.analyse(prog, cancelled)
        prog(97, "Finding the characters", None)
        P.find_characters(int(k) if k else None)
    return run_job("analyse", fn)


@app.post("/api/find")
def find():
    k = (request.get_json(silent=True) or {}).get("count")
    try:
        P.find_characters(int(k) if k else None)
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(chars=chars_json())


@app.post("/api/chars/<action>")
def chars(action):
    d = request.get_json(silent=True) or {}
    try:
        if action == "add":
            P.add_character(d.get("name"), float(d["t0"]), float(d["t1"]))
        elif action == "sample":
            P.set_sample(d["id"], float(d["t0"]), float(d["t1"]))
        elif action == "rename":
            P.rename(d["id"], d.get("name"))
        elif action == "delete":
            P.delete_char(d["id"])
        elif action == "merge":
            P.merge(int(d["src"]), int(d["dst"]))
        else:
            return jsonify(error="unknown"), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 400
    return jsonify(chars=chars_json())


@app.post("/api/untie")
def untie():
    return run_job("untie", lambda prog: P.untie(prog, cancelled))


@app.post("/api/edit")
def edit():
    d = request.get_json(silent=True) or {}
    return run_job("edit", lambda prog: P.add_edit(d.get("kind"), d.get("t0"), d.get("t1"), d.get("chars", []), prog))


@app.post("/api/edit/remove")
def edit_remove():
    i = (request.get_json(silent=True) or {}).get("index", -1)
    return run_job("edit", lambda prog: P.remove_edit(i, prog))


@app.post("/api/cancel")
def cancel():
    JOB["cancel"] = True
    return jsonify(ok=True)


@app.post("/api/export")
def export():
    d = request.get_json(silent=True) or {}
    folder = d.get("folder")
    if not folder or not os.path.isdir(folder):
        return jsonify(error="Choose a folder."), 400
    res = {}

    def fn(prog):
        res["files"] = P.export(folder, d.get("format", "pcm24"), bool(d.get("residual", True)), prog,
                                wavs=bool(d.get("wavs", True)), aaf=bool(d.get("aaf", False)),
                                start_tc=(d.get("start_tc") or None), fps=float(d.get("fps", 25)),
                                guide=bool(d.get("guide", True)))
        JOB["files"] = res["files"]
    return run_job("export", fn)


@app.get("/api/default_tc")
def default_tc():
    from .aaf_export import default_start_tc
    fps = float(request.args.get("fps", 25))
    if P.x is None:
        return jsonify(tc="01:00:00:00")
    return jsonify(tc=default_start_tc(P.time_ref, P.sr, fps)[0], stamped=P.time_ref is not None)


@app.get("/api/audio/<which>")
def audio(which):
    if which == "orig":
        if P.x is None:
            return "", 404
        return send_file(P.orig_preview, mimetype="audio/wav", conditional=True, max_age=0)
    if not P.rendered:
        return "", 404
    i = int(which)
    paths = P.stem_paths()
    if not 0 <= i < len(paths):
        return "", 404
    return send_file(paths[i], mimetype="audio/wav", conditional=True, max_age=0)
