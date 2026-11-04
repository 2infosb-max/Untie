"""Starts Un-tie as a normal Mac app: the page opens in its own window; closing it quits."""
import json
import logging
import os
import socket
import sys
import threading
import time

LOG_DIR = os.path.join(os.path.expanduser("~"), "Library", "Logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "Untie.log")
if getattr(sys, "frozen", False) and not os.environ.get("UNTIE_SELFTEST"):
    _log = open(LOG_PATH, "a", buffering=1)
    sys.stdout = sys.stderr = _log
    logging.basicConfig(stream=_log, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Api:
    """Things the page asks the Mac app to do."""

    def choose_file(self):
        import webview
        types = ("Audio files (*.wav;*.wave;*.bwf;*.aif;*.aiff;*.flac;*.mp3;*.m4a;*.caf;*.ogg;*.WAV;*.AIF;*.AIFF)",
                 "All files (*.*)")
        res = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, file_types=types)
        return res[0] if res else None

    def choose_folder(self):
        import webview
        res = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        if not res:
            return None
        return res[0] if isinstance(res, (list, tuple)) else res

    def reveal(self, path):
        import subprocess
        if sys.platform == "darwin" and os.path.exists(str(path)):
            subprocess.run(["open", "-R", str(path)])
        return True


def main():
    if os.environ.get("UNTIE_SELFTEST"):
        sys.stdout = sys.__stdout__
        from untie import selftest
        res = selftest.run(os.environ["UNTIE_SELFTEST"], int(os.environ.get("UNTIE_VOICES", "2")))
        print("SELFTEST_JSON " + json.dumps(res))
        return
    from untie.app import app
    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    threading.Thread(target=lambda: app.run(host="127.0.0.1", port=port, threaded=True, debug=False,
                                            use_reloader=False), daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    logging.info("serving at %s", url)
    try:
        with open(os.path.join(LOG_DIR, "Untie.url"), "w") as f:
            f.write(url)
    except OSError:
        pass
    try:
        import webview
        webview.create_window("Un-tie by Sound Brewery", url, js_api=Api(), width=1440, height=900,
                              min_size=(1080, 680), background_color="#0E1012")
        webview.start()
    except Exception:
        logging.exception("app window failed, falling back to the web browser")
        import webbrowser
        webbrowser.open(url)
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
