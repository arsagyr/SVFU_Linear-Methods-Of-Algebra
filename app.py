import importlib
import json
import os
import traceback

import numpy as np
from flask import Flask, abort, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider


# --- JSON-провайдер, умеющий работать с numpy ---
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


app = Flask(__name__)
app.json = NumpyJSONProvider(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "labs.json")


def load_labs():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f).get("labs", [])


def find_lab(lab_id):
    for lab in load_labs():
        if lab["id"] == lab_id:
            return lab
    return None


_handler_cache = {}


def get_handler(module_name, function_name):
    key = (module_name, function_name)
    if key not in _handler_cache:
        module = importlib.import_module(f"handlers.{module_name}")
        func = getattr(module, function_name, None)
        if not callable(func):
            raise AttributeError(f"handlers.{module_name}.{function_name} не найдена")
        _handler_cache[key] = func
    return _handler_cache[key]


@app.route("/")
def index():
    return render_template("index.html", labs=load_labs())


@app.route("/lab/<lab_id>")
def lab_view(lab_id):
    lab = find_lab(lab_id)
    if lab is None:
        abort(404)
    return render_template("lab.html", labs=load_labs(), current=lab_id, current_lab=lab)


@app.route("/lab/<lab_id>/run", methods=["POST"])
def lab_run(lab_id):
    lab = find_lab(lab_id)
    if lab is None:
        return jsonify({"ok": False, "error": "Лаба не найдена"}), 404

    cfg = lab.get("handler") or {}
    module_name, function_name = cfg.get("module"), cfg.get("function")
    if not module_name or not function_name:
        return jsonify({"ok": False, "error": "Обработчик не задан в labs.json"}), 500

    try:
        handler = get_handler(module_name, function_name)
        form_data = request.form.to_dict()
        result = handler(**form_data)
    except Exception as e:
        # Логируем полный трейсбек в консоль, но клиенту отдаём JSON
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc().split("\n")[-5:],  # последние 5 строк
        }), 500

    if not isinstance(result, dict):
        result = {"ok": True, "result": result}
    return jsonify(result)


@app.errorhandler(404)
def not_found(error):
    return render_template("index.html", labs=load_labs()), 404


@app.errorhandler(500)
def server_error(error):
    """Если Flask сам упал — возвращаем JSON вместо HTML."""
    return jsonify({"ok": False, "error": "Внутренняя ошибка сервера"}), 500


if __name__ == "__main__":
    app.run(debug=True)