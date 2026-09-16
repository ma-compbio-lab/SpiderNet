"""SpiderNet-Interactive — Flask app factory."""
from flask import Flask, render_template

import config
from core.datasets import discover_datasets
from modules.module1_basic import bp as m1_bp
from modules.module2_subtype import bp as m2_bp
from modules.module3_cascade import bp as m3_bp
from modules.module4_perturb import bp as m4_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "spidernet-interactive-2026"
    app.config["APP_TITLE"] = config.APP_TITLE

    print("🔍 Discovering datasets...")
    datasets = discover_datasets(config.SEARCH_ROOT)
    print(f"📊 Found {len(datasets)} dataset(s) under {config.SEARCH_ROOT}")
    for name, ds in datasets.items():
        print(f"   - {name}: {ds.run_dir.name} (dim {ds.dim_envir}, {ds.version})")
    app.config["DATASETS"] = datasets

    @app.route("/")
    def index():
        return render_template("index.html", datasets=datasets, app_title=config.APP_TITLE)

    @app.route("/team")
    def team():
        return render_template("team.html", app_title=config.APP_TITLE)

    @app.route("/paper")
    def paper():
        return render_template("paper.html", app_title=config.APP_TITLE)

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("404.html", app_title=config.APP_TITLE), 404

    @app.context_processor
    def inject_globals():
        return {"datasets": datasets, "app_title": config.APP_TITLE}

    app.register_blueprint(m1_bp)
    app.register_blueprint(m2_bp)
    app.register_blueprint(m3_bp)
    app.register_blueprint(m4_bp)

    return app


app = create_app()


if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print(f"🚀 Starting {config.APP_TITLE}")
    print(f"{'=' * 60}")
    print(f"🌐 Server: http://localhost:{config.PORT}")
    print(f"📂 Search: {config.SEARCH_ROOT}")
    print(f"{'=' * 60}\n")
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)
