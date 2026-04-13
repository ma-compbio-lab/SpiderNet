"""
SpiderNet Results Viewer - Flask Application
"""
from flask import Flask, render_template, send_from_directory, abort, url_for
from pathlib import Path
import config
from utils import discover_datasets, categorize_plots, organize_plots_by_section

app = Flask(__name__)
app.config['SECRET_KEY'] = 'spidernet-results-viewer-2026'

# Dataset discovery (cached on startup)
print("🔍 Discovering datasets...")
DATASETS = discover_datasets()
print(f"📊 Found {len(DATASETS)} dataset(s)")
for name, info in DATASETS.items():
    print(f"   - {name}: {info['plot_count']} plots")


@app.route('/')
def index():
    """Landing page with dataset selector"""
    return render_template('index.html',
                         datasets=DATASETS,
                         app_title=config.APP_TITLE)


@app.route('/team')
def team():
    """Team page"""
    return render_template('team.html',
                         app_title=config.APP_TITLE)


@app.route('/paper')
def paper():
    """Publication page"""
    return render_template('paper.html',
                         app_title=config.APP_TITLE)


@app.route('/dataset/<dataset_name>/basic')
def basic_analysis(dataset_name):
    """Basic analysis plots for selected dataset"""
    if dataset_name not in DATASETS:
        abort(404)

    plots = categorize_plots(DATASETS[dataset_name]['results_dir'])['basic']
    sections = organize_plots_by_section(plots, 'basic')

    return render_template(
        'basic_analysis.html',
        dataset_name=dataset_name,
        dataset=DATASETS[dataset_name],
        sections=sections,
        app_title=config.APP_TITLE
    )


@app.route('/dataset/<dataset_name>/subtype')
def subtype_analysis(dataset_name):
    """Subtype analysis plots for selected dataset"""
    if dataset_name not in DATASETS:
        abort(404)

    plots = categorize_plots(DATASETS[dataset_name]['results_dir'])['subtype']
    sections = organize_plots_by_section(plots, 'subtype')

    return render_template(
        'subtype_analysis.html',
        dataset_name=dataset_name,
        dataset=DATASETS[dataset_name],
        sections=sections,
        app_title=config.APP_TITLE
    )


@app.route('/dataset/<dataset_name>/cascade')
def cascade_analysis(dataset_name):
    """MI cascade analysis plots for selected dataset"""
    if dataset_name not in DATASETS:
        abort(404)

    plots = categorize_plots(DATASETS[dataset_name]['results_dir'])['cascade']
    sections = organize_plots_by_section(plots, 'cascade')

    return render_template(
        'cascade_analysis.html',
        dataset_name=dataset_name,
        dataset=DATASETS[dataset_name],
        sections=sections,
        app_title=config.APP_TITLE
    )


@app.route('/static/results/<dataset_name>/<path:filepath>')
def serve_result(dataset_name, filepath):
    """Serve plot images from results directories"""
    if dataset_name not in DATASETS:
        abort(404)

    # Construct full path
    results_dir = DATASETS[dataset_name]['results_dir']
    full_path = results_dir / filepath

    # Security: ensure file is within results directory
    try:
        full_path = full_path.resolve()
        results_dir = results_dir.resolve()
        if not str(full_path).startswith(str(results_dir)):
            abort(403)
    except:
        abort(403)

    if not full_path.exists():
        abort(404)

    return send_from_directory(full_path.parent, full_path.name)


@app.context_processor
def utility_processor():
    """Make utility functions available in templates"""
    def generate_plot_url(dataset_name, plot_path):
        return url_for('serve_result', dataset_name=dataset_name, filepath=plot_path)

    return dict(generate_plot_url=generate_plot_url, datasets=DATASETS)


@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error page"""
    return render_template('404.html', app_title=config.APP_TITLE), 404


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🚀 Starting {config.APP_TITLE}")
    print(f"{'='*60}")
    print(f"🌐 Server: http://localhost:{config.PORT}")
    print(f"📂 Root: {config.SPIDERNET_ROOT}")
    print(f"{'='*60}\n")
    print("Press Ctrl+C to stop the server\n")

    app.run(host='0.0.0.0', port=config.PORT, debug=config.DEBUG)
