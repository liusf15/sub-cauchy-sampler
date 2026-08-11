from pathlib import Path


def make_output_dirs(rootdir, plotdir, experiment, date):
    """Create and return result/plot directories for one dated experiment run."""
    result_dir = Path(rootdir) / experiment / date
    plot_dir = Path(plotdir) / experiment / date
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return str(result_dir), str(plot_dir)
