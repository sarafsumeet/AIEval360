"""
AIEval360 CLI — command-line entry point.

Usage:
    aieval360 --help
    aieval360 evaluate --dataset data/... --sector banking
    aieval360 safety    --sector banking --provider mock
    aieval360 version
"""

import sys
from pathlib import Path

try:
    import typer
except ImportError:  # pragma: no cover
    print("typer is required for the CLI. Run: pip install 'aieval360[cli]'", file=sys.stderr)
    sys.exit(1)

app = typer.Typer(
    name="aieval360",
    help="AIEval360 — Enterprise AI Evaluation & Assurance Platform",
    no_args_is_help=True,
)


@app.command("evaluate")
def run_evaluate(
    dataset: Path = typer.Option(..., help="Path to CSV dataset"),
    sector: str = typer.Option(..., help="Sector (banking/university/insurance)"),
    output: Path = typer.Option(Path("evidence/evaluation_runs"), help="Output directory"),
    config: Path = typer.Option(Path("config/metrics_config.yaml"), help="Metrics config"),
    provider: str = typer.Option("mock", help="Provider identifier"),
    model: str = typer.Option("mock", help="Model identifier"),
) -> None:
    """Run metric evaluation pipeline on a dataset."""
    import asyncio  # noqa: PLC0415

    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    from run_evaluation import main as _main  # noqa: PLC0415

    # Override sys.argv so argparse in run_evaluation picks up our args
    sys.argv = [
        "run_evaluation.py",
        "--dataset", str(dataset),
        "--sector", sector,
        "--output", str(output),
        "--config", str(config),
        "--provider", provider,
        "--model", model,
    ]
    exit_code = asyncio.run(_main())
    raise typer.Exit(exit_code)


@app.command("safety")
def run_safety(
    sector: str = typer.Option(..., help="Sector (banking/university/insurance)"),
    provider: str = typer.Option("mock", help="Provider (mock/openai/ollama)"),
    model: str = typer.Option("mock", help="Model identifier"),
    output: Path = typer.Option(Path("evidence/safety_reports"), help="Output directory"),
) -> None:
    """Run safety probe evaluation."""
    import asyncio  # noqa: PLC0415

    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    from run_safety_probes import main as _main  # noqa: PLC0415

    sys.argv = [
        "run_safety_probes.py",
        "--sector", sector,
        "--provider", provider,
        "--model", model,
        "--output", str(output),
    ]
    exit_code = asyncio.run(_main())
    raise typer.Exit(exit_code)


@app.command("version")
def show_version() -> None:
    """Show the installed AIEval360 version."""
    try:
        from importlib.metadata import version  # noqa: PLC0415
        ver = version("aieval360")
    except Exception:
        ver = "unknown"
    typer.echo(f"aieval360 {ver}")


if __name__ == "__main__":
    app()
