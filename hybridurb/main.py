# -*- coding: utf-8 -*-
"""command line interface for HybridUrb runners"""


import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import click
from .runners import NowcastRunner

# TODO: suport hindcast

### Below is the documentation for the commandline interface, see the CLICK-package.
@click.command(short_help="Run HybridUrb models")
@click.argument("root_dir", type=click.Path(exists=True, resolve_path=True, dir_okay=True, file_okay=False))
@click.option("--mode", type = str, default=None, help="Mode of the run: hindcast (not implemented), nowcast")
@click.option("--t0",  type = str, default=None, help="T0 of the run (%Y%m%d%H%M)")
@click.option("--export_to_fews",  type = bool, default=False,  is_flag=True, help="Export MayLayerFiles for Delft-FEWS.")
def run(root_dir: str, mode:str = None, t0: str = None, export_to_fews: bool = False):
    click.echo(f"run HybridUrb using the following arguments: {root_dir} --mode {mode} --t0 {t0} --export_to_fews {export_to_fews}")
    def _as_path(my_dir: str) -> Optional[Path]:
        if not my_dir:
            return None

        _my_dir = Path(my_dir)
        if not _my_dir.is_dir():
            raise NotADirectoryError(_my_dir)
        return _my_dir

    def _as_datetime(my_time:str):
        if not my_time:
            return None

        try:
            _my_time = datetime.strptime(my_time, '%Y%m%d%H%M')
        except:
            raise ValueError("t0 is not in the correct format. ")

    root_dir = _as_path(root_dir)
    _model_dir = _as_path(root_dir / 'model')
    _input_dir = _as_path(root_dir / 'input')
    _my_t0 = _as_datetime(t0)

    if mode == 'nowcast':
        _runner = NowcastRunner(root_dir = root_dir, t0 = t0)
        _runner.run()

    elif mode == 'hindcast':
        # _runner = HindcastRunner(root_dir = root_dir, t0 = t0)
        raise NotImplementedError

    else:
        Runner = NowcastRunner # FIXME: change this after runner structure is more clear
        _runner = Runner(root_dir = root_dir, t0 = t0)

    if export_to_fews is True:
        _runner.export_to_fews()


if __name__ == "__main__":
    try:
        run()
    except Exception as e_info:
        logging.error(str(e_info))
