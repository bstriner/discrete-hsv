import torch
import ignite
import os
import warnings
LOGS_FNAME = "logs.tsv"
PLOT_FNAME = "plot.svg"


def print_logs(output_dir, max_epoch, loader, pbar):
    def fn(engine):
        fname = os.path.join(output_dir, LOGS_FNAME)
        columns = ["iteration", ] + list(engine.state.metrics.keys())
        values = [str(engine.state.iteration), ] + [str(round(value, 5))
                                                    for value in engine.state.metrics.values()]

        with open(fname, "a") as f:
            if f.tell() == 0:
                print("\t".join(columns), file=f)
            print("\t".join(values), file=f)

        message = "[{epoch}/{max_epoch}][{i}/{max_i}]".format(
            epoch=engine.state.epoch, max_epoch=max_epoch, i=(engine.state.iteration % len(loader)), max_i=len(loader)
        )
        for name, value in zip(columns, values):
            message += " | {name}: {value}".format(name=name, value=value)

        pbar.log_message(message)
    return fn


def print_times(pbar, timer):
    def fn(engine):
        pbar.log_message("Epoch {} done. Time per batch: {:.3f}[s]".format(
            engine.state.epoch, timer.value()))
        timer.reset()
    return fn


def create_plots(output_dir):
    def fn(engine):
        try:
            import matplotlib as mpl

            mpl.use("agg")

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

        except ImportError:
            warnings.warn(
                "Loss plots will not be generated -- pandas or matplotlib not found")

        else:
            df = pd.read_csv(os.path.join(output_dir, LOGS_FNAME),
                             delimiter="\t", index_col="iteration")
            _ = df.plot(subplots=True, figsize=(20, 20))
            _ = plt.xlabel("Iteration number")
            fig = plt.gcf()
            path = os.path.join(output_dir, PLOT_FNAME)

            fig.savefig(path)
    return fn


def handle_exception(handler=None, **kwargs):
    def fn(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn("KeyboardInterrupt caught. Exiting gracefully.")
            create_plots(engine)
            if handler is not None:
                handler(engine, **kwargs)
        else:
            raise e
    return fn


def get_value(key):
    def getter(dictionary):
        return dictionary[key]
    return getter


class Args(object):
    def __init__(self, args):
        self.args = args

    def state_dict(self):
        return vars(self.args)
