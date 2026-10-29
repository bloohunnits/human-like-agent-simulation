# Shared helper functions across all packages
import multiprocessing
from contextlib import redirect_stdout, redirect_stderr

def _write_output_to_file(func, filename, **kwargs):
    with open(filename, "w") as f:
        with redirect_stdout(f), redirect_stderr(f):
            func(kwargs)

def start_and_log_process(func, output_fname, **kwargs):
    process = multiprocessing.Process(
        target=run_function_to_file,
        args=(my_function, output_fname)
    )

    process.start()
    return process
