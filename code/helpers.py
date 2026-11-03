# Shared helper functions across all packages
import multiprocessing
import os
import sys
import json

def _run_with_redirect(func, filename, kwargs):
    # Redirect at the file-descriptor level so C extensions/subprocesses are captured too
    with open(filename, "w", buffering=1) as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())
        func(**kwargs)


def start_and_log_process(func, output_fname, **kwargs):
    process = multiprocessing.Process(
        target=_run_with_redirect,
        args=(func, output_fname, kwargs),
    )
    process.start()
    return process

    process.start()
    return process

def extract_json_objects_as_str(text):
    decoder = json.JSONDecoder()
    objects = []
    i = 0
    while True:
        i = text.find("{", i)
        if i == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, i)
            objects.append(json.dumps(obj))
            i = end  # skip past the object we just parsed
        except json.JSONDecodeError:
            i += 1  # not valid JSON here, try the next "{"
    return objects
