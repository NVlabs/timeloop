# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from typing import Optional, Union, Dict, Any
import re
import ast
import contextlib
import joblib

from joblib import Parallel, delayed
import tqdm.notebook as tqdm

debugger_active = hasattr(sys, "gettrace") and sys.gettrace() is not None

N_THREADS = 1 if debugger_active else os.environ.get("LSB_MAX_NUM_PROCESSORS", 64)
N_PROCS = 1 if debugger_active else os.environ.get("LSB_MAX_NUM_PROCESSORS", 64)

if hasattr(sys, "gettrace") and sys.gettrace() is not None:
    N_PROCS = 1
    N_THREADS = 1


def pretty_sci_notation(s: str) -> str:
    return s.replace("e+0", "e").replace("e+", "e")


def recursive_sort(x: Union[list, dict]) -> Union[list, dict]:
    key = lambda y: str(y)
    if isinstance(x, (list, set)):
        return sorted([recursive_sort(i) for i in x], key=key)
    if isinstance(x, dict):
        keys = sorted([k for k in x.keys()], key=key)
        return {k: recursive_sort(x[k]) for k in keys}
    return x


def recursive_translate(
    x: Union[list, dict, str],
    translation: Dict[str, str],
) -> Union[list, dict, str]:
    if isinstance(x, (list, set, tuple)):
        return type(x)(recursive_translate(y, translation) for y in x)

    if isinstance(x, dict):
        return {
            recursive_translate(k, translation): recursive_translate(v, translation)
            for k, v in x.items()
        }
    return translation.get(x, x)


def literal_eval(x: str, on_fail: Optional[Any]) -> Any:
    if on_fail is None:
        return ast.literal_eval(x)
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return on_fail


def toint(x):
    found = re.findall(r"\d+", x)
    assert len(found) == 1, f">1 integers found in {x}"
    return int(found[0])


# Code from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
# user dano's answer
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def parallel_thread(x, pbar: str = None, n_jobs: int = 0, leave=True):
    if not pbar:
        return _parallel_thread(x, n_jobs=n_jobs)

    with tqdm_joblib(tqdm.tqdm(desc=pbar, total=len(x), leave=leave)) as pbar:
        return Parallel(n_jobs=N_PROCS, backend="threading")(x)


def parallel_proc(x, pbar: str = None, n_jobs: int = 0, leave=True):
    if not pbar:
        return _parallel_proc(x, n_jobs=n_jobs)

    with tqdm_joblib(tqdm.tqdm(desc=pbar, total=len(x), leave=leave)) as pbar:
        return Parallel(n_jobs=N_PROCS)(x)


def as_subproc(x):
    return parallel_proc([x, delayed(lambda: 0)()])[0]


def serial(x, pbar: str = None, n_jobs: int = 0, leave=True):
    if not pbar:
        return _serial(x, n_jobs=n_jobs, leave=leave)

    return [
        y[0](*y[1], **y[2]) for y in tqdm.tqdm(x, desc=pbar, total=len(x), leave=leave)
    ]


def _parallel_thread(x, pbar: str = None, n_jobs: int = 0):
    x = tqdm.tqdm(list(x), desc=pbar, leave=True) if pbar else x
    return Parallel(n_jobs=(n_jobs if n_jobs else N_THREADS), backend="threading")(x)


def _parallel_proc(x, pbar: str = None, n_jobs: int = 0):
    x = tqdm.tqdm(list(x), desc=pbar, leave=True) if pbar else x
    return Parallel(n_jobs=(n_jobs if n_jobs else N_PROCS), backend="loky")(x)


def as_subproc(x):
    return parallel_proc([x, delayed(lambda: 0)()])[0]


def _serial(x, pbar: str = None, n_jobs: int = 0):
    x = tqdm.tqdm(list(x), desc=pbar, leave=True) if pbar else x
    return Parallel(n_jobs=1)(x)
