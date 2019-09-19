# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import functools
import inspect
import os
import subprocess
import sys
import timeit

import libconf

# Output file names.
out_prefix = "timeloop."
log_file_name = out_prefix + "log";
stats_file_name = out_prefix + "stats.txt";
xml_file_name = out_prefix + "map+stats.xml";
map_txt_file_name = out_prefix + "map.txt";
map_cfg_file_name = out_prefix + "map.cfg";
map_cpp_file_name = out_prefix + "map.cpp";
output_file_names = [ log_file_name,
                      stats_file_name,
                      xml_file_name,
                      map_txt_file_name,
                      map_cfg_file_name,
                      map_cpp_file_name ]

def prod (l):
    return functools.reduce(lambda x, y: x*y, l)


def rewrite_workload_bounds(src, dst, workload_bounds):
    w, h, c, n, k, s, r, wpad, hpad, wstride, hstride = workload_bounds
    q = int((w - s + 2 * wpad) / wstride) + 1
    p = int((h - r + 2 * hpad) / hstride) + 1

    print('Workload Dimensions:')
    print('  W        =', w)
    print('  H        =', h)
    print('  C        =', c)
    print('  K        =', k)
    print('  S        =', s)
    print('  R        =', r)
    print('  P        =', p)
    print('  Q        =', q)
    print('  N        =', n)
    print('  W-pad    =', wpad)
    print('  H-pad    =', hpad)
    print('  W-stride =', wstride)
    print('  H-stride =', hstride)
    print()

    with open(src, "r") as f:
        config = libconf.load(f)
    config['problem']['R'] = r
    config['problem']['S'] = s
    config['problem']['P'] = p
    config['problem']['Q'] = q 
    config['problem']['C'] = c
    config['problem']['K'] = k
    config['problem']['N'] = n
    config['problem']['Wstride'] = wstride
    config['problem']['Hstride'] = hstride
    config['problem']['Wdilation'] = 1
    config['problem']['Hdilation'] = 1

    with open(dst, "w") as f:
        f.write(libconf.dumps(config))


def run_timeloop(dirname, configfile, logfile='timeloop.log', workload_bounds=None):
    configfile_path = os.path.join(dirname, os.path.basename(configfile))
    logfile_path = os.path.join(dirname, logfile)
    if workload_bounds:
        rewrite_workload_bounds(configfile, configfile_path, workload_bounds)
    else:
        subprocess.check_call(['cp', configfile, configfile_path])

    print('Running timeloop to get mapping')
    def stmt():
        with open(logfile_path, "w") as outfile:
            this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
            timeloop_executable_location = os.path.join(
                    os.path.dirname(this_file_path), '..', 'build', 'timeloop')
            status = subprocess.call([timeloop_executable_location, configfile_path], stdout = outfile, stderr = outfile)
            if status != 0:
                subprocess.check_call(['cat', logfile_path])
                print('Did you remember to build timeloop and set up your environment properly?')
                sys.exit(1)
    t = timeit.Timer(stmt)
    time = t.timeit(1)
    print('Time to run timeloop = ', time)
    
    # Move timeloop output files to the right directory
    for f in output_file_names:
        if os.path.exists(f):
            os.rename(f, dirname + '/' + f)
