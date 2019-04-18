#! /usr/bin/env python3

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

# This file enables testing changes in Timeloop's outputs relative to previous versions.
# It is intended for regression testing.

import argparse
import inspect
import pickle
import os
import subprocess
import random
import sys

import numpy as np
import libconf

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
root_dir = os.path.join(os.path.dirname(this_file_path), '..')

sys.path.append(os.path.join(root_dir, 'scripts'))
import timeloop
import parse_timeloop_output


test_suite = [
        'configs/sample.cfg',
        ]

def diff(ref, actual, location='stats'):
    assert(isinstance(ref, dict))
    assert(isinstance(actual, dict))

    found_difference = False

    missing = set(ref.keys()) - set(actual.keys())
    if missing:
        print('Within %s, test output missing keys: %s' % (location, str(list(missing))))
        found_difference = True

    extra = set(actual.keys()) - set(ref.keys())
    if extra:
        print('Within %s, test output has extra keys: %s' % (location, str(list(extra))))
        found_difference = True

    for key in set(actual.keys()) & set(ref.keys()):
        if isinstance(ref[key], dict):
            found_difference |= diff(ref[key], actual[key], '%s[%s]' % (location, repr(key)))
        else:
            is_same = (
                    np.allclose(np.nan_to_num(ref[key]),
                                np.nan_to_num(actual[key]))
                    if isinstance(ref[key], (np.ndarray, float))
                    else ref[key] == actual[key])
            if not is_same:
                print('difference at %s:\n  key: %s\n  test produced value: %s\n  expected reference value: %s'
                      % (location, str(key), str(actual[key]), str(ref[key])))
                found_difference = True

    return found_difference



def get_or_make_dir(test):
    test_config_abspath = os.path.join(root_dir, test)

    # Just test that path points to a valid config file.
    with open(test_config_abspath, 'r') as f:
        config = libconf.load(f)

    test_name = os.path.splitext(test)[0]
    test_name_str = test_name.replace(os.sep, '_')
    dirname = os.path.join(root_dir, 'tests', 'results', 'changes', test_name_str)
    subprocess.check_call(['mkdir', '-p', dirname])

    return dirname, test_config_abspath


def get_reference_path(dirname):
    return os.path.join(dirname, 'reference_stats.pkl')


def regenerate_reference():
    print('Overwriting all reference pickle files in tests/results/changes/ ...')
    for test in test_suite:
        dirname, config_abspath = get_or_make_dir(test)
        print('Running test in %s/ ...' % dirname)

        timeloop.run_timeloop(dirname, config_abspath)

        stats = parse_timeloop_output.parse_timeloop_stats(dirname)

        assert not diff(stats, stats)

        ref_path = get_reference_path(dirname)
        print('Writing results to %s ...' % ref_path)
        with open(ref_path, 'wb') as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
    print('Done writing reference pickle files.')


def run_tests():
    error_suggestion = '\n\nIf you intentionally changed the output or tests, please run ./%s --regenerate-reference\n\n' % os.path.relpath(this_file_path)
    print('Running tests against reference values in tests/results/changes/ ...')
    success = True
    for test in test_suite:
        dirname, config_abspath = get_or_make_dir(test)
        print('Preparing to run test in %s/ ...' % dirname)

        ref_path = get_reference_path(dirname)
        assert os.path.exists(ref_path), 'Cannot find ' + os.path.relpath(ref_path) + error_suggestion
        with open(ref_path, 'rb') as f:
            ref = pickle.load(f)

        timeloop.run_timeloop(dirname, config_abspath)

        stats = parse_timeloop_output.parse_timeloop_stats(dirname)

        if diff(ref, stats):
            print('Test failed in %s\n' % dirname)
            success = False
        else:
            print('Test passed in %s' % dirname)
    print('Done running tests in tests/results/changes/.')
    if success:
        print('All tests passed.')
    else:
        print('Some tests failed.')
        print(error_suggestion)


def main():
    parser = argparse.ArgumentParser(
            description='Compare timeloop output with past versions.')
    parser.add_argument('--regenerate-reference', action='store_true',
            help='overwrite reference output files with current output')
    options = parser.parse_args()

    if options.regenerate_reference:
        regenerate_reference()
    else:
        run_tests()

if __name__ == '__main__':
    main()
