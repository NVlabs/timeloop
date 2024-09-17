#!/usr/bin/env python3

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

import os
import sys
import datetime
import fnmatch
from functools import reduce

LICENSE = "../LICENSE"
LICENSE_SIGNATURE = "Copyright (c)"
LINESEP = "\n" # os.linesep
EXCLUDE_EXT = [
    "pyc",
    "log",
    "gitignore",
    "md",
    "LICENSE",
    "pdf",
    "xml",
    "txt",
    "csv",
    "pkl",
    "ipynb"
]
EXCLUDE_PATTERNS = [
    "core.*",
    "*.out",
    "README",
    "Makefile"
]
EXCLUDE_DIR = [
    "__pycache__",
    "build",
    ".git"
]
license_dispatcher = {
    'py'  : 'python_license',
    'cpp' : 'cpp_license',
    'hpp' : 'cpp_license',
    'cfg' : 'python_license',
    'SConstruct' : 'python_license',
    'SConscript' : 'python_license',
    'sh' : 'python_license',
    'csh' : 'python_license',
    'tcsh' : 'python_license',
    'bash' : 'python_license',
    'zsh' : 'python_license',
    'iscc' : 'python_license',
    'yaml' : 'python_license'
}
license_path = None

def main():
    if len(sys.argv) != 2:
        print("usage: apply_license.py <root>")
        sys.exit(0)
        
    dir_path = os.path.dirname(os.path.realpath(__file__))
    global license_path
    license_path = dir_path + "/" + LICENSE
    if not os.path.isfile(license_path):
        print("ERROR: license template not found at", license_path)
        sys.exit(1)
    else:
        print("OK: license template found at", license_path)
    walk(sys.argv[1])

def walk(start):
    for root, dirs, files in os.walk(start):
        dirs[:] = [ d for d in dirs if d not in EXCLUDE_DIR ]
        files[:] = [ f for f in files if f.split('.')[-1] not in EXCLUDE_EXT ]
        files[:] = [ f for f in files if f not in
                     reduce(set.union, [ set(fnmatch.filter(files, p)) for p in EXCLUDE_PATTERNS ]) ]
        for f in files:
            license_header = license_(f.split('.')[-1])
            if license_header == None:
                print("ERROR: unrecognized file type:",os.path.join(root, f))
                sys.exit(1)
                #continue
            path = os.path.join(root, f)
            if check_and_insert_license(path, license_header) == False:
                print("ABORTING")
                sys.exit(1)

def is_script(path):
    # Detect shebang.
    with open(path) as f:
        header = f.readline()
        if header[0:2] == "#!":
            return True
        else:
            return False
                
def check_and_insert_license(path, license_header):

    with open(path, 'r') as f:
        lines = f.readlines()
        
    license_lines = license_header.splitlines(True)

    # Find license signature.
    content_start = 1 if is_script(path) else 0    
    license_start = 0
    license_signature_found = False
    for line_no in range(content_start, len(lines)):
        line = lines[line_no]
        if line.strip():
            if LICENSE_SIGNATURE in line:
                license_start = line_no
                license_signature_found = True
            break

    if license_signature_found:
        # License signature found at start of file.
        # Make sure the entire license is consistent.
        error = False
        for line_no in range(0, len(license_lines)):
            ref_line = license_lines[line_no]
            line = lines[line_no + license_start]
            if line_no == 0:
                line = sub_year(line)
            if line.rstrip() != ref_line.rstrip():
                print("expected:", ref_line)
                print("found:", line)
                error = True
                break
        if error:
            print("ERROR: malformed license in file:", path)
            return False
        else:
            # print("OK: license correct in file:", path)
            return True
        
    # License signature not found at start of file.
    # Make sure it is nowhere else in the file.
    # found = False    
    # for line_no in range(content_start+1, len(lines)):
    #     if LICENSE_SIGNATURE in lines[line_no]:
    #         found = True
    #         break
    # if found:
    #     print("ERROR: malformed license in file:", path)
    #     return False
    
    # Insert license into file.
    prefix = [lines[0], "\n"] if is_script(path) else []
    separator = ["\n"] if lines[content_start].strip() else []
    content = lines[content_start:]

    with open(path, 'w+') as f:
        for line in prefix + license_lines + separator + content:
            f.write(line)
    
    print("OK: inserted license into file:", path)
    return True
        
def python_license():
    return license_lines("# ")

def cpp_license():
    outstr = ""
    outstr += license_lines(" * ", "/* ")
    outstr += " */" + LINESEP
    return outstr

def txt_license():
    return license_lines("")

def markdown_license():
    outstr = LINESEP
    outstr += license_lines("[//]: # (", None, ")")
    outstr += LINESEP
    return outstr

def license_(ext):
    if not ext in license_dispatcher:
        return None
    else:
        return globals()[license_dispatcher[ext]]()
    
def license_lines(prefix = "", first_line_prefix = None, suffix = ""):
    if first_line_prefix == None:
        first_line_prefix = prefix
    outstr = ""

    with open(license_path, 'r') as f:
        lines = f.readlines()
        
    outstr += license_line(lines[0], first_line_prefix, suffix)
    for line in lines[1:]:
        outstr += license_line(line, prefix, suffix)
        
    return outstr

def license_line(line, prefix, suffix):
    if LICENSE_SIGNATURE in line:
        line = sub_year(line)
    line = prefix + line.rstrip("\n") + suffix + "\n"
    return line

def sub_year(line):
    sig_start = line.find(LICENSE_SIGNATURE)
    assert(sig_start != -1)    
    year_start = sig_start + len(LICENSE_SIGNATURE) + 1
    fakeyear = line[year_start:year_start+4]
    assert(fakeyear[0].isdigit())
    assert(fakeyear[-1].isdigit())
    year = str(datetime.datetime.now().year)
    return line.replace(fakeyear, year)

# def sub_year(line):
#     fakeyear = line.split(' ')[2]
#     if not fakeyear[-1].isdigit():
#         fakeyear = fakeyear[:-1]
#     assert(fakeyear[0].isdigit())
#     assert(fakeyear[-1].isdigit())
#     year = str(datetime.datetime.now().year)
#     return line.replace(fakeyear, year)

main()

