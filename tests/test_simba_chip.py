#! /usr/bin/env python3

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

import inspect
import os
import platform
import pprint
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

# Ugly hack because Darwin kills the DYLD_FALLBACK_LIBRARY_PATH env variable
# and we need this for the timeloop binary to find its libraries.
if (platform.system() == 'Darwin'):
    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = os.path.join(os.environ['TIMELOOP_BASE_PATH'], 'lib')

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
os.chdir(this_directory)
sys.path.append(os.path.join(this_directory, '..', 'scripts'))
sys.path.append(os.path.join(os.environ['TIMELOOP_BASE_PATH'], 'scripts'))

from conv_problems import *
import timeloop
import parse_timeloop_output

# Output file names.
out_prefix = "timeloop-mapper."
xml_file_name = out_prefix + "map+stats.xml";

def top(problem_indices):

    for j in problem_indices:

        problem = inference_server_set[j]

        # Hack: only run problems with at least 16 input channels. This prevents
        # a sliding-window of inputs between the global buffer and input buffer,
        # which forces the results to align with the reference equations.
        # Also: only run problems with at least 16 output channels.
        C = problem[2]
        K = problem[4]
        if (C % 16) != 0:
            print("Skipping problem index", j, "because it has C =", C, \
                  "but we need it to be divisible by 16 to run the tests")
            continue
        if (K % 16) != 0:
            print("Skipping problem index", j, "because it has K =", K, \
                  "but we need it to be divisible by 16 to run the tests")
            continue

        # Hack: force batch size N==1 because our constraints assume N==1.
        problem_list = list(problem)
        problem_list[3] = 1
        problem = tuple(problem_list)

        #STRIDEH = problem[9]
        #STRIDEW = problem[10]
        #if STRIDEH != 1 or STRIDEW != 1:
        #  print("Skipping problem", j, "with non-unit stride")
        #  continue

        print("Running timeloop for problem index", j)

        dirname = 'simba_chip/problem_' + str(j) + '/'
        subprocess.check_call(['mkdir', '-p', dirname])

        timeloop.run_timeloop(dirname, configfile = 'simba_chip.cfg', workload_bounds = problem)
        
        mapping = parse_timeloop_mapping(dirname)
        if mapping == {}:
            print("ERROR: timeloop did not produce a mapping for problem index", j)
            sys.exit()
            # # Ugh. Timeloop probably didn't run long enough to generate a mapping.
            # # Skip and continue.
            # print("WARNING: timeloop did not produce a mapping, skipping this problem")
            # continue

        stats = parse_timeloop_output.parse_timeloop_stats(dirname)

        print("Mapping from timeloop")
        pprint.pprint(mapping)

        print("Statistics from timeloop")
        pprint.pprint(stats)
        
        STRIDEH = problem[9]
        STRIDEW = problem[10]
        #print("STRIDEH", STRIDEH)
        #print("STRIDEW", STRIDEW)        
        P3 = mapping['P3']
        Q3 = mapping['Q3']
        C3 = mapping['C3']
        K3 = mapping['K3']
        P2 = mapping['P2']
        Q2 = mapping['Q2']
        C2t = mapping['C2t']
        C2 = mapping['C2']
        K2t = mapping['K2t']
        K2 = mapping['K2']
        P1 = mapping['P1']
        Q1 = mapping['Q1']
        C1 = mapping['C1']
        K1 = mapping['K1']
        R  = mapping['R']
        S  = mapping['S']
        C0 = mapping['C0']
        K0 = mapping['K0']
        N3 = mapping['N3']
        P = P3*P2*P1
        Q = Q3*Q2*Q1
        K = K3*K2*K1*K0*K2t
        C = C3*C2*C1*C0*C2t
        H1 = (P1 - 1)*STRIDEH + R 
        W1 = (Q1 - 1)*STRIDEW + S

        # Access count formulas for SIMBA
        # Input buffer
        #   C0 = spatial dimension, convert vector accesses to scalar (reported by Timeloop)
        #   C2t = new temporal dimension added to prevent sliding windows
        gold_input_buffer_reads  = R*S*C1*K1*P1*Q1 * P2*Q2*C2*K2 * P3*Q3*C3*K3 * C0 * C2t * K2t
        gold_input_buffer_writes = H1*W1*C1        * P2*Q2*C2*K2 * P3*Q3*C3*K3 * C0 * C2t * K2t
 
        # Weight buffer
        #   C0 = spatial dimension, convert vector accesses to scalar (reported by Timeloop)
        #   K0 = spatial dimension (actually only need K[3] from Timeloop)
        #   C2t = new temporal dimension added to prevent sliding windows
        #         unfortunately, this destroys weight reuse as well, increasing the
        #         weight fills and DRAM reads for weights.
        gold_weight_buffer_reads  = R*S*C1*K1 * P2*Q2*C2*K2 * P3*Q3*C3*K3 * C0 * K0 * C2t * K2t
        gold_weight_buffer_writes = R*S*C1*K1 * P2*Q2*C2*K2 * P3*Q3*C3*K3 * C0 * K0 * C2t * K2t

        # Accumulation buffer
        # Reads  = local accumulation reads + final output reads + cross PE reduction reads
        #gold_accum_buffer_reads = ((R*S*C1 - 1)*K1*P1*Q1 * P2*Q2*C2*K2 * P3*Q3*C3*K3 + P*Q*int(K/K0) + (C2*C3 - 1)*P*Q*int(K/K0))*K0 * C2t
        #gold_accum_buffer_reads = (P1*Q1*C1*K1*R*S*C2t*K2t*P2*Q2*P3*Q3*C3*K3 - P1*Q1*K1) * (K0*K2*C2)
        # The formula above is what we used with Timeloop's inaccurate method to calculate reads-WITU,
        # where tile size was used as partition size.
        # The formula below is what we use with Timeloop's new partition size calculation. The
        # term subtracted is the partition size. Note the multiplication by (K0*K2*C2) at the end:
        # This is to accumulate across all spatial instances.
        gold_accum_buffer_reads = (P1*Q1*C1*K1*R*S*C2t*K2t*P2*Q2*P3*Q3*C3*K3 - P1*P2*P3 * Q1*Q2*Q3 * K1*K2t*K3) * (K0*K2*C2)

        # Writes = local accumulation writes + cross PE reduction writes
        # gold_accum_buffer_writes = K0*(R*S*C1*K1*P1*Q1 * P2*Q2*C2*K2 * P3*Q3*C3*K3 + (C2*C3 - 1)*P*Q*int(K/K0)) * C2t
        gold_accum_buffer_writes = (P1*Q1*C1*K1*R*S*C2t*K2t*P2*Q2*P3*Q3*C3*K3) * (K0*K2*C2)
        # Writes += fills from global buffer
        #gold_accum_buffer_writes += P*Q*K*C2*C3
        gold_accum_buffer_writes += (P1*Q1*K1*K2t*P2*Q2*P3*Q3*C3*K3 - P1*P2*P3 * Q1*Q2*Q3 * K1*K2t*K3) * (K0*K2*C2)
        
        # Get the above counts from timeloop
        print()
        print("Starting assertions")
        print()
 
        # Input buffer
        timeloop_input_buffer = stats['energy_breakdown_pJ']['InputBuffer']
        timeloop_input_buffer_reads = timeloop_input_buffer['reads_per_instance'][1]*timeloop_input_buffer['instances'][1]
        timeloop_input_buffer_writes = (timeloop_input_buffer['fills_per_instance'][1] + timeloop_input_buffer['updates_per_instance'][1])*timeloop_input_buffer['instances'][1]
        
        print("gold_input_buffer_reads\t\t", gold_input_buffer_reads)
        print("timeloop_input_buffer_reads\t", timeloop_input_buffer_reads)
        print()
        assert(timeloop_input_buffer_reads == gold_input_buffer_reads)

        print("gold_input_buffer_writes\t", gold_input_buffer_writes)
        print("timeloop_input_buffer_writes\t", timeloop_input_buffer_writes)
        print()
        assert(timeloop_input_buffer_writes == gold_input_buffer_writes)
        
        # Weight buffer
        timeloop_weight_buffer = stats['energy_breakdown_pJ']['WeightBuffer']
        timeloop_weight_buffer_reads = timeloop_weight_buffer['reads_per_instance'][0]*timeloop_weight_buffer['instances'][0]
        timeloop_weight_buffer_writes = (timeloop_weight_buffer['fills_per_instance'][0] + timeloop_weight_buffer['updates_per_instance'][0])*timeloop_weight_buffer['instances'][0]

        print("gold_weight_buffer_reads\t", gold_weight_buffer_reads)
        print("timeloop_weight_buffer_reads\t", timeloop_weight_buffer_reads)
        print()
        assert(timeloop_weight_buffer_reads == gold_weight_buffer_reads)

        print("gold_weight_buffer_writes\t", gold_weight_buffer_writes)
        print("timeloop_weight_buffer_writes\t", timeloop_weight_buffer_writes)
        print()
        assert(timeloop_weight_buffer_writes == gold_weight_buffer_writes)

        # Accumulation buffer 
        timeloop_accum_buffer = stats['energy_breakdown_pJ']['AccumulationBuffer']
        timeloop_accum_buffer_reads = timeloop_accum_buffer['reads_per_instance'][2]*timeloop_accum_buffer['instances'][2]
        timeloop_accum_buffer_writes = (timeloop_accum_buffer['fills_per_instance'][2] + timeloop_accum_buffer['updates_per_instance'][2])*timeloop_accum_buffer['instances'][2]

        print("gold_accum_buffer_reads\t\t", gold_accum_buffer_reads)
        print("timeloop_accum_buffer_reads\t", timeloop_accum_buffer_reads)
        print()
        assert(timeloop_accum_buffer_reads == gold_accum_buffer_reads)

        print("gold_accum_buffer_writes\t", gold_accum_buffer_writes)
        print("timeloop_accum_buffer_writes\t", timeloop_accum_buffer_writes)
        print()
        assert(timeloop_accum_buffer_writes == gold_accum_buffer_writes)

        # Registers

        # Global buffer

        # DRAM      
         
loop_name = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'N'}
spacetime_name = {0: 'temporal', 1: 'spatialx', 2: 'spatialy'}

def parse_timeloop_mapping(dirname):    
    tree = ET.parse(dirname + xml_file_name)
    root = tree.getroot()

    topology = root.findall('engine')[0].findall('topology_')[0]
    
    # Get the list of storage elements
    storage_levels = topology.findall('levels_')[0]
    num_storage_levels = int(storage_levels.findall('count')[0].text)    
    level_ptrs = storage_levels.findall('item')  

    out = []
    for level_ptr in level_ptrs:

        level = level_ptr.findall('px')[0]        
        
        # The XML structure is interesting. Every Level gets a <px>, but
        # only the first object of each type gets a full class_id descriptor.
        # For example, the first model::BufferLevel item will get:
        #    <px class_id="9" class_name="model::BufferLevel" tracking_level="1" version="0" object_id="_1">
        # but subsequent levels will get something like: 
	#     <px class_id_reference="9" object_id="_2">
        # with increasing object_ids. We can keep a table of new class_ids as
        # we encounter them, but for now we'll just hack something that works.
        
        # Is this the Arithmetic level (the only one)?
        if 'class_id' in level.attrib and level.attrib['class_name'] == "model::ArithmeticUnits":
            continue
            
        l = {'temporal': {},'spatialx': {},'spatialy': {}}

        for j in loop_name:
            for k in spacetime_name:
                l[spacetime_name[k]][loop_name[j]] = {'start': 0, 'end': 1, 'stride': 1}

        # Get the list of loops for each storage element
        for loop in level.findall('subnest_')[0].findall('item'):
            dimension = int(loop.findall('dimension')[0].text)
            start = int(loop.findall('start')[0].text)
            end = int(loop.findall('end')[0].text)
            stride = int(loop.findall('stride')[0].text)
            # Spacetime dimension is either temporal, spatial-x or spatial-y. It is an enum is in that order.
            txy = int(loop.findall('spacetime_dimension')[0].text)
            l[spacetime_name[txy]][loop_name[dimension]] = {'start': start, 'end': end, 'stride': stride}
        out = out + [l]
        
    if len(out) == 0:
      return {}

    # For SIMBA, 
    # out[0]: register
    #     1 : accumulation buffer
    #     2 : weight buffer
    #     3 : input buffer
    #     4 : global buffer
    #     5 : dram
    # c in accumulation buffer corresponds to spatial tiling
    # k in input buffer corresponds to spatial tiling
    
    mapping = {\
    'P3': out[5]['temporal']['P']['end'],\
    'Q3': out[5]['temporal']['Q']['end'],\
    'C3': out[5]['temporal']['C']['end'],\
    'K3': out[5]['temporal']['K']['end'],\
    'P2': out[4]['temporal']['P']['end'],\
    'Q2': out[4]['temporal']['Q']['end'],\
    'C2t': out[4]['temporal']['C']['end'],\
    'C2': out[4]['spatialx']['C']['end']*out[4]['spatialy']['C']['end'],\
    'K2t': out[4]['temporal']['K']['end'],\
    'K2': out[4]['spatialx']['K']['end']*out[4]['spatialy']['K']['end'],\
    'P1': timeloop.prod([out[j]['temporal']['P']['end'] for j in range(4)]),\
    'Q1': timeloop.prod([out[j]['temporal']['Q']['end'] for j in range(4)]),\
    'C1': timeloop.prod([out[j]['temporal']['C']['end'] for j in range(4)]),\
    'K1': timeloop.prod([out[j]['temporal']['K']['end'] for j in range(4)]),\
    'R' : timeloop.prod([out[j]['temporal']['R']['end'] for j in range(4)]),\
    'S' : timeloop.prod([out[j]['temporal']['S']['end'] for j in range(4)]),\
    'C0': timeloop.prod([out[j]['spatialx']['C']['end'] for j in range(4)]) * timeloop.prod([out[j]['spatialy']['C']['end'] for j in range(4)]),\
    'K0': timeloop.prod([out[j]['spatialx']['K']['end'] for j in range(4)]) * timeloop.prod([out[j]['spatialy']['K']['end'] for j in range(4)]),\
    'N3': out[5]['temporal']['N']['end']\
    }
    return mapping


top(range(0,150))
print("ALL TESTS PASSED.")
