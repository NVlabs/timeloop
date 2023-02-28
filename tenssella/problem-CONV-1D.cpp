/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "problem-CONV-1D.hpp"


void construct_problem(isl_ctx* context,
        map<string, DataPtr> & data_map,
        map<string, ProblemPtr> & einsum_map, int P, int R) {

    string cs_name = "Add0";
    string input = "Inputs";
    string output = "Outputs";
    shared_ptr<ProblemShape_CONV_1D> problem_test =
        make_shared<ProblemShape_CONV_1D>(context, cs_name, input, output, true, P, R);
    einsum_map[cs_name] = problem_test;

    // -- Data spaces.
    DataPtr I = make_shared<DataSpace>();
    DataPtr O = make_shared<DataSpace>();

    I->setInput();
    O->setOutput();
    I->name = input;
    O->name = output;

    I->num_ranks = 1;
    O->num_ranks = 1;

    // We need the following only for generating human-readable emulation code for transfer blocks.
    // w = "p+r" ??
    I->subscripts = { "w" };
    O->subscripts = { "p" };

    // -- Tensor accesses.
    auto I_read_projection = isl_map_read_from_str(context, str("{ [p,r] -> " + input + "[p+r] }"));
    auto O_read_projection = isl_map_read_from_str(context, str("{ [p,r] -> " + output + "[p] }"));

    //I.write_projection = nullptr;
    auto O_write_projection = isl_map_read_from_str(context, str("{ [p,r] -> " + output + "[p] }"));

    // ---- Limit the tensor accesses to the same domains as iteration space.
    //      e.g., A_read := A_read * iteration_space;
    isl_set* iteration_space_ = problem_test->IterationSpace();
    TRACE(1) << cs_name << std::endl;
    I->read_projection[cs_name] =
        isl_map_intersect_domain(I_read_projection, isl_set_copy(iteration_space_));
    O->read_projection[cs_name] =
        isl_map_intersect_domain(O_read_projection, isl_set_copy(iteration_space_));
    O->write_projection[cs_name] =
        isl_map_intersect_domain(O_write_projection, isl_set_copy(iteration_space_));
    data_map.insert({input, I});
    data_map.insert({output, O});

    // -- Compute dependencies.
    //    all_reads := A_read + B_read + Z_read;
    //    all_writes := Z_write;
    //    closure := all_writes.(all_reads^-1);
    //
    //    less_than := { [i] -> [j] : i < j };
    //    schedule_reference := { MAC[m,n,k] -> [m,n,k] };
    //    program_order := schedule_reference.less_than.(schedule_reference^-1);
    //
    //    dependencies := closure * program_order;
}

void construct_problem_fusion(isl_ctx* context,
        map<string, DataPtr> & data_map,
        map<string, ProblemPtr> & einsum_map, int P, int R) {

    string input = "Inputs";
    string tmp = "Tmp";
    string output = "Outputs";

    string cs_name_0 = "Add0";
    shared_ptr<ProblemShape_CONV_1D> problem_test_0 =
        make_shared<ProblemShape_CONV_1D>(context, cs_name_0, input, tmp, true, P+R-1, R);
    einsum_map[cs_name_0] = problem_test_0;

    string cs_name_1 = "Add1";
    shared_ptr<ProblemShape_CONV_1D> problem_test_1 =
        make_shared<ProblemShape_CONV_1D>(context, cs_name_1, tmp, output, true, P, R);
    einsum_map[cs_name_1] = problem_test_1;

    // -- Data spaces.
    DataPtr I = make_shared<DataSpace>();
    DataPtr Tmp = make_shared<DataSpace>();
    DataPtr O = make_shared<DataSpace>();

    I->setInput();
    O->setOutput();
    I->name = input;
    Tmp->name = tmp;
    O->name = output;

    I->num_ranks = 1;
    Tmp->num_ranks = 1;
    O->num_ranks = 1;

    // We need the following only for generating human-readable emulation code for transfer blocks.
    // w = "p+r" ??
    I->subscripts = { "w" };
    O->subscripts = { "p" };

    // -- Tensor accesses.
    auto I_read_projection = isl_map_read_from_str(context, str("{ [p,r] -> " + input + "[p+r] }"));
    auto TMP_read_projection = isl_map_read_from_str(context, str("{ [p,r] -> " + tmp+ "[p] }"));

    //I.write_projection = nullptr;
    auto TMP_write_projection = isl_map_read_from_str(context, str("{ [p,r] -> " + tmp+ "[p] }"));


    auto TMP_read_projection_1 = isl_map_read_from_str(context, str("{ [p,r] -> " + tmp + "[p+r] }"));
    auto O_read_projection = isl_map_read_from_str(context, str("{ [p,r] -> " + output + "[p] }"));

    //I.write_projection = nullptr;
    auto O_write_projection = isl_map_read_from_str(context, str("{ [p,r] -> " + output + "[p] }"));

    // ---- Limit the tensor accesses to the same domains as iteration space.
    //      e.g., A_read := A_read * iteration_space;
    isl_set* iteration_space_0 = problem_test_0->IterationSpace();
    isl_set* iteration_space_1 = problem_test_1->IterationSpace();
    //TRACE(1) << cs_name << std::endl;
    I->read_projection[cs_name_0] =
        isl_map_intersect_domain(cpy(I_read_projection),
                isl_set_copy(iteration_space_0));
    data_map.insert({input, I});

    Tmp->read_projection[cs_name_0] =
        isl_map_intersect_domain(cpy(TMP_read_projection),
                isl_set_copy(iteration_space_0));
    Tmp->read_projection[cs_name_1] =
        isl_map_intersect_domain(cpy(TMP_read_projection_1),
                isl_set_copy(iteration_space_1));
    Tmp->write_projection[cs_name_0] =
        isl_map_intersect_domain(cpy(TMP_write_projection),
                isl_set_copy(iteration_space_0));
    data_map.insert({tmp, Tmp});

    O->read_projection[cs_name_1] =
        isl_map_intersect_domain(cpy(O_read_projection),
                isl_set_copy(iteration_space_1));
    O->write_projection[cs_name_1] =
        isl_map_intersect_domain(cpy(O_write_projection),
                isl_set_copy(iteration_space_1));
    data_map.insert({output, O});



    // -- Compute dependencies.
    //    all_reads := A_read + B_read + Z_read;
    //    all_writes := Z_write;
    //    closure := all_writes.(all_reads^-1);
    //
    //    less_than := { [i] -> [j] : i < j };
    //    schedule_reference := { MAC[m,n,k] -> [m,n,k] };
    //    program_order := schedule_reference.less_than.(schedule_reference^-1);
    //
    //    dependencies := closure * program_order;
}
