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

#include "mapping-CONV-1D.hpp"
#include "utils.hpp"

std::map<std::string, MappingPtr> ConstructMapping2Level(isl_ctx* c, int P, int R) {
  shared_ptr<Mapping_CONV_1D_2Level> m =
      make_shared<Mapping_CONV_1D_2Level>(c, P, R, 1, 0);
  return {{"Add0", m}};
}

std::map<std::string, MappingPtr> ConstructMappingMulti2Level(isl_ctx* c, int P, int R) {
  auto m = make_shared<Mapping_CONV_1D_2Level>(c, P+R+1, R, 2, 0);
  auto n = make_shared<Mapping_CONV_1D_2Level>(c, P, R, 2, 1);
  return {{"Add0", m}, {"Add1", n}};
}

map<string, MappingPtr> ConstructMapping(isl_ctx* context,
        vector<int> p_tile_factors, vector<int> r_tile_factors) {

    map<string, MappingPtr> app_mapping;

    assert(p_tile_factors.size() == 3);
    assert(r_tile_factors.size() == 3);
    shared_ptr<Mapping_CONV_1D> mapping_ispace_0 =
        make_shared<Mapping_CONV_1D>(context,  p_tile_factors, r_tile_factors);
    app_mapping.insert({"Add0", mapping_ispace_0});

    return app_mapping;
}

map<string, MappingPtr> ConstructMappingMulti(isl_ctx* context,
        vector<int> p0_tile_factors, vector<int> r0_tile_factors,
        vector<int> p1_tile_factors, vector<int> r1_tile_factors) {

    map<string, MappingPtr> app_mapping;

    assert(p0_tile_factors.size() == 3);
    assert(r0_tile_factors.size() == 3);
    assert(p1_tile_factors.size() == 3);
    assert(r1_tile_factors.size() == 3);
    shared_ptr<Mapping_CONV_1D_3Level> mapping_ispace_0 =
        make_shared<Mapping_CONV_1D_3Level>(context,  p0_tile_factors, r0_tile_factors, 2, 0);
    app_mapping.insert({"Add0", mapping_ispace_0});

    shared_ptr<Mapping_CONV_1D_3Level> mapping_ispace_1 =
        make_shared<Mapping_CONV_1D_3Level>(context,  p1_tile_factors, r1_tile_factors, 2, 1);
    app_mapping.insert({"Add1", mapping_ispace_1});

    return app_mapping;
}

shared_ptr<Binding> CreateBinding() {
  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //

  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["Inputs"][3]["Add0"]= { "DRAM", 0};
  b->memory_binding_["Outputs"][3]["Add0"] = { "DRAM", 1};

  b->memory_binding_["Inputs"][2]["Add0"]= { "IRegfile", 0 };
  b->memory_binding_["Outputs"][2]["Add0"] = { "ORegfile", 0 };

  b->memory_binding_["Inputs"][1]["Add0"]= { "OperandA", 0 };
  b->memory_binding_["Outputs"][1]["Add0"] = { "Result", 0 };

  b->compute_binding_["Add0"] = { "Adder", 0 };
  return b;
}

shared_ptr<Binding> CreateBindingSeq() {
  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //
  //std::map<std::string, std::map<int, Partition>> binding_;
  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["Inputs"][2]["Add0"] = { "Memory", 0 };
  b->memory_binding_["Outputs"][2]["Add0"] = { "Memory", 1 };

  b->memory_binding_["Inputs"][1]["Add0"] = { "OperandA", 0 };
  b->memory_binding_["Outputs"][1]["Add0"] = { "Result", 0 };

  b->compute_binding_["Add0"] = { "ALU", 0 };
  return b;
}

shared_ptr<Binding> Create2StageBinding() {
  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //

  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["Inputs"][3]["Add0"]= { "DRAM", 0};
  b->memory_binding_["Outputs"][3]["Add1"] = { "DRAM", 1};

  b->memory_binding_["Inputs"][2]["Add0"]= { "IRegfile", 0 };

  b->memory_binding_["Tmp"][2]["Add0"]= { "TmpRegfile", 0 };
  b->memory_binding_["Tmp"][2]["Add1"]= { "TmpRegfile", 0 };
  b->memory_binding_["Outputs"][2]["Add1"] = { "ORegfile", 0 };

  b->memory_binding_["Inputs"][1]["Add0"]= { "OperandA", 0 };
  b->memory_binding_["Tmp"][1]["Add1"] = { "OperandA", 0};
  //FIXME: a data space may have two binding,
  //       need add another mapping
  b->memory_binding_["Tmp"][1]["Add0"] = { "Result", 0 };
  b->memory_binding_["Outputs"][1]["Add1"] = { "Result", 0 };

  b->compute_binding_["Add0"] = { "Adder", 0 };
  b->compute_binding_["Add1"] = { "Adder", 0 };
  return b;
}

shared_ptr<Binding> Create2Stage3LevelBindingSeq() {
  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //

  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["Inputs"][3]["Add0"]= { "DRAM", 0};
  b->memory_binding_["Outputs"][3]["Add1"] = { "DRAM", 0};

  b->memory_binding_["Inputs"][2]["Add0"]= { "Regfile", 0 };

  b->memory_binding_["Tmp"][2]["Add0"]= { "Regfile", 0 };
  b->memory_binding_["Tmp"][2]["Add1"]= { "Regfile", 0 };
  b->memory_binding_["Outputs"][2]["Add1"] = { "Regfile", 0 };

  b->memory_binding_["Inputs"][1]["Add0"]= { "OperandA", 0 };
  b->memory_binding_["Tmp"][1]["Add1"] = { "OperandA", 0};
  //FIXME: a data space may have two binding,
  //       need add another mapping
  b->memory_binding_["Tmp"][1]["Add0"] = { "Result", 0 };
  b->memory_binding_["Outputs"][1]["Add1"] = { "Result", 0 };

  b->compute_binding_["Add0"] = { "Adder", 0 };
  b->compute_binding_["Add1"] = { "Adder", 0 };
  return b;
}

shared_ptr<Binding> Create2Stage2LevelBindingSeq() {
  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //
  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["Inputs"][2]["Add0"] = { "DRAM", 0 };
  b->memory_binding_["Tmp"][2]["Add0"] = { "DRAM", 0};
  b->memory_binding_["Tmp"][2]["Add1"] = { "DRAM", 0 };
  b->memory_binding_["Outputs"][2]["Add1"] = { "DRAM", 0 };


  b->memory_binding_["Inputs"][1]["Add0"] = { "OperandA", 0 };
  b->memory_binding_["Tmp"][1]["Add0"] = { "Result", 0 };
  b->memory_binding_["Tmp"][1]["Add1"] = { "OperandA", 0 };
  b->memory_binding_["Outputs"][1]["Add1"] = { "Result", 0 };

  b->compute_binding_["Add0"] = { "Adder", 0 };
  b->compute_binding_["Add1"] = { "Adder", 0 };
  return b;
}

shared_ptr<Binding> Create2StageBindingSeqDiffID() {
  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //
  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["Inputs"][2]["Add0"] = { "DRAM", 0 };
  b->memory_binding_["Tmp"][2]["Add0"] = { "DRAM", 1};
  b->memory_binding_["Tmp"][2]["Add1"] = { "DRAM", 1 };
  b->memory_binding_["Outputs"][2]["Add1"] = { "DRAM", 2 };


  b->memory_binding_["Inputs"][1]["Add0"] = { "OperandA", 0 };
  b->memory_binding_["Tmp"][1]["Add0"] = { "Result", 0 };
  b->memory_binding_["Tmp"][1]["Add1"] = { "OperandA", 0 };
  b->memory_binding_["Outputs"][1]["Add1"] = { "Result", 0 };

  b->compute_binding_["Add0"] = { "Adder", 0 };
  b->compute_binding_["Add1"] = { "Adder", 0 };
  return b;
}
