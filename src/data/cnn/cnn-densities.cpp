/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

// ===============================================
//                   Densities
// ===============================================

#include <fstream>

#include "workload/workload.hpp"

#include "data/cnn/cnn-layers.hpp"
#include "data/cnn/cnn-densities.hpp"

namespace problem
{

std::map<std::string, std::map<problem::Shape::DataSpaceID, double>> densities = {
{"TEST",
  {{kDataSpaceWeight, 1},
   {kDataSpaceInput, 1}, 
   {kDataSpaceOutput, 1}}},
{"ALEX_conv1",
  {{kDataSpaceWeight, 0.710166},
   {kDataSpaceInput, 1}, 
   {kDataSpaceOutput, 0.491494}}},
{"ALEX_conv2_1",
  {{kDataSpaceWeight, 0.372552},
   {kDataSpaceInput, 0.906464}, 
   {kDataSpaceOutput, 0.222351}}},
{"ALEX_conv2_2",
  {{kDataSpaceWeight, 0.385306},
   {kDataSpaceInput, 0.965392}, 
   {kDataSpaceOutput, 0.222351}}},
{"ALEX_conv3",
  {{kDataSpaceWeight, 0.346133},
   {kDataSpaceInput, 0.527598}, 
   {kDataSpaceOutput, 0.389361}}},
{"ALEX_conv4",
  {{kDataSpaceWeight, 0.372449},
   {kDataSpaceInput, 0.389361}, 
   {kDataSpaceOutput, 0.429179}}},
{"ALEX_conv5",
  {{kDataSpaceWeight, 0.368811},
   {kDataSpaceInput, 0.429179}, 
   {kDataSpaceOutput, 0.160041}}},
{"VGG_conv1_1",
  {{kDataSpaceWeight, 0.565394},
   {kDataSpaceInput, 1}, 
   {kDataSpaceOutput, 0.506188}}},
{"VGG_conv1_2",
  {{kDataSpaceWeight, 0.217204},
   {kDataSpaceInput, 0.506188}, 
   {kDataSpaceOutput, 0.774166}}},
{"VGG_conv2_1",
  {{kDataSpaceWeight, 0.338243},
   {kDataSpaceInput, 0.896687}, 
   {kDataSpaceOutput, 0.786959}}},
{"VGG_conv2_2",
  {{kDataSpaceWeight, 0.354702},
   {kDataSpaceInput, 0.786959}, 
   {kDataSpaceOutput, 0.596098}}},
{"VGG_conv3_1",
  {{kDataSpaceWeight, 0.52947},
   {kDataSpaceInput, 0.813013}, 
   {kDataSpaceOutput, 0.638544}}},
{"VGG_conv3_2",
  {{kDataSpaceWeight, 0.235348},
   {kDataSpaceInput, 0.638544}, 
   {kDataSpaceOutput, 0.688107}}},
{"VGG_conv3_3",
  {{kDataSpaceWeight, 0.417079},
   {kDataSpaceInput, 0.688107}, 
   {kDataSpaceOutput, 0.492793}}},
{"VGG_conv4_1",
  {{kDataSpaceWeight, 0.320871},
   {kDataSpaceInput, 0.706623}, 
   {kDataSpaceOutput, 0.50143}}},
{"VGG_conv4_2",
  {{kDataSpaceWeight, 0.265933},
   {kDataSpaceInput, 0.50143}, 
   {kDataSpaceOutput, 0.451458}}},
{"VGG_conv4_3",
  {{kDataSpaceWeight, 0.342644},
   {kDataSpaceInput, 0.451458}, 
   {kDataSpaceOutput, 0.218371}}},
{"VGG_conv5_1",
  {{kDataSpaceWeight, 0.352268},
   {kDataSpaceInput, 0.392797}, 
   {kDataSpaceOutput, 0.330477}}},
{"VGG_conv5_2",
  {{kDataSpaceWeight, 0.28619},
   {kDataSpaceInput, 0.330477}, 
   {kDataSpaceOutput, 0.305674}}},
{"VGG_conv5_3",
  {{kDataSpaceWeight, 0.362868},
   {kDataSpaceInput, 0.305674}, 
   {kDataSpaceOutput, 0.141881}}},
{"inception_3a-1x1",
  {{kDataSpaceWeight, 0.378988},
   {kDataSpaceInput, 0.708978}, 
   {kDataSpaceOutput, 0.664441}}},
{"inception_3a-3x3",
  {{kDataSpaceWeight, 0.451443},
   {kDataSpaceInput, 0.663983}, 
   {kDataSpaceOutput, 0.489796}}},
{"inception_3a-3x3_reduce",
  {{kDataSpaceWeight, 0.41352},
   {kDataSpaceInput, 0.708978}, 
   {kDataSpaceOutput, 0.663983}}},
{"inception_3a-5x5",
  {{kDataSpaceWeight, 0.340547},
   {kDataSpaceInput, 0.774713}, 
   {kDataSpaceOutput, 0.486767}}},
{"inception_3a-5x5_reduce",
  {{kDataSpaceWeight, 0.355794},
   {kDataSpaceInput, 0.708978}, 
   {kDataSpaceOutput, 0.774713}}},
{"inception_3a-pool_proj",
  {{kDataSpaceWeight, 0.460775},
   {kDataSpaceInput, 0.956194}, 
   {kDataSpaceOutput, 0.464286}}},
{"inception_3b-1x1",
  {{kDataSpaceWeight, 0.407562},
   {kDataSpaceInput, 0.52989}, 
   {kDataSpaceOutput, 0.328882}}},
{"inception_3b-3x3",
  {{kDataSpaceWeight, 0.48815},
   {kDataSpaceInput, 0.515814}, 
   {kDataSpaceOutput, 0.202932}}},
{"inception_3b-3x3_reduce",
  {{kDataSpaceWeight, 0.46225},
   {kDataSpaceInput, 0.52989}, 
   {kDataSpaceOutput, 0.515814}}},
{"inception_3b-5x5",
  {{kDataSpaceWeight, 0.46056},
   {kDataSpaceInput, 0.651985}, 
   {kDataSpaceOutput, 0.163571}}},
{"inception_3b-5x5_reduce",
  {{kDataSpaceWeight, 0.382446},
   {kDataSpaceInput, 0.52989}, 
   {kDataSpaceOutput, 0.651985}}},
{"inception_3b-pool_proj",
  {{kDataSpaceWeight, 0.460327},
   {kDataSpaceInput, 0.865588}, 
   {kDataSpaceOutput, 0.146225}}},
{"inception_4a-1x1",
  {{kDataSpaceWeight, 0.450499},
   {kDataSpaceInput, 0.554401}, 
   {kDataSpaceOutput, 0.305511}}},
{"inception_4a-3x3",
  {{kDataSpaceWeight, 0.480341},
   {kDataSpaceInput, 0.618622}, 
   {kDataSpaceOutput, 0.168907}}},
{"inception_4a-3x3_reduce",
  {{kDataSpaceWeight, 0.448047},
   {kDataSpaceInput, 0.554401}, 
   {kDataSpaceOutput, 0.618622}}},
{"inception_4a-5x5",
  {{kDataSpaceWeight, 0.457188},
   {kDataSpaceInput, 0.760842}, 
   {kDataSpaceOutput, 0.235013}}},
{"inception_4a-5x5_reduce",
  {{kDataSpaceWeight, 0.417708},
   {kDataSpaceInput, 0.554401}, 
   {kDataSpaceOutput, 0.760842}}},
{"inception_4a-pool_proj",
  {{kDataSpaceWeight, 0.479036},
   {kDataSpaceInput, 0.868665}, 
   {kDataSpaceOutput, 0.133131}}},
{"inception_4b-1x1",
  {{kDataSpaceWeight, 0.346252},
   {kDataSpaceInput, 0.221859}, 
   {kDataSpaceOutput, 0.691773}}},
{"inception_4b-3x3",
  {{kDataSpaceWeight, 0.413699},
   {kDataSpaceInput, 0.720891}, 
   {kDataSpaceOutput, 0.455198}}},
{"inception_4b-3x3_reduce",
  {{kDataSpaceWeight, 0.344308},
   {kDataSpaceInput, 0.221859}, 
   {kDataSpaceOutput, 0.720891}}},
{"inception_4b-5x5",
  {{kDataSpaceWeight, 0.421927},
   {kDataSpaceInput, 0.785927}, 
   {kDataSpaceOutput, 0.365115}}},
{"inception_4b-5x5_reduce",
  {{kDataSpaceWeight, 0.350342},
   {kDataSpaceInput, 0.221859}, 
   {kDataSpaceOutput, 0.785927}}},
{"inception_4b-pool_proj",
  {{kDataSpaceWeight, 0.398407},
   {kDataSpaceInput, 0.555026}, 
   {kDataSpaceOutput, 0.36081}}},
{"inception_4c-1x1",
  {{kDataSpaceWeight, 0.37326},
   {kDataSpaceInput, 0.506069}, 
   {kDataSpaceOutput, 0.63062}}},
{"inception_4c-3x3",
  {{kDataSpaceWeight, 0.419098},
   {kDataSpaceInput, 0.568878}, 
   {kDataSpaceOutput, 0.396026}}},
{"inception_4c-3x3_reduce",
  {{kDataSpaceWeight, 0.391769},
   {kDataSpaceInput, 0.506069}, 
   {kDataSpaceOutput, 0.568878}}},
{"inception_4c-5x5",
  {{kDataSpaceWeight, 0.399766},
   {kDataSpaceInput, 0.65051}, 
   {kDataSpaceOutput, 0.336336}}},
{"inception_4c-5x5_reduce",
  {{kDataSpaceWeight, 0.364258},
   {kDataSpaceInput, 0.506069}, 
   {kDataSpaceOutput, 0.65051}}},
{"inception_4c-pool_proj",
  {{kDataSpaceWeight, 0.408997},
   {kDataSpaceInput, 0.821847}, 
   {kDataSpaceOutput, 0.288186}}},
{"inception_4d-1x1",
  {{kDataSpaceWeight, 0.381801},
   {kDataSpaceInput, 0.433733}, 
   {kDataSpaceOutput, 0.364933}}},
{"inception_4d-3x3",
  {{kDataSpaceWeight, 0.430296},
   {kDataSpaceInput, 0.443452}, 
   {kDataSpaceOutput, 0.228759}}},
{"inception_4d-3x3_reduce",
  {{kDataSpaceWeight, 0.412652},
   {kDataSpaceInput, 0.433733}, 
   {kDataSpaceOutput, 0.443452}}},
{"inception_4d-5x5",
  {{kDataSpaceWeight, 0.425176},
   {kDataSpaceInput, 0.638552}, 
   {kDataSpaceOutput, 0.298868}}},
{"inception_4d-5x5_reduce",
  {{kDataSpaceWeight, 0.385498},
   {kDataSpaceInput, 0.433733}, 
   {kDataSpaceOutput, 0.638552}}},
{"inception_4d-pool_proj",
  {{kDataSpaceWeight, 0.399719},
   {kDataSpaceInput, 0.74992}, 
   {kDataSpaceOutput, 0.136081}}},
{"inception_4e-1x1",
  {{kDataSpaceWeight, 0.342537},
   {kDataSpaceInput, 0.254909}, 
   {kDataSpaceOutput, 0.287966}}},
{"inception_4e-3x3",
  {{kDataSpaceWeight, 0.371354},
   {kDataSpaceInput, 0.642953}, 
   {kDataSpaceOutput, 0.261304}}},
{"inception_4e-3x3_reduce",
  {{kDataSpaceWeight, 0.395135},
   {kDataSpaceInput, 0.254909}, 
   {kDataSpaceOutput, 0.642953}}},
{"inception_4e-5x5",
  {{kDataSpaceWeight, 0.422871},
   {kDataSpaceInput, 0.805644}, 
   {kDataSpaceOutput, 0.15896}}},
{"inception_4e-5x5_reduce",
  {{kDataSpaceWeight, 0.36991},
   {kDataSpaceInput, 0.254909}, 
   {kDataSpaceOutput, 0.805644}}},
{"inception_4e-pool_proj",
  {{kDataSpaceWeight, 0.387636},
   {kDataSpaceInput, 0.544846}, 
   {kDataSpaceOutput, 0.173788}}},
{"inception_5a-1x1",
  {{kDataSpaceWeight, 0.312002},
   {kDataSpaceInput, 0.499951}, 
   {kDataSpaceOutput, 0.384407}}},
{"inception_5a-3x3",
  {{kDataSpaceWeight, 0.361775},
   {kDataSpaceInput, 0.565051}, 
   {kDataSpaceOutput, 0.30963}}},
{"inception_5a-3x3_reduce",
  {{kDataSpaceWeight, 0.327141},
   {kDataSpaceInput, 0.499951}, 
   {kDataSpaceOutput, 0.565051}}},
{"inception_5a-5x5",
  {{kDataSpaceWeight, 0.357646},
   {kDataSpaceInput, 0.694515}, 
   {kDataSpaceOutput, 0.269611}}},
{"inception_5a-5x5_reduce",
  {{kDataSpaceWeight, 0.313815},
   {kDataSpaceInput, 0.499951}, 
   {kDataSpaceOutput, 0.694515}}},
{"inception_5a-pool_proj",
  {{kDataSpaceWeight, 0.334266},
   {kDataSpaceInput, 0.799303}, 
   {kDataSpaceOutput, 0.199777}}},
{"inception_5b-1x1",
  {{kDataSpaceWeight, 0.319455},
   {kDataSpaceInput, 0.309581}, 
   {kDataSpaceOutput, 0.215986}}},
{"inception_5b-3x3",
  {{kDataSpaceWeight, 0.366761},
   {kDataSpaceInput, 0.301339}, 
   {kDataSpaceOutput, 0.205198}}},
{"inception_5b-3x3_reduce",
  {{kDataSpaceWeight, 0.329233},
   {kDataSpaceInput, 0.309581}, 
   {kDataSpaceOutput, 0.301339}}},
{"inception_5b-5x5",
  {{kDataSpaceWeight, 0.346413},
   {kDataSpaceInput, 0.459609}, 
   {kDataSpaceOutput, 0.155772}}},
{"inception_5b-5x5_reduce",
  {{kDataSpaceWeight, 0.329928},
   {kDataSpaceInput, 0.309581}, 
   {kDataSpaceOutput, 0.459609}}},
{"inception_5b-pool_proj",
  {{kDataSpaceWeight, 0.314744},
   {kDataSpaceInput, 0.569761}, 
   {kDataSpaceOutput, 0.1478}}},
};

// Function to get the layer density from a layer name.
Workload::Densities GetLayerDensities(std::string layer_name)
{
  std::map<Shape::DataSpaceID, double> avg_dens;
  Workload::Densities dens;
  try
  {
    avg_dens = densities.at(layer_name);
    for (unsigned d = 0; d < GetShape()->NumDataSpaces; d++){
      YAML::Node ynode;
      ynode["distribution"] = "fixed";
      ynode["density"] = avg_dens;
      auto density_specs = problem::DensityDistributionFactory::ParseSpecs(config::CompoundConfigNode(nullptr, ynode,
                                                                                                      new config::CompoundConfig("dummy.yaml")));
      dens[d]= problem::DensityDistributionFactory::Construct(density_specs);
    }

  }
  catch (const std::out_of_range& oor)
  {
    std::cerr << "Out of Range error: " << oor.what() << std::endl;
    std::cerr << "Layer " << layer_name << " not found in dictionary." << std::endl;
    exit(1);
  }

  return dens;
}

// Read CSV files
void ReadDensities(std::string filename)
{
  std::ifstream file(filename);
  std::string buf;

  while (getline(file, buf, ','))
  {
    std::string layer = buf;
    
    getline(file, buf, ',');
    densities.at(layer).at(kDataSpaceWeight) = atof(buf.data());

    getline(file, buf, ',');
    densities.at(layer).at(kDataSpaceInput) = atof(buf.data());

    getline(file, buf);
    densities.at(layer).at(kDataSpaceOutput) = atof(buf.data());
  }

  file.close();
}

// Dump densities.
void DumpDensities(std::string filename)
{
  std::ofstream file(filename);

  for (auto & layer : densities)
  {
    file << layer.first << ", ";
    file << layer.second.at(kDataSpaceWeight) << ", ";
    file << layer.second.at(kDataSpaceInput) << ", ";
    file << layer.second.at(kDataSpaceOutput) << std::endl;
  }

  file.close();
}

// Dump densities.
void DumpDensities_CPP(std::string filename)
{
  std::ofstream file(filename);

  // file << "#include \"cnn-layers.hpp\"" << std::endl;
  // file << std::endl;

  // file << "const unsigned kDataSpaceWeight = " << kDataSpaceWeight << ";" << std::endl;
  // file << "const unsigned kDataSpaceInput = " << kDataSpaceInput << ";" << std::endl;
  // file << "const unsigned kDataSpaceOutput = " << kDataSpaceOutput << ";" << std::endl;
  // file << std::endl;
  
  file << "std::map<std::string, Workload::Densities> densities = {" << std::endl;
  
  for (auto & layer : densities)
  {
    file << "{\"" << layer.first << "\"," << std::endl;
    file << "  {{" << "kDataSpaceWeight, " << layer.second.at(kDataSpaceWeight) << "}," << std::endl;
    file << "   {" << "kDataSpaceInput, " << layer.second.at(kDataSpaceInput) << "}, " << std::endl;
    file << "   {" << "kDataSpaceOutput, " << layer.second.at(kDataSpaceOutput) << "}}}," << std::endl;
  }

  file << "};" << std::endl;
  
  file.close();
}

} // namespace problem
