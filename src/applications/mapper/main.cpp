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

#include <iostream>
#include <csignal>
#include <cstring>

#include "applications/mapper/mapper.hpp"
#include "util/banner.hpp"
#include "util/args.hpp"
#include "compound-config/compound-config.hpp"

extern bool gTerminate;
extern bool gTerminateEval;

void handler(int s)
{
  if (!gTerminate)
  {
    std::cerr << strsignal(s) << " caught. Mapper threads will terminate after "
              << "completing any ongoing evaluations." << std::endl;
    gTerminate = true;
  }
  else if (!gTerminateEval)
  {
    std::cerr << "Second " << strsignal(s) << " caught. Mapper threads will "
              << "abandon ongoing evaluations and terminate immediately."
              << std::endl;
    gTerminateEval = true;
  }
  else
  {
    std::cerr << "Third " << strsignal(s) << " caught. Existing disgracefully."
              << std::endl;
    exit(0);
  }
}

//--------------------------------------------//
//                    MAIN                    //
//--------------------------------------------//

int main(int argc, char* argv[])
{
  assert(argc >= 2);

  struct sigaction action;
  action.sa_handler = handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  sigaction(SIGINT, &action, NULL);

  std::vector<std::string> input_files;
  std::string output_dir = ".";
  bool success = ParseArgs(argc, argv, input_files, output_dir);
  if (!success)
  {
    std::cerr << "ERROR: error parsing command line." << std::endl;
    exit(1);
  }

  auto config = new config::CompoundConfig(input_files);
  
  for (auto& line: banner)
  {
    std::cout << line << std::endl;
  }
  std::cout << std::endl;
  
  application::Mapper application(config, output_dir);
  
  const auto result = application.Run();

  // Output file names.
  std::string out_prefix = output_dir + "/" + "timeloop-mapper";

  const auto fname_to_string = std::map<std::string, const std::string&>({
    {"stats.txt", result.stats_string},
    {"map+stats.xml", result.xml_mapping_stats_string},
    {"map.txt", result.mapping_string},
    {"map.yaml", result.mapping_yaml_string},
    {"map.cpp", result.mapping_cpp_string},
    {"map.tensella.txt", result.tensella_string},
    {"orojenesis.csv", result.orojenesis_string}
  });

  for (const auto& [fname_suffix, content_string] : fname_to_string)
  {
    std::ofstream file(out_prefix + "." + fname_suffix);
    file << content_string;
    file.close();
  }

  return 0;
}
