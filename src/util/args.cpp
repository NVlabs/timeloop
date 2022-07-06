/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <sys/stat.h>

#include "util/args.hpp"

bool ParseArgs(int argc, char* argv[],
               std::vector<std::string>& input_files,
               std::string& output_dir)
{
  // Very rudimentary argument parsing. The only recognized pattern is "-o <odir>"
  // and a set of .yaml or .cfg files.
  std::vector<std::string> input_args(argv + 1, argv + argc);
  for (auto arg = input_args.begin(); arg != input_args.end(); arg++)
  {
    if (arg->compare("-o") == 0)
    {
      arg++;
      output_dir = *arg;
      struct stat info;
      if (stat(output_dir.c_str(), &info) != 0)
      {
        std::cerr << "ERROR: cannot access output directory: " << output_dir << std::endl;
        return false;
      }
      else if (!(info.st_mode & S_IFDIR))
      {
        std::cerr << "ERROR: non-existent output directory: " << output_dir << std::endl;
        return false;
      }
    }
    else if (arg->compare("timeloop-mapper.map.yaml") == 0)
    {
      std::cerr << "WARNING: found timeloop-mapper.map.yaml in input file list, ignoring."
                << std::endl;
    }
    else
    {
      input_files.push_back(*arg);
    }
  }

  for (auto& file: input_files)
  {
    std::cout << "input file: " << file << std::endl;
  }

  return true;
}
