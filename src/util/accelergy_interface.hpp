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

#pragma once

#include <iostream>
#include <cstring>
#include <stdexcept>

namespace accelergy
{
  std::string exec(const char* cmd) {
    std::string result = "";
    char buffer[128];
    FILE* pipe = popen("which accelergy", "r");
    if (!pipe) {
      std::cout << "popen(" << cmd << ") failed" << std::endl;
      exit(0);
    }

    try {
      while (fgets(buffer, 128, pipe) != nullptr) {
        result += buffer;
      }
    } catch (...) {
      pclose(pipe);
    }
    pclose(pipe);
    return result;
  }

  void invokeAccelergy(std::vector<std::string> input_files, std::string out_prefix, std::string out_dir) {
#ifdef USE_ACCELERGY
    std::string accelergy_path = exec("which accelergy");
    // if `which` does not find it, we will try env
    if (accelergy_path.find("accelergy") == std::string::npos) {
      accelergy_path = exec("echo $ACCELERGYPATH");
      accelergy_path += "accelergy";
    }
    //std::cout << "Invoke Accelergy at: " << accelergy_path << std::endl;
    std::string cmd = accelergy_path.substr(0, accelergy_path.size() - 1);
    for (auto input_file : input_files) {
      cmd += " " + input_file;
    }
    cmd += " --oprefix " + out_prefix + ".";
    cmd += " -o " + out_dir + "/ > " + out_prefix + ".accelergy.log 2>&1";
    std::cout << "execute:" << cmd << std::endl;
    int ret = system(cmd.c_str());
    if (ret) {
      std::cout << "Failed to run Accelergy. Did you install Accelergy or specify ACCELERGYPATH correctly? Or check accelergy.log to see what went wrong" << std::endl;
      exit(0);
    }
#else
    (void) input_files;
    (void) out_prefix;
    (void) out_dir;
#endif
    return;
  }
} // namespace accelergy
