/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "evaluator.hpp"
#include "util/banner.hpp"

bool gTerminate = false;
bool gTerminateEval = false;

void handler(int s)
{
  if (!gTerminate)
  {
    std::cerr << strsignal(s) << " caught. Terminating after "
              << "completing ongoing evaluation." << std::endl;
    gTerminate = true;
  }
  else if (!gTerminateEval)
  {
    std::cerr << "Second " << strsignal(s) << " caught. Abandoning "
              << "ongoing evaluation and terminating immediately."
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
  assert(argc == 2 || argc == 3);

  struct sigaction action;
  action.sa_handler = handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  sigaction(SIGINT, &action, NULL);
  
  char* config_file = argv[1];

  libconfig::Config config;
  config.readFile(config_file);
  
  // Should we override the layer to be evaluated?
  if (argc == 3)
  {
    libconfig::Setting& root = config.getRoot();
    if (!root.exists("problem"))
      root.add("problem", libconfig::Setting::TypeGroup);
    libconfig::Setting& problem = root["problem"];
    if (!problem.exists("layer"))
      problem.add("layer", libconfig::Setting::TypeString);
    problem["layer"] = argv[2];
  }

  for (auto& line: banner)
  {
    std::cout << line << std::endl;
  }
  std::cout << std::endl;
  
  Application application(config);
  
  application.Run();

  return 0;
}
