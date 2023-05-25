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

#include "utils.hpp"
#include <limits>

std::vector<std::string> parse_aff(const std::string& aff) {
  //cout << "pt: " << pt << endl;
  std::regex cm("\\{ \\[(.*)\\] -> \\[(.*)\\] \\}");
  std::smatch match;
  auto res = regex_search(aff, match, cm);

  if (res) {

    std::string coefs = match[2];
    std::vector<std::string> coords;

    auto terms = split_at(coefs, ", ");
    for (auto term: terms) {
      coords.push_back(term);
    }

    return coords;
  } else {
    std::cout << "DEBUG the ISL aff parser" << std::endl;
    std::exit(1);
  }
}


std::map<std::string, int> ParseFactorLine(std::string line)
{
  std::map<std::string, int> factors;

  std::regex re("([A-Za-z]+)[[:space:]]*[=]*[[:space:]]*([0-9]+)(,([0-9]+))?", std::regex::extended);
  std::smatch sm;

  while (std::regex_search(line, sm, re))
  {
    std::string dimension_name = sm[1];

    int end = std::stoi(sm[2]);
    assert(end > 0);

    factors[dimension_name] = end;

    line = sm.suffix().str();
  }

  return factors;
}

