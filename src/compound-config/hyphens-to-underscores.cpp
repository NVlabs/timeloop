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

#include "compound-config/hyphens-to-underscores.hpp"
#include <sstream>

namespace hyphens2underscores
{

std::string hyphens2underscores(std::string input) {
    std::regex regex;
    std::string target_underscore;
    std::string target_hyphen;
    for (std::size_t i = 0; i < std::size(HYPHEN_TO_UNDERSCORE_TARGETS); i++) {
        target_underscore = HYPHEN_TO_UNDERSCORE_TARGETS[i];
        target_hyphen = std::regex_replace(target_underscore, std::regex("_"), "-");
        std::string regex_str = "\\b" + target_hyphen + "\\b(?!-)(?!>)(?!\\.hpp)(?!\\.cpp)";
        regex = std::regex(regex_str);
        if (std::regex_search(input, regex)) {
            std::cout 
                << "Warning: Hyphenated keyword " << target_hyphen << " found in inputs. "
                << "Hyphenated keywords are deprecated and to be replaced with "
                << "underscores. Please use the the scripts/hyphens2underscores.py in the "
                << "in the Timeloop repository to update your inputs.\n";
        }
        input = std::regex_replace(input, regex, target_underscore);
    }
    return input;
}

std::string hyphens2underscores_from_file(const char* inputFile)
{
    std::ifstream file(inputFile);
    if (!file.is_open()) return "";

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string input = buffer.str();

    return hyphens2underscores(input);
}

}