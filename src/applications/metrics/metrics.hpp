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

#pragma once

#include "mapping/parser.hpp"

#include <fstream>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/bitset.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

class Application
{
 protected:
  model::Engine::Specs arch_specs_;
  model::Engine engine_;
  
 public:

  Application(libconfig::Config& config)
  {
    try
    {
      // Architecture configuration.
      libconfig::Setting& arch = config.lookup("arch");
      arch_specs_ = model::Engine::ParseSpecs(arch);
      engine_.Spec(arch_specs_);
      std::cout << "Architecture configuration complete." << std::endl;
    }
    catch (const libconfig::SettingTypeException& e)
    {
      std::cerr << "ERROR: setting type exception at: " << e.getPath() << std::endl;
      exit(1);
    }
    catch (const libconfig::SettingNotFoundException& e)
    {
      std::cerr << "ERROR: setting not found: " << e.getPath() << std::endl;
      exit(1);
    }
    catch (const libconfig::SettingNameException& e)
    {
      std::cerr << "ERROR: setting name exception at: " << e.getPath() << std::endl;
      exit(1);
    }    
  }

  // This class does not support being copied
  Application(const Application&) = delete;
  Application& operator=(const Application&) = delete;

  ~Application()
  {
  }

  // Run the evaluation.
  void Run()
  {
    std::cout << engine_ << std::endl;
  }
};

