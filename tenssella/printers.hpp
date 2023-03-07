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
#include <fstream>
#include <cassert>

#include <isl/space.h>
#include <isl/constraint.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/ast_build.h>
#include <isl/aff.h>
#include <isl/polynomial.h>

// -------------------------
//    Console and Tracing
// -------------------------

void InitPrinters(isl_ctx* context);
void UninitPrinters();

std::ostream& TRACE(int level);

std::ostream& operator << (std::ostream& out, isl_space* space);
std::ostream& operator << (std::ostream& out, isl_constraint* constraint);
std::ostream& operator << (std::ostream& out, isl_point* point);
std::ostream& operator << (std::ostream& out, isl_basic_set* basic_set);
std::ostream& operator << (std::ostream& out, isl_set* set);
std::ostream& operator << (std::ostream& out, isl_map* map);
std::ostream& operator << (std::ostream& out, isl_union_map* union_map);
std::ostream& operator << (std::ostream& out, isl_ast_node* tree);
std::ostream& operator << (std::ostream& out, isl_aff* aff);
std::ostream& operator << (std::ostream& out, isl_pw_aff* pwaff);
std::ostream& operator << (std::ostream& out, isl_qpolynomial* x);
std::ostream& operator << (std::ostream& out, isl_pw_qpolynomial* x);

// ----------------------
//     Print to file
// ----------------------

class Printer
{
 private:
  isl_ctx* ctx_ = nullptr;
  isl_printer* p_ = nullptr;
  FILE* cstream_ = nullptr;
  bool i_opened_the_file_ = false;
  bool emit_ast_ = false;

 public:
  Printer() = delete;

  Printer(isl_ctx* ctx, std::string filename, bool emit_ast = false) :
      ctx_(ctx), emit_ast_(emit_ast)
  {
    cstream_ = fopen(filename.c_str(), "w");
    i_opened_the_file_ = true;
    Init();
  }

  Printer(isl_ctx* ctx, FILE* cstream) :
      ctx_(ctx), cstream_(cstream)
  {
    Init();
  }

  ~Printer()
  {
    if (p_)
      isl_printer_free(p_);
    p_ = nullptr;
    if (i_opened_the_file_)
      fclose(cstream_);
  }

  void Init()
  {
    p_ = isl_printer_to_file(ctx_, cstream_);  
    p_ = isl_printer_set_indent(p_, 2);
  }

  Printer& operator << (isl_set* set)
  {
    assert(!emit_ast_);
    fprintf(cstream_, "// ");
    int format = isl_printer_get_output_format(p_);
    p_ = isl_printer_set_output_format(p_, ISL_FORMAT_ISL);
    p_ = isl_printer_print_set(p_, set);
    p_ = isl_printer_set_output_format(p_, format);
    fprintf(cstream_, "\n");
    return *this;
  }

  Printer& operator << (isl_map* map)
  {
    assert(!emit_ast_);
    fprintf(cstream_, "// ");
    int format = isl_printer_get_output_format(p_);
    p_ = isl_printer_set_output_format(p_, ISL_FORMAT_ISL);
    p_ = isl_printer_print_map(p_, map);
    p_ = isl_printer_set_output_format(p_, format);
    fprintf(cstream_, "\n");
    return *this;
  }

  Printer& operator << (isl_ast_node* tree)
  {
    int format = isl_printer_get_output_format(p_);
    p_ = isl_printer_set_output_format(p_, emit_ast_ ? ISL_FORMAT_ISL : ISL_FORMAT_C);
    p_ = isl_printer_print_ast_node(p_, tree);
    p_ = isl_printer_set_output_format(p_, format);
    return *this;
  }

  Printer& operator << (std::string str)
  {
    fprintf(cstream_, "%s", str.c_str());
    return *this;
  }
};
