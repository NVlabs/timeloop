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

#include "printers.hpp"

isl_printer* console;
int TRACE_LEVEL = 0;
std::ofstream NULL_STREAM;

void InitPrinters(isl_ctx* context)
{
  // Setup tracing.
  NULL_STREAM.setstate(std::ios_base::badbit);
  char* trace_level = getenv("TENSSELLA_TRACE_LEVEL");
  if (trace_level != NULL)
  {
    TRACE_LEVEL = atoi(trace_level);
  }
  console = isl_printer_to_file(context, stdout);
  console = isl_printer_set_output_format(console, ISL_FORMAT_ISL);  
}

void UninitPrinters()
{
  isl_printer_free(console);
}

std::ostream& operator << (std::ostream& out, isl_space* space)
{
  if (!out.bad()) console = isl_printer_print_space(console, space);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_constraint* constraint)
{
  if (!out.bad()) console = isl_printer_print_constraint(console, constraint);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_point* point)
{
  if (!out.bad()) console = isl_printer_print_point(console, point);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_basic_set* basic_set)
{
  if (!out.bad()) console = isl_printer_print_basic_set(console, basic_set);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_set* set)
{
  if (!out.bad()) console = isl_printer_print_set(console, set);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_map* map)
{
  if (!out.bad()) console = isl_printer_print_map(console, map);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_union_map* union_map)
{
  if (!out.bad()) console = isl_printer_print_union_map(console, union_map);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_ast_node* tree)
{
  if (!out.bad()) console = isl_printer_set_output_format(console, ISL_FORMAT_C);
  if (!out.bad()) console = isl_printer_print_ast_node(console, tree);
  if (!out.bad()) console = isl_printer_set_output_format(console, ISL_FORMAT_ISL);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_aff* aff)
{
  if (!out.bad()) console = isl_printer_print_aff(console, aff);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_pw_aff* pwaff)
{
  if (!out.bad()) console = isl_printer_print_pw_aff(console, pwaff);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_qpolynomial* x)
{
  if (!out.bad()) console = isl_printer_print_qpolynomial(console, x);
  return out;
}

std::ostream& operator << (std::ostream& out, isl_pw_qpolynomial* x)
{
  if (!out.bad()) console = isl_printer_print_pw_qpolynomial(console, x);
  return out;
}

std::ostream& TRACE(int level)
{
  if (level <= TRACE_LEVEL)
    return std::cout;
  else
    return NULL_STREAM;
}

