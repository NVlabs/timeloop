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

#include <cstdlib>
#include <fstream>

#include "cnn-layers.hpp"

namespace problem
{

// FIXME: Add stride parameter U
// Alexnet layers from Eyeriss ISCA Paper Table II
// Batch size = 1 in these definitions. We will
// use the appropriate batch size based on config file.

std::map<std::string, Bounds> layers = {

  {"TEST", {{problem::Dimension::R, 3},
            {problem::Dimension::S, 3},
            {problem::Dimension::P, 40}, // 56
            {problem::Dimension::Q, 40}, // 56
            {problem::Dimension::C, 64},// 256
            {problem::Dimension::K, 1}, // 36
            {problem::Dimension::N, 1}}},
  
  // ========
  // Alex Net
  // ========
  
  {"ALEX_conv1", {{problem::Dimension::R, 3},
                  {problem::Dimension::S, 3},
                  {problem::Dimension::P, 57},
                  {problem::Dimension::Q, 57},
                  {problem::Dimension::C, 48},
                  {problem::Dimension::K, 96},
                  {problem::Dimension::N, 1}}},

  {"ALEX_conv2_1", {{problem::Dimension::R, 5},
                    {problem::Dimension::S, 5},
                    {problem::Dimension::P, 27},
                    {problem::Dimension::Q, 27},
                    {problem::Dimension::C, 48},
                    {problem::Dimension::K, 128},
                    {problem::Dimension::N, 1}}},

  {"ALEX_conv2_2", {{problem::Dimension::R, 5},
                    {problem::Dimension::S, 5},
                    {problem::Dimension::P, 27},
                    {problem::Dimension::Q, 27},
                    {problem::Dimension::C, 48},
                    {problem::Dimension::K, 128},
                    {problem::Dimension::N, 1}}},

  {"ALEX_conv3", {{problem::Dimension::R, 3},
                  {problem::Dimension::S, 3},
                  {problem::Dimension::P, 13},
                  {problem::Dimension::Q, 13},
                  {problem::Dimension::C, 256},
                  {problem::Dimension::K, 384},
                  {problem::Dimension::N, 1}}},

  {"ALEX_conv4", {{problem::Dimension::R, 3},
                  {problem::Dimension::S, 3},
                  {problem::Dimension::P, 13},
                  {problem::Dimension::Q, 13},
                  {problem::Dimension::C, 192},
                  {problem::Dimension::K, 384},
                  {problem::Dimension::N, 1}}},

  {"ALEX_conv5", {{problem::Dimension::R, 3},
                  {problem::Dimension::S, 3},
                  {problem::Dimension::P, 13},
                  {problem::Dimension::Q, 13},
                  {problem::Dimension::C, 192},
                  {problem::Dimension::K, 256},
                  {problem::Dimension::N, 1}}},

  // ========
  //  VGG 16
  // ========
  
  {"VGG_conv1_1", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 224},
                   {problem::Dimension::Q, 224},
                   {problem::Dimension::C, 3},
                   {problem::Dimension::K, 64},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv1_2", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 224},
                   {problem::Dimension::Q, 224},
                   {problem::Dimension::C, 64},
                   {problem::Dimension::K, 64},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv2_1", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 112},
                   {problem::Dimension::Q, 112},
                   {problem::Dimension::C, 64},
                   {problem::Dimension::K, 128},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv2_2", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 112},
                   {problem::Dimension::Q, 112},
                   {problem::Dimension::C, 128},
                   {problem::Dimension::K, 128},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv3_1", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 56},
                   {problem::Dimension::Q, 56},
                   {problem::Dimension::C, 128},
                   {problem::Dimension::K, 256},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv3_2", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 56},
                   {problem::Dimension::Q, 56},
                   {problem::Dimension::C, 256},
                   {problem::Dimension::K, 256},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv3_3", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 56},
                   {problem::Dimension::Q, 56},
                   {problem::Dimension::C, 256},
                   {problem::Dimension::K, 256},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv4_1", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 28},
                   {problem::Dimension::Q, 28},
                   {problem::Dimension::C, 256},
                   {problem::Dimension::K, 512},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv4_2", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 28},
                   {problem::Dimension::Q, 28},
                   {problem::Dimension::C, 512},
                   {problem::Dimension::K, 512},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv4_3", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 28},
                   {problem::Dimension::Q, 28},
                   {problem::Dimension::C, 512},
                   {problem::Dimension::K, 512},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv5_1", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 14},
                   {problem::Dimension::Q, 14},
                   {problem::Dimension::C, 512},
                   {problem::Dimension::K, 512},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv5_2", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 14},
                   {problem::Dimension::Q, 14},
                   {problem::Dimension::C, 512},
                   {problem::Dimension::K, 512},
                   {problem::Dimension::N, 1}}},

  {"VGG_conv5_3", {{problem::Dimension::R, 3},
                   {problem::Dimension::S, 3},
                   {problem::Dimension::P, 14},
                   {problem::Dimension::Q, 14},
                   {problem::Dimension::C, 512},
                   {problem::Dimension::K, 512},
                   {problem::Dimension::N, 1}}},

  // =========
  // GoogLeNet
  // =========

  // Inception 3a
  
  {"inception_3a-pool_proj", {{problem::Dimension::R, 1},
                              {problem::Dimension::S, 1},
                              {problem::Dimension::P, 28},
                              {problem::Dimension::Q, 28},
                              {problem::Dimension::C, 192},
                              {problem::Dimension::K, 32},
                              {problem::Dimension::N, 1}}},

  {"inception_3a-1x1", {{problem::Dimension::R, 1},
                        {problem::Dimension::S, 1},
                        {problem::Dimension::P, 28},
                        {problem::Dimension::Q, 28},
                        {problem::Dimension::C, 192},
                        {problem::Dimension::K, 64},
                        {problem::Dimension::N, 1}}},

  {"inception_3a-3x3_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 28},
                               {problem::Dimension::Q, 28},
                               {problem::Dimension::C, 192},
                               {problem::Dimension::K, 96},
                               {problem::Dimension::N, 1}}},

  {"inception_3a-3x3", {{problem::Dimension::R, 3},
                        {problem::Dimension::S, 3},
                        {problem::Dimension::P, 28},
                        {problem::Dimension::Q, 28},
                        {problem::Dimension::C, 96},
                        {problem::Dimension::K, 128},
                        {problem::Dimension::N, 1}}},

  {"inception_3a-5x5_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 28},
                               {problem::Dimension::Q, 28},
                               {problem::Dimension::C, 192},
                               {problem::Dimension::K, 16},
                               {problem::Dimension::N, 1}}},

  {"inception_3a-5x5", {{problem::Dimension::R, 5},
                        {problem::Dimension::S, 5},
                        {problem::Dimension::P, 28},
                        {problem::Dimension::Q, 28},
                        {problem::Dimension::C, 16},
                        {problem::Dimension::K, 32},
                        {problem::Dimension::N, 1}}},

  // Inception 3b
  
  {"inception_3b-pool_proj", {{problem::Dimension::R, 1},
                              {problem::Dimension::S, 1},
                              {problem::Dimension::P, 28},
                              {problem::Dimension::Q, 28},
                              {problem::Dimension::C, 256},
                              {problem::Dimension::K, 64},
                              {problem::Dimension::N, 1}}},

  {"inception_3b-1x1", {{problem::Dimension::R, 1},
                        {problem::Dimension::S, 1},
                        {problem::Dimension::P, 28},
                        {problem::Dimension::Q, 28},
                        {problem::Dimension::C, 256},
                        {problem::Dimension::K, 128},
                        {problem::Dimension::N, 1}}},

  {"inception_3b-3x3_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 28},
                               {problem::Dimension::Q, 28},
                               {problem::Dimension::C, 256},
                               {problem::Dimension::K, 128},
                               {problem::Dimension::N, 1}}},

  {"inception_3b-3x3", {{problem::Dimension::R, 3},
                        {problem::Dimension::S, 3},
                        {problem::Dimension::P, 28},
                        {problem::Dimension::Q, 28},
                        {problem::Dimension::C, 128},
                        {problem::Dimension::K, 192},
                        {problem::Dimension::N, 1}}},

  {"inception_3b-5x5_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 28},
                               {problem::Dimension::Q, 28},
                               {problem::Dimension::C, 256},
                               {problem::Dimension::K, 32},
                               {problem::Dimension::N, 1}}},

  {"inception_3b-5x5", {{problem::Dimension::R, 5},
                        {problem::Dimension::S, 5},
                        {problem::Dimension::P, 28},
                        {problem::Dimension::Q, 28},
                        {problem::Dimension::C, 32},
                        {problem::Dimension::K, 96},
                        {problem::Dimension::N, 1}}},

  // Inception 4a
  
  {"inception_4a-pool_proj", {{problem::Dimension::R, 1},
                              {problem::Dimension::S, 1},
                              {problem::Dimension::P, 14},
                              {problem::Dimension::Q, 14},
                              {problem::Dimension::C, 480},
                              {problem::Dimension::K, 64},
                              {problem::Dimension::N, 1}}},

  {"inception_4a-1x1", {{problem::Dimension::R, 1},
                        {problem::Dimension::S, 1},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 480},
                        {problem::Dimension::K, 192},
                        {problem::Dimension::N, 1}}},

  {"inception_4a-3x3_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 14},
                               {problem::Dimension::Q, 14},
                               {problem::Dimension::C, 480},
                               {problem::Dimension::K, 96},
                               {problem::Dimension::N, 1}}},

  {"inception_4a-3x3", {{problem::Dimension::R, 3},
                        {problem::Dimension::S, 3},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 96},
                        {problem::Dimension::K, 208},
                        {problem::Dimension::N, 1}}},

  {"inception_4a-5x5_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 14},
                               {problem::Dimension::Q, 14},
                               {problem::Dimension::C, 480},
                               {problem::Dimension::K, 16},
                               {problem::Dimension::N, 1}}},

  {"inception_4a-5x5", {{problem::Dimension::R, 5},
                        {problem::Dimension::S, 5},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 16},
                        {problem::Dimension::K, 48},
                        {problem::Dimension::N, 1}}},

  // Inception 4b
  
  {"inception_4b-pool_proj", {{problem::Dimension::R, 1},
                              {problem::Dimension::S, 1},
                              {problem::Dimension::P, 14},
                              {problem::Dimension::Q, 14},
                              {problem::Dimension::C, 512},
                              {problem::Dimension::K, 64},
                              {problem::Dimension::N, 1}}},

  {"inception_4b-1x1", {{problem::Dimension::R, 1},
                        {problem::Dimension::S, 1},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 512},
                        {problem::Dimension::K, 160},
                        {problem::Dimension::N, 1}}},

  {"inception_4b-3x3_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 14},
                               {problem::Dimension::Q, 14},
                               {problem::Dimension::C, 512},
                               {problem::Dimension::K, 112},
                               {problem::Dimension::N, 1}}},

  {"inception_4b-3x3", {{problem::Dimension::R, 3},
                        {problem::Dimension::S, 3},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 112},
                        {problem::Dimension::K, 224},
                        {problem::Dimension::N, 1}}},

  {"inception_4b-5x5_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 14},
                               {problem::Dimension::Q, 14},
                               {problem::Dimension::C, 512},
                               {problem::Dimension::K, 24},
                               {problem::Dimension::N, 1}}},

  {"inception_4b-5x5", {{problem::Dimension::R, 5},
                        {problem::Dimension::S, 5},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 24},
                        {problem::Dimension::K, 64},
                        {problem::Dimension::N, 1}}},

  // Inception 4c
  
  {"inception_4c-pool_proj", {{problem::Dimension::R, 1},
                              {problem::Dimension::S, 1},
                              {problem::Dimension::P, 14},
                              {problem::Dimension::Q, 14},
                              {problem::Dimension::C, 512},
                              {problem::Dimension::K, 64},
                              {problem::Dimension::N, 1}}},

  {"inception_4c-1x1", {{problem::Dimension::R, 1},
                        {problem::Dimension::S, 1},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 512},
                        {problem::Dimension::K, 128},
                        {problem::Dimension::N, 1}}},

  {"inception_4c-3x3_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 14},
                               {problem::Dimension::Q, 14},
                               {problem::Dimension::C, 512},
                               {problem::Dimension::K, 128},
                               {problem::Dimension::N, 1}}},

  {"inception_4c-3x3", {{problem::Dimension::R, 3},
                        {problem::Dimension::S, 3},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 128},
                        {problem::Dimension::K, 256},
                        {problem::Dimension::N, 1}}},

  {"inception_4c-5x5_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 14},
                               {problem::Dimension::Q, 14},
                               {problem::Dimension::C, 512},
                               {problem::Dimension::K, 24},
                               {problem::Dimension::N, 1}}},

  {"inception_4c-5x5", {{problem::Dimension::R, 5},
                        {problem::Dimension::S, 5},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 24},
                        {problem::Dimension::K, 64},
                        {problem::Dimension::N, 1}}},

  // Inception 4d
  
  {"inception_4d-pool_proj", {{problem::Dimension::R, 1},
                              {problem::Dimension::S, 1},
                              {problem::Dimension::P, 14},
                              {problem::Dimension::Q, 14},
                              {problem::Dimension::C, 512},
                              {problem::Dimension::K, 64},
                              {problem::Dimension::N, 1}}},

  {"inception_4d-1x1", {{problem::Dimension::R, 1},
                        {problem::Dimension::S, 1},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 512},
                        {problem::Dimension::K, 112},
                        {problem::Dimension::N, 1}}},

  {"inception_4d-3x3_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 14},
                               {problem::Dimension::Q, 14},
                               {problem::Dimension::C, 512},
                               {problem::Dimension::K, 144},
                               {problem::Dimension::N, 1}}},

  {"inception_4d-3x3", {{problem::Dimension::R, 3},
                        {problem::Dimension::S, 3},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 144},
                        {problem::Dimension::K, 288},
                        {problem::Dimension::N, 1}}},

  {"inception_4d-5x5_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 14},
                               {problem::Dimension::Q, 14},
                               {problem::Dimension::C, 512},
                               {problem::Dimension::K, 32},
                               {problem::Dimension::N, 1}}},

  {"inception_4d-5x5", {{problem::Dimension::R, 5},
                        {problem::Dimension::S, 5},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 32},
                        {problem::Dimension::K, 64},
                        {problem::Dimension::N, 1}}},

  // Inception 4e
  
  {"inception_4e-pool_proj", {{problem::Dimension::R, 1},
                              {problem::Dimension::S, 1},
                              {problem::Dimension::P, 14},
                              {problem::Dimension::Q, 14},
                              {problem::Dimension::C, 528},
                              {problem::Dimension::K, 128},
                              {problem::Dimension::N, 1}}},

  {"inception_4e-1x1", {{problem::Dimension::R, 1},
                        {problem::Dimension::S, 1},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 528},
                        {problem::Dimension::K, 256},
                        {problem::Dimension::N, 1}}},

  {"inception_4e-3x3_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 14},
                               {problem::Dimension::Q, 14},
                               {problem::Dimension::C, 528},
                               {problem::Dimension::K, 160},
                               {problem::Dimension::N, 1}}},

  {"inception_4e-3x3", {{problem::Dimension::R, 3},
                        {problem::Dimension::S, 3},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 160},
                        {problem::Dimension::K, 320},
                        {problem::Dimension::N, 1}}},

  {"inception_4e-5x5_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 14},
                               {problem::Dimension::Q, 14},
                               {problem::Dimension::C, 528},
                               {problem::Dimension::K, 32},
                               {problem::Dimension::N, 1}}},

  {"inception_4e-5x5", {{problem::Dimension::R, 5},
                        {problem::Dimension::S, 5},
                        {problem::Dimension::P, 14},
                        {problem::Dimension::Q, 14},
                        {problem::Dimension::C, 32},
                        {problem::Dimension::K, 128},
                        {problem::Dimension::N, 1}}},

  // Inception 5a
  
  {"inception_5a-pool_proj", {{problem::Dimension::R, 1},
                              {problem::Dimension::S, 1},
                              {problem::Dimension::P, 7},
                              {problem::Dimension::Q, 7},
                              {problem::Dimension::C, 832},
                              {problem::Dimension::K, 128},
                              {problem::Dimension::N, 1}}},

  {"inception_5a-1x1", {{problem::Dimension::R, 1},
                        {problem::Dimension::S, 1},
                        {problem::Dimension::P, 7},
                        {problem::Dimension::Q, 7},
                        {problem::Dimension::C, 832},
                        {problem::Dimension::K, 256},
                        {problem::Dimension::N, 1}}},

  {"inception_5a-3x3_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 7},
                               {problem::Dimension::Q, 7},
                               {problem::Dimension::C, 832},
                               {problem::Dimension::K, 160},
                               {problem::Dimension::N, 1}}},

  {"inception_5a-3x3", {{problem::Dimension::R, 3},
                        {problem::Dimension::S, 3},
                        {problem::Dimension::P, 7},
                        {problem::Dimension::Q, 7},
                        {problem::Dimension::C, 160},
                        {problem::Dimension::K, 320},
                        {problem::Dimension::N, 1}}},

  {"inception_5a-5x5_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 7},
                               {problem::Dimension::Q, 7},
                               {problem::Dimension::C, 832},
                               {problem::Dimension::K, 32},
                               {problem::Dimension::N, 1}}},

  {"inception_5a-5x5", {{problem::Dimension::R, 5},
                        {problem::Dimension::S, 5},
                        {problem::Dimension::P, 7},
                        {problem::Dimension::Q, 7},
                        {problem::Dimension::C, 32},
                        {problem::Dimension::K, 128},
                        {problem::Dimension::N, 1}}},

  // Inception 5b
  
  {"inception_5b-pool_proj", {{problem::Dimension::R, 1},
                              {problem::Dimension::S, 1},
                              {problem::Dimension::P, 7},
                              {problem::Dimension::Q, 7},
                              {problem::Dimension::C, 832},
                              {problem::Dimension::K, 128},
                              {problem::Dimension::N, 1}}},

  {"inception_5b-1x1", {{problem::Dimension::R, 1},
                        {problem::Dimension::S, 1},
                        {problem::Dimension::P, 7},
                        {problem::Dimension::Q, 7},
                        {problem::Dimension::C, 832},
                        {problem::Dimension::K, 384},
                        {problem::Dimension::N, 1}}},

  {"inception_5b-3x3_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 7},
                               {problem::Dimension::Q, 7},
                               {problem::Dimension::C, 832},
                               {problem::Dimension::K, 192},
                               {problem::Dimension::N, 1}}},

  {"inception_5b-3x3", {{problem::Dimension::R, 3},
                        {problem::Dimension::S, 3},
                        {problem::Dimension::P, 7},
                        {problem::Dimension::Q, 7},
                        {problem::Dimension::C, 192},
                        {problem::Dimension::K, 384},
                        {problem::Dimension::N, 1}}},

  {"inception_5b-5x5_reduce", {{problem::Dimension::R, 1},
                               {problem::Dimension::S, 1},
                               {problem::Dimension::P, 7},
                               {problem::Dimension::Q, 7},
                               {problem::Dimension::C, 832},
                               {problem::Dimension::K, 48},
                               {problem::Dimension::N, 1}}},

  {"inception_5b-5x5", {{problem::Dimension::R, 5},
                        {problem::Dimension::S, 5},
                        {problem::Dimension::P, 7},
                        {problem::Dimension::Q, 7},
                        {problem::Dimension::C, 48},
                        {problem::Dimension::K, 128},
                        {problem::Dimension::N, 1}}}
};

std::ostream& operator << (std::ostream& out, Bounds& bounds)
{
  out << problem::Dimension::R << " = " << bounds[problem::Dimension::R] << std::endl;
  out << problem::Dimension::S << " = " << bounds[problem::Dimension::S] << std::endl;
  out << problem::Dimension::P << " = " << bounds[problem::Dimension::P] << std::endl;
  out << problem::Dimension::Q << " = " << bounds[problem::Dimension::Q] << std::endl;
  out << problem::Dimension::C << " = " << bounds[problem::Dimension::C] << std::endl;
  out << problem::Dimension::K << " = " << bounds[problem::Dimension::K] << std::endl;
  out << problem::Dimension::N << " = " << bounds[problem::Dimension::N] << std::endl;

  return out;
}

// ===============================================
//                   Densities
// ===============================================
#include "cnn-densities.hpp" 

const std::map<uint32_t, uint32_t> nearest_composite = {
  {11, 12}, {13, 15}, {27, 28}, {55, 56}, {57, 60}};

// Function to get the layer config from a layer name.
Bounds GetLayerBounds(std::string layer_name, bool pad_primes)
{
  Bounds prob;

  try
  {
    prob = layers.at(layer_name);
  }
  catch (const std::out_of_range& oor)
  {
    std::cerr << "Out of Range error: " << oor.what() << std::endl;
    std::cerr << "Layer " << layer_name << " not found in dictionary." << std::endl;
    exit(1);
  }

  if (pad_primes)
  {
    for (int pd = 0; pd < int(problem::Dimension::Num); pd++)
    {
      if (nearest_composite.count(prob[problem::Dimension(pd)]) != 0)
      {
        prob[problem::Dimension(pd)] =
            nearest_composite.at(prob[problem::Dimension(pd)]);
      }
    }
  }

  return prob;
}

// Function to get the layer density from a layer name.
Densities GetLayerDensities(std::string layer_name)
{
  Densities dens;

  try
  {
    dens = densities.at(layer_name);
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
    densities.at(layer).at(problem::DataType::Weight) = atof(buf.data());

    getline(file, buf, ',');
    densities.at(layer).at(problem::DataType::Input) = atof(buf.data());

    getline(file, buf);
    densities.at(layer).at(problem::DataType::Output) = atof(buf.data());
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
    file << layer.second.at(problem::DataType::Weight) << ", ";
    file << layer.second.at(problem::DataType::Input) << ", ";
    file << layer.second.at(problem::DataType::Output) << std::endl;
  }

  file.close();
}

// Dump densities.
void DumpDensities_CPP(std::string filename)
{
  std::ofstream file(filename);

  file << "#include \"cnn-layers.hpp\"" << std::endl;
  file << "std::map<std::string, Densities> densities = {" << std::endl;
  
  for (auto & layer : densities)
  {
    file << "{\"" << layer.first << "\"," << std::endl;
    file << "  {{" << "problem::DataType::Weight, " << layer.second.at(problem::DataType::Weight) << "}," << std::endl;
    file << "   {" << "problem::DataType::Input, " << layer.second.at(problem::DataType::Input) << "}, " << std::endl;
    file << "   {" << "problem::DataType::Output, " << layer.second.at(problem::DataType::Output) << "}}}," << std::endl;
  }

  file << "};" << std::endl;
  
  file.close();
}

// Libconfig Parsers.
void ParseConfig(libconfig::Setting& config, WorkloadConfig &workload)
{
  Bounds bounds;
  std::string layer_name = "";
  if (config.lookupValue("layer", layer_name))
  {
    bool pad_primes = true;
    config.lookupValue("padPrimes", pad_primes);
    bounds = GetLayerBounds(layer_name, pad_primes);

    // Optional overrides.
    config.lookupValue("R", bounds[problem::Dimension::R]);
    config.lookupValue("S", bounds[problem::Dimension::S]);
    config.lookupValue("P", bounds[problem::Dimension::P]);
    config.lookupValue("Q", bounds[problem::Dimension::Q]);
    config.lookupValue("C", bounds[problem::Dimension::C]);
    config.lookupValue("K", bounds[problem::Dimension::K]);
    config.lookupValue("N", bounds[problem::Dimension::N]);
  }
  else
  {
    assert(config.lookupValue("R", bounds[problem::Dimension::R]));
    assert(config.lookupValue("S", bounds[problem::Dimension::S]));
    assert(config.lookupValue("P", bounds[problem::Dimension::P]));
    assert(config.lookupValue("Q", bounds[problem::Dimension::Q]));
    assert(config.lookupValue("C", bounds[problem::Dimension::C]));
    assert(config.lookupValue("K", bounds[problem::Dimension::K]));
    assert(config.lookupValue("N", bounds[problem::Dimension::N]));
  }
  workload.setBounds(bounds);

  int Wstride = 1, Hstride = 1;
  int Wdilation = 1, Hdilation = 1;
  config.lookupValue("Wstride", Wstride);
  config.lookupValue("Hstride", Hstride);
  config.lookupValue("Wdilation", Wdilation);
  config.lookupValue("Hdilation", Hdilation);
  workload.setWstride(Wstride);
  workload.setHstride(Hstride);
  workload.setWdilation(Wdilation);
  workload.setHdilation(Hdilation);
  
  Densities densities;
  // See if user wants to override default densities.
  double common_density;
  if (config.lookupValue("commonDensity", common_density))
  {
    densities[problem::DataType::Weight] = common_density;
    densities[problem::DataType::Input] = common_density;
    densities[problem::DataType::Output] = common_density;
  }
  else if (config.exists("densities"))
  {
    libconfig::Setting &config_densities = config.lookup("densities");
    assert(config_densities.lookupValue("weights", densities[problem::DataType::Weight]));
    assert(config_densities.lookupValue("inputs", densities[problem::DataType::Input]));
    assert(config_densities.lookupValue("outputs", densities[problem::DataType::Output]));
  }
  else if (layer_name != "")
  {
    densities = GetLayerDensities(layer_name);
  }
  else
  {
    densities[problem::DataType::Weight] = 1.0;
    densities[problem::DataType::Input] = 1.0;
    densities[problem::DataType::Output] = 1.0;
  }
  workload.setDensities(densities);
}


} // namespace problem
