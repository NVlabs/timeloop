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

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "cnn-layers.hpp"

namespace problem
{

// FIXME: Add stride coefficient U
// Alexnet layers from Eyeriss ISCA Paper Table II
// Batch size = 1 in these definitions. We will
// use the appropriate batch size based on config file.

const unsigned kDimensionR = 0;
const unsigned kDimensionS = 1;
const unsigned kDimensionP = 2;
const unsigned kDimensionQ = 3;
const unsigned kDimensionC = 4;
const unsigned kDimensionK = 5;
const unsigned kDimensionN = 6;

const unsigned kDataSpaceWeight = 0;
const unsigned kDataSpaceInput = 1;
const unsigned kDataSpaceOutput = 2;

std::map<std::string, Workload::FactorizedBounds> layers = {

  {"TEST", {{kDimensionR, 3},
            {kDimensionS, 3},
            {kDimensionP, 40}, // 56
            {kDimensionQ, 40}, // 56
            {kDimensionC, 64},// 256
            {kDimensionK, 1}, // 36
            {kDimensionN, 1}}},
  
  // ========
  // Alex Net
  // ========
  
  {"ALEX_conv1", {{kDimensionR, 3},
                  {kDimensionS, 3},
                  {kDimensionP, 57},
                  {kDimensionQ, 57},
                  {kDimensionC, 48},
                  {kDimensionK, 96},
                  {kDimensionN, 1}}},

  {"ALEX_conv2_1", {{kDimensionR, 5},
                    {kDimensionS, 5},
                    {kDimensionP, 27},
                    {kDimensionQ, 27},
                    {kDimensionC, 48},
                    {kDimensionK, 128},
                    {kDimensionN, 1}}},

  {"ALEX_conv2_2", {{kDimensionR, 5},
                    {kDimensionS, 5},
                    {kDimensionP, 27},
                    {kDimensionQ, 27},
                    {kDimensionC, 48},
                    {kDimensionK, 128},
                    {kDimensionN, 1}}},

  {"ALEX_conv3", {{kDimensionR, 3},
                  {kDimensionS, 3},
                  {kDimensionP, 13},
                  {kDimensionQ, 13},
                  {kDimensionC, 256},
                  {kDimensionK, 384},
                  {kDimensionN, 1}}},

  {"ALEX_conv4", {{kDimensionR, 3},
                  {kDimensionS, 3},
                  {kDimensionP, 13},
                  {kDimensionQ, 13},
                  {kDimensionC, 192},
                  {kDimensionK, 384},
                  {kDimensionN, 1}}},

  {"ALEX_conv5", {{kDimensionR, 3},
                  {kDimensionS, 3},
                  {kDimensionP, 13},
                  {kDimensionQ, 13},
                  {kDimensionC, 192},
                  {kDimensionK, 256},
                  {kDimensionN, 1}}},

  // ========
  //  VGG 16
  // ========
  
  {"VGG_conv1_1", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 224},
                   {kDimensionQ, 224},
                   {kDimensionC, 3},
                   {kDimensionK, 64},
                   {kDimensionN, 1}}},

  {"VGG_conv1_2", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 224},
                   {kDimensionQ, 224},
                   {kDimensionC, 64},
                   {kDimensionK, 64},
                   {kDimensionN, 1}}},

  {"VGG_conv2_1", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 112},
                   {kDimensionQ, 112},
                   {kDimensionC, 64},
                   {kDimensionK, 128},
                   {kDimensionN, 1}}},

  {"VGG_conv2_2", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 112},
                   {kDimensionQ, 112},
                   {kDimensionC, 128},
                   {kDimensionK, 128},
                   {kDimensionN, 1}}},

  {"VGG_conv3_1", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 56},
                   {kDimensionQ, 56},
                   {kDimensionC, 128},
                   {kDimensionK, 256},
                   {kDimensionN, 1}}},

  {"VGG_conv3_2", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 56},
                   {kDimensionQ, 56},
                   {kDimensionC, 256},
                   {kDimensionK, 256},
                   {kDimensionN, 1}}},

  {"VGG_conv3_3", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 56},
                   {kDimensionQ, 56},
                   {kDimensionC, 256},
                   {kDimensionK, 256},
                   {kDimensionN, 1}}},

  {"VGG_conv4_1", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 28},
                   {kDimensionQ, 28},
                   {kDimensionC, 256},
                   {kDimensionK, 512},
                   {kDimensionN, 1}}},

  {"VGG_conv4_2", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 28},
                   {kDimensionQ, 28},
                   {kDimensionC, 512},
                   {kDimensionK, 512},
                   {kDimensionN, 1}}},

  {"VGG_conv4_3", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 28},
                   {kDimensionQ, 28},
                   {kDimensionC, 512},
                   {kDimensionK, 512},
                   {kDimensionN, 1}}},

  {"VGG_conv5_1", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 14},
                   {kDimensionQ, 14},
                   {kDimensionC, 512},
                   {kDimensionK, 512},
                   {kDimensionN, 1}}},

  {"VGG_conv5_2", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 14},
                   {kDimensionQ, 14},
                   {kDimensionC, 512},
                   {kDimensionK, 512},
                   {kDimensionN, 1}}},

  {"VGG_conv5_3", {{kDimensionR, 3},
                   {kDimensionS, 3},
                   {kDimensionP, 14},
                   {kDimensionQ, 14},
                   {kDimensionC, 512},
                   {kDimensionK, 512},
                   {kDimensionN, 1}}},

  // =========
  // GoogLeNet
  // =========

  // Inception 3a
  
  {"inception_3a-pool_proj", {{kDimensionR, 1},
                              {kDimensionS, 1},
                              {kDimensionP, 28},
                              {kDimensionQ, 28},
                              {kDimensionC, 192},
                              {kDimensionK, 32},
                              {kDimensionN, 1}}},

  {"inception_3a-1x1", {{kDimensionR, 1},
                        {kDimensionS, 1},
                        {kDimensionP, 28},
                        {kDimensionQ, 28},
                        {kDimensionC, 192},
                        {kDimensionK, 64},
                        {kDimensionN, 1}}},

  {"inception_3a-3x3_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 28},
                               {kDimensionQ, 28},
                               {kDimensionC, 192},
                               {kDimensionK, 96},
                               {kDimensionN, 1}}},

  {"inception_3a-3x3", {{kDimensionR, 3},
                        {kDimensionS, 3},
                        {kDimensionP, 28},
                        {kDimensionQ, 28},
                        {kDimensionC, 96},
                        {kDimensionK, 128},
                        {kDimensionN, 1}}},

  {"inception_3a-5x5_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 28},
                               {kDimensionQ, 28},
                               {kDimensionC, 192},
                               {kDimensionK, 16},
                               {kDimensionN, 1}}},

  {"inception_3a-5x5", {{kDimensionR, 5},
                        {kDimensionS, 5},
                        {kDimensionP, 28},
                        {kDimensionQ, 28},
                        {kDimensionC, 16},
                        {kDimensionK, 32},
                        {kDimensionN, 1}}},

  // Inception 3b
  
  {"inception_3b-pool_proj", {{kDimensionR, 1},
                              {kDimensionS, 1},
                              {kDimensionP, 28},
                              {kDimensionQ, 28},
                              {kDimensionC, 256},
                              {kDimensionK, 64},
                              {kDimensionN, 1}}},

  {"inception_3b-1x1", {{kDimensionR, 1},
                        {kDimensionS, 1},
                        {kDimensionP, 28},
                        {kDimensionQ, 28},
                        {kDimensionC, 256},
                        {kDimensionK, 128},
                        {kDimensionN, 1}}},

  {"inception_3b-3x3_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 28},
                               {kDimensionQ, 28},
                               {kDimensionC, 256},
                               {kDimensionK, 128},
                               {kDimensionN, 1}}},

  {"inception_3b-3x3", {{kDimensionR, 3},
                        {kDimensionS, 3},
                        {kDimensionP, 28},
                        {kDimensionQ, 28},
                        {kDimensionC, 128},
                        {kDimensionK, 192},
                        {kDimensionN, 1}}},

  {"inception_3b-5x5_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 28},
                               {kDimensionQ, 28},
                               {kDimensionC, 256},
                               {kDimensionK, 32},
                               {kDimensionN, 1}}},

  {"inception_3b-5x5", {{kDimensionR, 5},
                        {kDimensionS, 5},
                        {kDimensionP, 28},
                        {kDimensionQ, 28},
                        {kDimensionC, 32},
                        {kDimensionK, 96},
                        {kDimensionN, 1}}},

  // Inception 4a
  
  {"inception_4a-pool_proj", {{kDimensionR, 1},
                              {kDimensionS, 1},
                              {kDimensionP, 14},
                              {kDimensionQ, 14},
                              {kDimensionC, 480},
                              {kDimensionK, 64},
                              {kDimensionN, 1}}},

  {"inception_4a-1x1", {{kDimensionR, 1},
                        {kDimensionS, 1},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 480},
                        {kDimensionK, 192},
                        {kDimensionN, 1}}},

  {"inception_4a-3x3_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 14},
                               {kDimensionQ, 14},
                               {kDimensionC, 480},
                               {kDimensionK, 96},
                               {kDimensionN, 1}}},

  {"inception_4a-3x3", {{kDimensionR, 3},
                        {kDimensionS, 3},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 96},
                        {kDimensionK, 208},
                        {kDimensionN, 1}}},

  {"inception_4a-5x5_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 14},
                               {kDimensionQ, 14},
                               {kDimensionC, 480},
                               {kDimensionK, 16},
                               {kDimensionN, 1}}},

  {"inception_4a-5x5", {{kDimensionR, 5},
                        {kDimensionS, 5},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 16},
                        {kDimensionK, 48},
                        {kDimensionN, 1}}},

  // Inception 4b
  
  {"inception_4b-pool_proj", {{kDimensionR, 1},
                              {kDimensionS, 1},
                              {kDimensionP, 14},
                              {kDimensionQ, 14},
                              {kDimensionC, 512},
                              {kDimensionK, 64},
                              {kDimensionN, 1}}},

  {"inception_4b-1x1", {{kDimensionR, 1},
                        {kDimensionS, 1},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 512},
                        {kDimensionK, 160},
                        {kDimensionN, 1}}},

  {"inception_4b-3x3_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 14},
                               {kDimensionQ, 14},
                               {kDimensionC, 512},
                               {kDimensionK, 112},
                               {kDimensionN, 1}}},

  {"inception_4b-3x3", {{kDimensionR, 3},
                        {kDimensionS, 3},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 112},
                        {kDimensionK, 224},
                        {kDimensionN, 1}}},

  {"inception_4b-5x5_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 14},
                               {kDimensionQ, 14},
                               {kDimensionC, 512},
                               {kDimensionK, 24},
                               {kDimensionN, 1}}},

  {"inception_4b-5x5", {{kDimensionR, 5},
                        {kDimensionS, 5},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 24},
                        {kDimensionK, 64},
                        {kDimensionN, 1}}},

  // Inception 4c
  
  {"inception_4c-pool_proj", {{kDimensionR, 1},
                              {kDimensionS, 1},
                              {kDimensionP, 14},
                              {kDimensionQ, 14},
                              {kDimensionC, 512},
                              {kDimensionK, 64},
                              {kDimensionN, 1}}},

  {"inception_4c-1x1", {{kDimensionR, 1},
                        {kDimensionS, 1},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 512},
                        {kDimensionK, 128},
                        {kDimensionN, 1}}},

  {"inception_4c-3x3_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 14},
                               {kDimensionQ, 14},
                               {kDimensionC, 512},
                               {kDimensionK, 128},
                               {kDimensionN, 1}}},

  {"inception_4c-3x3", {{kDimensionR, 3},
                        {kDimensionS, 3},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 128},
                        {kDimensionK, 256},
                        {kDimensionN, 1}}},

  {"inception_4c-5x5_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 14},
                               {kDimensionQ, 14},
                               {kDimensionC, 512},
                               {kDimensionK, 24},
                               {kDimensionN, 1}}},

  {"inception_4c-5x5", {{kDimensionR, 5},
                        {kDimensionS, 5},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 24},
                        {kDimensionK, 64},
                        {kDimensionN, 1}}},

  // Inception 4d
  
  {"inception_4d-pool_proj", {{kDimensionR, 1},
                              {kDimensionS, 1},
                              {kDimensionP, 14},
                              {kDimensionQ, 14},
                              {kDimensionC, 512},
                              {kDimensionK, 64},
                              {kDimensionN, 1}}},

  {"inception_4d-1x1", {{kDimensionR, 1},
                        {kDimensionS, 1},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 512},
                        {kDimensionK, 112},
                        {kDimensionN, 1}}},

  {"inception_4d-3x3_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 14},
                               {kDimensionQ, 14},
                               {kDimensionC, 512},
                               {kDimensionK, 144},
                               {kDimensionN, 1}}},

  {"inception_4d-3x3", {{kDimensionR, 3},
                        {kDimensionS, 3},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 144},
                        {kDimensionK, 288},
                        {kDimensionN, 1}}},

  {"inception_4d-5x5_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 14},
                               {kDimensionQ, 14},
                               {kDimensionC, 512},
                               {kDimensionK, 32},
                               {kDimensionN, 1}}},

  {"inception_4d-5x5", {{kDimensionR, 5},
                        {kDimensionS, 5},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 32},
                        {kDimensionK, 64},
                        {kDimensionN, 1}}},

  // Inception 4e
  
  {"inception_4e-pool_proj", {{kDimensionR, 1},
                              {kDimensionS, 1},
                              {kDimensionP, 14},
                              {kDimensionQ, 14},
                              {kDimensionC, 528},
                              {kDimensionK, 128},
                              {kDimensionN, 1}}},

  {"inception_4e-1x1", {{kDimensionR, 1},
                        {kDimensionS, 1},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 528},
                        {kDimensionK, 256},
                        {kDimensionN, 1}}},

  {"inception_4e-3x3_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 14},
                               {kDimensionQ, 14},
                               {kDimensionC, 528},
                               {kDimensionK, 160},
                               {kDimensionN, 1}}},

  {"inception_4e-3x3", {{kDimensionR, 3},
                        {kDimensionS, 3},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 160},
                        {kDimensionK, 320},
                        {kDimensionN, 1}}},

  {"inception_4e-5x5_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 14},
                               {kDimensionQ, 14},
                               {kDimensionC, 528},
                               {kDimensionK, 32},
                               {kDimensionN, 1}}},

  {"inception_4e-5x5", {{kDimensionR, 5},
                        {kDimensionS, 5},
                        {kDimensionP, 14},
                        {kDimensionQ, 14},
                        {kDimensionC, 32},
                        {kDimensionK, 128},
                        {kDimensionN, 1}}},

  // Inception 5a
  
  {"inception_5a-pool_proj", {{kDimensionR, 1},
                              {kDimensionS, 1},
                              {kDimensionP, 7},
                              {kDimensionQ, 7},
                              {kDimensionC, 832},
                              {kDimensionK, 128},
                              {kDimensionN, 1}}},

  {"inception_5a-1x1", {{kDimensionR, 1},
                        {kDimensionS, 1},
                        {kDimensionP, 7},
                        {kDimensionQ, 7},
                        {kDimensionC, 832},
                        {kDimensionK, 256},
                        {kDimensionN, 1}}},

  {"inception_5a-3x3_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 7},
                               {kDimensionQ, 7},
                               {kDimensionC, 832},
                               {kDimensionK, 160},
                               {kDimensionN, 1}}},

  {"inception_5a-3x3", {{kDimensionR, 3},
                        {kDimensionS, 3},
                        {kDimensionP, 7},
                        {kDimensionQ, 7},
                        {kDimensionC, 160},
                        {kDimensionK, 320},
                        {kDimensionN, 1}}},

  {"inception_5a-5x5_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 7},
                               {kDimensionQ, 7},
                               {kDimensionC, 832},
                               {kDimensionK, 32},
                               {kDimensionN, 1}}},

  {"inception_5a-5x5", {{kDimensionR, 5},
                        {kDimensionS, 5},
                        {kDimensionP, 7},
                        {kDimensionQ, 7},
                        {kDimensionC, 32},
                        {kDimensionK, 128},
                        {kDimensionN, 1}}},

  // Inception 5b
  
  {"inception_5b-pool_proj", {{kDimensionR, 1},
                              {kDimensionS, 1},
                              {kDimensionP, 7},
                              {kDimensionQ, 7},
                              {kDimensionC, 832},
                              {kDimensionK, 128},
                              {kDimensionN, 1}}},

  {"inception_5b-1x1", {{kDimensionR, 1},
                        {kDimensionS, 1},
                        {kDimensionP, 7},
                        {kDimensionQ, 7},
                        {kDimensionC, 832},
                        {kDimensionK, 384},
                        {kDimensionN, 1}}},

  {"inception_5b-3x3_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 7},
                               {kDimensionQ, 7},
                               {kDimensionC, 832},
                               {kDimensionK, 192},
                               {kDimensionN, 1}}},

  {"inception_5b-3x3", {{kDimensionR, 3},
                        {kDimensionS, 3},
                        {kDimensionP, 7},
                        {kDimensionQ, 7},
                        {kDimensionC, 192},
                        {kDimensionK, 384},
                        {kDimensionN, 1}}},

  {"inception_5b-5x5_reduce", {{kDimensionR, 1},
                               {kDimensionS, 1},
                               {kDimensionP, 7},
                               {kDimensionQ, 7},
                               {kDimensionC, 832},
                               {kDimensionK, 48},
                               {kDimensionN, 1}}},

  {"inception_5b-5x5", {{kDimensionR, 5},
                        {kDimensionS, 5},
                        {kDimensionP, 7},
                        {kDimensionQ, 7},
                        {kDimensionC, 48},
                        {kDimensionK, 128},
                        {kDimensionN, 1}}}
};

std::ostream& operator << (std::ostream& out, Workload::FactorizedBounds& bounds)
{
  out << kDimensionR << " = " << bounds[kDimensionR] << std::endl;
  out << kDimensionS << " = " << bounds[kDimensionS] << std::endl;
  out << kDimensionP << " = " << bounds[kDimensionP] << std::endl;
  out << kDimensionQ << " = " << bounds[kDimensionQ] << std::endl;
  out << kDimensionC << " = " << bounds[kDimensionC] << std::endl;
  out << kDimensionK << " = " << bounds[kDimensionK] << std::endl;
  out << kDimensionN << " = " << bounds[kDimensionN] << std::endl;

  return out;
}

// ===============================================
//                   Densities
// ===============================================
#include "cnn-densities.hpp" 

const std::map<uint32_t, uint32_t> nearest_composite = {
  {11, 12}, {13, 15}, {27, 28}, {55, 56}, {57, 60}};

// Function to get the layer config from a layer name.
Workload::FactorizedBounds GetLayerBounds(std::string layer_name, bool pad_primes)
{
  Workload::FactorizedBounds prob;

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
    for (int pd = 0; pd < int(problem::GetShape()->NumFactorizedDimensions); pd++)
    {
      if (nearest_composite.count(prob[problem::Shape::FactorizedDimensionID(pd)]) != 0)
      {
        prob[problem::Shape::FactorizedDimensionID(pd)] =
            nearest_composite.at(prob[problem::Shape::FactorizedDimensionID(pd)]);
      }
    }
  }

  return prob;
}

// Function to get the layer density from a layer name.
Workload::Densities GetLayerDensities(std::string layer_name)
{
  std::map<Shape::DataSpaceID, double> avg_dens;
  Workload::Densities dens;
  try
  {
    avg_dens = densities.at(layer_name);
    for (unsigned d = 0; d < GetShape()->NumDataSpaces; d++){
      YAML::Node ynode;
      ynode["distribution"] = "fixed";
      ynode["density"] = avg_dens;
      auto density_specs = problem::DensityDistributionFactory::ParseSpecs(config::CompoundConfigNode(nullptr, ynode,
                                                                                                      new config::CompoundConfig("dummy.yaml")));
      dens[d]= problem::DensityDistributionFactory::Construct(density_specs);
    }

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
    densities.at(layer).at(kDataSpaceWeight) = atof(buf.data());

    getline(file, buf, ',');
    densities.at(layer).at(kDataSpaceInput) = atof(buf.data());

    getline(file, buf);
    densities.at(layer).at(kDataSpaceOutput) = atof(buf.data());
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
    file << layer.second.at(kDataSpaceWeight) << ", ";
    file << layer.second.at(kDataSpaceInput) << ", ";
    file << layer.second.at(kDataSpaceOutput) << std::endl;
  }

  file.close();
}

// Dump densities.
void DumpDensities_CPP(std::string filename)
{
  std::ofstream file(filename);

  // file << "#include \"cnn-layers.hpp\"" << std::endl;
  // file << std::endl;

  // file << "const unsigned kDataSpaceWeight = " << kDataSpaceWeight << ";" << std::endl;
  // file << "const unsigned kDataSpaceInput = " << kDataSpaceInput << ";" << std::endl;
  // file << "const unsigned kDataSpaceOutput = " << kDataSpaceOutput << ";" << std::endl;
  // file << std::endl;
  
  file << "std::map<std::string, Workload::Densities> densities = {" << std::endl;
  
  for (auto & layer : densities)
  {
    file << "{\"" << layer.first << "\"," << std::endl;
    file << "  {{" << "kDataSpaceWeight, " << layer.second.at(kDataSpaceWeight) << "}," << std::endl;
    file << "   {" << "kDataSpaceInput, " << layer.second.at(kDataSpaceInput) << "}, " << std::endl;
    file << "   {" << "kDataSpaceOutput, " << layer.second.at(kDataSpaceOutput) << "}}}," << std::endl;
  }

  file << "};" << std::endl;
  
  file.close();
}

// Libconfig Parsers.
void ParseConfig(config::CompoundConfigNode config, Workload &workload)
{
  Workload::FactorizedBounds bounds;
  std::string layer_name = "";
  if (config.lookupValue("layer", layer_name))
  {
    bool pad_primes = true;
    config.lookupValue("padPrimes", pad_primes);
    bounds = GetLayerBounds(layer_name, pad_primes);

    // Optional overrides.
    config.lookupValue("R", bounds[kDimensionR]);
    config.lookupValue("S", bounds[kDimensionS]);
    config.lookupValue("P", bounds[kDimensionP]);
    config.lookupValue("Q", bounds[kDimensionQ]);
    config.lookupValue("C", bounds[kDimensionC]);
    config.lookupValue("K", bounds[kDimensionK]);
    config.lookupValue("N", bounds[kDimensionN]);
  }
  else
  {
    assert(config.lookupValue("R", bounds[kDimensionR]));
    assert(config.lookupValue("S", bounds[kDimensionS]));
    assert(config.lookupValue("P", bounds[kDimensionP]));
    assert(config.lookupValue("Q", bounds[kDimensionQ]));
    assert(config.lookupValue("C", bounds[kDimensionC]));
    assert(config.lookupValue("K", bounds[kDimensionK]));
    assert(config.lookupValue("N", bounds[kDimensionN]));
  }
  workload.SetFactorizedBounds(bounds);

  Workload::Coefficients coefficients;
  coefficients[0] = 1;
  coefficients[1] = 1;
  coefficients[2] = 1;
  coefficients[3] = 1;
  config.lookupValue("Wstride", coefficients[0]);
  config.lookupValue("Hstride", coefficients[1]);
  config.lookupValue("Wdilation", coefficients[2]);
  config.lookupValue("Hdilation", coefficients[3]);
  workload.SetCoefficients(coefficients);
  
  Workload::Densities densities;
  // See if user wants to override default densities.
  double common_density;
  YAML::Node ynode;
  if (config.lookupValue("commonDensity", common_density))
  {
    ynode["distribution"] = "fixed";
    ynode["density"] = common_density;
    auto density_specs = problem::DensityDistributionFactory::ParseSpecs(config::CompoundConfigNode(nullptr, ynode,
                                                                                                    new config::CompoundConfig("dummy.yaml")));
    auto shared_density_distribution = problem::DensityDistributionFactory::Construct(density_specs);

    densities[kDataSpaceWeight]= shared_density_distribution;
    densities[kDataSpaceInput]= shared_density_distribution;
    densities[kDataSpaceOutput]= shared_density_distribution;
  }
  else if (config.exists("densities"))
  {
    config::CompoundConfigNode config_densities = config.lookup("densities");
    
    double weights_average_density;
    double input_average_density;
    double output_average_density;

    assert(config_densities.lookupValue("weights", weights_average_density));
    assert(config_densities.lookupValue("inputs", input_average_density));
    assert(config_densities.lookupValue("outputs", output_average_density));

    ynode["distribution"] = "fixed";

    ynode["density"] = weights_average_density;
    auto density_specs = problem::DensityDistributionFactory::ParseSpecs(config::CompoundConfigNode(nullptr, ynode,
                                                                                                    new config::CompoundConfig("dummy.yaml")));
    densities[kDataSpaceWeight] = problem::DensityDistributionFactory::Construct(density_specs);

    ynode["density"] = input_average_density;
    density_specs = problem::DensityDistributionFactory::ParseSpecs(config::CompoundConfigNode(nullptr, ynode,
                                                                                                    new config::CompoundConfig("dummy.yaml")));
    densities[kDataSpaceInput] = problem::DensityDistributionFactory::Construct(density_specs);

    ynode["density"] = output_average_density;
    density_specs = problem::DensityDistributionFactory::ParseSpecs(config::CompoundConfigNode(nullptr, ynode,
                                                                                                    new config::CompoundConfig("dummy.yaml")));
    densities[kDataSpaceOutput] = problem::DensityDistributionFactory::Construct(density_specs);

  }
  else if (layer_name != "")
  {
    densities = GetLayerDensities(layer_name);
  }
  else
  {

    ynode["distribution"] = "fixed";
    ynode["density"] = 1.0;
    auto density_specs = problem::DensityDistributionFactory::ParseSpecs(config::CompoundConfigNode(nullptr, ynode,
                                                                                                    new config::CompoundConfig("dummy.yaml")));
    auto shared_density_distribution = problem::DensityDistributionFactory::Construct(density_specs);

    densities[kDataSpaceWeight]= shared_density_distribution;
    densities[kDataSpaceInput]= shared_density_distribution;
    densities[kDataSpaceOutput]= shared_density_distribution;

  }
  workload.SetDensities(densities);
}

} // namespace problem
