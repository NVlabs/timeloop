#include <iostream>
#include <map>
#include <vector>
#include <tuple>

#include "conv_problems.h"

void gen_dict(
  std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
  unsigned int, unsigned int, unsigned int,
  unsigned int, unsigned int, unsigned int, unsigned int>>& prob_set,
  std::string prob_set_name)
{
  std::cout << prob_set_name << " = [" << std::endl;
  for (auto prob = training_set.begin(); prob != training_set.end(); prob++)
  {
    auto [w, h, c, n, k, s, r, padw, padh, wstride, hstride] = *prob;
    char buf[256];
    snprintf(buf, 255, "    (%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d)",
             w, h, c, n, k, s, r, padw, padh, wstride, hstride);
    std::cout << buf;
    if (prob != std::prev(training_set.end()))
      std::cout << "," << std::endl;
    else
      std::cout << "]" << std::endl;
  }
  std::cout << std::endl;
}

int main()
{
  gen_dict(training_set, "training_set");
  gen_dict(inference_server_set, "inference_server_set");
  gen_dict(inference_device_set, "inference_device_set");
  return 0;
}
