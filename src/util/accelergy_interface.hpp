#include <cstring>

namespace accelergy
{
  void invokeAccelergy(std::vector<std::string> inputFiles) {
    std::string cmd = std::string(ACCELERGY_PATH) + "/accelergy";
    for (auto inputFile : inputFiles) {
      cmd += " " + inputFile;
    }
    cmd += " -o ./ > ./accelergy.log 2>&1";
    int ret = system(cmd.c_str());
    if (ret) {
      std::cout << "Cannot invoke Accelergy. Do you specify ACCELERGYPATH correctly?" << std::endl;
      exit(0);
    }
    return;
  }
} // namespace accelergy
