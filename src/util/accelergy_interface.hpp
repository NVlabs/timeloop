#include <cstring>

namespace accelergy
{
  void invokeAccelergy(std::vector<std::string> inputFiles) {
#ifdef ACCELERGY_PATH
    std::string cmd = std::string(ACCELERGY_PATH) + "/accelergy";
    for (auto inputFile : inputFiles) {
      cmd += " " + inputFile;
    }
    cmd += " -o ./ > ./accelergy.log 2>&1";
    int ret = system(cmd.c_str());
    if (ret) {
      std::cout << "Cannot invoke Accelergy. Do you specify ACCELERGYPATH correctly? Or check accelergy.log to see what went wrong" << std::endl;
      exit(0);
    }
#else
    (void) inputFiles;
#endif
    return;
  }
} // namespace accelergy
