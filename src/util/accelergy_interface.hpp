#include <iostream>
#include <cstring>
#include <stdexcept>

namespace accelergy
{
  std::string exec(const char* cmd) {
    std::string result = "";
    char buffer[128];
    FILE* pipe = popen("which accelergy", "r");
    if (!pipe) {
      std::cout << "popen(" << cmd << ") failed" << std::endl;
      exit(0);
    }

    try {
      while (fgets(buffer, 128, pipe) != nullptr) {
        result += buffer;
      }
    } catch (...) {
      pclose(pipe);
    }
    pclose(pipe);
    return result;
  }

  void invokeAccelergy(std::vector<std::string> inputFiles) {
#ifdef USE_ACCELERGY
    std::string accelergy_path = exec("which accelergy");
    // if `which` does not find it, we will try env
    if (accelergy_path.find("accelergy") == std::string::npos) {
      accelergy_path = exec("echo $ACCELERGYPATH");
      accelergy_path += "accelergy";
    }
    //std::cout << "Invoke Accelergy at: " << accelergy_path << std::endl;
    std::string cmd = accelergy_path.substr(0, accelergy_path.size() - 1);
    for (auto inputFile : inputFiles) {
      cmd += " " + inputFile;
    }
    cmd += " -o ./ > ./accelergy.log 2>&1";
    //std::cout << "execute:" << cmd << std::endl;
    int ret = system(cmd.c_str());
    if (ret) {
      std::cout << "Failed to run Accelergy. Did you install Accelergy or specify ACCELERGYPATH correctly? Or check accelergy.log to see what went wrong" << std::endl;
      exit(0);
    }
#else
    (void) inputFiles;
#endif
    return;
  }
} // namespace accelergy
