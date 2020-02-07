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

  void invokeAccelergy(std::vector<std::string> input_files, std::string out_prefix) {
#ifdef USE_ACCELERGY
    std::string accelergy_path = exec("which accelergy");
    // if `which` does not find it, we will try env
    if (accelergy_path.find("accelergy") == std::string::npos) {
      accelergy_path = exec("echo $ACCELERGYPATH");
      accelergy_path += "accelergy";
    }
    //std::cout << "Invoke Accelergy at: " << accelergy_path << std::endl;
    std::string cmd = accelergy_path.substr(0, accelergy_path.size() - 1);
    for (auto input_file : input_files) {
      cmd += " " + input_file;
    }
    cmd += " --oprefix " + out_prefix + ".";
    cmd += " -o ./ > ./" + out_prefix + ".accelergy.log 2>&1";
    //std::cout << "execute:" << cmd << std::endl;
    int ret = system(cmd.c_str());
    if (ret) {
      std::cout << "Failed to run Accelergy. Did you install Accelergy or specify ACCELERGYPATH correctly? Or check accelergy.log to see what went wrong" << std::endl;
      exit(0);
    }
#else
    (void) input_files;
    (void) out_prefix;
#endif
    return;
  }
} // namespace accelergy
