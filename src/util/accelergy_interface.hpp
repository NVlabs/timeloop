
namespace accelergy
{
  void invokeAccelergy(std::vector<std::string> inputFiles) {
    std::string cmd = std::string(ACCELERGY_PATH) + "accelergy";
    for (auto inputFile : inputFiles) {
      cmd += " " + inputFile;
    }
    cmd += " -o ./ > ./accelergy.log 2>&1";
    system(cmd.c_str());
    return;
  }
} // namespace accelergy
