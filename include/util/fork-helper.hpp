#include <functional>
#include <optional>
#include <iostream>
#include <unistd.h>


/**
 * @brief Helper for keeping track of pipe files.
 */
struct PipeFd
{
  int pipefd_read;
  int pipefd_write;

  void CloseReadEnd()
  {
    if (pipefd_read >= 0)
    {
      close(pipefd_read);
      pipefd_read = -1;
    }
  }

  void CloseWriteEnd()
  {
    if (pipefd_write >= 0)
    {
      close(pipefd_write);
      pipefd_write = -1;
    }
  }

  ~PipeFd()
  {
    CloseReadEnd();
    CloseWriteEnd();
  }
};

/**
 * @brief Create a pipe for two processes to communicate.
 */
PipeFd GetPipe()
{
  int pipefd[2];
  auto stat = pipe(pipefd);
  if (stat < 0)
  {
    throw std::logic_error("pipe creation error!");
  }
  return PipeFd{pipefd[0], pipefd[1]};
}

/**
 * @brief Reads a typed object from the pipe. The object cannot be a container.
 * 
 * @note T has to have a copy-ctor.
 * @note A better way to do this is via ser-des.
 */
template<typename T>
std::optional<T> ReadFromPipe(int pipefd_read)
{
  char buf[sizeof(T)];
  auto n_read_so_far = 0;
  while (true)
  {
    auto n_read = read(pipefd_read,
                       buf + n_read_so_far,
                       sizeof(T) - n_read_so_far);
    if (n_read < 0) // ERROR!
    {
      throw std::logic_error("pipe read error!");
    }
    else if (n_read == 0) // EOF
    {
      break;
    }
    n_read_so_far += n_read;
  }

  if (n_read_so_far < sizeof(T))
  {
    throw std::logic_error("recv'd bytes did not match expected type");
  }

  // Creates the result via copy-ctor
  T result = *reinterpret_cast<const T*>(buf)
  return reinterpret_cast<T>(buf);
}


/**
 * @brief Executes a closure in another process.
 * 
 * If communication with parent process is needed, the closure can capture a
 * pipe, socket, etc.
 */
template<typename T>
pid_t ExecInAnotherProcess(const std::function<T()>& f)
{
  auto pid = fork();
  if (pid < 0)
  {
    std::cout << "Bad fork!" << std::endl;
  }
  else if (pid == 0)
  {
    f();
  }
  else
  {
    return pid;
  }
}
