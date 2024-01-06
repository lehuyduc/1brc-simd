#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <immintrin.h>
#include <thread>
#include <new>
#include <unordered_map>
#include <cstring>
using namespace std;

const uint8_t* mmap2array(string file_path, size_t& file_size)
{
  int fd = open(file_path.c_str(), O_RDONLY);
  struct stat file_stat;
  fstat(fd, &file_stat);
  file_size = file_stat.st_size;

  void* mapped_data_void = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);

  const uint8_t* data = reinterpret_cast<uint8_t*>(mapped_data_void);
  return data;
}

int main(int argc, char* argv[])
{  
  if (argc <= 1) {
    cout << "Need argument result file name\n";
    exit(1);
  }

  string output = string(argv[1]);
  string ref = "ref.txt"; // replace with "tests/measurements-20.txt" for example

  size_t file_size, file_size_ref;
  auto data = mmap2array(output, file_size);
  auto data_ref = mmap2array(ref, file_size_ref);

  int N = min(file_size, file_size_ref);
  for (int i = 0; i < N; i++) if (data[i] != data_ref[i]) {
    cout << "diff at " << i << ":" << data[i] << "|" << data_ref[i] << "|\n";
  }

  if (file_size != file_size_ref) {
    cout << "Different file size: out = " << file_size << ", ref = " << file_size_ref << "\n";
    cout << "Check end of file characters when you copy\n";
  }
    
  return 0;
}