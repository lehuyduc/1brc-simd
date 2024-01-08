 #include "my_timer.h"
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

constexpr int N_THREADS = 32;
volatile int dummy = 0;
volatile int dummies[N_THREADS];
constexpr size_t N = 3'200'000'000ULL - (3'200'000'000ULL % N_THREADS);
constexpr size_t N_LOOP = 10;
int* A;
const int* B;

void __attribute__ ((noinline)) func(int* __restrict__ A, const int* __restrict__ B, size_t from_int, size_t to_int, int tid)
{
  dummies[tid] = tid;
  for (size_t i = from_int; i < to_int; i++) A[i] = B[i];
  A[dummies[tid]] = dummies[tid];
}

void parallel_copy(size_t tid, int* __restrict__ A, const int* __restrict__ B)
{
  constexpr size_t BLOCK_SIZE = N / N_THREADS;
  size_t from_int = BLOCK_SIZE * tid;
  size_t to_int = BLOCK_SIZE * (tid + 1);

  for (size_t t = 1; t <= N_LOOP; t++) {   
    func(A, B, from_int, to_int, tid);
    dummies[tid] = A[rand() % N] % N_THREADS;
  }
}


int main(int argc, char* argv[])
{
  MyTimer timer;

  string file_path = "measurements.txt";
  if (argc > 1) file_path = string(argv[1]);

  int fd = open(file_path.c_str(), O_RDONLY);
  struct stat file_stat;
  fstat(fd, &file_stat);
  size_t file_size = file_stat.st_size;

  void* mapped_data_void = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  //----------------------

  timer.startCounter();
  A = new int[N];    
  for (size_t i = 0; i < N; i++) A[i] = 0;

  B = reinterpret_cast<int*>(mapped_data_void);
  cout << "Time to init data = " << timer.getCounterMsPrecise() << "ms" << std::endl;
  
  timer.startCounter();
  vector<std::thread> threads;
  for (size_t tid = 0; tid < N_THREADS; tid++) {
    threads.emplace_back([tid, &A, &B]() {
      parallel_copy(tid, A, B);
    });
  }

  for (auto& thread : threads) thread.join();
  double time_sec = timer.getCounterMsPrecise() / 1000.0;

  size_t N_BYTES = N_LOOP * N * sizeof(int);
  cout << "Bandwidth = " << (N_BYTES / time_sec) << " byte/s\n";
  return 0;
}

