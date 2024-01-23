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

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

constexpr uint32_t SMALL = 276187; // >> 20
constexpr int MAX_KEY_LENGTH = 100;
constexpr uint32_t NUM_BINS = 16384;

#ifndef N_THREADS_PARAM
constexpr int N_THREADS = 8; // to match evaluation server
#else
constexpr int N_THREADS = N_THREADS_PARAM;
#endif


struct Stats {
    int64_t sum;
    int cnt;
    int max;
    int min;

    Stats() {
        cnt = 0;
        sum = 0;
        max = -1024;
        min = -1024;
    }

    bool operator < (const Stats& other) const {
        return min < other.min;
    }
};

struct HashBin {
    Stats stats;
    int len;
    uint8_t key[MAX_KEY_LENGTH];    

    HashBin() {
      // C++ zero-initialize global variable by default
      // len = 0;
      // memset(key, 0, sizeof(key));
    }
};
static_assert(sizeof(HashBin) == 128); // faster array indexing if struct is power of 2

constexpr int N_AGGREGATE = (N_THREADS >= 16) ? (N_THREADS >> 2) : 1;
constexpr int N_AGGREGATE_LV2 = (N_AGGREGATE >= 32) ? (N_AGGREGATE >> 2) : 1;
std::unordered_map<string, Stats> partial_stats[N_AGGREGATE];
std::unordered_map<string, Stats> final_recorded_stats;

alignas(4096) uint32_t pow_small[64];

alignas(4096) HashBin hmaps[N_THREADS][NUM_BINS];

void init_pow_small() {
    uint32_t b[40];
    b[0] = 1;
    for (int i = 1; i <= 32; i++) b[i] = b[i - 1] * SMALL;

    for (int i = 0; i < 32; i++) pow_small[i] = b[31 - i];
    for (int i = 32; i < 64; i++) pow_small[i] = 0;
}


// https://en.algorithmica.org/hpc/simd/reduction/
//inline uint32_t __attribute__((always_inline)) hsum(__m256i x) {
uint32_t hsum(__m256i x) {
    __m128i l = _mm256_extracti128_si256(x, 0);
    __m128i h = _mm256_extracti128_si256(x, 1);
    l = _mm_add_epi32(l, h);
    l = _mm_hadd_epi32(l, l);
    return (uint32_t)_mm_extract_epi32(l, 0) + (uint32_t)_mm_extract_epi32(l, 1);
}

// 16 all 1s followed by 16 all 0s
// This is used to mask characters past the end of a string later
alignas(4096) const uint8_t strcmp_mask[32] = {
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

// force inline here make performance more consistent, ~2% lower average
template <bool SAFE_HASH>
inline void __attribute__((always_inline)) hmap_insert(HashBin* hmap, uint32_t hash_value, const uint8_t* key, int len, int value)
{
  if (likely(!SAFE_HASH && len <= 16)) {
    __m128i chars = _mm_loadu_si128((__m128i*)key);
    __m128i mask = _mm_loadu_si128((__m128i*)(strcmp_mask + 16 - len));
    __m128i key_chars = _mm_and_si128(chars, mask);

    __m128i bin_chars = _mm_loadu_si128((__m128i*)hmap[hash_value].key);
    if (likely(_mm_testc_si128(bin_chars, key_chars) || hmap[hash_value].len == 0)) {
      // consistent 2.5% improvement in `user` time by testing first bin before loop
    }
    else {
      hash_value = (hash_value + 1) % NUM_BINS; // previous one failed
      while (hmap[hash_value].len > 0) {
        // SIMD string comparison      
        __m128i bin_chars = _mm_loadu_si128((__m128i*)hmap[hash_value].key);
        if (likely(_mm_testc_si128(bin_chars, key_chars))) break;
        hash_value = (hash_value + 1) % NUM_BINS;    
      }
    }
  } else {
    while (hmap[hash_value].len > 0) {
      // check if this slot is mine
      if (likely(hmap[hash_value].len == len)) {
          bool equal = true;
          for (int i = 0; i < len; i++) if (key[i] != hmap[hash_value].key[i]) {
              equal = false;
              break;
          }
          if (likely(equal)) break;
      }
      hash_value = (hash_value + 1) % NUM_BINS;
    }
  }

  auto& stats = hmap[hash_value].stats;
  stats.cnt++;
  stats.sum += value;
  stats.max = max(stats.max, value);
  stats.min = max(stats.min, -value);

  // each key will only be free 1 first time, so it's unlikely
  if (unlikely(hmap[hash_value].len == 0)) {        
      hmap[hash_value].len = len;
      memcpy((char*)hmap[hash_value].key, (char*)key, len);        
  }
}

uint32_t slow_hash(const uint8_t* data, uint32_t* return_pos)
{
  uint8_t chars[32];

  int pos = 0;
  uint32_t myhash = 0;
  while (data[pos] != ';') pos++;

  int L = min(pos, 16);
  for (int i = 0; i < L; i++) chars[i] = data[i];
  for (int i = L; i < 16; i++) chars[i] = 0;

  myhash = 0;
  for (int i = 0; i < 8; i++) chars[i] += chars[i + 8];
  uint64_t value;
  memcpy(&value, chars, 8);
  myhash = (value * SMALL) >> 20;

  for (int i = 16; i < pos; i++) myhash = myhash * SMALL + data[i];
  *return_pos = pos;
  return myhash;
}

template <bool SAFE_HASH = false>
inline void handle_line(const uint8_t* data, HashBin* hmap, size_t &data_idx)
{
  uint32_t pos = 16;
  uint32_t myhash;

  // we read 16 bytes at a time with SIMD, so if the final line has < 16 bytes,
  // this cause out-of-bound read.
  // Most of the time it doesn't cause any error, but if the last extra bytes are past
  // the final memory page provided by mmap, it will cause SIGBUS.
  // So for the last few lines, we use safe code.
  if constexpr (SAFE_HASH) {
    myhash = slow_hash(data, &pos);    
  }
  else {
    __m128i chars = _mm_loadu_si128((__m128i*)data);
    __m128i separators = _mm_set1_epi8(';');        
    __m128i compared = _mm_cmpeq_epi8(chars, separators);
    uint32_t separator_mask = _mm_movemask_epi8(compared);

    //__m256i pow_vec1 = _mm256_loadu_si256((__m256i*)(pow_small + 24));

    if (likely(separator_mask)) pos = __builtin_ctz(separator_mask);

    // sum the 2 halves of 16 characters together, then hash the resulting 8 characters
    // this save 1 _mm256_mullo_epi32 instruction, improving performance by ~3%
    __m128i mask = _mm_loadu_si128((__m128i*)(strcmp_mask + 16 - pos));    
    __m128i key_chars = _mm_and_si128(chars, mask);    
    __m128i sumchars = _mm_add_epi8(key_chars, _mm_srli_si128(key_chars, 8));

    // __m256i data_vec1 = _mm256_cvtepu8_epi32(sumchars);
    // myhash = hsum(_mm256_mullo_epi32(pow_vec1, data_vec1));

    // we change hashing method, completely dropping SIMD multiplication, which is slow.
    // This method will cause more hash collision, but we already paid for hash-collision handling,
    // so we will use hash-collision handling :D
    // myhash = (uint64_t(_mm_extract_epi64(sumchars, 0)) * SMALL) >> 20;

    // faster
    // uint64_t temp;
    // memcpy(&temp, &sumchars, 8);
    // myhash = (temp * SMALL) >> 20;
        
    // It's not illegal to dereference __m128i, yay. 0.3% faster than memcpy
    // Maybe it's just noise, but I measure best time instead of average FOR THIS CONTEST, so every millisecond counts.
    // https://stackoverflow.com/questions/52112605/is-reinterpret-casting-between-hardware-simd-vector-pointer-and-the-correspond
    myhash = (*(reinterpret_cast<uint64_t*>(&sumchars)) * SMALL) >> 20;

    if (unlikely(!separator_mask)) {      
      while (data[pos] != ';') {
        myhash = myhash * SMALL + data[pos];
        pos++;
      }
    }
  }

  // data[pos] = ';'.
  // There are 4 cases: ;9.1, ;92.1, ;-9.1, ;-92.1
  int key_end = pos;
  pos += (data[pos + 1] == '-'); // after this, data[pos] = position right before first digit
  int sign = (data[pos] == '-') ? -1 : 1;
  myhash %= NUM_BINS; // let pos be computed first beacause it's needed earlier

  int case1 = 10 * (data[pos + 1] - 48) + (data[pos + 3] - 48); // 9.1
  int case2 = 100 * (data[pos + 1] - 48) + 10 * (data[pos + 2] - 48) + (data[pos + 4] - 48); // 92.1
  int value = case2 * (data[pos + 3] == '.') + case1 * (!(data[pos + 3] == '.'));
  value *= sign;

  // intentionally move index updating before hmap_insert
  // to improve register dependency chain
  data_idx += pos + 5 + (data[pos + 3] == '.');
  
  hmap_insert<SAFE_HASH>(hmap, myhash, data, key_end, value);
}

void handle_line_raw(int tid, const uint8_t* data, size_t from_byte, size_t to_byte, size_t file_size)
{
    size_t idx = from_byte;
    // always start from beginning of a line
    if (from_byte != 0 && data[from_byte - 1] != '\n') {
        while (data[idx] != '\n') idx++;
        idx++;
    }
    if (idx >= to_byte) {
        // this should never happen since if dataset is too small, we use 1 thread
        throw std::runtime_error("idx >= to_byte error");        
    }

    // Thread that process end block must not use SIMD in the last few lines
    // to prevent potential out-of-range access error.
    // This can happen if the file size satisfy: (file_size % page_size) > page_size - 16
    if (tid == N_THREADS - 1) to_byte -= 2 * MAX_KEY_LENGTH;

    while (idx < to_byte) {
        handle_line<false>(data + idx, hmaps[tid], idx);
    }

    if (tid == N_THREADS - 1) {
        while (idx < file_size) {
            handle_line<true>(data + idx, hmaps[tid], idx);
        }
    }
}

void parallel_aggregate(int tid)
{
  constexpr int BLOCK_SIZE = (N_THREADS / N_AGGREGATE);
  int start_idx = tid * BLOCK_SIZE;
  int end_idx = (tid + 1) * BLOCK_SIZE;

  for (int hmap_idx = start_idx; hmap_idx < end_idx; hmap_idx++) {
    for (int h = 0; h < NUM_BINS; h++) if (hmaps[hmap_idx][h].len > 0) {
      auto& bin = hmaps[hmap_idx][h];
      auto& stats = partial_stats[tid][string(bin.key, bin.key + bin.len)];
      stats.cnt += bin.stats.cnt;
      stats.sum += bin.stats.sum;
      stats.max = max(stats.max, bin.stats.max);
      stats.min = max(stats.min, bin.stats.min);
    }
  }
}

void parallel_aggregate_lv2(int tid)
{
  for (int idx = N_AGGREGATE_LV2 + tid; idx < N_AGGREGATE; idx += N_AGGREGATE_LV2) {
    for (auto& [key, value] : partial_stats[idx]) {
      auto& stats = partial_stats[tid][key];
      stats.cnt += value.cnt;
      stats.sum += value.sum;
      stats.max = max(stats.max, value.max);
      stats.min = max(stats.min, value.min);
    }
  }
}

float roundTo1Decimal(float number) {
    return std::round(number * 10.0) / 10.0;
}

int main(int argc, char* argv[])
{
  cout << "Using " << N_THREADS << " threads\n";
  MyTimer timer, timer2;
  timer.startCounter();    
  init_pow_small();

  string file_path = "measurements.txt";
  if (argc > 1) file_path = string(argv[1]);

  int fd = open(file_path.c_str(), O_RDONLY);
  struct stat file_stat;
  fstat(fd, &file_stat);
  size_t file_size = file_stat.st_size;

  void* mapped_data_void = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);

  const uint8_t* data = reinterpret_cast<uint8_t*>(mapped_data_void);
  cout << "init mmap file cost = " << timer.getCounterMsPrecise() << "ms\n";
  
  //----------------------
  timer2.startCounter();
  size_t idx = 0;
  int n_threads = N_THREADS;
  if (file_size / n_threads < 4 * MAX_KEY_LENGTH) n_threads = 1;
  
  size_t remaining_bytes = file_size - idx;
  size_t bytes_per_thread = remaining_bytes / n_threads + 1;
  vector<size_t> tstart, tend;
  vector<std::thread> threads;
  for (size_t tid = 0; tid < n_threads; tid++) {
      size_t starter = idx + tid * bytes_per_thread;
      size_t ender = idx + (tid + 1) * bytes_per_thread;
      if (ender > file_size) ender = file_size;
      threads.emplace_back([tid, data, starter, ender, file_size]() {
          handle_line_raw(tid, data, starter, ender, file_size);
      });
  }

  for (auto& thread : threads) thread.join();
  cout << "Parallel process file cost = " << timer.getCounterMsPrecise() << "ms\n";

  //----------------------
  timer2.startCounter();
  if constexpr(N_AGGREGATE > 1) {
    threads.clear();
    for (int tid = 0; tid < N_AGGREGATE; tid++) {
      threads.emplace_back([tid]() {
        parallel_aggregate(tid);
      });
    }
    for (auto& thread : threads) thread.join();

    //----- parallel reduction again
    threads.clear();
    for (int tid = 0; tid < N_AGGREGATE_LV2; tid++) {
      threads.emplace_back([tid]() {
        parallel_aggregate_lv2(tid);
      });
    }
    for (auto& thread : threads) thread.join();
    // now, the stats are aggregated into partial_stats[0 : N_AGGREGATE_LV2]

    for (int tid = 0; tid < N_AGGREGATE_LV2; tid++) {
      for (auto& [key, value] : partial_stats[tid]) {
        auto& stats = final_recorded_stats[key];
        stats.cnt += value.cnt;
        stats.sum += value.sum;
        stats.max = max(stats.max, value.max);
        stats.min = max(stats.min, value.min);
      }
    }
  } else {
    for (int tid = 0; tid < n_threads; tid++) {
      for (int h = 0; h < NUM_BINS; h++) if (hmaps[tid][h].len > 0) {
          auto& bin = hmaps[tid][h];            
          auto& stats = final_recorded_stats[string(bin.key, bin.key + bin.len)];            
          stats.cnt += bin.stats.cnt;
          stats.sum += bin.stats.sum;
          stats.max = max(stats.max, bin.stats.max);
          stats.min = max(stats.min, bin.stats.min);
      }
    }
  }
  cout << "Aggregate stats cost = " << timer2.getCounterMsPrecise() << "ms\n";

  timer2.startCounter();
  vector<pair<string, Stats>> results;
  for (auto& [key, value] : final_recorded_stats) {
      results.emplace_back(key, value);
  }
  sort(results.begin(), results.end());

  // {Abha=-37.5/18.0/69.9, Abidjan=-30.0/26.0/78.1,  
  ofstream fo("result_valid11.txt");
  fo << fixed << setprecision(1);
  fo << "{";
  for (size_t i = 0; i < results.size(); i++) {
      const auto& result = results[i];
      const auto& station_name = result.first;
      const auto& stats = result.second;
      float avg = roundTo1Decimal((double)stats.sum / 10.0 / stats.cnt);
      float mymax = roundTo1Decimal(stats.max / 10.0);
      float mymin = roundTo1Decimal(-stats.min / 10.0);

      fo << station_name << "=" << mymin << "/" << avg << "/" << mymax;
      if (i < results.size() - 1) fo << ", ";
  }
  fo << "}\n";
  fo.close();
  cout << "Output stats cost = " << timer2.getCounterMsPrecise() << "ms\n";

  cout << "Runtime inside main = " << timer.getCounterMsPrecise() << "ms\n";

  timer.startCounter();
  munmap(mapped_data_void, file_size);
  cout << "Time to munmap = " << timer.getCounterMsPrecise() << "\n";
  return 0;
}

// Pointer cast instead of _mm_extract. Is this legal?
// Using 32 threads
// init mmap file cost = 0.038864ms
// Parallel process file cost = 498.699ms
// Aggregate stats cost = 1.93794ms
// Output stats cost = 0.980052ms
// Runtime inside main = 501.687ms
// Time to munmap = 151.639
// real	0m0.685s
// user	0m15.009s
// sys	0m0.759s

// Using 32 threads
// init mmap file cost = 0.032371ms
// Parallel process file cost = 495.93ms
// Aggregate stats cost = 1.84298ms
// Output stats cost = 0.732642ms
// Runtime inside main = 498.568ms
// Time to munmap = 157.28
// real	0m0.692s
// user	0m15.027s
// sys	0m0.678s

// Using 32 threads
// init mmap file cost = 0.035077ms
// Parallel process file cost = 496.84ms
// Aggregate stats cost = 1.89636ms
// Output stats cost = 0.722013ms
// Runtime inside main = 499.515ms
// Time to munmap = 155.853
// real	0m0.688s
// user	0m14.993s
// sys	0m0.712s

// Using 32 threads
// init mmap file cost = 0.035057ms
// Parallel process file cost = 495.233ms
// Aggregate stats cost = 1.99891ms
// Output stats cost = 0.739335ms
// Runtime inside main = 498.032ms
// Time to munmap = 155.353
// real	0m0.684s
// user	0m14.982s
// sys	0m0.702s

// memcpy instead of deref
// Using 32 threads
// init mmap file cost = 0.040357ms
// Parallel process file cost = 498.324ms
// Aggregate stats cost = 1.96971ms
// Output stats cost = 1.00153ms
// Runtime inside main = 501.363ms
// Time to munmap = 151.798
// real	0m0.684s
// user	0m14.885s
// sys	0m0.794s

// Using 32 threads
// init mmap file cost = 0.028825ms
// Parallel process file cost = 499.88ms
// Aggregate stats cost = 1.98679ms
// Output stats cost = 1.54208ms
// Runtime inside main = 503.487ms
// Time to munmap = 152.384
// real	0m0.686s
// user	0m15.089s
// sys	0m0.689s

// Using 32 threads
// init mmap file cost = 0.034225ms
// Parallel process file cost = 496.12ms
// Aggregate stats cost = 1.95508ms
// Output stats cost = 1.30793ms
// Runtime inside main = 499.448ms
// Time to munmap = 152.345
// real	0m0.685s
// user	0m14.985s
// sys	0m0.769s

// Using 32 threads
// init mmap file cost = 0.034015ms
// Parallel process file cost = 497.599ms
// Aggregate stats cost = 1.90578ms
// Output stats cost = 1.29208ms
// Runtime inside main = 500.871ms
// Time to munmap = 156.892
// real	0m0.688s
// user	0m14.880s
// sys	0m0.789s

// Using 1 threads
// init mmap file cost = 0.015239ms
// Parallel process file cost = 10460.6ms
// Aggregate stats cost = 0.240646ms
// Output stats cost = 1.51426ms
// Runtime inside main = 10462.4ms
// Time to munmap = 157.086

// 1
//  Performance counter stats for './main':

//          10,613.37 msec task-clock                #    0.999 CPUs utilized          
//              1,025      context-switches          #    0.097 K/sec                  
//                  3      cpu-migrations            #    0.000 K/sec                  
//            211,177      page-faults               #    0.020 M/sec                  
//     45,279,768,774      cycles                    #    4.266 GHz                      (37.50%)
//        116,998,032      stalled-cycles-frontend   #    0.26% frontend cycles idle     (37.49%)
//     36,811,389,158      stalled-cycles-backend    #   81.30% backend cycles idle      (37.49%)
//     93,927,959,176      instructions              #    2.07  insn per cycle         
//                                                   #    0.39  stalled cycles per insn  (37.49%)
//      6,113,016,649      branches                  #  575.973 M/sec                    (37.50%)
//         54,446,103      branch-misses             #    0.89% of all branches          (37.51%)
//     21,267,234,146      L1-dcache-loads           # 2003.815 M/sec                    (37.51%)
//        552,760,583      L1-dcache-load-misses     #    2.60% of all L1-dcache hits    (37.51%)
//    <not supported>      LLC-loads                                                   
//    <not supported>      LLC-load-misses                                             

//       10.623010661 seconds time elapsed

//       10.219247000 seconds user
//        0.391511000 seconds sys

// 32
//  Performance counter stats for './main':

//          15,713.73 msec task-clock                #   22.296 CPUs utilized          
//              1,869      context-switches          #    0.119 K/sec                  
//                 26      cpu-migrations            #    0.002 K/sec                  
//            227,210      page-faults               #    0.014 M/sec                  
//     58,752,234,378      cycles                    #    3.739 GHz                      (37.98%)
//        737,402,692      stalled-cycles-frontend   #    1.26% frontend cycles idle     (38.14%)
//     12,200,248,818      stalled-cycles-backend    #   20.77% backend cycles idle      (37.67%)
//     94,714,174,971      instructions              #    1.61  insn per cycle         
//                                                   #    0.13  stalled cycles per insn  (37.29%)
//      6,209,576,972      branches                  #  395.169 M/sec                    (37.21%)
//         55,074,139      branch-misses             #    0.89% of all branches          (37.08%)
//     21,169,947,951      L1-dcache-loads           # 1347.226 M/sec                    (37.06%)
//        800,876,968      L1-dcache-load-misses     #    3.78% of all L1-dcache hits    (37.58%)
//    <not supported>      LLC-loads                                                   
//    <not supported>      LLC-load-misses                                             

//        0.704783243 seconds time elapsed

//       14.939946000 seconds user
//        0.766560000 seconds sys

// 16
//  Performance counter stats for './main':

//          11,793.81 msec task-clock                #   12.002 CPUs utilized          
//              1,145      context-switches          #    0.097 K/sec                  
//                  7      cpu-migrations            #    0.001 K/sec                  
//            218,933      page-faults               #    0.019 M/sec                  
//     46,466,230,961      cycles                    #    3.940 GHz                      (37.47%)
//        183,222,902      stalled-cycles-frontend   #    0.39% frontend cycles idle     (37.26%)
//     36,037,440,295      stalled-cycles-backend    #   77.56% backend cycles idle      (37.18%)
//     95,047,349,354      instructions              #    2.05  insn per cycle         
//                                                   #    0.38  stalled cycles per insn  (37.03%)
//      6,194,625,591      branches                  #  525.244 M/sec                    (37.33%)
//         54,539,392      branch-misses             #    0.88% of all branches          (37.76%)
//     21,108,851,736      L1-dcache-loads           # 1789.825 M/sec                    (38.12%)
//        564,074,957      L1-dcache-load-misses     #    2.67% of all L1-dcache hits    (37.84%)
//    <not supported>      LLC-loads                                                   
//    <not supported>      LLC-load-misses                                             

//        0.982642727 seconds time elapsed

//       11.250367000 seconds user
//        0.540113000 seconds sys
