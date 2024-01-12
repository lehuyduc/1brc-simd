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
#include <atomic>
#include <malloc.h>
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
constexpr bool DEBUG = 1;


struct Stats {
    int64_t sum;
    int cnt;
    int max;
    int min;

    Stats() {
        cnt = 0;
        sum = 0;
        max = -1024;
        min = 1024;
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
      len = 0;
      memset(key, 0, sizeof(key));
    }
};
static_assert(sizeof(HashBin) == 128); // faster array indexing if struct is power of 2

constexpr int N_AGGREGATE = (N_THREADS >= 16) ? (N_THREADS >> 2) : 1;
constexpr int N_AGGREGATE_LV2 = (N_AGGREGATE >= 32) ? (N_AGGREGATE >> 2) : 1;
std::unordered_map<string, Stats> partial_stats[N_AGGREGATE];
std::unordered_map<string, Stats> final_recorded_stats;

HashBin* global_hmaps;
alignas(4096) HashBin* hmaps[N_THREADS];

// force inline here make performance more consistent, ~2% lower average
template <bool SAFE_HASH>
inline void __attribute__((always_inline)) hmap_insert(HashBin* hmap, uint32_t hash_value, const uint8_t* key, int len, int value)
{
  if (likely(!SAFE_HASH && len <= 16)) {
    __m128i chars = _mm_loadu_si128((__m128i*)key);
    __m128i index = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    //__m128i mask = _mm_loadu_si128((__m128i*)(strcmp_mask + 16 - len));
    __m128i mask = _mm_cmplt_epi8(index, _mm_set1_epi8(len));
    __m128i key_chars = _mm_and_si128(chars, mask);

    __m128i bin_chars = _mm_loadu_si128((__m128i*)hmap[hash_value].key);
    // WTF THIS CODE ALWAYS HAS BEEN WRONG FROM THE START HOW DID IT PASSED SO MANY TESTS
    __m128i neq = _mm_xor_si128(bin_chars, key_chars);
    if (likely(_mm_test_all_zeros(neq, neq) || hmap[hash_value].len == 0)) {
      // consistent 2.5% improvement in `user` time by testing first bin before loop
    }
    else {
      hash_value = (hash_value + 1) % NUM_BINS; // previous one failed
      while (hmap[hash_value].len > 0) {
        // SIMD string comparison      
        __m128i bin_chars = _mm_loadu_si128((__m128i*)hmap[hash_value].key);
        __m128i neq = _mm_xor_si128(bin_chars, key_chars);
        if (likely(_mm_test_all_zeros(neq, neq))) break;
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
  stats.min = min(stats.min, value);

  // each key will only be free 1 first time, so it's unlikely
  if (unlikely(hmap[hash_value].len == 0)) {
      hmap[hash_value].len = len;
      memcpy(hmap[hash_value].key, key, len);
      memset(hmap[hash_value].key + len, 0, 100 - len);
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
inline void __attribute__((always_inline)) handle_line(const uint8_t* data, HashBin* hmap, size_t &data_idx)
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
    __m128i index = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m128i separators = _mm_set1_epi8(';');
    __m128i compared = _mm_cmpeq_epi8(chars, separators);
    uint32_t separator_mask = _mm_movemask_epi8(compared);

    if (likely(separator_mask)) pos = __builtin_ctz(separator_mask);

    // sum the 2 halves of 16 characters together, then hash the resulting 8 characters
    // this save 1 _mm256_mullo_epi32 instruction, improving performance by ~3%
    //__m128i mask = _mm_loadu_si128((__m128i*)(strcmp_mask + 16 - pos));
    __m128i mask = _mm_cmplt_epi8(index, _mm_set1_epi8(pos));
    __m128i key_chars = _mm_and_si128(chars, mask);
    __m128i sumchars = _mm_add_epi8(key_chars, _mm_srli_si128(key_chars, 8));
        
    // It's not illegal to dereference __m128i, yay. 0.3% faster than memcpy
    // Maybe it's just noise, but I measure best time instead of average FOR THIS CONTEST, so every millisecond counts.
    // https://stackoverflow.com/questions/52112605/is-reinterpret-casting-between-hardware-simd-vector-pointer-and-the-correspond
    //myhash = (*(reinterpret_cast<uint64_t*>(&sumchars)) * SMALL) >> 20;

    // okay so it was actually technically illegal. Am I stupid?
    myhash = (uint64_t(_mm_cvtsi128_si64(sumchars)) * SMALL) >> 20;

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
  
  
  int value;
  if constexpr(SAFE_HASH) {
    if (data[pos + 2] == '.') value = 10 * (data[pos + 1] - 48) + (data[pos + 3] - 48); // 9.1
    else value = 100 * (data[pos + 1] - 48) + 10 * (data[pos + 2] - 48) + (data[pos + 4] - 48); // 92.1    
  } else {
    // PhD code from curiouscoding.nl 
    memcpy(&value, data + pos + 1, 4);
    value <<= 8 * (data[pos + 2] == '.');
    constexpr uint64_t C = 1 + (10 << 16) + (100 << 24); // holy hell
    value &= 0x0f000f0f; // new mask just dropped
    value = ((value * C) >> 24) & ((1 << 10) - 1); // actual branchless
    value *= sign;
  }

  // intentionally move index updating before hmap_insert
  // to improve register dependency chain
  data_idx += pos + 5 + (data[pos + 3] == '.');
  
  hmap_insert<SAFE_HASH>(hmap, myhash, data, key_end, value);
}

void handle_line_raw(int tid, const uint8_t* data, size_t from_byte, size_t to_byte, size_t file_size)
{
    // if use size_t(tid) * NUM_BINS instead of tid * NUM_BINS, total time becomes 2% slower. Wtf
    hmaps[tid] = global_hmaps + tid * NUM_BINS;

    // use malloc because we don't need to fill key with 0
    for (int i = 0; i < NUM_BINS; i++) {
      hmaps[tid][i].stats.sum = 0;
      hmaps[tid][i].stats.cnt = 0;
      hmaps[tid][i].stats.max = -1024;
      hmaps[tid][i].stats.min = 1024;
      hmaps[tid][i].len = 0;      
    }

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
      stats.min = min(stats.min, bin.stats.min);
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
      stats.min = min(stats.min, value.min);
    }
  }
}

float roundTo1Decimal(float number) {
    return std::round(number * 10.0) / 10.0;
}

volatile int dummy;

int main(int argc, char* argv[])
{
  cout << "Using " << N_THREADS << " threads\n";
  MyTimer timer, timer2;
  timer.startCounter();

  // doing this is faster than letting each thread malloc once
  timer2.startCounter();
  global_hmaps = (HashBin*)memalign(sizeof(HashBin), (size_t)N_THREADS * NUM_BINS * sizeof(HashBin));
  if constexpr(DEBUG) cout << "Malloc cost = " << timer.getCounterMsPrecise() << "\n";

  timer2.startCounter();
  string file_path = "measurements.txt";
  if (argc > 1) file_path = string(argv[1]);

  int fd = open(file_path.c_str(), O_RDONLY);
  struct stat file_stat;
  fstat(fd, &file_stat);
  size_t file_size = file_stat.st_size;

  void* mapped_data_void = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);

  const uint8_t* data = reinterpret_cast<uint8_t*>(mapped_data_void);
  if constexpr(DEBUG) cout << "init mmap file cost = " << timer2.getCounterMsPrecise() << "ms\n";
  
  //----------------------
  timer2.startCounter();
  size_t idx = 0;
  int n_threads = N_THREADS;
  if (file_size / n_threads < 4 * MAX_KEY_LENGTH) n_threads = 1;
  
  size_t remaining_bytes = file_size - idx;
  size_t bytes_per_thread = remaining_bytes / n_threads + 1;
  vector<size_t> tstart, tend;
  vector<std::thread> threads;
  for (int64_t tid = n_threads - 1; tid >= 0; tid--) {
      size_t starter = idx + tid * bytes_per_thread;
      size_t ender = idx + (tid + 1) * bytes_per_thread;
      if (ender > file_size) ender = file_size;
      if (tid) {
        threads.emplace_back([tid, data, starter, ender, file_size]() {
          handle_line_raw(tid, data, starter, ender, file_size);
        });
      } else handle_line_raw(tid, data, starter, ender, file_size);
  }

  for (auto& thread : threads) thread.join();
  if constexpr(DEBUG) cout << "Parallel process file cost = " << timer.getCounterMsPrecise() << "ms\n";

  //----------------------
  timer2.startCounter();
  if constexpr(N_AGGREGATE > 1) {
    threads.clear();
    for (int tid = 1; tid < N_AGGREGATE; tid++) {
      threads.emplace_back([tid]() {
        parallel_aggregate(tid);
      });
    }
    parallel_aggregate(0);
    for (auto& thread : threads) thread.join();

    //----- parallel reduction again
    threads.clear();
    for (int tid = 1; tid < N_AGGREGATE_LV2; tid++) {
      threads.emplace_back([tid]() {
        parallel_aggregate_lv2(tid);
      });
    }
    parallel_aggregate_lv2(0);
    for (auto& thread : threads) thread.join();
    // now, the stats are aggregated into partial_stats[0 : N_AGGREGATE_LV2]

    for (int tid = 0; tid < N_AGGREGATE_LV2; tid++) {
      for (auto& [key, value] : partial_stats[tid]) {
        auto& stats = final_recorded_stats[key];
        stats.cnt += value.cnt;
        stats.sum += value.sum;
        stats.max = max(stats.max, value.max);
        stats.min = min(stats.min, value.min);
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
          stats.min = min(stats.min, bin.stats.min);
      }
    }
  }
  if constexpr(DEBUG) cout << "Aggregate stats cost = " << timer2.getCounterMsPrecise() << "ms\n";

  timer2.startCounter();
  vector<pair<string, Stats>> results;
  for (auto& [key, value] : final_recorded_stats) {
      results.emplace_back(key, value);
  }
  sort(results.begin(), results.end());

  // {Abha=-37.5/18.0/69.9, Abidjan=-30.0/26.0/78.1,  
  ofstream fo("result_valid14.txt");
  fo << fixed << setprecision(1);
  fo << "{";
  for (size_t i = 0; i < results.size(); i++) {
      const auto& result = results[i];
      const auto& station_name = result.first;
      const auto& stats = result.second;
      float avg = roundTo1Decimal((double)stats.sum / 10.0 / stats.cnt);
      float mymax = roundTo1Decimal(stats.max / 10.0);
      float mymin = roundTo1Decimal(stats.min / 10.0);

      fo << station_name << "=" << mymin << "/" << avg << "/" << mymax;
      if (i < results.size() - 1) fo << ", ";
  }
  fo << "}\n";
  fo.close();
  if constexpr(DEBUG) cout << "Output stats cost = " << timer2.getCounterMsPrecise() << "ms\n";

  if constexpr(DEBUG) cout << "Runtime inside main = " << timer.getCounterMsPrecise() << "ms\n";

  timer.startCounter();
  munmap(mapped_data_void, file_size);
  if constexpr(DEBUG) cout << "Time to munmap = " << timer.getCounterMsPrecise() << "\n";

  timer.startCounter();  
  free(global_hmaps);
  if constexpr(DEBUG) cout << "Time to free memory = " << timer.getCounterMsPrecise() << "\n";
  return 0;
}

// Use safe instruction instead of trying to deref uint64_t* pointing to a __m128i*
// Using 32 threads
// Malloc cost = 0.006663
// init mmap file cost = 0.01609ms
// Parallel process file cost = 458.101ms
// Aggregate stats cost = 1.73987ms
// Output stats cost = 0.716246ms
// Runtime inside main = 460.619ms
// Time to munmap = 151.386
// Time to free memory = 4.24478
// real    0m0.619s
// user    0m13.435s
// sys     0m0.843s

// Using 32 threads
// Malloc cost = 0.00514
// init mmap file cost = 0.013185ms
// Parallel process file cost = 457.532ms
// Aggregate stats cost = 1.90608ms
// Output stats cost = 0.715004ms
// Runtime inside main = 460.216ms
// Time to munmap = 150.891
// Time to free memory = 4.20632

// real    0m0.618s
// user    0m13.414s
// sys     0m0.883s

// Using 32 threads
// Malloc cost = 0.005971
// init mmap file cost = 0.009929ms
// Parallel process file cost = 454.998ms
// Aggregate stats cost = 1.87568ms
// Output stats cost = 1.32216ms
// Runtime inside main = 458.27ms
// Time to munmap = 153.662
// Time to free memory = 4.24626

// real    0m0.619s
// user    0m13.495s
// sys     0m0.813s