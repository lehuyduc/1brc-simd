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
#include <sys/types.h>
#include <sys/wait.h>
#include <malloc.h>
using namespace std;

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

constexpr uint32_t SMALL = 749449;
constexpr uint32_t SHL_CONST = 18;
constexpr int MAX_KEY_LENGTH = 100;
constexpr uint32_t NUM_BINS = 16384 * 8; // for 10k key cases. For 413 key cases, 16384 is enough.

#ifndef N_THREADS_PARAM
constexpr int MAX_N_THREADS = 8; // to match evaluation server
#else
constexpr int MAX_N_THREADS = N_THREADS_PARAM;
#endif

#ifndef N_CORES_PARAM
constexpr int N_CORES = MAX_N_THREADS;
#else
constexpr int N_CORES = N_CORES_PARAM;
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
    uint8_t key_short[32]; // std::string Short String Optimization, pogchamp
    int64_t sum;
    int cnt;
    int max;
    int min;    
    int len;
    uint8_t* key_long;

    HashBin() {
      // C++ zero-initialize global variable by default
      len = 0;
      memset(key_short, 0, sizeof(key_short));
      key_long = nullptr;
    }
};
static_assert(sizeof(HashBin) == 64); // faster array indexing if struct is power of 2

std::unordered_map<string, Stats> partial_stats[MAX_N_THREADS];
std::unordered_map<string, Stats> final_recorded_stats;

HashBin* global_hmaps;
alignas(4096) HashBin* hmaps[MAX_N_THREADS];

alignas(4096) uint8_t strcmp_mask32[64] = {
  255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
};

inline int __attribute__((always_inline)) mm128i_equal(__m128i a, __m128i b)
{
  __m128i neq = _mm_xor_si128(a, b);
  return _mm_test_all_zeros(neq, neq);
}

inline int __attribute__((always_inline)) mm256i_equal(__m256i a, __m256i b)
{
  __m256i neq = _mm256_xor_si256(a, b);
  return _mm256_testz_si256(neq, neq);
}

inline void __attribute__((always_inline)) handle_line(const uint8_t* data, HashBin* hmap, size_t &data_idx)
{
  uint32_t pos = 16;
  uint32_t myhash;

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
  __m128i sumchars = _mm_add_epi8(key_chars, _mm_unpackhi_epi64(key_chars, key_chars)); // 0.7% faster total program time compared to srli

  // okay so it was actually technically illegal. Am I stupid?
  myhash = (uint64_t(_mm_cvtsi128_si64(sumchars)) * SMALL) >> SHL_CONST;

  if (unlikely(data[pos] != ';')) {
    __m256i chars32_1 = _mm256_loadu_si256((__m256i*)(data + 17));
    __m256i chars32_2 = _mm256_loadu_si256((__m256i*)(data + 17 + 32));
    __m256i separators32 = _mm256_set1_epi8(';');

    __m256i compared32_1 = _mm256_cmpeq_epi8(chars32_1, separators32);
    __m256i compared32_2 = _mm256_cmpeq_epi8(chars32_2, separators32);

    uint64_t separator_mask64 = uint64_t(_mm256_movemask_epi8(compared32_1)) | (uint64_t(_mm256_movemask_epi8(compared32_2)) << 32);
    if (likely(separator_mask64)) pos = 17 + __builtin_ctzll(separator_mask64);      
    else {
      __m256i chars32_3 = _mm256_loadu_si256((__m256i*)(data + 81));
      __m256i compared32_3 = _mm256_cmpeq_epi8(chars32_3, separators32);
      uint32_t separator_mask_final = _mm256_movemask_epi8(compared32_3);
      pos = 81 + __builtin_ctz(separator_mask_final);      
    }
  }

  PARSE_VALUE:
  // DO NOT MOVE HASH TABLE PROBE TO BEFORE VALUE PARSING
  // IT'S LIKE 5% SLOWER
  // data[pos] = ';'.
  // There are 4 cases: ;9.1, ;92.1, ;-9.1, ;-92.1
  int len = pos;
  pos += (data[pos + 1] == '-'); // after this, data[pos] = position right before first digit
  int sign = (data[pos] == '-') ? -1 : 1;
  myhash %= NUM_BINS; // let pos be computed first beacause it's needed earlier  

  // PhD code from curiouscoding.nl. Must use uint32_t else undefined behavior
  uint32_t uvalue;
  memcpy(&uvalue, data + pos + 1, 4);
  uvalue <<= 8 * (data[pos + 2] == '.');
  constexpr uint64_t C = 1 + (10 << 16) + (100 << 24); // holy hell
  uvalue &= 0x0f000f0f; // new mask just dropped
  uvalue = ((uvalue * C) >> 24) & ((1 << 10) - 1); // actual branchless
  int value = int(uvalue) * sign;

  // intentionally move index updating before hmap_insert
  // to improve register dependency chain
  data_idx += pos + 5 + (data[pos + 3] == '.');

  if (likely(len <= 16)) {
    // loading everything and calculate twice is consistently 1% faster than just
    // using old result (comment all 4 lines)
    // Keep all these 4 lines is faster than commenting any of them, even though
    // all 4 variables were already calculated before. HUH???
    __m128i chars = _mm_loadu_si128((__m128i*)data);
    __m128i index = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m128i mask = _mm_cmplt_epi8(index, _mm_set1_epi8(len));
    __m128i key_chars = _mm_and_si128(chars, mask);

    __m128i bin_chars = _mm_load_si128((__m128i*)hmap[myhash].key_short);
    __m128i neq = _mm_xor_si128(bin_chars, key_chars);
    if (likely(_mm_test_all_zeros(neq, neq) || hmap[myhash].len == 0)) {
      // consistent 2.5% improvement in `user` time by testing first bin before loop
    }
    else {
      myhash = (myhash + 1) % NUM_BINS; // previous one failed
      while (hmap[myhash].len > 0) {
        // SIMD string comparison      
        __m128i bin_chars = _mm_load_si128((__m128i*)hmap[myhash].key_short);
        if (likely(mm128i_equal(key_chars, bin_chars))) break;        
        myhash = (myhash + 1) % NUM_BINS;    
      }
    }
  }
  else if (likely(len <= 32 && hmap[myhash].len != 0)) {
    __m256i chars = _mm256_loadu_si256((__m256i*)data);
    __m256i mask = _mm256_loadu_si256((__m256i*)(strcmp_mask32 + 32 - len));
    __m256i key_chars = _mm256_and_si256(chars, mask);

    __m256i bin_chars = _mm256_load_si256((__m256i*)hmap[myhash].key_short);
    if (likely(mm256i_equal(key_chars, bin_chars))) {}
    else {
      myhash = (myhash + 1) % NUM_BINS;
      while (hmap[myhash].len > 0) {
        // SIMD string comparison      
        __m256i bin_chars = _mm256_load_si256((__m256i*)hmap[myhash].key_short);
        if (likely(mm256i_equal(key_chars, bin_chars))) break;            
        myhash = (myhash + 1) % NUM_BINS;    
      }
    }
  }
  else {
    while (hmap[myhash].len > 0) {
      // check if this slot is mine
      if (likely(hmap[myhash].len == len)) {
        int idx = 0;        
        while (idx + 32 < len) {
          __m256i chars = _mm256_loadu_si256((__m256i*)(data + idx));          
          __m256i bin_chars = _mm256_loadu_si256((__m256i*)(hmap[myhash].key_long + idx));
          if (unlikely(!mm256i_equal(chars, bin_chars))) goto NEXT_LOOP;
          idx += 32;
        }
        
        if (likely(idx <= 64)) {
          __m256i mask = _mm256_loadu_si256((__m256i*)(strcmp_mask32 + 32 - (len - idx)));
          __m256i chars = _mm256_loadu_si256((__m256i*)(data + idx));
          __m256i key_chars = _mm256_and_si256(chars, mask);
          __m256i bin_chars = _mm256_loadu_si256((__m256i*)(hmap[myhash].key_long + idx));
          if (likely(mm256i_equal(key_chars, bin_chars))) break;
        } else {
          // len must be >= 97
          bool equal = true;
          for (int i = idx; i < len; i++) if (data[i] != hmap[myhash].key_long[i]) {
              equal = false;
              break;
          }
          if (likely(equal)) break;
        }
      }
      NEXT_LOOP:
      myhash = (myhash + 1) % NUM_BINS;
    }
  }
  
  hmap[myhash].cnt++;
  hmap[myhash].sum += value;
  if (unlikely(hmap[myhash].max < value)) hmap[myhash].max = value;
  if (unlikely(hmap[myhash].min > value)) hmap[myhash].min = value;  

  // each key will only be free 1 first time, so it's unlikely
  if (unlikely(hmap[myhash].len == 0)) {
      hmap[myhash].len = len;      
      hmap[myhash].sum = value;
      hmap[myhash].cnt = 1;
      hmap[myhash].max = value;
      hmap[myhash].min = value;
      if (len <= 32) {
        memcpy(hmap[myhash].key_short, data, len);
        memset(hmap[myhash].key_short + len, 0, 32 - len);
      } else {
        hmap[myhash].key_long = new uint8_t[MAX_KEY_LENGTH];
        memcpy(hmap[myhash].key_long, data, len);
        memset(hmap[myhash].key_long + len, 0, MAX_KEY_LENGTH - len);
      }
  }
}

void find_next_line_start(const uint8_t* data, size_t N, size_t &idx)
{
  if (idx == 0) return;
  while (idx < N && data[idx - 1] != '\n') idx++;
}

size_t handle_line_raw(int tid, const uint8_t* data, size_t from_byte, size_t to_byte, size_t file_size, bool inited)
{
  if (!inited) {
    hmaps[tid] = global_hmaps + tid * NUM_BINS;
    // use malloc because we don't need to fill key with 0
    for (int i = 0; i < NUM_BINS; i++) hmaps[tid][i].len = 0;
  }

  size_t start_idx = from_byte;
  find_next_line_start(data, to_byte, start_idx);  


  constexpr size_t ILP_LEVEL = 2;
  size_t BYTES_PER_THREAD = (to_byte - start_idx) / ILP_LEVEL;
  size_t idx0 = start_idx;
  size_t idx1 = start_idx + BYTES_PER_THREAD * 1;
  
  find_next_line_start(data, to_byte, idx1);  

  size_t end_idx0 = idx1 - 1;
  size_t end_idx1 = to_byte;

  while (likely(idx0 < end_idx0 && idx1 < end_idx1)) {
    handle_line(data + idx0, hmaps[tid], idx0);
    handle_line(data + idx1, hmaps[tid], idx1);    
  }

  while (idx0 < end_idx0) {
    handle_line(data + idx0, hmaps[tid], idx0);
  }
  while (idx1 < end_idx1) {
    handle_line(data + idx1, hmaps[tid], idx1);
  }

  return idx1; // return the beginning of the first line of the next block
}

void parallel_aggregate(int tid, int n_threads, int n_aggregate)
{
  for (int idx = tid; idx < n_threads; idx += n_aggregate) {
    for (int h = 0; h < NUM_BINS; h++) if (hmaps[idx][h].len > 0) {
      auto& bin = hmaps[idx][h];
      string key;
      if (bin.len <= 32) key = string(bin.key_short, bin.key_short + bin.len);
      else key = string(bin.key_long, bin.key_long + bin.len);
      auto& stats = partial_stats[tid][key];
      stats.cnt += bin.cnt;
      stats.sum += bin.sum;
      stats.max = max(stats.max, bin.max);
      stats.min = min(stats.min, bin.min);
    }
  }
}

void parallel_aggregate_lv2(int tid, int n_aggregate, int n_aggregate_lv2)
{
  for (int idx = n_aggregate_lv2 + tid; idx < n_aggregate; idx += n_aggregate_lv2) {
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

int do_everything(int argc, char* argv[])
{
  if constexpr(DEBUG) cout << "Using " << MAX_N_THREADS << " threads\n";
  if constexpr(DEBUG) cout << "PC has " << N_CORES << " physical cores\n";
  MyTimer timer, timer2;
  timer.startCounter();

  // doing this is faster than letting each thread malloc once
  timer2.startCounter();
  global_hmaps = (HashBin*)memalign(sizeof(HashBin), (size_t)MAX_N_THREADS * NUM_BINS * sizeof(HashBin));
  if constexpr(DEBUG) cout << "Malloc cost = " << timer2.getCounterMsPrecise() << "\n";

  timer2.startCounter();
  string file_path = "measurements.txt";
  if (argc > 1) file_path = string(argv[1]);

  int fd = open(file_path.c_str(), O_RDONLY);
  struct stat file_stat;
  fstat(fd, &file_stat);
  size_t file_size = file_stat.st_size;

  void* mapped_data_void = mmap(nullptr, file_size + 128, PROT_READ, MAP_SHARED, fd, 0);

  const uint8_t* data = reinterpret_cast<uint8_t*>(mapped_data_void);
  if constexpr(DEBUG) cout << "init mmap file cost = " << timer2.getCounterMsPrecise() << "ms\n";
  
  //----------------------
  timer2.startCounter();
  size_t idx = 0;

  bool tid0_inited = false;
  int n_threads = MAX_N_THREADS;
  if (file_size > 100'000'000 && MAX_N_THREADS > N_CORES) {
    // when there are too many hash collision, hyper threading will make the program slower
    // due to L3 cache problem.
    // So, we use the first 1MB to gather statistics about the file.
    // If num_unique_keys >= X then use hyper threading (MAX_N_THREADS)
    // Else, use physical cores only (N_CORES)
    // The program still works on all inputs. But this check let it works faster for hard inputs
    tid0_inited = true;
    size_t to_byte = 50'000;
    idx = handle_line_raw(0, data, 0, to_byte, file_size, false);

    int unique_key_cnt = 0;
    for (int h = 0; h < NUM_BINS; h++) if (hmaps[0][h].len > 0) unique_key_cnt++;
    if (unique_key_cnt > 500) n_threads = N_CORES;    
  }

  if constexpr(DEBUG) cout << "n_threads = " << n_threads << "\n";
  if constexpr(DEBUG) cout << "Gather key stats cost = " << timer2.getCounterMsPrecise() << "\n";

  timer2.startCounter();
  size_t remaining_bytes = file_size - idx;
  if (remaining_bytes / n_threads < 4 * MAX_KEY_LENGTH) n_threads = 1;  
  size_t bytes_per_thread = remaining_bytes / n_threads + 1;
  vector<size_t> tstart, tend;
  vector<std::thread> threads;
  for (int64_t tid = n_threads - 1; tid >= 0; tid--) {
      size_t starter = idx + tid * bytes_per_thread;
      size_t ender = idx + (tid + 1) * bytes_per_thread;
      if (ender > file_size) ender = file_size;
      if (tid) {
        threads.emplace_back([tid, data, starter, ender, file_size]() {
          handle_line_raw(tid, data, starter, ender, file_size, false);
        });
      } else handle_line_raw(0, data, starter, ender, file_size, tid0_inited);
  }

  for (auto& thread : threads) thread.join();
  if constexpr(DEBUG) cout << "Parallel process file cost = " << timer2.getCounterMsPrecise() << "ms\n";

  //----------------------
  timer2.startCounter();
  const int N_AGGREGATE = (n_threads >= 16 && n_threads % 4 == 0) ? (n_threads >> 2) : 1;
  const int N_AGGREGATE_LV2 = (N_AGGREGATE >= 32 && N_AGGREGATE % 4 == 0) ? (N_AGGREGATE >> 2) : 1;

  if (N_AGGREGATE > 1) {
    threads.clear();
    for (int tid = 1; tid < N_AGGREGATE; tid++) {
      threads.emplace_back([tid, n_threads, N_AGGREGATE]() {
        parallel_aggregate(tid, n_threads, N_AGGREGATE);
      });
    }
    parallel_aggregate(0, n_threads, N_AGGREGATE);
    for (auto& thread : threads) thread.join();

    //----- parallel reduction again
    threads.clear();
    for (int tid = 1; tid < N_AGGREGATE_LV2; tid++) {
      threads.emplace_back([tid, N_AGGREGATE, N_AGGREGATE_LV2]() {
        parallel_aggregate_lv2(tid, N_AGGREGATE, N_AGGREGATE_LV2);
      });
    }
    parallel_aggregate_lv2(0, N_AGGREGATE, N_AGGREGATE_LV2);
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
          string key;
          if (bin.len <= 32) key = string(bin.key_short, bin.key_short + bin.len);
          else key = string(bin.key_long, bin.key_long + bin.len);   
          auto& stats = final_recorded_stats[key];
          stats.cnt += bin.cnt;
          stats.sum += bin.sum;
          stats.max = max(stats.max, bin.max);
          stats.min = min(stats.min, bin.min);
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
  ofstream fo("result.txt");
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
  kill(getppid(), SIGUSR1);

  timer.startCounter();
  munmap(mapped_data_void, file_size);
  if constexpr(DEBUG) cout << "Time to munmap = " << timer.getCounterMsPrecise() << "\n";

  timer.startCounter();  
  free(global_hmaps);
  if constexpr(DEBUG) cout << "Time to free memory = " << timer.getCounterMsPrecise() << "\n";
  return 0;
}

void handleSignal(int signum) {
    if (signum == SIGUSR1) {
        printf("Received SIGUSR1 signal. Exiting.\n");
        exit(0);
    }
}

int main(int argc, char* argv[])
{
  signal(SIGUSR1, handleSignal);

  int pid = fork();
  int status;
  if (pid == 0) {
    do_everything(argc, argv);
  } else {
    waitpid(pid, &status, 0);
  }
}
