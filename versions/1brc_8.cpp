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
using namespace std;

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)


const string FILE_PATH = "measurements.txt";
// 67153 14
// 779347 15
uint32_t SMALL = 779347;

struct Stats {
    int cnt;
    float sum;
    float max;
    float min;

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

unordered_map<uint32_t, string> hash_names;
constexpr int HASH_TRUNCATE = 15;
constexpr int N_THREADS = 32;
constexpr int HASH_TABLE_SIZE = (1 << (32 - HASH_TRUNCATE)) + 1;
alignas(4096) Stats* thread_recorded_stats[N_THREADS];
alignas(4096) Stats final_recorded_stats[HASH_TABLE_SIZE];

alignas(4096) uint32_t pow_small[64];

void init_pow_small() {
  uint32_t b[40];
  b[0] = 1;
  for (int i = 1; i <= 32; i++) b[i] = b[i - 1] * SMALL;

  for (int i = 0; i < 32; i++) pow_small[i] = b[31 - i];
  for (int i = 32; i < 64; i++) pow_small[i] = 0;
}

// https://en.algorithmica.org/hpc/simd/reduction/
uint32_t hsum(__m256i x) {
    __m128i l = _mm256_extracti128_si256(x, 0);
    __m128i h = _mm256_extracti128_si256(x, 1);
    l = _mm_add_epi32(l, h);
    l = _mm_hadd_epi32(l, l);
    return (uint32_t)_mm_extract_epi32(l, 0) + (uint32_t)_mm_extract_epi32(l, 1);
}

template <bool STORE_NAME = false, bool SAFE_HASH = false>
inline int handle_line(const uint8_t* data, Stats* my_recorded_stats)
{
    int pos;
    uint32_t myhash;

    // we read 16 bytes at a time with SIMD, so if the final line has < 16 bytes,
    // this cause out-of-bound read.
    // Most of the time it doesn't cause any error, but if the last extra bytes are past
    // the final memory page provided by mmap, it will cause SIGBUS.
    // So for the last few lines, we use safe code.
    if constexpr(SAFE_HASH) {
        pos = 0;
        myhash = 0;
        while (data[pos] != ';') {
            myhash = myhash * SMALL + data[pos];
            pos++;
        }
        myhash >>= HASH_TRUNCATE;        
    } else {
        __m128i separators = _mm_set1_epi8(';');
        __m128i chars = _mm_loadu_si128((__m128i*)data);
        __m128i compared = _mm_cmpeq_epi8(chars, separators);
        uint32_t separator_mask = _mm_movemask_epi8(compared);

        if (likely(separator_mask)) {
            pos = __builtin_ctz(separator_mask);
            uint32_t pow_start = 32 - pos;
                        
            __m256i pow_vec1 = _mm256_loadu_si256((__m256i*)(pow_small + pow_start));
            __m256i data_vec1 = _mm256_cvtepu8_epi32(chars);
            __m256i summer1 = _mm256_mullo_epi32(pow_vec1, data_vec1);

            __m256i pow_vec2 = _mm256_loadu_si256((__m256i*)(pow_small + pow_start + 8));
            __m256i data_vec2 = _mm256_cvtepu8_epi32(_mm_srli_si128(chars, 8));
            __m256i summer2 = _mm256_mullo_epi32(pow_vec2, data_vec2);

            __m256i summer = _mm256_add_epi32(summer1, summer2);
            myhash = hsum(summer);     
        } else {
            pos = 0;
            myhash = 0;
            while (data[pos] != ';') {
                myhash = myhash * SMALL + data[pos];
                pos++;
            }            
        }
     
        myhash >>= HASH_TRUNCATE;
    }

    if constexpr(STORE_NAME) {        
        hash_names[myhash] = string(data, data + pos);
    }

    // data[pos] = ';'.
    // There are 4 cases: ;9.1, ;92.1, ;-9.1, ;-92.1
    pos += (data[pos + 1] == '-'); // after this, data[pos] = position right before first digit
    bool negative = data[pos] == '-';
    float case1 = (data[pos + 1] - 48) + 0.1f * (data[pos + 3] - 48); // 9.1
    float case2 = (10 * (data[pos + 1] - 48) + (data[pos + 2] - 48)) + 0.1f * (data[pos + 4] - 48); // 92.1
    float value = (data[pos + 2] == '.') ? case1 : case2;
    if (negative) value = -value;

    auto& stats = my_recorded_stats[myhash];
    stats.cnt++;
    stats.sum += value;
    stats.max = max(stats.max, value);
    stats.min = min(stats.min, value);

    return pos + 3 + (data[pos + 3] == '.') + 1 + 1;
}

void handle_line_raw(int tid, const uint8_t* data, size_t from_byte, size_t to_byte, size_t file_size)
{
    if (tid != 0) thread_recorded_stats[tid] = new Stats[HASH_TABLE_SIZE];

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

    if (tid == N_THREADS - 1) to_byte -= 100;

    while (idx < to_byte) {
        idx += handle_line<false>(data + idx, thread_recorded_stats[tid]);
    }

    if (tid == N_THREADS - 1) {
        while (idx < file_size) {
            idx += handle_line<false, true>(data + idx, thread_recorded_stats[tid]);
        }
    }
}

float roundTo1Decimal(float number) {
    return std::round(number * 10.0) / 10.0;
}

int main()
{
    MyTimer timer;
    timer.startCounter();
    init_pow_small();
    
    int fd = open(FILE_PATH.c_str(), O_RDONLY);
    struct stat file_stat;
    fstat(fd, &file_stat);
    size_t file_size = file_stat.st_size;

    void* mapped_data_void = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);

    const uint8_t *data = reinterpret_cast<uint8_t*>(mapped_data_void);
    //----------------------
    
    thread_recorded_stats[0] = new Stats[HASH_TABLE_SIZE];

    size_t idx = 0;
    constexpr size_t SMALL_LIMIT = 30'000'000;
    bool small_file = file_size <= SMALL_LIMIT;
    size_t build_hash_name_size = min(file_size, SMALL_LIMIT);

    if (file_size < SMALL_LIMIT + 100) {
        while (idx < build_hash_name_size)
            idx += handle_line<true, true>(data + idx, thread_recorded_stats[0]);
    } else {
        while (idx < build_hash_name_size) {
            idx += handle_line<true, false>(data + idx, thread_recorded_stats[0]);
        }
    }

    if (!small_file) {
        size_t remaining_bytes = file_size - idx;
        size_t bytes_per_thread = remaining_bytes / N_THREADS + 1;
        vector<size_t> tstart, tend;
        vector<std::thread> threads;    
        for (size_t tid = 0; tid < N_THREADS; tid++) {
            size_t starter = idx + tid * bytes_per_thread;
            size_t ender = idx + (tid + 1) * bytes_per_thread;
            if (ender > file_size) ender = file_size;
            threads.emplace_back([tid, data, starter, ender, file_size]() {
                handle_line_raw(tid, data, starter, ender, file_size);
            });
        }

        for (auto& thread : threads) thread.join();
    }


    //----------------------
    int n_threads = N_THREADS;
    if (small_file) n_threads = 1;
    for (auto &[key, value] : hash_names) {        
        for (int tid = 0; tid < n_threads; tid++) {
            final_recorded_stats[key].cnt += thread_recorded_stats[tid][key].cnt;
            final_recorded_stats[key].sum += thread_recorded_stats[tid][key].sum;
            final_recorded_stats[key].min = min(final_recorded_stats[key].min, thread_recorded_stats[tid][key].min);
            final_recorded_stats[key].max = max(final_recorded_stats[key].max, thread_recorded_stats[tid][key].max);
        }
    }


    vector<pair<string, Stats>> results;
    for (auto &[key, value] : hash_names) {
        results.emplace_back(value, final_recorded_stats[key]);
    }
    sort(results.begin(), results.end());

    // {Abha=-37.5/18.0/69.9, Abidjan=-30.0/26.0/78.1,
    ofstream fo("result_assume.txt");
    fo << fixed << setprecision(1);
    fo << "{";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& result = results[i];
        string station_name = result.first;
        Stats stats = result.second;
        float avg = roundTo1Decimal(stats.sum / stats.cnt);
        float mymax = roundTo1Decimal(stats.max);
        float mymin = roundTo1Decimal(stats.min);

        fo << station_name << "=" << mymin << "/" << avg << "/" << mymax;
        if (i < results.size() - 1) fo << ", ";        
    }
    fo << "}";
    fo.close();

    cout << "Runtime inside main = " << timer.getCounterMsPrecise() << "ms\n";
    return 0;
}

