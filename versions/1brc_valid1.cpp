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

constexpr uint32_t SMALL = 779347;
constexpr int MAX_KEY_LENGTH = 100;
constexpr int NUM_BINS = 16384 * 2;

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

struct HashBin {
    int len;
    uint8_t key[MAX_KEY_LENGTH + 1];
    Stats stats;

    HashBin() {
        len = 0;
        memset(key, 0, sizeof(key));        
    }
};

constexpr int N_THREADS = 32;
std::unordered_map<string, Stats> final_recorded_stats;

alignas(4096) uint32_t pow_small[64];

HashBin hmaps[N_THREADS][NUM_BINS];

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

void hmap_insert(HashBin* hmap, uint32_t hash_value, const uint8_t* key, int len, float value)
{
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

    // each key will only be free 1 first time, so it's unlikely
    if (unlikely(hmap[hash_value].len == 0)) {        
        hmap[hash_value].len = len;
        memcpy((char*)hmap[hash_value].key, (char*)key, len);        
    }
       
    auto& stats = hmap[hash_value].stats;
    stats.cnt++;
    stats.sum += value;
    stats.max = max(stats.max, value);
    stats.min = min(stats.min, value);
}

template <bool SAFE_HASH = false>
inline int handle_line(const uint8_t* data, HashBin* hmap)
{
    int pos;
    uint32_t myhash;

    // we read 16 bytes at a time with SIMD, so if the final line has < 16 bytes,
    // this cause out-of-bound read.
    // Most of the time it doesn't cause any error, but if the last extra bytes are past
    // the final memory page provided by mmap, it will cause SIGBUS.
    // So for the last few lines, we use safe code.
    if constexpr (SAFE_HASH) {
        pos = 0;
        myhash = 0;
        while (data[pos] != ';') {
            myhash = myhash * SMALL + data[pos];
            pos++;
        }
        myhash %= NUM_BINS;
    }
    else {
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
        }
        else {
            pos = 0;
            myhash = 0;
            while (data[pos] != ';') {
                myhash = myhash * SMALL + data[pos];
                pos++;
            }
        }

        myhash %= NUM_BINS;
    }

    // data[pos] = ';'.
    // There are 4 cases: ;9.1, ;92.1, ;-9.1, ;-92.1
    int key_end = pos;
    pos += (data[pos + 1] == '-'); // after this, data[pos] = position right before first digit
    bool negative = data[pos] == '-';
    float case1 = (data[pos + 1] - 48) + 0.1f * (data[pos + 3] - 48); // 9.1
    float case2 = (10 * (data[pos + 1] - 48) + (data[pos + 2] - 48)) + 0.1f * (data[pos + 4] - 48); // 92.1
    float value = (data[pos + 2] == '.') ? case1 : case2;
    if (negative) value = -value;
    
    hmap_insert(hmap, myhash, data, key_end, value);

    return pos + 3 + (data[pos + 3] == '.') + 1 + 1;
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
        idx += handle_line<false>(data + idx, hmaps[tid]);
    }

    if (tid == N_THREADS - 1) {
        while (idx < file_size) {
            idx += handle_line<true>(data + idx, hmaps[tid]);
        }
    }
}

float roundTo1Decimal(float number) {
    return std::round(number * 10.0) / 10.0;
}

int main(int argc, char* argv[])
{
    MyTimer timer;
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
    //----------------------

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

    //----------------------
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

    vector<pair<string, Stats>> results;
    for (auto& [key, value] : final_recorded_stats) {
        results.emplace_back(key, value);
    }
    sort(results.begin(), results.end());

    // {Abha=-37.5/18.0/69.9, Abidjan=-30.0/26.0/78.1,
    ofstream fo("result_valid1.txt");
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
    fo << "}\n";
    fo.close();

    cout << "Runtime inside main = " << timer.getCounterMsPrecise() << "ms\n";
    return 0;
}

