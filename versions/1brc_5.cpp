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
using namespace std;

const string FILE_PATH = "measurements.txt";
uint32_t SMALL = 113;

template <typename T>
void print_m256i(__m256i data)
{
  alignas(32) T arr[32 / sizeof(T)];
  _mm256_store_si256((__m256i*)arr, data);
  for (int i = 0; i < 16 / sizeof(T); i++) cout << uint64_t(arr[i]) << " ";
  cout << "\n";
}

struct Stats {
    int cnt;
    float sum;
    float max;
    float min;

    Stats() {
        max = -1024;
        min = 1024;
        sum = 0;
        cnt = 0;
    }

    bool operator < (const Stats& other) const {
        return min < other.min;
    }
};

unordered_map<uint32_t, string> hash_names;
constexpr int HASH_TRUNCATE = 11;
constexpr int N_THREADS = 32;
constexpr int HASH_TABLE_SIZE = (1 << (32 - HASH_TRUNCATE)) + 4096;
alignas(4096) Stats thread_recorded_stats[N_THREADS][HASH_TABLE_SIZE];
alignas(4096) Stats final_recorded_stats[HASH_TABLE_SIZE];
//unordered_map<uint32_t, Stats> recorded_stats;
int len_cnt[1000];

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

alignas(4096) const uint32_t pow_small[64] = {
    1796972177, 2258407457, 3858850481, 490251841, 308406993, 4183670881, 3495802609, 3983826561, 757417745, 1375010977, 620305201, 3882362561, 2770973521, 4167454945, 3457650545, 3147300609,
    1966288785, 4122325281, 3875345329, 3683116865, 3909467089, 870785377, 4226656241, 3116097409, 3714406417, 3187581345, 1244482609, 163047361, 1442897, 12769, 113, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

uint32_t hsum(__m256i x) {
    __m128i l = _mm256_extracti128_si256(x, 0);
    __m128i h = _mm256_extracti128_si256(x, 1);
    l = _mm_add_epi32(l, h);
    l = _mm_hadd_epi32(l, l);
    return (uint32_t)_mm_extract_epi32(l, 0) + (uint32_t)_mm_extract_epi32(l, 1);
}

bool kt = false;
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

    // data[pos] = ';', ;9.1, ;92.1, ;-9.1, ;-92.1
    pos += (data[pos + 1] == '-');
    bool negative = data[pos] == '-';
    float case1 = (data[pos + 1] - 48) + 0.1f * (data[pos + 3] - 48);
    float case2 = (10 * (data[pos + 1] - 48) + (data[pos + 2] - 48)) + 0.1f * (data[pos + 4] - 48);
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
    size_t idx = from_byte;
    while (data[idx] != '\n') idx++;
    idx++;

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
    int fd = open(FILE_PATH.c_str(), O_RDONLY);
    struct stat file_stat;
    fstat(fd, &file_stat);
    size_t file_size = file_stat.st_size;

    void* mapped_data_void = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);

    const uint8_t *data = reinterpret_cast<uint8_t*>(mapped_data_void);
    cout << "Cost 0 = " << timer.getCounterMsPrecise() << "\n";

    timer.startCounter();
    size_t idx = 0;
    int l = 0;
    while (l < 1'000'000) {
        idx += handle_line<true>(data + idx, thread_recorded_stats[0]);
        l++;
    }
    cout << "Cost 1 = " << timer.getCounterMsPrecise() << "\n";
    cout << "Number of keys = " << hash_names.size() << "\n";

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
    cout << "Cost 2 = " << timer.getCounterMsPrecise() << "\n";

    timer.startCounter();
    for (auto &[key, value] : hash_names) {
        for (int tid = 0; tid < N_THREADS; tid++) {
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
    ofstream fo("result5.txt");
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
    cout << "Cost 3 = " << timer.getCounterMsPrecise() << "\n";
    return 0;
}
