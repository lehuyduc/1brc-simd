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
using namespace std;

const string FILE_PATH = "measurements.txt";
uint32_t SMALL = 113;

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
unordered_map<uint32_t, Stats> recorded_stats;
int len_cnt[1000];

template <bool STORE_NAME = false>
inline int handle_line(const uint8_t* data)
{
    int pos = 1;
    uint32_t myhash = data[0];
    while (data[pos] != ';') {
        myhash = myhash * SMALL + data[pos];
        pos++;
    }
    myhash >>= 8;
    if constexpr(STORE_NAME) {
        hash_names[myhash] = string(data, data + pos);
    }

    // data[pos] = ';', ;9.1, ;92.1, ;-9.1, ;-92.1
    bool negative = data[pos + 1] == '-';
    if (data[pos + 1] == '-') pos++;
    float case1 = (data[pos + 1] - 48) + 0.1f * (data[pos + 3] - 48);
    float case2 = 10 * (data[pos + 1] - 48) + (data[pos + 2] - 48) + 0.1f * (data[pos + 4] - 48);
    float value = (data[pos + 2] == '.') ? case1 : case2;
    if (negative) value = -value;

    auto& stats = recorded_stats[myhash];
    stats.cnt++;
    stats.sum += value;
    stats.max = max(stats.max, value);
    stats.min = min(stats.min, value);

    //cout << data[pos] << "\n";
    return pos + 3 + (data[pos + 3] == '.') + 1 + 1;
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
    int j = 0;
    while (idx < file_size) {
        idx += handle_line<true>(data + idx);
        j++;
        if (j >= 10'000'000) break;
    }
    cout << "Cost 1 = " << timer.getCounterMsPrecise() << "\n";

    timer.startCounter();
    while (idx < file_size) {
        idx += handle_line<false>(data + idx);
    }
    cout << "Cost 2 = " << timer.getCounterMsPrecise() << "\n";

    timer.startCounter();
    vector<pair<string, Stats>> results;
    for (auto &[key, value] : recorded_stats) {
        results.emplace_back(hash_names[key], value);
    }
    sort(results.begin(), results.end());

    // {Abha=-37.5/18.0/69.9, Abidjan=-30.0/26.0/78.1,
    ofstream fo("result2.txt");
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
