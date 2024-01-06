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

unordered_map<string, Stats> recorded_stats;
unordered_map<string, uint32_t> hashed_name;
int len_cnt[1000];

inline int handle_line(const uint8_t* data)
{
    int pos = 0;
    while (data[pos] != ';') pos++;
    string station_name = string(data, data + pos);

    int num_start = pos + 1;
    pos++;
    while (data[pos] != '\n') pos++;
    string value_str = string(data + num_start, data + pos);
    float value = std::stof(value_str);
    
    auto& stats = recorded_stats[station_name];
    stats.cnt++;
    stats.sum += value;
    stats.max = max(stats.max, value);
    stats.min = min(stats.min, value);

    len_cnt[station_name.length()]++;
    uint32_t myhash = 0;
    for (size_t i = 0; i < station_name.length(); i++) {
        myhash = myhash * SMALL + station_name[i];
    }
    myhash >>= 8;

    if (hashed_name.find(station_name) == hashed_name.end()) {
        hashed_name[station_name] = myhash;
    } else if (hashed_name[station_name] != myhash) {
        cout << "ERROR COLLISION: " << myhash << " " << hashed_name[station_name] << " " << station_name << "\n";
        exit(1);
    }

    return pos + 1;
}

float roundTo1Decimal(float number) {
    return std::round(number * 10.0) / 10.0;
}

int main(int argc, char* argv[])
{
    string file_path = "measurements.txt";
    if (argc > 1) file_path = string(argv[1]);

    int fd = open(file_path.c_str(), O_RDONLY);
    struct stat file_stat;
    fstat(fd, &file_stat);
    size_t file_size = file_stat.st_size;

    void* mapped_data_void = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);

    const uint8_t *data = reinterpret_cast<uint8_t*>(mapped_data_void);

    size_t idx = 0;
    while (idx < file_size) {
        idx += handle_line(data + idx);        
    }

    vector<pair<string, Stats>> results;
    for (auto &[key, value] : recorded_stats) {
        results.emplace_back(key, value);
    }
    sort(results.begin(), results.end());

    ofstream fo("result1.txt");
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
    
    cout << "station count = " << recorded_stats.size() << "\n";
    for (int i = 0; i < 1000; i++) if (len_cnt[i] > 0) cout << i << " " << len_cnt[i] << "\n";
    return 0;
}
