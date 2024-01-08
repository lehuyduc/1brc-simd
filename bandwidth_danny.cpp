
#include <random>
#include <chrono>
#include <iostream>
#include <future>

using uint = unsigned int;
using ulong = unsigned long;

// Size of data buffer used to test.
const std::size_t N = 1'000'000'000;
//uint data[N];
uint* data;
size_t sizeof_data = N * sizeof(uint);

// Worker function.  Sums a chunk of memory.
unsigned int
sum(const uint *const begin, const uint *const end) {

    uint sum = 0;
    for (const uint *p = begin; p < end; p++) {
        sum += *p;
    }

    return sum;
}

// Wrapper function that spawns threads and times them.
void
time(int n_threads) {

    std::vector<std::future<uint>> futures;

    // Make it a double because it might not divide evenly.
    double chunk_size = double(N)/n_threads;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_threads; i++) {
        futures.push_back(std::async(sum, data + ulong(i*chunk_size), data + ulong((i + 1)*chunk_size)));
    }

    // Add up all the individual sums.
    uint sum = 0;
    for (auto &f : futures) {
        sum += f.get();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> secs = stop - start;
    std::cerr << "    " << sizeof_data / secs.count() << std::endl;

    std::cerr << "    To prevent optimizing out all ops: " << sum << std::endl;
}

int
main() {
    data = new uint[N];
    /*
     * Fill with some random numbers.  PRNGs are slow, though,
     * so mostly just use the index.
     */

    std::default_random_engine eng;
    std::uniform_int_distribution<uint> dist(0, 0xffffffffU);

    for (std::size_t i = 0; i < 1'000'000'000; i++) {
        // Only set every 1000 numbers to a random number.
        if (i%1000 == 0) {
            data[i] = dist(eng);
        } else {
            data[i] = i;
        }
    }

    /*
     * Now do the timing.
     */

    for (int i = 1; i < 32; i++) {
        std::cerr << i << " thread(s):" << std::endl;
        time(i);
    }
}

//
// 1 thread(s):
//     1.21548e+10
//     To prevent optimizing out all ops: 2268145617
// 2 thread(s):
//     2.25055e+10
//     To prevent optimizing out all ops: 2268145617
// 3 thread(s):
//     3.13692e+10
//     To prevent optimizing out all ops: 2268145617
// 4 thread(s):
//     3.83841e+10
//     To prevent optimizing out all ops: 2268145617
// 5 thread(s):
//     4.18454e+10
//     To prevent optimizing out all ops: 2268145617
// 6 thread(s):
//     4.12414e+10
//     To prevent optimizing out all ops: 2268145617
// 7 thread(s):
//     4.57211e+10
//     To prevent optimizing out all ops: 2268145617
// 8 thread(s):
//     5.02745e+10
//     To prevent optimizing out all ops: 2268145617
// 9 thread(s):
//     4.53361e+10
//     To prevent optimizing out all ops: 2268145617
// 10 thread(s):
//     4.44753e+10
//     To prevent optimizing out all ops: 2268145617
// 11 thread(s):
//     4.92731e+10
//     To prevent optimizing out all ops: 2268145617
// 12 thread(s):
//     5.17069e+10
//     To prevent optimizing out all ops: 2268145617
// 13 thread(s):
//     4.72197e+10
//     To prevent optimizing out all ops: 2268145617
// 14 thread(s):
//     4.65148e+10
//     To prevent optimizing out all ops: 2268145617
// 15 thread(s):
//     5.00228e+10
//     To prevent optimizing out all ops: 2268145617
// 16 thread(s):
//     5.15139e+10
//     To prevent optimizing out all ops: 2268145617
// 17 thread(s):
//     4.73933e+10
//     To prevent optimizing out all ops: 2268145617
// 18 thread(s):
//     4.7023e+10
//     To prevent optimizing out all ops: 2268145617
// 19 thread(s):
//     4.26685e+10
//     To prevent optimizing out all ops: 2268145617
// 20 thread(s):
//     5.10202e+10
//     To prevent optimizing out all ops: 2268145617
// 21 thread(s):
//     4.85223e+10
//     To prevent optimizing out all ops: 2268145617
// 22 thread(s):
//     4.7792e+10
//     To prevent optimizing out all ops: 2268145617
// 23 thread(s):
//     4.94533e+10
//     To prevent optimizing out all ops: 2268145617
// 24 thread(s):
//     5.10361e+10
//     To prevent optimizing out all ops: 2268145617
// 25 thread(s):
//     4.8397e+10
//     To prevent optimizing out all ops: 2268145617
// 26 thread(s):
//     4.77104e+10
//     To prevent optimizing out all ops: 2268145617
// 27 thread(s):
//     4.92636e+10
//     To prevent optimizing out all ops: 2268145617
// 28 thread(s):
//     5.02862e+10
//     To prevent optimizing out all ops: 2268145617
// 29 thread(s):
//     4.83424e+10
//     To prevent optimizing out all ops: 1268145618
// 30 thread(s):
//     4.88658e+10
//     To prevent optimizing out all ops: 2268145617
// 31 thread(s):
//     4.93164e+10
//     To prevent optimizing out all ops: 2268145617
