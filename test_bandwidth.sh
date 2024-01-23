g++ -o test_copy_bandwidth copy_bandwidth.cpp -O3 -march=native -m64 -lpthread
time ./test_copy_bandwidth

# Result on Dual EPYC 9354
# root@C.8156873:~/1brc-simd$ ./test_copy_bandwidth 
# Time to init data = 3116.87ms
# Bandwidth = 1.06749e+11 byte/s