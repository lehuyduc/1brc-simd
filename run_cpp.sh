rm main
g++ -o main 1brc_final_valid.cpp -O3 -std=c++17 -march=native -m64 -lpthread
#g++ -S -o main.s 1brc_final_valid.cpp -O3 -std=c++17 -march=native -fverbose-asm -g -lpthread
#g++ -o test_copy_bandwidth copy_bandwidth.cpp -O3 -march=native -m64 -lpthread
time ./main measurements.txt
