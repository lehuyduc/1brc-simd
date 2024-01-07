rm main
g++ -o main 1brc_final_valid.cpp -O3 -std=c++17 -march=native -m64 -lpthread
time ./main measurements.txt
