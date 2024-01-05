rm main
g++ -o main 1brc_final.cpp -O3 -std=c++17 -march=native -mavx2 -m64 -ffast-math
#g++ -S -o main.s 1brc_final.cpp -O3 -std=c++17 -march=native -mavx2  -fverbose-asm -g
time ./main
