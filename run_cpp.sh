# Check if an input parameter is provided
if [ -n "$1" ]; then
    # Use the provided input parameter as the number of threads
    num_threads="$1"
else
    # Use nproc to get the number of CPU threads
    num_threads=$(nproc --all)
fi

rm -f main
rm -f result.txt
g++ -o main 1brc_final_valid.cpp -O3 -std=c++17 -march=native -m64 -lpthread -DN_THREADS_PARAM=$num_threads
time ./main measurements.txt
sha256sum result.txt