#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

int main()
{
    cout << std::fixed << setprecision(9);
    float a = 0;
    for (int i = 1; i <= 1'000'000'000; i++) {        
        a += 1;
        if (a >= 16777215) cout << a << "\n";
        if (round(a) != i) {
            cout << i << " " << a << " " << (int(round(a))) << "\n";
            exit(1);
        }
    }
}