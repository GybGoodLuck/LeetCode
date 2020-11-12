#include <iostream>
#include <vector>

using namespace std;

class QuickSort {

public:
    static int quickSort(int* arr, int left, int right) {

        if (left >= right) {

            for (int i = 0; i < 9; i++) {
                cout << arr[i] << " ";
            }
            cout << endl;
            return 1;
        }

        int base = arr[left];

        int m = left;
        int n = right;

        while (n > m) {

            while (n > m && arr[n] > base) {
                n--;
            }

            while (n > m && arr[m] <= base) {
                m++;
            }

            if (n > m) {
                int temp = arr[n];
                arr[n] = arr[m];
                arr[m] = temp;
            }
        }

        arr[left] = arr[m];
        arr[m] = base;
        
        quickSort(arr, left, m - 1);
        quickSort(arr, m + 1, right);

        return 0;
    }

};