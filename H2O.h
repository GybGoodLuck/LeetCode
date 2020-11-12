#include <atomic>
#include <thread>

class H2O {
    std::atomic<int> h2;
public:
    H2O() {
        h2=0;
    }

    void hydrogen() {
        
        while(h2.load()>1) std::this_thread::yield();
        h2++;
    }

    void oxygen() {
        
        while(h2.load()!=2) std::this_thread::yield();
        h2.store(0);
    }
};
