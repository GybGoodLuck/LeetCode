#include <map>
using namespace std;

class LFUCache {
public:
    LFUCache(int capacity) {
    }
    
    int get(int key) {

        if (m_cacheMap.count(key)) {
            return key;
        }
        return -1;
    }
    
    void put(int key, int value) {

        if (m_cacheMap.find(key) != m_cacheMap.end()) {
            m_cacheMap[key] = value;
        }
    }

private:
    map<int, int> m_cacheMap;

};