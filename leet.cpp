#include <iostream>
#include <string.h>
#include <unordered_map>
#include <vector>
#include <stack>
#include <queue>
#include <deque>
#include <limits.h>
#include <memory>
#include <set>
#include <list>
#include <algorithm>

#include "QuickSort.h"
#include "LFUCache.h"
#include "Twitter.h"

using namespace std;

struct Coordinate {
    int x;
    int y;
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct TreeNode {
    int val;
    TreeNode *left;     
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

void gameOfLife(vector<vector<int>>& board) {
    int M = board[0].size(); // 行
    int N = board.size();    // 列

    int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            
            int num = 0;
            for (int k = 0; k < 8; k++) {
                int nx = j + dx[k];
                int ny = i + dy[k];

                if (nx >=0 && nx < M && ny >=0 && ny < N) {
                    if (board[ny][nx] > 0) num++;
                }
            }

            switch (board[i][j])
            {
            case 0:
                if (num == 3) {
                    board[i][j] = -1;
                }
                break;
            case 1:
                if (num < 2 || num > 3) {
                    board[i][j] = 2;
                }
                break;
            default:
                break;
            }

        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (board[i][j] == 2) { board[i][j] = 0; }
            if (board[i][j] == -1) { board[i][j] = 1; }
            cout << board[i][j] << " ";
        }
        cout << endl;
    }
}

int maximalRectangle(vector<vector<char>>& matrix) {

    int ans = 0;
    int N = matrix.size();
    if (N < 1) return ans;
    int M = matrix[0].size();

    vector<Coordinate> v;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < M; x++)
        {
            if (matrix[y][x] == '1') v.push_back({x, y});
        }
    }

    for (auto coord : v) {

        int left = coord.x; int right = coord.x; int height = 1;

        for (int i = coord.x; i < M; i++) {
            if (matrix[coord.y][i] == '1') 
                right = i;
            else
                break;
        }

        bool stop = false;
        for (int j = coord.y + 1; j < N; j++) {
            for (int i = left; i <= right; i++) {
                if (matrix[j][i] == '0') {
                    stop = true;
                    break;
                }
            }
          
            if (stop) break;
            height++;
        }

        stop = false;
        for (int j = coord.y - 1; j >= 0; j--) {
            for (int i = left; i <= right; i++) {
                if (matrix[j][i] == '0') {
                    stop = true;
                    break;
                }
            }

            if (stop) break;
            height++;
        }
       
        ans = max(ans, (right - left + 1) * height);
    }

    return ans;
}


// 84. 柱状图中最大的矩形
int largestRectangleArea(vector<int>& heights) {

    int N = heights.size();

    if (!N) {
        return 0;
    }

    stack<int> mStack;
    mStack.push(0);
    int ans = heights[0];

    for (int i = 0; i < N; i++) {

        while (!mStack.empty() && heights[i] <= heights[mStack.top()]) {
            auto height = heights[mStack.top()];
            mStack.pop();
            int weight = mStack.empty() ? i : i - mStack.top() - 1;
            ans = max(ans, height * weight);
        }
        mStack.push(i);
    }

    while (!mStack.empty()) {
        auto height = heights[mStack.top()];
        mStack.pop();
        int weight = mStack.empty() ? N : N - mStack.top() - 1;
        ans = max(ans, height * weight);
    }
    
    return ans;
}

vector<vector<string>> ans;

bool isValide(vector<string>& board, int row, int col) {
    int n = board[row].size();

    for (int i = 0; i < n; i++) {
        if (board[row][i] == 'Q') {
            return false;
        }
    }

    for (int i = 0; i < n; i++) {
        if (board[i][col] == 'Q') {
            return false;
        }
    }

    for (int i = row - 1, j = col + 1; 
        i >= 0 && j < n; i--, j++) {
        if (board[i][j] == 'Q')
            return false;
    }

    for (int i = row - 1, j = col - 1;
        i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q')
            return false;
    }

    for (int i = row + 1, j = col + 1; 
        i < n && j < n; i++, j++) {
        if (board[i][j] == 'Q')
            return false;
    }

    for (int i = row + 1, j = col - 1;
        i < n && j >= 0; i++, j--) {
        if (board[i][j] == 'Q')
            return false;
    }

    return true;
}

void backtrack(vector<string>& board, int row) {
    
    if (row == board.size()) {
        ans.push_back(board);
        return;
    }

    int n = board[row].size();

    for (int col = 0; col < n; col++) {
        
        if (!isValide(board, row, col))
        {
            continue;
        }
        
        board[row][col] = 'Q';
        backtrack(board, row + 1);
        board[row][col] = '.';
    }

}

vector<vector<string>> solveNQueens(int n) {
    vector<string> board(n, string(n, '.'));
    backtrack(board, 0);

    // for (auto v : ans) {
    //     for (int i = 0; i < v.size(); i++) {
    //         cout << v[i] << endl;
    //     }
    //     cout << "*************** "<< endl;
    // }
    return ans;
}

int zero = '0';

// 37. 解数独
bool sudoKubackstrack(vector<vector<char>>& board, deque<Coordinate>& rc,
    vector<int>& row, vector<int>& col, vector<int>& gong) {

    if (rc.empty()) {

        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                cout << board[i][j] << " ";
            }
            cout << endl;
        }
        return true;
    }

    auto coord = rc.front();
    
    char c = '1';
    while (c <= zero + 9) {
        int g = (coord.y / 3) * 3 + coord.x / 3;
        int p = c - zero;
        if (((row[coord.x] >> p) & 1) || ((col[coord.y] >> p) & 1) || ((gong[g] >> p) & 1)) {
            c++;
            continue;
        }

        row[coord.x] ^= (1 << p); col[coord.y] ^= (1 << p); gong[g] ^= (1 << p);
        board[coord.y][coord.x] = c;
        rc.pop_front();
        if (sudoKubackstrack(board, rc, row, col, gong)) return true;
        row[coord.x] ^= (1 << p); col[coord.y] ^= (1 << p); gong[g] ^= (1 << p);
        board[coord.y][coord.x] = '.';
        rc.push_front(coord);
        c++;
    }

    return false;
}

void solveSudoku(vector<vector<char>>& board) {

    vector<int> row(9, 0);  // 行
    vector<int> col(9, 0);  // 列　
    vector<int> gong(9, 0); // 格
    deque<Coordinate> rc;

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (board[i][j] != '.') {
                int p = board[i][j] - zero;
                row[j] |= (1 << p);
                col[i] |= (1 << p);
                gong[(i / 3) * 3 + j / 3] |= (1 << p);
            } else {
                rc.push_back({j, i});
            }
        }
    }

    sudoKubackstrack(board, rc, row, col, gong);
}

int myAtoi(string str) {

    long ans = 0;
    bool isFirst = true;
    char sign = '+';

    for (char c : str) {

        if ('0' <= c && c <= '9') {
            ans *= 10;
            ans += c - '0';

            if (ans > INT32_MAX) {
                ans = sign == '+' ? INT32_MAX : INT32_MIN;
                return ans;
            }

            if (isFirst) isFirst = false; 
            continue;
        }

        if (c != ' ' && isFirst) {
            isFirst = false;
            if (c == '+' || c == '-') {
                sign = c;
                continue;
            }
            return ans;
        }

        if (!isFirst) break;
    }
    
    ans = sign == '+' ? ans : ~ans + 1;

    return ans;
}

// 42. 接雨水
int trap(vector<int>& height) {

    int N = height.size();
    if (N < 1) { return 0; }

    int ans = 0;
    stack<int> mStack;
    int reference = height[0];
    mStack.push(0);

    for (int i = 1; i < N; i++) {

        if (height[i] > reference) {
            while (!mStack.empty()) {
                ans += reference - height[mStack.top()];
                mStack.pop();
            }
            reference = height[i];
        }
        mStack.push(i);
    }

    if (!mStack.empty()) {
        reference = height[mStack.top()];
        mStack.pop();

        while (!mStack.empty()) {

            if (height[mStack.top()] > reference) {
                reference = height[mStack.top()];
            } else {
                ans += reference - height[mStack.top()];
            }
            mStack.pop();
        }
    }

    return ans;
}

// 反转链表
ListNode* reverseList(ListNode* head) {

    ListNode* pre = nullptr;
    ListNode* tmp = nullptr;
    auto cur = head;

    while(cur) {
        tmp = cur->next;
        cur->next = pre;
        pre = cur;
        cur = tmp;
    }
    return pre;
}

// 剑指 Offer 06. 从尾到头打印链表
vector<int> reversePrint(ListNode* head) {

    ListNode* pre = nullptr;
    ListNode* curr = head;

    while (curr) {
        ListNode* temp = curr->next;
        curr->next = pre;
        pre = curr;
        curr = temp;
    }

    vector<int> ans;
    while (pre)
    {
        ans.push_back(pre->val);
        pre = pre->next;
    }
    return ans;
}

// 面试题 01.07. 旋转矩阵
void rotate(vector<vector<int>>& matrix) {

    int M = 0;
    int N = matrix.size() - 1;
    
    while (M < N)
    {
        for (int i = M; i < N;) {
            for (int j = N; j > M;) {
                int temp = matrix[M][i];
                matrix[M][i] = matrix[j][M];
                matrix[j][M] = matrix[N][j];
                matrix[N][j] = matrix[i][N];
                matrix[i][N] = temp;
                i++;
                j--;
            }
        }

        M++;
        N--;
    }
}

vector<string> findRepeatedDnaSequences(string s) {

    vector<string> result;
    unordered_map<string, int> cache;

    if (s.length() <= 10) {
        return result;
    }

    int head = 0;
    auto str = s.substr(head, 10);
    cache.insert({str, 1});

    while (head < s.length() - 9) {
        head++;
        auto temp = s.substr(head, 10);

        auto it = cache.find(temp);
        if (it != cache.end() && it->second != 2) 
        {
            result.push_back(temp);
            it->second = 2;
        } else {
            cache.insert({temp, 1});
        }
    }
    return result;
}

set<int> appear;
bool isHappy(int n) {

    if (n == 1) return true;

    auto it = appear.find(n);
    if (it != appear.end()) return false;
    appear.insert(n);

    int num = 0;
    while (n != 0) {
        int x = n % 10;
        num += x * x;
        n = n / 10;
    }

   return isHappy(num);
}


bool isIsomorphic(string s, string t) {

    vector<int> src(128, -1);
    vector<int> dest(128, -1);

    for (int i = 0; i < s.length(); i++) {

        if (src[s[i]] != -1 && src[s[i]] != t[i]) return false;
        if (dest[t[i]] != -1 && dest[t[i]] != s[i]) return false; 

        src[s[i]] = t[i];
        dest[t[i]] = s[i];
    }

    return true;
}

int movingCount(int m, int n, int k) {

    int dx[] = {0, -1, 0, 1};
    int dy[] = {-1, 0, 1, 0};

    vector<vector<int>> visited(m, vector<int>(n, 0));

    queue<Coordinate> que;
    que.push({0, 0});
    visited[0][0] = 1;
    int ans = 1;

    while (!que.empty()) {
        auto coord = que.front();
        que.pop();

        for (int i = 0; i < 4; i++) {
            int nx = coord.x + dx[i];
            int ny = coord.y + dy[i];

            if (nx >=0 && nx < n && ny >=0 && ny < m) {

                if (visited[ny][nx]) continue;
                

                int num = 0;
                int temp = nx;

                while (temp != 0) {
                    int x = temp % 10;
                    num += x;
                    temp = temp / 10;
                }

                temp = ny;
                while (temp != 0) {
                    int y = temp % 10;
                    num += y;
                    temp = temp / 10;
                }

                if (num <= k) {
                    que.push({nx, ny});
                    visited[ny][nx] = 1;
                    ans++;
                }
            }
        }
    }

    return ans;

}

void backtrackP(vector<string>& ans, string& str, int open, int close, int n) {

    if (str.size() == n << 1) {
        ans.push_back(str);
        return;
    }

    if (open < n) {
        str.push_back('(');
        backtrackP(ans, str, open + 1, close, n);
        str.pop_back();
    }

    if (close < open) {
        str.push_back(')');
        backtrackP(ans, str, open, close + 1, n);
        str.pop_back();
    }
}

vector<string> generateParenthesis(int n) {
    vector<string> result;
    string current;
    backtrackP(result, current, 0, 0 ,n);
    return result;
}

string reverseWords(string s) {

    string ans;
    vector<string> cache;
    bool head = false;
    string temp;

    for (int i = 0 ; i < s.length(); i++) {

        if (s[i] == ' ') {

            if (head) {
                cache.push_back(temp);
                temp.clear();
                head = false;
            }
            continue;
        } else {
            if (!head) head = true;
            temp.push_back(s[i]);
        }
    }

    if (temp.size() > 0) cache.push_back(temp);

    for (int i = cache.size() - 1; i >= 0; i--) {
        ans.append(cache[i]);
        if (i != 0) ans.push_back(' ');
    }

    return ans;
}

int divide(int dividend, int divisor) {

    if (dividend == INT32_MIN) {
        if (divisor == 1) return INT32_MIN;
        if (divisor == -1) return INT32_MAX;
    }

    int A, B = 0;
    if (dividend > 0) A = -dividend;
    if (divisor > 0) B = -divisor;

    int a = A >> 1;
    int ans = 0;
    if (a < B) {

        int temp = B << 1;
        ans = 2;
        while ((temp << 1) > A) {
            temp = temp << 1;
            ans = ans << 1;
        }

        temp = temp - dividend;

        while (temp < B) {
            temp = B - temp;
            ans++;
        }
    } else {

    }

    if ((dividend > 0 && divisor < 0) || (dividend < 0 && divisor > 0)) {
        ans = -abs(ans);
    }

    return ans;
}

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    
    vector<int> n1;
    vector<int> n2;

    n1.push_back(l1->val);

    while (l1->next) {
        l1 = l1->next;
        n1.push_back(l1->val);
    }

    n2.push_back(l2->val);
  
    while (l2->next) {
        l2 = l2->next;
        n2.push_back(l2->val);
    }

    int index1 = n1.size() - 1;
    int index2 = n2.size() - 1;
    int temp = n1[index1] + n2[index2];
    int x = 0;

    if (temp >= 10) {
        x = 1;
        temp -= 10;
    }
    ListNode* ans = new ListNode(temp);

    while (index1 - 1 >=0 || index2 - 1 >= 0 || x > 0) {
        index1 -= 1; index2 -= 1;
        int v1 = 0; int v2 = 0;

        if (index1 >= 0) {
            v1 = n1[index1];
        }

        if (index2 >= 0) {
            v2 = n2[index2];
        }

        temp = v1 + v2 + x;

        if (temp >= 10) {
            x = 1;
            temp -= 10;
        } else {
            x = 0;
        }

        ListNode* node = new ListNode(temp);
        node->next = ans;
        ans = node;
    }

    return ans;
}


vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {

    int M = matrix.size();
    int N = matrix[0].size();

    vector<vector<int>> dist(M, vector<int>(N, 0));
    vector<vector<int>> seen(M, vector<int>(N, 0));

    int dx[] = {0, -1, 0, 1};
    int dy[] = {-1, 0, 1, 0};

    queue<Coordinate> que;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (matrix[i][j] == 0) {
                que.push({j, i});
                seen[i][j] = 1;
            }
        }
    }

    while (!que.empty())
    {
        auto coord = que.front();
        que.pop();

        for (int i = 0; i < 4; i++) {
            int nx = dx[i] + coord.x;
            int ny = dy[i] + coord.y;

            if (nx >=0 && nx < N && ny >=0 && ny < M && !seen[ny][nx]) {
                dist[ny][nx] = dist[coord.y][coord.x] + 1;
                que.push({nx, ny});
                seen[ny][nx] = 1;
            }
        }
    }

    return dist;
}

vector<vector<int>> merge(vector<vector<int>>& intervals) {
    
    if (intervals.size() == 0) {
        return {};
    }

    sort(intervals.begin(), intervals.end());
    vector<vector<int>> merged;
    for (int i = 0; i < intervals.size(); ++i) {
        int L = intervals[i][0], R = intervals[i][1];
        if (!merged.size() || merged.back()[1] < L) {
            merged.push_back({L, R});
        } else {
            merged.back()[1] = max(merged.back()[1], R);
        }
    }
    return merged;
}

bool canJump(vector<int>& nums) {

    int N = nums.size();

    if (N <= 1) return true;
    
    int temp = 1;

    for (int i = nums.size() - 2; i >= 0; i--) {

        if (nums[i] < temp) {
            temp++;
        } else {
            temp = 1;
        }
    }

    return temp == 1;
}

vector<vector<int>> subsets(vector<int>& nums) {

    vector<vector<int>> ans;
    
    for (int num : nums) {
        int n = ans.size();
        for (int i = 0; i < n; ++i) {
            ans.push_back(ans[i]);
            ans[i].push_back(num);
        }
        ans.push_back({ num });
    }
    ans.push_back({});
    return ans;
}

// 200. 岛屿数量
int numIslands(vector<vector<char>>& grid) {
    
    int N = grid.size();
    if (N < 1) return 0;
    int M = grid[0].size();

    int dx[] = {0, -1, 0, 1};
    int dy[] = {-1, 0, 1, 0};

    queue<Coordinate> que;
    int ans = 0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++) {
            if (grid[i][j] == '1') {
                que.push({j, i});

                while (!que.empty()) {
                    auto coord = que.front();
                    que.pop();

                    for (int i = 0; i < 4; i++) {

                        int nx = dx[i] + coord.x;
                        int ny = dy[i] + coord.y;

                        if (nx < 0 || nx >= M || ny < 0 || ny >= N) continue;

                        if (grid[ny][nx] == '1') {
                            que.push({nx, ny});
                            grid[ny][nx] = '0';
                        }
                    }
                }
                ans++;
            }
        }
    }
    
    return ans;
}

unordered_map <char, int> ori, cnt;

bool check() {
    for (const auto &p: ori) {
        if (cnt[p.first] < p.second) {
            return false;
        }
    }
    return true;
}

string minWindow(string s, string t) {
    for (const auto &c: t) {
        ++ori[c];
    }

    int l = 0, r = -1;
    int len = INT_MAX, ansL = -1, ansR = -1;

    while (r < int(s.size())) {
        if (ori.find(s[++r]) != ori.end()) {
            ++cnt[s[r]];
        }
        while (check() && l <= r) {
            if (r - l + 1 < len) {
                len = r - l + 1;
                ansL = l;
            }
            if (ori.find(s[l]) != ori.end()) {
                --cnt[s[l]];
            }
            ++l;
        }
    }

    return ansL == -1 ? string() : s.substr(ansL, len);
}

//　四数之和
vector<vector<int>> fourSum(vector<int>& nums, int target) {

    vector<vector<int>> result;
    int N = nums.size();
    if (N < 4) return result;
    sort(nums.begin(), nums.end());

    for (int i = 0; i < N - 1; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;

        for (int j = i + 1; j < N; j++) {
            if (j > i + 1 && nums[j] == nums[j - 1]) continue;

            int l = j + 1;
            int r = N - 1;

            while (l < r)
            {
                int sum = nums[i] + nums[j] + nums[l] + nums[r];
                if (sum == target) {
                    result.push_back(vector<int>({nums[l], nums[r], nums[i], nums[j]}));
                    while (l < r && nums[l] == nums[l + 1] || l == i || l == j) l++;
                    while (l < r && nums[r] == nums[r - 1] || r == i || r == j) r--;
                    l++;
                    r--;
                } else {
                    if (sum > target) r--;
                    if (sum < target) l++;
                }
            } 
        }
    }

    return result;
}

vector<string> getValidT9Words(string num, vector<string>& words) {

    for (unsigned int i = 0; i < num.length(); i++) {

        int n = num[i] - '2';
        int k = (n == 5 || n == 7) ? 4 : 3;
        n = n <= 5 ? 3 * n : 3 * n + 1;

        for (auto it = words.begin(); it != words.end();) {
            char c = 'a' + n;
            bool hasWord = false;

            for (int j = 0; j < k; j++) {
                if (c == (*it)[i]) {
                    hasWord = true;
                    break;
                }
                c++;
            }

            if (!hasWord) {
                words.erase(it);
            } else {
                it++;
            }
        }
    }

    return words;
}

int findDuplicate(vector<int>& nums) {
    int fast = 0;
    int slow = 0;

    while (1) {
        fast = nums[nums[fast]];
        slow = nums[slow];

        if (fast == slow) {
            break;
        }
    }

    int finder = 0;
    while (1) {
        finder = nums[finder];
        slow = nums[slow];
        if (slow == finder) {
            break;
        }
    }

    return finder;
}


// P[i - 1] % K = P[j] % K
int subarraysDivByK(vector<int>& A, int K) {

    std::unordered_map<int, int> m_map = {{0, 1}};
    int ans = 0; int sum = 0;

    for (auto i : A) {
        sum += i;
        auto mol = (sum % K + K) % K;
        if (m_map.count(mol)) {
            ans += m_map[mol];
        }
        m_map[mol]++;
    }

    return ans;
}

struct DS
{
    int num;
    string str;
};

string decodeString(string s) {
    string ans;
    stack<DS> ms;
    int num = 0;

    for (auto c : s) {

        if (c >= '0' && c <= '9') {
            if (num != 0) {
                num = num * 10 + (c - '0');
            } else {
                num = c - '0';
            }
            continue;
        } 

        if (c == '[') {
            DS ds;
            ds.num = num;
            ms.push(ds);
            num = 0;
            continue;
        }

        if (c == ']') {
            string temp;
            auto ds = ms.top(); ms.pop();

            for (int i = 0; i < ds.num; i++) {
                temp += ds.str;
            }

            if (!ms.empty()) {
                ms.top().str += temp;
            } else {
                ans += temp;
            }

            continue;
        }

        if (!ms.empty()) {
            ms.top().str += c;
        } else {
            ans += c;
        }
    }

    return ans;
}

bool isValidSudoku(vector<vector<char>>& board) {
    vector<int> row(9, 0);
    vector<int> col(9, 0);
    vector<int> gong(9, 0);

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (board[i][j] != '.') {
                int a = board[i][j] - '0';

                if ((row[j] & 1 << a) || (col[i] & 1 << a) || (gong[(i / 3) * 3 + j / 3] & 1 << a)) {
                    return false;
                }

                row[j] |= 1 << a;
                col[i] |= 1 << a;
                gong[(i / 3) * 3 + j / 3] |= 1 << a;
            }
        }
    }

    return true;
}

string countAndSay(int n) {
    
    string num = "1";
    int k = 1;

    while (k < n) {
        int count = 1;
        char c = num[0];
        string temp;

        for (unsigned int i = 1; i < num.length(); i++) {

            if (num[i] == num[i - 1]) {
                count++;
            } else {
                temp = temp + to_string(count) + c;
                c = num[i];
                count = 1;
            }
        }

        num = temp + to_string(count) + c;
        k++;
    }

    return num;
}


void backtrackA(vector<vector<int>>& res, vector<int>& output, int first, int len){

    if (first == len) {
        res.emplace_back(output);
        return;
    }

    for (int i = first; i < len; i++) {
        swap(output[i], output[first]);
        backtrackA(res, output, first + 1, len);
        swap(output[i], output[first]);
    }
}

vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> res;
    backtrackA(res, nums, 0, (int)nums.size());
    
    for (int i = 0; i < res.size(); i++) {
        for (int j = 0; j < res[i].size(); j++) {
            std::cout << res[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return res;
}
                                  
int rob(vector<int>& nums) {
    if (nums.empty()) {
        return 0;
    }
    int size = nums.size();
    if (size == 1) {
        return nums[0];
    }
    int first = nums[0], second = max(nums[0], nums[1]);
    for (int i = 2; i < size; i++) {
        int temp = second;
        second = max(first + nums[i], second);
        first = temp;
    }
    return second;
}

int minPathSum(vector<vector<int>>& grid) {

    int m = grid.size();
    int n = grid[0].size();

    if (m == 0 && n == 0) return 0;

    auto dp = vector<vector<int>>(m, vector<int>(n, 0));
    dp[0][0] = grid[0][0];
    for (int i = 1; i < m; i++) {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }
    for (int i = 1; i < n; i++) {
        dp[0][i] = dp[0][i - 1] + grid[0][i];
    }
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = grid[i][j] + std::min(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    return dp[m - 1][n - 1];
}

vector<int> topKFrequent(vector<int>& nums, int k) {

    std::unordered_map<int, int> uomap;

    for (auto num : nums) {
        
        if (uomap.count(num)) {
            uomap[num]++;
        } else {
            uomap.insert({num, 1});
        }
    }

    vector<int> result;
    int min = 0;
    bool flag = true;

    for (auto it = uomap.begin(); it != uomap.end(); it++) {

        if (result.size() < k) {
            result.push_back(it->first);
        } else {
            if (flag) {
                for (int i = 0; i < k; i++) {
                    if (uomap[result[min]] > uomap[result[i]]) {
                        min = i;
                    }
                }
                flag = false;
            }
            if (uomap[result[min]] < it->second) {
                result[min] = it->first;
                flag = true;
            }
        }
    }

    return result;
}

void composRes(vector<vector<int>>& results, vector<int>* result, vector<int>& candidates, int target) {

    for (auto it = candidates.begin(); it != candidates.end();) {

        auto newTarget = target - (*it);

        if (newTarget == 0) {
            vector<int> res;
            if (result) {
                res = *result;
            }
            res.push_back((*it));
            results.push_back(res);
            break;
        }

        if (newTarget > 0) {
            vector<int> res;
            if (result) {
                res = *result;
            }
            res.push_back((*it));
            auto can = candidates;
            composRes(results, &res, can, newTarget);
        }

        if (candidates.size() == 1) break;
        candidates.erase(it);
    }

}

vector<vector<int>> combinationSum(vector<int>& candidates, int target) {

    std::sort(candidates.begin(), candidates.end());

    vector<vector<int>> results;
    composRes(results, nullptr, candidates, target);
    return results;
}

void composRes2(vector<vector<int>>& results, vector<int>* result, vector<int>& candidates, int target) {

    int lastNum = 0;

    for (auto it = candidates.begin(); it != candidates.end();) {

        while (lastNum == (*it)) {
            if (candidates.size() == 1) return;
            candidates.erase(it);
        }

        lastNum = (*it);

        auto newTarget = target - (*it);

        if (newTarget < 0) {
            return;
        }

        if (newTarget == 0) {
            vector<int> res;
            if (result) {
                res = *result;
            }
            res.push_back((*it));
            results.push_back(res);
            return;
        }

        if (candidates.size() == 1) return;

        if (newTarget > 0) {
            vector<int> res;
            if (result) {
                res = *result;
            }
            res.push_back((*it));
            candidates.erase(it);
            auto can = candidates;
            composRes2(results, &res, can, newTarget);
        }
    }
}

vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    std::sort(candidates.begin(), candidates.end());
    
    vector<vector<int>> results;
    composRes2(results, nullptr, candidates, target);
    return results;
}


// 1024. 视频拼接
int videoStitching(vector<vector<int>>& clips, int T) {

    int result = 0;

    int start = 0;
    int end = 0;

    while (end < T) {
        for (int i = 0; i < clips.size(); i++) {
            if (clips[i][0] <= start) {
                if (T == 0) return 1;
                if (clips[i][1] > end) {
                    end = clips[i][1];
                }
            }
        }
        if (end == start) return -1;
        start = end;
        result++;
    }
    
    return result;
}

// 1365
vector<int> smallerNumbersThanCurrent(vector<int>& nums) {

    vector<int> counts(101, 0);
    vector<int> result;

    for (auto num : nums) {
        counts[num]++;
    }

    for (auto num : nums) {
        int count = 0;
        for (int i = 0; i < num; i++) {
            count += counts[i];
        }
        result.push_back(count);
    }

    return result;
}

// 1207
bool uniqueOccurrences(vector<int>& arr) {

    int min = 1000;
    int max = -1000;
    for (auto i : arr) {
        if (i > max) max = i;
        if (i < min) min = i;
    }
 
    vector<int> vs((max - min + 1), 0);
    int maxLen = 0;
    for (auto i : arr) {
        if (++vs[i - min] > maxLen) maxLen = vs[i - min];
    }

    vector<int> vsl(maxLen + 1, 0);
    for (auto i : vs) {
        if (i == 0) continue;
        if (vsl[i] == 1) return false;
        vsl[i] = 1;
    }

    return true;
}

// 463
int islandPerimeter(vector<vector<int>>& grid) {

    int N = grid.size();
    int M = grid[0].size();

    int dx[] = {0, -1, 0, 1};
    int dy[] = {-1, 0, 1, 0};

    queue<Coordinate> mQue;

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < M; x++) {
            if (grid[y][x] == 1) mQue.push({x, y});
        }
    }

    int ans = 0;
    while (!mQue.empty())
    {
        auto coord = mQue.front();
        mQue.pop();    
        for (int i = 0; i < 4; i++) {
            int nx = coord.x + dx[i];
            int ny = coord.y + dy[i];
            if (nx < 0 || ny < 0 || nx >= M || ny >= N) {
                ans++;
            } else {
                if (grid[ny][nx] == 0) ans++;
            }
        }
    }
    
    return ans;
}

// 349
vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {

    vector<int> res;
    if (nums1.size() == 0 || nums2.size() == 0) return res;

    map<int, int> map;

    for (auto num : nums1) {
        map[num] = 0;
    }
    for (auto num : nums2) {
        auto it = map.find(num);
        if (it != map.end()) {
            map[num] = 1;
        }
    }
    for (auto it = map.begin(); it != map.end(); it++) {
        if (it->second == 1) {
            res.push_back(it->first);
        }
    }

    return res;
}

// 941
bool validMountainArray(vector<int>& A) {

    int max = 0;

    for (int i = 1; i < A.size(); i++) {
        if (A[i] > A[max]) {
            max = i;
        } else if (A[i] == A[max]) {
            return false;
        } else {
            break;
        }
    }

    if (max == 0 || max == A.size() - 1) return false;
    
    int min = max;

    for (int i = max + 1; i < A.size(); i++) {
        if (A[i] < A[min]) {
            min = i;
        } else  {
            return false;
        }
    }

    return true;
}

// 973
vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
    sort(points.begin(), points.end(), [](const vector<int>& u, const vector<int>& v) {
        return u[0] * u[0] + u[1] * u[1] < v[0] * v[0] + v[1] * v[1];
    });
    return {points.begin(), points.begin() + K};
}

// 31
void nextPermutation(vector<int>& nums) {
    int i = nums.size() - 2;
    while (i >= 0 && nums[i] >= nums[i + 1]) {
        i--;
    }
    if (i >= 0) {
        int j = nums.size() - 1;
        while (j >= 0 && nums[i] >= nums[j]) {
            j--;
        }
        swap(nums[i], nums[j]);
    }
    reverse(nums.begin() + i + 1, nums.end());
}

// 剑指 Offer 22 链表中倒数第k个节点  
ListNode* getKthFromEnd(ListNode* head, int k) {

    ListNode* fast = head;
    ListNode* slow = head;
    
    int index = 0;

    while (fast) {
        index++;
        if (index > k) slow = slow->next;
        fast = fast->next;
    }

    return slow;
}

// 922. 按奇偶排序数组 II
vector<int> sortArrayByParityII(vector<int>& A) {

    int evenIndex = 0;
    int oddIndex = 0;

    vector<int> result;

    for (int i = 0; i < A.size(); i++) {
        if (i % 2 == 0) {
            for (evenIndex; evenIndex < A.size(); evenIndex++) {
                if (A[evenIndex] % 2 == 0) {
                    result.push_back(A[evenIndex]);
                    evenIndex++;
                    break;
                }
            }
        } else {
            for (oddIndex; oddIndex < A.size(); oddIndex++) {
                if (A[oddIndex] % 2 != 0) {
                    result.push_back(A[oddIndex]);
                    oddIndex++;
                    break;
                }
            }
        }
    }

    return result;
}

// 23. 合并K个升序链表
ListNode* mergeKLists(vector<ListNode*>& lists) {
    
    int N = lists.size();
    ListNode* res = nullptr;
    ListNode* curr = nullptr;
    vector<int> indexs;
    while (1)
    {
        for (int i = 0; i < N; i++) {
            if (!lists[i]) continue;
            if (indexs.empty()) {
                indexs.push_back(i);
                continue;
            }
            if (lists[indexs[0]]->val > lists[i]->val) {
                indexs.clear();
                indexs.push_back(i);
            } else if (lists[indexs[0]]->val == lists[i]->val) {
                indexs.push_back(i);
            }
        }
        if (indexs.empty()) {
            break;
        }
        for (auto index : indexs) {
            if (!res) {
                res = lists[index];
                curr = res;
            } else {
                curr->next = lists[index];
                curr = curr->next;
            }
            lists[index] = lists[index]->next;
        }
        indexs.clear();
    }

    return res;
}

// 328. 奇偶链表
ListNode* oddEvenList(ListNode* head) {
    if (!head) return head;

    ListNode* evenHead = head->next;
    ListNode* odd = head;
    ListNode* even = evenHead;

    while (even && even->next)
    {
        odd->next = even->next;
        odd = odd->next;
        even->next = odd->next;
        even = even->next;
    }
    odd->next = evenHead;
    return head;
}

// 406. 根据身高重建队列
vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
    std::sort(people.begin(), people.end(), [](const vector<int>& a, const vector<int>& b) {
        return (a[0] > b[0]) || (a[0] == b[0] && a[1] < b[1]);
    });
    vector<vector<int>> res;
    for (auto person : people) {
        res.insert(res.begin() + person[1], person);
    }
    return res;
}

// 剑指 Offer 03. 数组中重复的数字
int findRepeatNumber(vector<int>& nums) {
    set<int> numsSet;

    for (auto n : nums) {
        auto it = numsSet.find(n);
        if (it != numsSet.end()) {
            return n;
        } else {
            numsSet.insert(n);
        }
    }
    return 0;
}

// 剑指 Offer 07. 重建二叉树
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    if (preorder.size() == 0) return NULL; 
    TreeNode* root = new TreeNode(preorder[0]);
    if (preorder.size() == 1) return root;

    vector<int> leftInorder;
    vector<int> rightInorder;
    bool isLeft = true;
    for (auto i : inorder) {
        if (i == root->val) {
            isLeft = false;
        } else {
            if (isLeft) {
                leftInorder.push_back(i);
            } else {
                rightInorder.push_back(i);
            }
        }
    }

    vector<int> leftPreorder;
    vector<int> rightPreorder;
    isLeft = leftInorder.size() == 0 ? false : true;
    for (int p = 1; p < preorder.size(); p++) {
        if (isLeft) {
            leftPreorder.push_back(preorder[p]);
            if (p == leftInorder.size()) {
                isLeft = false;
            }
        } else {
            rightPreorder.push_back(preorder[p]);
        }
    }

    root->left = buildTree(leftPreorder, leftInorder);
    root->right = buildTree(rightPreorder, rightInorder);

    return root;
}


// 剑指 Offer 47. 礼物的最大价值
int maxValue(vector<vector<int>>& grid) {
    int M = grid.size();
    if (M == 0) return 0;
    int N = grid[0].size();
    if (N == 0) return 0;

    vector<vector<int>> cache(M, vector<int>(N, 0));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int up = 0; int left = 0;
            if (i > 0) left = cache[i - 1][j];
            if (j > 0) up = cache[i][j - 1];
            cache[i][j] = max(left, up) + grid[i][j];
        }
    }

    return cache[M - 1][N - 1];
}

// 剑指 Offer 27. 二叉树的镜像
TreeNode* mirrorTree(TreeNode* root) {
    if (!root) return NULL;
    if (root->left || root->right) {
        TreeNode* temp = root->left;
        root->left = root->right;
        root->right = temp;
    }

    if (root->left) mirrorTree(root->left);
    if (root->right) mirrorTree(root->right);

    return root;
}

bool isChildrenSymmetric(TreeNode* left, TreeNode* right) {

    if (left) {
        if (right) {
            if (left->val != right->val) return false;
            return isChildrenSymmetric(left->left, right->right) && isChildrenSymmetric(left->right, right->left);
        } else { 
            return false;
        }
    } else {
        return right == nullptr ? true : false;
    }
}

// 剑指 Offer 28. 对称的二叉树
bool isSymmetric(TreeNode* root) {
    if (!root) return true;
    return isChildrenSymmetric(root->left, root->right);
}

// 1030. 距离顺序排列矩阵单元格
vector<vector<int>> allCellsDistOrder(int R, int C, int r0, int c0) {

    vector<vector<int>> ans;
    vector<vector<int>> record(R, vector<int>(C, 0));
    queue<Coordinate> coords;
    coords.push({r0, c0});

    int dx[] = {0, -1, 0, 1};
    int dy[] = {-1, 0, 1, 0};

    while (!coords.empty())
    {
        auto coord = coords.front();
        coords.pop();
        if (record[coord.x][coord.y]) continue; 
        ans.push_back({coord.x, coord.y});
        record[coord.x][coord.y] = 1;

        for (int i = 0; i < 4; i++) {
            int nx = coord.x + dx[i];
            int ny = coord.y + dy[i];
            if (nx < 0 || nx > R - 1 || ny < 0 || ny > C - 1) continue;
            coords.push({nx, ny});
        }
    }
    
    return ans;
}

// 134 加油站  
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {

    for (int i = 0; i < gas.size(); i++) {
        if (gas[i] < cost[i]) continue;
        int curr = i;
        int end = curr == 0 ? gas.size() - 1 : curr - 1;
        int totle = gas[curr];
        int can = true;                       

        while (curr != end) {
            if (totle >= cost[curr]) {
                totle -= cost[curr];
                if (curr == cost.size() - 1) {
                    curr = 0;
                } else {
                    curr++;
                }
                totle += gas[curr];

            } else {
                can = false;
                break;
            } 
        }

        if (can && totle >= cost[end]) return i;
    }
    
    return -1;
}

// 283. 移动零
void moveZeroes(vector<int>& nums) {
    bool hasZero = false;
    int index = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] == 0) {
            if (!hasZero) {
                hasZero = true;
                index = i;
            }
        } else {
            if (hasZero) {
                int temp = nums[i];
                nums[i] = nums[index];
                nums[index] = temp;
                index++;
            }
        }
    }
}

// 先序遍历
vector<int> preorderTraversal(TreeNode* root) {
    vector<int> res;
    stack<TreeNode*> m_stk;
    TreeNode* node = root;

    while (!m_stk.empty() || node != nullptr) {

        while (node)
        {
            res.push_back(node->val);
            m_stk.emplace(node);
            node = node->left;
        }
        node = m_stk.top();
        m_stk.pop();
        node = node->right;
    }

    return res;
}

// 面试题 02.07. 链表相交
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    auto tempA = headA;
    auto tempB = headB;

    while (tempA != tempB) {
        tempA = tempA == nullptr ? headB : tempA->next;
        tempB = tempB == nullptr ? headA : tempB->next;
    }

    return tempA;
}

// 147. 对链表进行插入排序
ListNode* insertionSortList(ListNode* head) {

    auto curr = head;
    
    while (curr)
    {
        cout << curr->val << endl;
        if (curr->next && curr->next->val < curr->val) {
            auto temp = curr->next;
            curr->next = curr->next->next;
            auto node = head;
            if (temp->val < node->val) {
                temp->next = node;
                head = temp;
            } else {
                while (temp->val >= node->next->val)
                {
                    node = node->next;
                }
                temp->next = node->next;
                node->next = temp;
            }
        } else {
            curr = curr->next;
        }
    }
    
    return head;
}

// 剑指 Offer 56 - I. 数组中数字出现的次数
vector<int> singleNumbers(vector<int>& nums) {
    int ret = 0;
    for (int n : nums)
        ret ^= n;
    int div = 1;
    while ((div & ret) == 0)
        div <<= 1;
    int a = 0, b = 0;
    for (int n : nums)
        if (div & n)
            a ^= n;
        else
            b ^= n;
    return vector<int>{a, b};
}

// 452. 用最少数量的箭引爆气球
int findMinArrowShots(vector<vector<int>>& points) {

    if (points.size() == 0) {
        return 0;
    }

    sort(points.begin(), points.end(), [](const vector<int>& u, const vector<int>& v) {
        return u[1] < v[1];
    });

    int res = 1;
    int edge = points[0][1];
    for (auto point : points) {
        if (point[0] > edge) {
            edge = point[1];
            res++;
        }
    }

    return res;
}

// 70. 爬楼梯
map<int, int> climbCaches;
int climbStairs(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    if (n == 2) return 2;

    auto it = climbCaches.find(n);
    if (it != climbCaches.end()) {
        return it->second;
    } else {
        int ans =  climbStairs(n - 1) + climbStairs(n - 2);
        climbCaches.insert({n, ans});
        return ans;
    }
}

// 面试题 08.01. 三步问题
int waysToStep(int n) {

    if (n <= 2) {
        return n;
    }
    if (n == 3) {
        return 4;
    }

    vector<long> dp(n + 1, 0);
    dp[1] = 1;
    dp[2] = 2;
    dp[3] = 4;
    for (int i = 4; i <= n; i++) {
        dp[i] = (dp[i - 1] + dp[i - 2] + dp[i - 3]) % 1000000007;
    }

    return (int)dp[n];
}

// 222. 完全二叉树的节点个数
int countNodes(TreeNode* root) {
    if (!root) return 0;

    int ans = 1;
    if (root->left) ans = ans + countNodes(root->left);
    if (root->right) ans = ans + countNodes(root->right);

    return ans;
}

// 剑指 Offer 25. 合并两个排序的链表
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {

    auto curr1 = l1;
    auto curr2 = l2;
    ListNode* ans = nullptr;
    ListNode* currAns = nullptr;

    while (curr1 || curr2)
    {
        if (!curr2 || (curr1 && curr1->val <= curr2->val)) {
            if (!ans) {
                ans = curr1;
                currAns = ans;
            } else {
                currAns->next = curr1;
                currAns = currAns->next;
            }
            curr1 = curr1->next;
        } else if (!curr1 || (curr2 && curr1->val > curr2->val)) {
            if (!ans) {
                ans = curr2;
                currAns = ans;
            } else {
                currAns->next = curr2;
                currAns = currAns->next;
            }
            curr2 = curr2->next;
        }
    }
    return ans;
}

// 剑指 Offer 31. 栈的压入、弹出序列
bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
    
    stack<int> help;
    int pos = 0;

    for (int i = 0; i < pushed.size(); i++) {
        help.push(pushed[i]);
        while (pos < popped.size() && !help.empty() && popped[pos] == help.top()) {
            help.pop();
            pos++;
        }
    }

    return help.empty();
}

// 1370. 上升下降字符串
string sortString(string s) {

    string ans;
    vector<int> barrel(26, 0);

    for (auto c : s) {
        barrel[c - 'a']++;
    }

    bool end = false;
    while (!end) {
        end = true;
        for (int i = 0; i < 26; i++) {
            if (barrel[i] > 0) {
                ans.push_back(i + 'a');
                if (barrel[i]-- > 0) end = false;
            }
        }
        for (int i = 25; i >= 0; i--) {
            if (barrel[i] > 0) {
                ans.push_back(i + 'a');
                if (barrel[i]-- > 0) end = false;
            }
        }
    }

    return ans;
}

int main(int argc, char** argv) {
    std::cout << sortString("leetcode") << std::endl;
    return 0;
}