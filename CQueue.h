#pragma once

#include <stack>

using namespace std;

// 剑指 Offer 09. 用两个栈实现队列
class CQueue {
public:
    stack<int> m_stack1;
    stack<int> m_stack2;

    CQueue() {

    }
    
    void appendTail(int value) {
        m_stack1.push(value);
    }

    int deleteHead() {
        if (m_stack2.empty()) {
            while (!m_stack1.empty()) {
                int top = m_stack1.top();
                m_stack1.pop();
                m_stack2.push(top);
            }
        }
        if (m_stack2.empty()) return -1;

        int res = m_stack2.top();
        m_stack2.pop();

        return res;
    }
};