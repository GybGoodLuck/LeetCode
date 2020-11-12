#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

struct Item
{
    int userId;
    int tweetId;
};


class Twitter {

    vector<Item> tweets;
    unordered_map<int, unordered_set<int>> users;

public:
    /** Initialize your data structure here. */
    Twitter() {

    }
    
    /** Compose a new tweet. */
    void postTweet(int userId, int tweetId) {
        Item item;
        item.userId = userId;
        item.tweetId = tweetId;
        tweets.push_back(item);
    }
    
    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    vector<int> getNewsFeed(int userId) {
        vector<int> res;
        int count = 0;
        int index = tweets.size() - 1;

        auto it = users.find(userId);

        while (count < 10) {
            if (index >= 0) {
                if (tweets[index].userId == userId) {
                    res.push_back(tweets[index].tweetId);
                    count++;
                } else {
                    if (it != users.end()) {
                        auto iter = it->second.find(tweets[index].userId);
                        if (iter != it->second.end()) {
                            res.push_back(tweets[index].tweetId);
                            count++;
                        }
                    }
                }
                index--;
            } else {
                break;
            }
        }

        return res;
    }
    
    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    void follow(int followerId, int followeeId) {
        auto it = users.find(followerId);

        if (it != users.end()) {
            it->second.insert(followeeId);
        } else {
            unordered_set<int> followees;
            followees.insert(followeeId);
            users.insert({followerId, followees});
        }
    }
    
    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    void unfollow(int followerId, int followeeId) {
        auto it = users.find(followerId);

        if (it != users.end()) {

            auto iter = it->second.find(followeeId);

            if (iter != it->second.end()) {
                it->second.erase(iter);
            }
        }
    }
};