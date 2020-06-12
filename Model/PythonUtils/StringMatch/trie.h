#ifndef __TRIE_H_
#define __TRIE_H_

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
using namespace std;

class Trie
{
    vector<unordered_map<char, size_t>> next;
    vector<int> node2word_id;
    vector<string> words;

    size_t addNode();

public:
    Trie();

    void insert(const string& word);
    int insertFromFile(const string& filename, char separator, bool hasHeader);
    size_t size();
    pair<int, int> maxMatch(const string& s, int st);
    vector<int> matchWholeString(const string& s);
    string getWord(int id);
    void dumpDictionary(const string& filename);
};

#endif