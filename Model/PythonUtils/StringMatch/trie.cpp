#include "trie.h"

// Utils

inline FILE* tryOpen(string filename, string param = "r")
{
    FILE* f = fopen(filename.c_str(), param.c_str());
    if (f == NULL) {
        fprintf(stderr, "[Fatal Error] File Not Found! %s\n", filename.c_str());
        assert(false);
    }
    return f;
}


const int MAX_LENGTH = 100000000;

char line[MAX_LENGTH + 1];

inline bool getLine(FILE* in)
{
    bool hasNext = fgets(line, MAX_LENGTH, in);
    int length = strlen(line);
    while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) {
        -- length;
    }
    line[length] = 0;
    return hasNext;
}


// Trie

size_t Trie::addNode()
{
    size_t newNodeID = next.size();
    node2word_id.push_back(-1);
    next.push_back(unordered_map<char, size_t>());
    return newNodeID;
}

Trie::Trie()
{
    // this is the root node
    addNode();
    // make sure that the word 0 corresponds to the UNK token
    insert("__unk__");
}

void Trie::insert(const string& word)
{
    size_t u = 0;
    for (char ch : word) {
        if (!next[u].count(ch)) {
            size_t newNodeID = addNode();
            next[u][ch] = newNodeID;
        }
        u = next[u][ch];
    }
    if (node2word_id[u] != -1) {
        // fprintf(stderr, "[Warning] %s vs. %s\n", word.c_str(), words[node2word_id[u]].c_str());
    } else {
        node2word_id[u] = words.size();
        words.push_back(word);
    }
}

int Trie::insertFromFile(const string& filename, char separator, bool hasHeader)
{
    // fprintf(stderr, "Building dictionary...");
    // fprintf(stderr, "\tFilename: %s\n", filename.c_str());
    // fprintf(stderr, "\tSeparator: '%c'\n", separator);
    // fprintf(stderr, "\tHas Header?: %s\n", hasHeader ? "True" : "False");
    FILE* in = tryOpen(filename, "r");
    int line_num = 0;
    for (; getLine(in); ++ line_num) {
        if (hasHeader && line_num == 0) {
            continue;
        }
        for (int i = 0; line[i]; ++ i) {
            if (line[i] == separator) {
                line[i] = 0;
                break;
            }
        }
        if (line[0]) {
            // avoid empty strings
            this->insert(line);
        }
    }
    fclose(in);
    return line_num;
    // fprintf(stderr, "\t# of words = %d\n", line_num);
    // fprintf(stderr, "\t# of trie nodes = %d\n", this->size());
    // fprintf(stderr, "Done.\n");
}

size_t Trie::size()
{
    return next.size();
}

pair<int, int> Trie::maxMatch(const string& s, int st)
{
    size_t u = 0;
    int ret = -1, wordID = -1;
    for (int i = st; i < s.size(); ++ i) {
        char ch = s[i];
        if (next[u].count(ch)) {
            u = next[u][ch];
        } else {
            break;
        }
        if (node2word_id[u] != -1) {
            ret = i;
            wordID = node2word_id[u];
        }
    }
    return make_pair(ret, wordID);
}

struct MatchResult
{
    int st, ed;
    int wordID;

    MatchResult(int st, int ed, int wordID) : st(st), ed(ed), wordID(wordID) {
    }
};

bool byLength(const MatchResult& a, const MatchResult& b)
{
    return a.ed - a.st > b.ed - b.st || a.ed - a.st == b.ed - b.st && a.wordID < b.wordID;
}

int findNext(vector<int>& next, int x)
{
    return next[x] == x ? x : next[x] = findNext(next, next[x]);
}

vector<int> Trie::matchWholeString(const string& s)
{
    vector<MatchResult> match;
    int lastEd = -1;
    for (int st = 0; st < s.size(); ++ st) {
        if (isspace(s[st])) {
            continue;
        }
        pair<int, int> ret = this->maxMatch(s, st);
        int ed = ret.first, wordID = ret.second;
        if (ed != -1) {
            // [st, ed]
            if (ed > lastEd) {
                match.push_back(MatchResult(st, ed, wordID));
                lastEd = ed;
            }
        }
    }

    sort(match.begin(), match.end(), byLength);
    vector<int> next(s.size() + 1), ret(s.size(), -1);
    for (int i = 0; i <= s.size(); ++ i) {
        next[i] = i;
    }
    for (const MatchResult& m : match) {
        for (int i = findNext(next, m.st); i <= m.ed && i < s.size(); i = findNext(next, i)) {
            assert(ret[i] == -1);
            ret[i] = m.wordID;
            next[i] = i + 1;
        }
    }
    for (int i = 0; i < ret.size(); ++ i) {
        if (ret[i] == -1 && !isspace(s[i])) {
            ret[i] = 0; // __unk__
        }
    } 
    return ret;
}

string Trie::getWord(int id)
{
    return words[id];
}

void Trie::dumpDictionary(const string& filename)
{
    FILE* out = tryOpen(filename, "w");
    for (const string& word : words) {
        fprintf(out, "%s\n", word.c_str());
    }
    fclose(out);
}

