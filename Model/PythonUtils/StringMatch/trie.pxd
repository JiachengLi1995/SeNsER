
# Using a .pxd file gives us a separate namespace for
# the C++ declarations. Using a .pxd file also allows
# us to reuse the declaration in multiple .pyx modules.

from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "trie.h":

    cppclass Trie:
        Trie() except +  # NB! std::bad_alloc will be converted to MemoryError
        void insert(string word)
        void insertFromFile(string filename, char separator, bool has_header)
        size_t size()
        pair[int, int] maxMatch(string s, int st)
        vector[int] matchWholeString(string s)
        string getWord(int id)
        void dumpDictionary(string filename)




