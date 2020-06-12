
# distutils: language = c++
# distutils: sources = trie.cpp

import torch_scope
import logging

logger = logging.getLogger(__name__)

from libcpp.string cimport string
from libcpp cimport bool

from trie cimport Trie

cdef class PyTrie:
    
    """ 
    Cython wrapper class for C++ class Trie
    """

    cdef:
        Trie *_thisptr

    def __cinit__(PyTrie self):
        # Initialize the "this pointer" to NULL so __dealloc__
        # knows if there is something to deallocate. Do not 
        # call new Trie() here.
        self._thisptr = NULL
        
    def __init__(PyTrie self):
        # Constructing the C++ object might raise std::bad_alloc
        # which is automatically converted to a Python MemoryError
        # by Cython. We therefore need to call "new Trie()" in
        # __init__ instead of __cinit__.
        self._thisptr = new Trie() 

    def __dealloc__(PyTrie self):
        # Only call del if the C++ object is alive, 
        # or we will get a segfault.
        if self._thisptr != NULL:
            del self._thisptr
            
    cdef int _check_alive(PyTrie self) except -1:
        # Beacuse of the context manager protocol, the C++ object
        # might die before PyTestClass self is reclaimed.
        # We therefore need a small utility to check for the
        # availability of self._thisptr
        if self._thisptr == NULL:
            raise RuntimeError("Wrapped C++ object is deleted")
        else:
            return 0    

    def insert(PyTrie self, string word):
        self._check_alive()
        self._thisptr.insert(word)

    def insert_from_file(PyTrie self, string filename, char separator, bool has_header):
        logger.info("Importing the dictionary from file...")
        logger.info("Filename: {}".format(filename))
        logger.info("Separator: {}".format(<bytes>separator))
        logger.info("First line is header: {}".format(has_header))

        self._check_alive()
        line_number = self._thisptr.insertFromFile(filename, separator, has_header)

        logger.info("# of imported words: {}".format(line_number))
        logger.info("# of trie nodes: {}".format(self._thisptr.size()))
        logger.info("Import is completed.")


    def size(PyTrie self):
        self._check_alive()
        return self._thisptr.size();

    def max_match(PyTrie self, string s, int st):
        self._check_alive()
        return self._thisptr.maxMatch(s, st)

    def match_whole_string(PyTrie self, string s):
        self._check_alive()
        return self._thisptr.matchWholeString(s)

    def get_word(PyTrie self, int id):
        self._check_alive()
        return self._thisptr.getWord(id)

    def dump_dictionary(PyTrie self, string filename):
        self._check_alive()
        self._thisptr.dumpDictionary(filename)
    
    # The context manager protocol allows us to precisely
    # control the liftetime of the wrapped C++ object. del
    # is called deterministically and independently of 
    # the Python garbage collection.

    def __enter__(PyTrie self):
        self._check_alive()
        return self
    
    def __exit__(PyTrie self, exc_tp, exc_val, exc_tb):
        if self._thisptr != NULL:
            del self._thisptr 
            self._thisptr = NULL # inform __dealloc__
        return False # propagate exceptions

