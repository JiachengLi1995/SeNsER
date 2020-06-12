# String Match in C++

Assuming you have Python, Cython, and a C++ compiler.

```
$ python setup.py build_ext --inplace
$ python test.py
```

The expected output should be
```
Inserting ab c
Inserting b cd e
[1, 2, 2, 2, 2, 2, 2, 0, -1, 1, 1, 1, 1]
```

If you want to install it globally, please run
```
$ python setup.py install
```

## Tips

`str` in `Python` cannot be passed as the `string` argument in ``C++``. For example, `trie.insert('abc')` will fail, but `trie.insert(b'abc')` is ok.
