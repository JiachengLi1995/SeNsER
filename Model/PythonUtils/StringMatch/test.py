from pytrie import PyTrie

test = PyTrie()
print('Inserting ab c')
test.insert(b'ab c')
print('Inserting b cd e')
test.insert(b'b cd e')

matches = test.match_whole_string(b'ab cd ef ab c')
print(matches)

test = PyTrie()

# The following one loads the whole file into the Trie.
# test.insert_from_file(b'../TwitterNER/2015_dict.txt', '\n', false)
# print(test.size())

# The following one loads the whole word2vec embedding format file into the Trie.
test.insert_from_file(b'test_dict.txt', ord(' '), True)
print(test.size())

matches = test.match_whole_string(b'why not __url__')
print(matches)

print(test.get_word(3))