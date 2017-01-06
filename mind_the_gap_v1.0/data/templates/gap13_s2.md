# Unigrams
B 0 tag & B 0 form
B 1 tag & B 1 form
B 2 tag & B 2 form
B 3 tag & B 3 form

W 0 top tag & W 0 top cat
W 0 top form & W 0 top cat

S 0 top tag & S 0 top cat
S 0 top form & S 0 top cat

S 1 top tag & S 1 top cat
S 1 top form & S 1 top cat

S 2 top tag & S 2 top cat
S 2 top form & S 2 top cat

S 0 left form & S 0 left cat
S 0 right form & S 0 right cat

W 0 left form & W 0 left cat
W 0 right form & W 0 right cat

# Bigrams

S 0 top form & W 0 top form
S 0 top form & W 0 top cat
S 0 top cat & W 0 top form
S 0 top cat & W 0 top cat

W 0 top form & B 0 form
W 0 top form & B 0 tag
W 0 top cat & B 0 form
W 0 top cat & B 0 tag

S 0 top form & B 0 form
S 0 top form & B 0 tag
S 0 top cat & B 0 form
S 0 top cat & B 0 tag

B 0 form & B 1 form
B 0 form & B 1 tag
B 0 tag & B 1 form
B 0 tag & B 1 tag

# Trigrams

W 0 top cat & S 0 top cat & S 1 top form
W 0 top cat & S 0 top cat & S 1 top cat

W 0 top cat & S 0 top cat & B 0 form
W 0 top cat & S 0 top cat & B 0 tag

W 0 top cat & S 0 top form & B 0 form
W 0 top cat & S 0 top form & B 0 tag

# something weird for this last pair
W 0 top form & S 0 top cat & S 1 top cat
W 0 top form & S 0 top cat & B 0 tag



# more context

S 3 top tag & S 3 top cat
S 3 top form & S 3 top cat

S 1 left form & S 1 left cat
S 1 right form & S 1 right cat

W 1 top tag & W 1 top cat
W 1 top form & W 1 top cat

W 2 top tag & W 2 top cat
W 2 top form & W 2 top cat

# cat sequences

W 0 top cat & S 0 top cat & S 1 top cat
W 0 top cat & S 0 top cat & S 1 top cat & S 2 top cat

W 0 top cat & S 0 top cat & W 1 top cat
W 0 top cat & S 0 top cat & S 1 top cat & W 1 top cat





# span features adapted from Crabbé EMNLP15


W 0 top cat & W 0 left_corner form & W 0 right_corner form
S 0 top cat & S 0 left_corner form & S 0 right_corner form

W 0 top cat & W 0 left_corner form & S 0 right_corner form
W 0 top cat & W 0 right_corner form & S 0 left_corner form

B 0 form & W 0 left_corner form & W 0 right_corner form
B 1 form & W 0 left_corner form & W 0 right_corner form

W 0 top cat & W 0 right_corner form & S 0 left_corner_out form


W 0 top cat & W 0 left_corner tag & W 0 right_corner form
W 0 top cat & W 0 left_corner form & W 0 right_corner tag
W 0 top cat & W 0 left_corner tag & W 0 right_corner tag

S 0 top cat & S 0 left_corner tag & S 0 right_corner form
S 0 top cat & S 0 left_corner form & S 0 right_corner tag
S 0 top cat & S 0 left_corner tag & S 0 right_corner tag

W 0 top cat & W 0 left_corner tag & S 0 right_corner form
W 0 top cat & W 0 left_corner form & S 0 right_corner tag
W 0 top cat & W 0 left_corner tag & S 0 right_corner tag

W 0 top cat & W 0 right_corner tag & S 0 left_corner form
W 0 top cat & W 0 right_corner form & S 0 left_corner tag
W 0 top cat & W 0 right_corner tag & S 0 left_corner tag

B 0 tag & W 0 left_corner form & W 0 right_corner form
B 1 tag & W 0 left_corner form & W 0 right_corner form

W 0 top cat & W 0 left_corner_out form
W 0 top cat & W 0 left_corner_out tag

S 0 top cat & S 0 top right_corner_out form
S 0 top cat & S 0 top right_corner_out tag













