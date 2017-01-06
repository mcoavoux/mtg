
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


# cat sequences

W 0 top cat & S 0 top cat & S 1 top cat
W 0 top cat & S 0 top cat & S 1 top cat & S 2 top cat

W 0 top cat & S 0 top cat & W 1 top cat
W 0 top cat & S 0 top cat & S 1 top cat & W 1 top cat
