# GAP minimal templates, inspired by Maier 2015
# (shifted to take second stack into account), also no difference between unary descendant and left descendent
# BEWARE: not exactly identical to java_prototype/gap_minimal_templates.
#    java : top stack is indexed with -1, -2 etc
#    c++  :                            0,  1 etc   
#       (don't ask why)
#####
## unigrams
#   s 0 tc, s 0 wc, s 1 tc, s 1 wc, s 2 tc, s 2 wc, s 3 tc, s 3 wc,
#   q 0 wt, q 1 wt, q 2 wt, q 3 wt,
#   s 0 lwc, s 0 rwc, s 0 uwc, s 1 lwc, s 1 rwc, s 1 uwc
# bigrams
#   s 0 ws 1 w, s 0 ws 1 c, s 0 cs 1 w, s 0 cs 1 c, s 0 wq 0 w, s 0 wq 0 t,
#   s 0 cq 0 w, s 0 cq 0 t, s 1 wq 0 w, s 1 wq 0 t, s 1 cq 0 w, s 1 cq 0 t,
#   q 0 wq 1 w, q 0 wq 1 t, q 0 tq 1 w, q 0 tq 1 t
# trigrams
#   s 0 cs 1 cs 2 w, s 0 cs 1 cs 2 c, s 0 cs 1 cq 0 w, s 0 cs 1 cq 0 t,
#   s 0 cs 1 wq 0 w, s 0 cs 1 wq 0 t, s 0 ws 1 cs 2 c, s 0 ws 1 cq 0 t
# extended
#   s 0 llwc, s 0 lrwc, s 0 luwc, s 0 rlwc, s 0 rrwc,
#   s 0 ruwc, s 0 ulwc, s 0 urwc, s 0 uuwc, s 1 llwc,
#   s 1 lrwc, s 1 luwc, s 1 rlwc, s 1 rrwc, s 1 ruwc
# separator
#   s 0 wp, s 0 wcp, s 0 wq, s 0 wcq, s 0 cs 1 cp, s 0 cs 1 cq
#   s 1 wp, s 1 wcp, s 1 wq, s 1 wc

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

B 0 tag & W 0 left_corner form & W 0 right_corner form
B 1 tag & W 0 left_corner form & W 0 right_corner form

