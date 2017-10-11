
This is the code of the multilingual shift-reduce constituency parser described in the following article:

Maximin Coavoux, Benoît Crabbé. Multilingual Lexicalized Constituency Parsing with Word-Level Auxiliary Tasks. EACL 2017 (short). [[pdf]](http://www.aclweb.org/anthology/E/E17/E17-2053.pdf) [[bib]](http://www.aclweb.org/anthology/E/E17/E17-2053.bib)


Quick start
-----------

Download headers for Eigen and Boost:

```bash
cd lib
bash get_dependencies.bash
cd ..
```

Compile:

```bash
mkdir bin
cd src
make wstring
cd ..
```

NB: without `wstring`, strings will be encoded with basic `char` instead
of `wchar_t`, which will confuse the character bi-lstm.


### Parse

```bash
cd bin
./mtg_parse -x ../data/french_ex.raw -b 1 -o corpus.parsed -m ../pretrained_models/FRENCH -F 1
# -x <input> -b <beamsize> -o <output> -m <model> -F <format indication>
```

This will output 2 files:
- `corpus.parsed` is a discbracket format corpus (use [discodop](/home/mcoavoux/Documents/MTG_eacl/mtg/mind_the_gap_v1.1/pretrained_models/)
    to convert it to a ptb style treebank).
- `corpus.parsed.conll` is the corresponding labelled dependency corpus
    (see paper for details).


For models which need only tokens as input (i.e. no tag), you can also
read stdin and output to stdout. This will only output constituency trees:

```bash
echo "Le chat mange une pomme ." | ./mtg_parse -m ../pretrained_models/FRENCH -b 1
```

See additional options with `--help` or `-h` (disclaimer: help message might not be up-to-date).

### Train a model


Sample data are available in `/data/sample/`. It must be in a pseudo-xml
format, where each has its own line and has a set of attributes.
The header specifies the name of the attributes.

To train a model, use the following command line:

    ./mtg -t ../data/sample/train.tbk -d ../data/sample/dev.tbk -i 140 -f ../data/spmrl_templates4.md -o mymodel -N ../data/hyperparameters -F 1 -r
    # ./mtg -t <train data>  -d <dev data> -i <num iterations> -f <feature templates> -o <output>  -N <hyperparameter file>

The option `-F 1` is a format indication and `-r` tells the parser to use a projective
transition system (default is shift-reduce-gap).


#### Template file specification

See `/data/spmrl_templates4.md` for an example.
Positions in the buffer are addressed with `B 0`, `B 1`, etc...
Positions in the stack are addressed with `W 0`, `S 0`, `S 1`, etc...

    W 0 top cat     # non terminal on first element in stack
    S 0 top cat     # non terminal on second element in stack
    S 1 top cat     # non terminal on third element in stack

    B 0 form        # first token in buffer 

    W 0 top form    # token that is the head of the first element in stack
    S 0 top form    # token that is the head of the second element in stack

    S 1 right_corner form  # token that is the rightmost token of the third constituent in stack

    S 0 left_corner form   # token that is the leftmost token of the second constituent in stack
    S 0 right_corner form  # token that is the rightmost token of the second constituent in stack

    W 0 right_corner form  # token that is the rightmost token of the first constituent in stack
    W 0 left_corner form   # token that is the leftmost token of the first constituent in stack

#### Hyperparameter specification

See `/data/hyperparameters` for an example.

    learning rate 0.02                # typical values: 0.01, 0.02
    decrease constant 1e-06           # 1e-6 / 0
    gradient clipping 1               # use hard gradient clipping (0 or 1)
    clip value 5                      # clip every coefficient in gradient with abs value > 5
    gaussian noise 1                  # adds some gaussian noise to gradient (0 or 1)
    gaussian noise eta 0.01           # value for gaussian noise
    hidden layers 2                   # number of hidden layers in action classifier (typical values: 1, 2)
    size hidden layers 32             # size of hidden layers for action classifier (32, 64, 128, 256)
    embedding sizes 16 32 16 4 4 4 4  # size of symmbol embeddings [non-terminal, token, tag, morph1, morpho2, etc] (when predicted morpho is used as input)
    bi-rnn 1                          # use a bi-rnn feature 
    cell type 2                       # use LSTM cells in bi-rnn (0: SRNN, 1: GRU)
    rnn depth 4                       # number of layers of bi-rnn (2: 1 layer, 4: 2 layers)
    rnn state size 32                 # size of lstm states (forward and backward), (64, 128, 256)
    number of token feature (rnn) 1   # number of items used as input to the bi-rnn: 1=only use token, 2=use token+tag, 3=use token+tag+morph1, etc.
    char rnn 1                        # use a character bi-rnn to construct char-based embeddings (on top on standard embeddings). value: 0 or 1
    char embedding size 16            # size of character embeddings
    char based embedding size 16      # size of rnn states for character bi-rnn
    auxiliary task 1                  # use auxiliary tasks
    auxiliary task max idx 20         # index of 


For example, if the data has this header: 
    
    word	tag	case	degree	gender	mood	number	person	tense	gdeprel


With the following options:

    number of token feature (rnn)	1
    auxiliary task	1
    auxiliary task max idx	20

The input to the bi-rnn is only the sequence of tokens. Auxiliary tasks
consist in predicting the POS tag, the case, the degree, ... the tense, and the dependency relation
of each token.


With the following options:

    number of token feature (rnn)	9
    auxiliary task	1
    auxiliary task max idx	20

The input to the bi-rnn for each token is the concatenation [word embedding, tag embedding, .... tense embedding].
And the only auxiliary task is to predict the dependency relation.




