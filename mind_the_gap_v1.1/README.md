
This is the code of the multilingual shift-reduce constituency parser described in the following article:

Maximin Coavoux, Benoît Crabbé. Multilingual Lexicalized Constituency Parsing with Word-Level Auxiliary Tasks. EACL 2017 (short). (pdf soon)


Quick start
-----------

Download headers for Eigen and Boost:

```bash
cd lib
bash get_dependencies.sh
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


Parse:

```bash
cd ..
cd bin
./mtg_parse -x ../data/french_ex.raw -b 1 -o corpus.parsed -m ../pretrained_models/FRENCH -F 1
# -x <input> -b <beamsize> -o <output> -m <model> -F <format indication>
```

This will output 2 files:
- `corpus.parsed` is a discbracket format corpus (use [discodop](/home/mcoavoux/Documents/MTG_eacl/mtg/mind_the_gap_v1.1/pretrained_models/)
    to convert it to a ptb style treebank).
- `corpus.parsed.conll` is the corresponding labelled dependency corpus
    (see paper for details).

See additional options with `--help` or `-h` (disclaimer: help message might not be up-to-date).

Train a model (later):

