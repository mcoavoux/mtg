
**Mind the Gap** is a statistical parser for discontinuous constituent parsing.

If you use this parser please cite (forthcoming):

    Maximin Coavoux, Benoît Crabbé. Incremental Discontinuous Phrase Structure Parsing with the GAP Transition. EACL 2017.

This release is meant to replicate the experiments in the article.
If you are interested in parsing standard constituency trees, or if you want to try
mind the gap with bi-lstm neural networks, checkout a later version of this parser instead (coming soon).


Compile
-------

To compile, you need `g++ >= 4.7` or a recent `clang++`, as well as Boost and Eigen libraries.
If the libraries are not installed on your system:

    cd lib
    bash get_dependencies.bash
    cd ..

This will download all the headers you need in `/lib` folder. Then:

    cd src
    make all
    cd ..

This will create the training and parsing front-ends in `/bin`.


Generate data sets
------------------

To generate datasets, you need [discodop](https://github.com/andreasvc/disco-dop/) (Van Cranenburgh, 2016), [treetools](https://github.com/wmaier/treetools) and python>=3.3 installed on your system (note: `treetools` needs python2.7).
The scripts uses `python3` command to call Python. You might need to create it as an alias / symlink to `python` if it is not available on your machine.

To generate the Hall and Nivre's (2008) split and Maier's (2015) split
(with gold tags, marmot predicted tags, and spmrl organizers' predicted tags).

    cd data
    bash generate_tiger_data.sh
    cd ..

To generate the split for the Negra corpus (assuming the archive `negra-corpus.tar.gz`
is in the data folder).

    cd data
    bash generate_negra_data.sh
    cd ..

Each folder in `/data` (`negraall`, `negra30`, `tigerM15`, `tigerHN8`, `tigerM15pred_spmrl`)
will contain discbracket format corpora (`train.mrg`, `dev.mrg`, `test.mrg`)
and raw parsing data (word/tag format: `dev.raw`, `test.raw`).

Train and parse
---------------

```bash
cd bin

# train
./mtg_gcc -t ../data/tigerM15/train.mrg -d ../data/tigerM15/dev.mrg -i 30 -b 4 -f ../data/templates/gap13_s2fix.md -u -o model
# -t <train> -d <dev> -i <iterations> -b <beamsize> -f <templates> -o <output>
# -u replaces hapaxes in train by an "UNKNOWN" pseudo-word
# ../data/templates/gap13_s2fix.md is the template config file for the full model (baseline+extended+spans, see article)

# parse
./mtg_parse_gcc -x ../data/tigerM15/test.raw -b 4 -o model/parse.discbracket -m model
# -x <raw test data> -b <beamsize> -o <output> -m <model(output of mtg_gcc)>

# evaluate
discodop eval ../data/tigerM15/test.mrg model/parse.discbracket ../data/proper.prm --fmt=discbracket
```

See additional options with `--help` or `-h`.


Transition systems
------------------

- **Gap** transition system (shift, reduce-left(X),reduce-right(X), reduce-unary(X), gap, idle).
    This transition system uses a padding strategy (Zhu et al. 2013) to handle
    derivations of different lengths.
- **Compound Gap** transition system (shift, ghost-reduce, reduce-left(X),reduce-right(X),
    reduce-unary(X), cgap(0 <= i <= n)).
    Every derivation has same length. A shift is necessarily followed
    by either ghost-reduce or reduce-unary.
    A reduce-lr must be preceded by a unique cgap (possibly of order 0).
    Derivations are 4n-2 long (n shifts, n gr/ru, n-1 RR/RL, n-1 cgaps).

Classifiers
-----------

This version of Mind the Gap only supports an averaged perceptron.
Checkout later versions for bi-lstm models.


Some technical details
----------------------

### Feature hashing

Single features are coded on a 32-bit integer. The first bits store
the template unique id, and the rest stores the value for this template.

n-grams templates consists in n integers hashed by some hashing function
(Jenkins). Hashing collision are not resolved. Ths size of the hash table
is fixed (experiments should take less than 6 Go of memory).
To get a more memory-efficient parser you can change the constant
`KERNEL_SIZE` in `src/hash_utils.h`, though it usually harms accuracy.


### Tree-structured stack (TSS)

The full stack (S + D in the article) is structured as a tree to factorize
common prefixes.
In standard implementations (continuous parsing), the TSS can be encoded
directly in parse states, with adequate pointers.
This seems not to be possible for discontinuous parsing because
each time some reordering happens, a new branch grows on the TSS
(it is hard to reconstruct the stack from the parse state and its predecessors,
due to this reordering).

In our implementation, each parse state has a pointer to its
predecessor state, and 2 pointers to the stack items representing
the top of D and S.

Acknowledgements
----------------

This archive includes:

- evaluation config files taken from [discodop](https://github.com/andreasvc/disco-dop/) (`proper.prm`),
and from [the spmrl evaluator](http://pauillac.inria.fr/~seddah/evalb_spmrl2013.tar.gz) (`spmrl.prm`).
- a script to produce Hall and Nivre's (2008) split from [Andreas van Cranenburgh's github page](https://gist.github.com/andreasvc/7507135#file-tigersplit-py)
- headrules from [discodop](https://github.com/andreasvc/disco-dop/) (`negra.headrules`)





References
----------

[Wolfgang Maier. 2015. Discontinuous incremental
shift-reduce parsing. In Proceedings of the 53rd An-
nual Meeting of the Association for Computational
Linguistics and the 7th International Joint Confer-
ence on Natural Language Processing (Volume 1:
Long Papers), pages 1202–1212, Beijing, China,
July. Association for Computational Linguistics.](http://www.aclweb.org/anthology/P/P15/P15-1116.pdf)

[Andreas van Cranenburgh, Remko Scha, and Rens
Bod. 2016. Data-oriented parsing with discontin-
uous constituents and function tags. J. Language
Modelling, 4(1):57–111.](http://jlm.ipipan.waw.pl/index.php/JLM/article/view/100)

[Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang,
and Jingbo Zhu. 2013. Fast and accurate shift-
reduce constituent parsing. In ACL (1), pages 434–
443. The Association for Computer Linguistics.](http://www.aclweb.org/anthology/P/P13/P13-1043.pdf)


