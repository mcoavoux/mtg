

+ `mind_the_gap_v1.2`: code for [Maximin Coavoux, Benoît Crabbé. Représentation et analyse automatique des discontinuités syntaxiques dans les corpus arborés en constituants du français. TALN 2017](http://taln2017.cnrs.fr/wp-content/uploads/2017/06/actes_TALN_2017-vol1-1.pdf#page=87)
    and for the experiments presented in my dissertation (forthcoming).
    + Lexicalized and unlexicalized structure-label transition systems
    + Joint constituency parsing, functional labelling and morphological analysis.
    + State-of-the-art pretrained models for
        + discontinuous constituency parsing: French, English (DPTB), German (Negra, Tiger)
        + projective constituency parsing: Arabic, Basque, French, German, Hebrew, Hungarian, Korean, Polish, Swedish (SPMRL)

Older versions:

+ `mind_the_gap_v1.0`: code for [Maximin Coavoux, Benoît Crabbé. Incremental Discontinuous Phrase Structure Parsing with the GAP Transition. EACL 2017.](http://www.aclweb.org/anthology/E/E17/E17-1118.pdf)
    + Discontinuous parsing with the Shift-reduce-gap transition system and a structured perceptron.
+ `mind_the_gap_v1.1`: code for [Maximin Coavoux, Benoît Crabbé. Multilingual Lexicalized Constituency Parsing with Word-Level Auxiliary Tasks. EACL 2017 (short).](http://www.aclweb.org/anthology/E/E17/E17-2053.pdf)
    + Joint (CFG) parsing and tagging with a bi-lstm sentence encoder and a character bi-lstm.
    + Includes some pretrained models
    + Outputs constituency trees and labelled dependency trees
