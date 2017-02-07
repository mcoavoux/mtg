#ifndef GRAMMAR_H
#define GRAMMAR_H


#include <vector>
#include <unordered_set>
#include <tuple>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <boost/tokenizer.hpp>
#include <ostream>
#include <fstream>
#include "utils.h"
#include "str_utils.h"
#include "tree.h"

using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::pair;
using std::make_pair;
using std::ostream;
using std::ifstream;
using std::ofstream;


class Node;

// Grammar contains information about binarization
// which symbols are root
// which symbols are tmp
// which are punctuation (REMINDER: corpus dependant, for now, negra and tiger should be fine, other corpora are not supported)
// link and reverse links between tmp and non tmp
class Grammar{
public:
    static const int MAX_SIZE = 3000;            // assume there are at most MAX_SIZE non terminals (including tags). Most languages: 300 should be enough, German SPMRL -> 2000, Korean SPMRL -> 3000
#ifdef WSTRING
    static constexpr Char UNARY_CODE = L'@';     // unary chains symbols e.g. S@NP
    static constexpr Char TMP_CODE = L':';       // temporary symbols, e.g. NP:
    static constexpr Char DOLLAR = L'$';
#else
    static constexpr Char UNARY_CODE = '@';     // unary chains symbols e.g. S@NP
    static constexpr Char TMP_CODE = ':';       // temporary symbols, e.g. NP:
    static constexpr Char DOLLAR = '$';
#endif
private:
    vector<bool> axioms;    // true if symbol is axiom
    vector<bool> tmps;      // true if symbol is tmp
    vector<bool> unary;     // unary chain
    vector<bool> punct;     // true if punctuation symbol

    vector<pair<STRCODE, STRCODE>> unary_splits; // un_decomp[i] decomposes in 2 symbols A@B@C -> A@B and C

    vector<unordered_set<STRCODE>> tmps2nontmps; // set of possible non tmps for a tmp, e.g. NP: -> NP, ROOT@NP
    vector<STRCODE> nontmps2tmps;                // gives corresponding tmp

    bool binarise_;
    //bool merge_unary_chains_;
public:
    Grammar();

    Grammar(const string &outdir);

    bool binarise();
    void do_not_binarise();
    //bool merge_unaries(){return merge_unary_chains_;}


    STRCODE merge_unary_chain(STRCODE nt1, STRCODE nt2);    // return nt1@nt2
    pair<STRCODE,STRCODE> get_unary_split(STRCODE nt);      // return nt1,nt2  (nt corresponds to nt1@nt2
    STRCODE get_tmp(STRCODE non_tmp);                       // get tmp symbol corresponding to non_tmp:
    void add_axiom(STRCODE code);

    // Test types of symbols (is a possible axiom? is tmp ? is merged unary chain ?)
    bool is_axiom(STRCODE code) const;
    bool is_tmp(STRCODE code) const;
    bool is_unary_chain(STRCODE code) const;
    bool is_punct(STRCODE code) const;

    void mark_punct();

    static void print_bit_vec(const string &outfile, vector<bool> &vec);
    static void import_bit_vec(const string &outfile, vector<bool> &vec);
    void export_model(const string &outdir);



    friend ostream& operator<<(ostream &os, const Grammar &grammar);
};


enum {LEFT_TO_RIGHT, RIGHT_TO_LEFT};  // order for head rules

struct RulePriority{
    int direction;                  // left to right or right to left
    vector<STRCODE> priority;       // priority list

    RulePriority(int direction, const vector<STRCODE> &priority);

    friend ostream& operator<<(ostream& os, const RulePriority &rp);
};

class HeadRules{
    vector<vector<RulePriority>> rules;
    Grammar *grammar_;
public:
    HeadRules(const string &filename, Grammar* grammar);
    ~HeadRules();

    void add(STRCODE nt, const RulePriority &rp);

    int find_head(Node &n) const;

    void read_from_file(const string &filename);

    void parse_line(const string &buffer);

    const Grammar * grammar()const;

    friend ostream& operator<<(ostream &os, const HeadRules &hr);

    static void test(const string &filename);
};




#endif // GRAMMAR_H
