#ifndef TREEBANK_H
#define TREEBANK_H

#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <memory>
#include <unordered_map>

#include "str_utils.h"
#include "tree.h"
#include "utils.h"
#include "grammar.h"
#include "random_utils.h"


//const int SEED  = 1;

using std::ofstream;
using std::endl;


// Treebank class stores a collection of trees
//  and some frequency information about tokens
class Treebank{
    vector<Tree> trees_;
    //unordered_map<string, int> frequencies;
    vector<unordered_map<String, int>> field_freqs;
    vector<int> voc_sizes;

    int unknown; // if true: replace tokens with less than one occurrence by UNKNOWN pseudoword
    bool train;
    const int MIN_FREQUENCY = 2;

    static std::default_random_engine random;
    vector<string> header;

    // constant string (typed according to main string type)
    static const String LBRACKET;
    static const String RBRACKET;
    static const String EQUAL;
    static const String HEAD;

#ifdef WSTRING
    static constexpr Char LANGLE = L'<';
    static constexpr Char RANGLE = L'>';
    static constexpr Char SLASH = L'/';
#else
    static constexpr Char LANGLE = '<';
    static constexpr Char RANGLE = '>';
    static constexpr Char SLASH = '/';
#endif

public:
    friend struct TreebankStats;
    enum {ENCODE_EVERYTHING, CUTOFF, UNKNOWN_CODING}; // policy for unknown words

    enum {DISCBRACKET, TBK, OLD_TBK}; // treebank format

    ~Treebank();
    Treebank();
    Treebank(const string & filename, int unknown, int format, bool train);
    Treebank(const string & filename, int unknown, int format, vector<string> &header, bool train);

    Treebank(Treebank &tbk); // deep copy
    Treebank& operator=(Treebank &tbk); // deep copy

    void get_raw_inputs(vector<vector<shared_ptr<Node>>> &v);

    Tree* operator[](int i);

    int size()const;

    void add_tree(const Tree &t);

    void clear();

    int rank();

    int gap_degree();

    void shuffle();

    void subset(int n_sentences, Treebank &tbk);

    void encode_all_from_freqs();
    void encode_known_from_freqs();
    void get_vocsizes(vector<int> &vocsizes);

    void annotate_heads(const HeadRules &hr);
    void annotate_parent_ptr();
    void transform(Grammar &grammar);
    void detransform(Grammar &grammar);


    void read_discbracket_treebank(const string &filename);

    void parse_discbracket_tree(const String &line);

    void parse_tokens_disco(const vector<String> &tokens, int d, int f, shared_ptr<Node> &res);

    void update_frequencies(const String &line);

    void update_frequencies_tokens(const vector<String> &tokens, int d, int f);

    void write(const string& filename);

    void write(const string& filename, vector<vector<std::pair<String,String>>> &str_sentences, bool prob=false);  // write in output file, if unknown token -> use str_sentences to retrieve correct string
    void write_conll(const string& filename, vector<vector<std::pair<String, String> > > &str_sentences, bool unlex);


    void read_tbk_treebank(const string &filename, bool train_set);
    void parse_tbk_tree(const vector<String> &sentence);
    //bool get_tbk_tree(const vector<String> &sentence, int d, int f, int &idx, shared_ptr<Node> &res);
    bool get_tbk_tree(const vector<String> &sentence, int d, int f, shared_ptr<Node> &res);

    void read_old_tbk_treebank(const string &filename, bool train_set);
    void parse_old_tbk_tree(const vector<String> &sentence);
    bool get_old_tbk_tree(const vector<String> &sentence, int d, int f, int &idx, shared_ptr<Node> &res);


    static int find_closing_bracket(int i, const vector<String> &sentence, string head_sep = "^");

    void get_header(vector<string> &header);

private:
    static int aux(const vector<String> &tokens, int d, int f);

public:
    static void test_binarization(const string &tbk_filename, const string &hr_filename);


    // format: tok1/tag1 tok2/tag2 ..... tokn/tagn
    static void read_raw_input_sentences(
            const string &filename,
            vector<vector<shared_ptr<Node>>> &raw_test,
            vector<vector<std::pair<String,String>>> &str_sentences,
            int format,
            int unknown);

    static void read_raw_input_sentences_tbk(
            const string &filename,
            vector<vector<shared_ptr<Node>>> &raw_test,
            vector<vector<std::pair<String,String>>> &str_sentences,
            int unknown);

    static void read_raw_sentence_tbk(
            vector<vector<String>> &sentence,
            vector<vector<shared_ptr<Node>>> &sentences,
            vector<vector<std::pair<String,String>>> &str_sentences,
            int unknown);

    static void read_raw_input_sentences_discbracket(
            const string &filename,
            vector<vector<shared_ptr<Node>>> &raw_test,
            vector<vector<std::pair<String, String> > > &str_sentences,
            int unknown);

    static void read_raw_sentence_discbracket(
            const String &line,
            vector<shared_ptr<Node>> &sentence,
            vector<std::pair<String,String>> &str_sentence,
            int unknown);

    static void test_raw_sentence_reading(const string &filename);

    static bool is_xml_markup(const String &s);
    static bool is_xml_beg(const String &s);
    static bool is_xml_end(const String &s);
    static bool get_xml_label(const String &s, String &label, string head_sep="^"); // return true if is head
    static bool is_head(String &s, string sep="^");

    void encode_fields(const vector<String> &fields, vector<STRCODE> &enc_fields);
};


// Stores some stats about a treebank (gap-degree, number of tokens, rank etc)
struct TreebankStats{
    int rank;
    int gap_degree;
    int n_tokens;
    int n_sentences;
    int longest_sent;
    int num_constituents;
    int num_disco_constituents;
    vector<int> sentence_lengths;
    vector<int> gap_degrees;

    TreebankStats(Treebank &tbk);

    friend ostream & operator<<(ostream &os, TreebankStats &ts);
};


#endif // TREEBANK_H
