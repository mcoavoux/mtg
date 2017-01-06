#ifndef TREEBANK_H
#define TREEBANK_H

#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <memory>
#include <unordered_map>
#include "tree.h"
#include "utils.h"
#include "grammar.h"


const int SEED  = 1;

using std::ofstream;
using std::endl;


class Treebank{
    vector<Tree> trees_;
    unordered_map<string, int> frequencies;
    int unknown; // if true: replace tokens with less than one occurrence by UNKNOWN pseudoword
    const int MIN_FREQUENCY = 2;

    static std::default_random_engine random;

public:
    friend struct TreebankStats;
    enum {ENCODE_EVERYTHING, CUTOFF, UNKNOWN_CODING};

    ~Treebank();
    Treebank();
    Treebank(const string & filename, int unknown);

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

    void annotate_heads(const HeadRules &hr);
    void annotate_parent_ptr();
    void transform(Grammar &grammar);
    void detransform(Grammar &grammar);


    void read_discbracket_treebank(const string &filename);

    void parse_discbracket_tree(const string &line);

    //shared_ptr<Node> parse_tokens_disco(const vector<string> &tokens, int d, int f){
    void parse_tokens_disco(const vector<string> &tokens, int d, int f, shared_ptr<Node> &res);

    void update_frequencies(const string &line);

    void update_frequencies_tokens(const vector<string> &tokens, int d, int f);

    void write(const string& filename);

    void write(const string& filename, vector<vector<std::pair<string,string>>> &str_sentences);  // write in output file, if unknown token -> use str_sentences to retrieve correct string


private:
    static int aux(const vector<string> &tokens, int d, int f);

public:
    static void test_binarization(const string &tbk_filename, const string &hr_filename);


    // format: tok1/tag1 tok2/tag2 ..... tokn/tagn
    static void read_raw_input_sentences(const string &filename, vector<vector<shared_ptr<Node>>> &raw_test, vector<vector<std::pair<string,string>>> &str_sentences);
    static void read_raw_sentence(const string &line, vector<shared_ptr<Node>> &sentence, vector<std::pair<string,string>> &str_sentence);
    static void test_raw_sentence_reading(const string &filename);
};



struct TreebankStats{
    int rank;
    int gap_degree;
    int n_tokens;
    int n_sentences;
    int longest_sent;
    vector<int> sentence_lengths;
    vector<int> gap_degrees;

    TreebankStats(Treebank &tbk);

    friend ostream & operator<<(ostream &os, TreebankStats &ts);
};


#endif // TREEBANK_H
