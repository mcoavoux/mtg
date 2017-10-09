#ifndef UTILS_H
#define UTILS_H

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <assert.h>
//#include <memory>

#include "str_utils.h"

#define DBG(x) cerr << x << endl;

using std::unordered_map;
using std::vector;
using std::string;
using std::wstring;
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::ostream;
//using std::unique_ptr;


#ifdef WSTRING
typedef wstring String;
typedef wchar_t Char;
#else
typedef string String;
typedef char Char;
#endif

typedef unsigned int STRCODE;

// Functions that handle coding typed string on integers
namespace enc{
    const int MAX_FIELDS = 40;
    enum {CAT, TOK, TAG, DEP};  // predefined types
    enum {UNKNOWN, UNDEF};      // predefined values

    class TypedStrEncoder;

    // String -> int dictionary
    struct StrDict{
        unordered_map<String,int> encoder;
        vector<String> decoder;
        int size_;

        StrDict();

        STRCODE code(String s);

        STRCODE code_unknown(String s);

        String decode(STRCODE i);

        int size();

        friend ostream & operator<<(ostream &os, StrDict &ts);
    };

    struct Frequencies{
        vector<float> counts;
        double total;
        void update(STRCODE code, double count);
        double freq(STRCODE code);
    };

    // String, type (int) -> int dictionary
    class TypedStrEncoder{
        vector<StrDict> encoders;
        vector<string> header;
        Frequencies freqs;
    public:
        TypedStrEncoder();
        STRCODE code(String s, int type);
        STRCODE code_unknown(String s, int type);
        String decode(STRCODE i, int type);
        string decode_to_str(STRCODE i, int type);
        int size(int type);
        void vocsizes(vector<int> &sizes);
        void reset();
        void export_model(const string &outdir);
        void import_model(const string &outdir);
        void set_header(vector<string> &header);
        int get_dep_idx();
        void update_wordform_frequencies(unordered_map<String, int> &freqdict);
        double get_freq(STRCODE code);
        string get_header(int i);
    };

    extern TypedStrEncoder hodor;
}


struct Tokenizer{
    enum {NOTHING, CHAR, TOKEN, SUFFIX, LAZY_CHAR};
    int type;
    Tokenizer();
    Tokenizer(int type);
    void operator()(String s, vector<String> &segments);
    void tokenize_on_chars(String s, vector<String> &segments);
    void tokenize_on_tokens(String s, vector<String> &segments);
    void tokenize_suffix(String s, vector<String> &segments);
    void tokenize_lazy_chars(String s, vector<String> &segments);
};




// Base class for splitting a token into components
// (EDU -> words, words -> chars, etc)
struct SequenceEncoder{
    enum {CHAR_LSTM, TOKEN_LSTM};

    enc::StrDict encoder;

    vector<vector<int>> dictionary;
    Tokenizer tokenizer;

    SequenceEncoder();
    SequenceEncoder(int i);
    SequenceEncoder(const string &outdir);

    void init();

    void export_model(const string &outdir);

    vector<int>* operator()(int code);

    int char_voc_size();
};



#endif // UTILS_H
