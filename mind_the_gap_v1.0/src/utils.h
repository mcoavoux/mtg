#ifndef UTILS_H
#define UTILS_H

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <assert.h>

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



typedef string STRING;
typedef char CHAR;
typedef unsigned int STRCODE;

namespace enc{
    const int MAX_FIELDS = 40;
    enum {CAT, TAG, TOK, DEP};
    enum {UNKNOWN, UNDEF};

    class TypedStrEncoder;
    struct StrDict{
        unordered_map<STRING,int> encoder;
        vector<STRING> decoder;
        int size_;

        StrDict();

        STRCODE code(STRING s);

        STRCODE code_unknown(STRING s);

        STRING decode(STRCODE i);
        int size();

        friend ostream & operator<<(ostream &os, StrDict &ts);
    };

    class TypedStrEncoder{
        vector<StrDict> encoders;
    public:
        TypedStrEncoder();
        STRCODE code(STRING s, int type);
        STRCODE code_unknown(STRING s, int type);
        STRING decode(STRCODE i, int type);
        int size(int type);
        void reset();
        void export_model(const string &outdir);
        void import_model(const string &outdir);
    };

    extern TypedStrEncoder hodor;
}


#endif // UTILS_H
