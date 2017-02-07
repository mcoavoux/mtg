#ifndef STR_UTILS_H
#define STR_UTILS_H


#include <string>
#include <vector>
#include <boost/tokenizer.hpp>

#include "utf8.h"

using std::wstring;
using std::string;
using std::vector;

namespace str{

////////
/// utf8

void decode(wstring &to,string const &from);
wstring decode(string const &from);
void encode(string &to,wstring const &from);
string encode(wstring const &from);

char encode(wchar_t fromc);

void split(const string &s, const string &delimiter, const string &keepimiter, vector<string> &result);
void split(const wstring &s, const string &delimiter, const string &keepimiter, vector<wstring> &result);

////////////
//PSEUDO-XML

//bool xml_beg(wstring const &token);       //tells if a token has XML begin pattern <...>
//bool xml_end(wstring const &token);       //tells if a token has XML end pattern </...>
//bool is_head(wstring const &token);       //tells if the pattern has head annotation
//wstring xml_strip(wstring const &token);  //removes brackets and annotated stuff
//bool is_tag(wstring const &token);         //says if a tag is head annotated
//wstring strip_tag(wstring const &token);  //removes head annotations from tags

//bool xml_beg(string const &token);       //tells if a token has XML begin pattern <...>
//bool xml_end(string const &token);       //tells if a token has XML end pattern </...>
//bool is_head(string const &token);       //tells if the pattern has head annotation
//string xml_strip(string const &token);  //removes brackets and annotated stuff
//bool is_tag(string const &token);         //says if a tag is head annotated
//string strip_tag(string const &token);  //removes head annotations from tags

}

#endif
