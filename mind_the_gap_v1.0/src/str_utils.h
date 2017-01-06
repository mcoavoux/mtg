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

  //UTF8/32
  inline void decode(wstring &to,string const &from){
    utf8::utf8to32(from.begin(),from.end(), back_inserter(to));
  }
  inline wstring decode(string const &from){
      wstring to;
      utf8::utf8to32(from.begin(),from.end(), back_inserter(to));
      return to;
  }
  
  //Encodes an utf-32 wstring to an utf8-string
  //Used at ouput
  inline void encode(string &to,wstring const &from){
    utf8::utf32to8(from.begin(),from.end(), back_inserter(to));
  }
	
  inline string encode(wstring const &from){
    string to;
    utf8::utf32to8(from.begin(),from.end(),back_inserter(to));
    return to;
  }

    inline void split(const string &s, const string &delimiter, const string &keepimiter, vector<string> &result){
        boost::char_separator<char> sep(delimiter.c_str(), keepimiter.c_str());
        boost::tokenizer<boost::char_separator<char>> toks(s, sep);
        result = vector<string>(toks.begin(), toks.end());
    }

////////////////////////////////////////////////////////////////////////////////////////////
//Line tokenizer

/**
 * This implements a simple lexer, useful for processing data formats in many NLP tasks.
 * It is made for processing UTF-8 documents.
 * The lexer splits a string into tokens, performs normalisation and removes comments.
 * It yields vectors of tokens encoded in UTF-32
 */
  
  class SimpleTokenizer{
        
  public:

    /**
     * @param tok_delimiters: a set of chars used as token delimiters
     * @param comment_delimiters: a char used a comment delimiter
     * @return a vector of strings
     */
    SimpleTokenizer(wstring const &tok_delimiters = L" \t\f",wchar_t line_comment_delim=L'#');
        
    //processes an UTF-8 line, converts it in UTF-32
    //and fills the tokens vector with lexed content on this line
    //returns the arity (#tokens read on this line)
    unsigned int llex(string const &line,vector<wstring> &tokens);
    
    //processes an UTF-32 line and fills the tokens vector with lexed content on this line
    //returns the arity (#tokens read on this line)
    unsigned int llex(wstring const &line,vector<wstring> &tokens)const;
           
  private:
    static const wstring whitespace;
    
    wstring tok_delim;
    wstring all_delim;
    wchar_t com_delim;
    wstring leading_delim;
  }; 

////////////////////////////////////////////////////////////////////////////////////////////
//PSEUDO-XML
  
  bool xml_beg(wstring const &token);       //tells if a token has XML begin pattern <...>
  bool xml_end(wstring const &token);       //tells if a token has XML end pattern </...>
  bool is_head(wstring const &token);       //tells if the pattern has head annotation
  wstring xml_strip(wstring const &token);  //removes brackets and annotated stuff
  bool is_tag(wstring const &token);         //says if a tag is head annotated
  wstring strip_tag(wstring const &token);  //removes head annotations from tags

}

#endif
