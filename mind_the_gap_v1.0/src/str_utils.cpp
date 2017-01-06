#include "str_utils.h"

namespace str{

  inline char encode(wchar_t fromc){
    wstring from;
    from += fromc;
    string to;
    utf8::utf32to8(from.begin(),from.end(), back_inserter(to));
    return to[0];
  }

  SimpleTokenizer::SimpleTokenizer(wstring const &tok_delimiters,wchar_t line_comment_delim){
    tok_delim = tok_delimiters;
    com_delim = line_comment_delim;
    all_delim = tok_delim;
    all_delim.push_back(com_delim);
    leading_delim = tok_delim+whitespace;
  }

  unsigned int SimpleTokenizer::llex(string const &line,vector<wstring> &tokens){
    wstring u32;
    str::decode(u32,line);
    return llex(u32,tokens);
  }
    
  const wstring SimpleTokenizer::whitespace = L" \t\f\v\n\r";

    
  unsigned int SimpleTokenizer::llex(wstring const &line,vector<wstring> &tokens)const{
    tokens.clear();
    unsigned int arity = 0;
    //tokenize
    size_t lastPos = line.find_first_not_of(leading_delim,0);//skips consecutive leading seps & whitespace
    if (lastPos == wstring::npos){return arity;}
    if (line[lastPos] == com_delim){return arity;}//leading comments
    size_t nextPos = line.find_first_of(all_delim,lastPos+1);
    while(nextPos != wstring::npos){
      tokens.push_back(line.substr(lastPos,nextPos-lastPos));
      ++arity;
      if (line[nextPos] == com_delim){return arity;}//abort on comment
      lastPos = line.find_first_not_of(tok_delim,nextPos+1);
      if (lastPos == wstring::npos || line[lastPos] == com_delim){return arity;}//abort on comment
      nextPos = line.find_first_of(all_delim,lastPos+1);
    }
    if (lastPos != string::npos){tokens.push_back(line.substr(lastPos));}
    ++arity;
    return arity;
  }

  bool xml_beg(wstring const &token){     //tells if a token has XML begin pattern <...>
    return (token[0] == L'<') && (token.back() == L'>') && (token[1] != L'/');
  }

  bool xml_end(wstring const &token){     //tells if a token has XML end pattern </...>
    return token[0] == L'<' && token.back() == L'>' && token[1] == L'/';
  }

  bool is_head(wstring const &token){       //tells if the pattern has head annotation '-head'
    return token.find_first_of(L"-") != wstring::npos;
  }

  wstring xml_strip(wstring const &token){  //removes brackets and annotated stuff
    unsigned int beg = 0;
    unsigned int len = 0;
    unsigned N = token.size();
    if (N > 0 && token[0] == L'<'){
      beg = 1;
      if (N > 1 && token[1] == L'/'){
	beg=2;
      }
    }
    len = N-beg;
    if(token.back()==L'>'){
      len--;
    }
    return token.substr(beg,len);
  }

  wstring strip_tag(wstring const &token){  //removes head annotations from tags
    unsigned int idx = token.find_first_of(L"-");
    return token.substr(0,idx);
  }
}
