#include "str_utils.h"

namespace str{

void decode(wstring &to,string const &from){
    utf8::utf8to32(from.begin(),from.end(), back_inserter(to));
}

wstring decode(string const &from){
    wstring to;
    utf8::utf8to32(from.begin(),from.end(), back_inserter(to));
    return to;
}

void encode(string &to,wstring const &from){
    utf8::utf32to8(from.begin(),from.end(), back_inserter(to));
}

string encode(wstring const &from){
    string to;
    utf8::utf32to8(from.begin(),from.end(),back_inserter(to));
    return to;
}

char encode(wchar_t fromc){
    wstring from;
    from += fromc;
    string to;
    utf8::utf32to8(from.begin(),from.end(), back_inserter(to));
    return to[0];
}

void split(const string &s, const string &delimiter, const string &keepimiter, vector<string> &result){
    boost::char_separator<char> sep(delimiter.c_str(), keepimiter.c_str());
    boost::tokenizer<boost::char_separator<char>> toks(s, sep);
    result = vector<string>(toks.begin(), toks.end());
}

void split(const wstring &s, const std::string &delimiter, const std::string &keepimiter, vector<wstring> &result){
    boost::char_separator<wchar_t> sep(str::decode(delimiter).c_str(), str::decode(keepimiter).c_str());
    boost::tokenizer<boost::char_separator<wchar_t>,std::wstring::const_iterator, std::wstring> toks(s, sep);
    result = vector<wstring>(toks.begin(), toks.end());
}





//bool xml_beg(wstring const &token){     //tells if a token has XML begin pattern <...>
//    return (token[0] == L'<') && (token.back() == L'>') && (token[1] != L'/');
//}

//bool xml_end(wstring const &token){     //tells if a token has XML end pattern </...>
//    return token[0] == L'<' && token.back() == L'>' && token[1] == L'/';
//}

//bool is_head(wstring const &token){       //tells if the pattern has head annotation '-head'
//    return token.find_first_of(L"-") != wstring::npos;
//}

//wstring xml_strip(wstring const &token){  //removes brackets and annotated stuff
//    unsigned int beg = 0;
//    unsigned int len = 0;
//    unsigned N = token.size();
//    if (N > 0 && token[0] == L'<'){
//        beg = 1;
//        if (N > 1 && token[1] == L'/'){
//            beg=2;
//        }
//    }
//    len = N-beg;
//    if(token.back()==L'>'){
//        len--;
//    }
//    return token.substr(beg,len);
//}

//wstring strip_tag(wstring const &token){  //removes head annotations from tags
//    unsigned int idx = token.find_first_of(L"-");
//    return token.substr(0,idx);
//}


//bool xml_beg(string const &token){     //tells if a token has XML begin pattern <...>
//    return (token[0] == '<') && (token.back() == '>') && (token[1] != '/');
//}

//bool xml_end(string const &token){     //tells if a token has XML end pattern </...>
//    return token[0] == '<' && token.back() == '>' && token[1] == '/';
//}

//bool is_head(string const &token){       //tells if the pattern has head annotation '-head'
//    return token.find_first_of("-") != string::npos;
//}

//string xml_strip(string const &token){  //removes brackets and annotated stuff
//    unsigned int beg = 0;
//    unsigned int len = 0;
//    unsigned N = token.size();
//    if (N > 0 && token[0] == '<'){
//        beg = 1;
//        if (N > 1 && token[1] == '/'){
//            beg=2;
//        }
//    }
//    len = N-beg;
//    if(token.back()=='>'){
//        len--;
//    }
//    return token.substr(beg,len);
//}

//string strip_tag(string const &token){  //removes head annotations from tags
//    unsigned int idx = token.find_first_of("-");
//    return token.substr(0,idx);
//}

}
