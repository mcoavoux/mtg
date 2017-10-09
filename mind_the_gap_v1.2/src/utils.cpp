
#include "utils.h"


namespace enc{

TypedStrEncoder hodor;

StrDict::StrDict() : size_(0){
#ifdef WSTRING
    code(L"UNKNOWN");
    code(L"UNDEF");
#else
    code("UNKNOWN");
    code("UNDEF");
#endif
}

STRCODE StrDict::code(String s){
    auto it = encoder.find(s);
    if (it == encoder.end()){
        encoder[s] = size_ ++;
        decoder.push_back(s);
        return size_-1;
    }else{
        return it->second;
    }
}

STRCODE StrDict::code_unknown(String s){
    auto it = encoder.find(s);
    if (it == encoder.end()){
        return UNKNOWN;
    }else{
        return it->second;
    }
}

String StrDict::decode(STRCODE i){
    assert(i < encoder.size() && size_ == encoder.size() && "hodor error: decoding unknown code");
    return decoder.at(i);
}

int StrDict::size(){
    return size_;
}

ostream & operator<<(ostream &os, StrDict &ts){
    for (int i = 0; i < ts.decoder.size(); i++){
#ifdef WSTRING
        os << str::encode(ts.decoder[i]) << endl;
#else
        os << ts.decoder[i] << endl;
#endif
    }
    return os;
}

void Frequencies::update(STRCODE code, double count){
    while (code >= counts.size()){
        counts.push_back(0.0);
    }
    counts[code] += count;
    total += count;
}

double Frequencies::freq(STRCODE code){
    assert(code < counts.size());
    //return counts[code] / total;
    return counts[code];
}



TypedStrEncoder::TypedStrEncoder() : encoders(MAX_FIELDS){
#ifdef DEBUG
    cerr << "Hold the door!" << endl;
#endif
}

STRCODE TypedStrEncoder::code(String s, int type){
    return encoders[type].code(s);
}
STRCODE TypedStrEncoder::code_unknown(String s, int type){
    return encoders[type].code_unknown(s);
}
String TypedStrEncoder::decode(STRCODE i, int type){
    assert(type < encoders.size() && "hodor error: type unknown");
    return encoders[type].decode(i);
}

string TypedStrEncoder::decode_to_str(STRCODE i, int type){
#ifdef WSTRING
    return str::encode(decode(i,type));
#else
    return decode(i,type);
#endif
}

int TypedStrEncoder::size(int type){
    return encoders[type].size();
}

void TypedStrEncoder::vocsizes(vector<int> &sizes){
    sizes.clear();
    for (int i = 0; i < encoders.size(); i++){
        sizes.push_back(encoders[i].size());
    }
}

void TypedStrEncoder::reset(){
    cerr << "Warning : string encoder has reset." << endl;
    encoders = vector<StrDict>(MAX_FIELDS);
}

void TypedStrEncoder::export_model(const string &outdir){
    ofstream os(outdir + "/encoder_id");
    for (int i = 0; i < encoders.size(); i++){
        if (encoders[i].size() > 2){
            os << i << endl;
            ofstream ost(outdir + "/encoder_t" + std::to_string(i));
            ost << encoders[i];
            ost.close();
        }
    }
    os.close();

    ofstream os_h(outdir + "/encoder_header");
    if (header.size() > 0){
        os_h << header[0];
        for (int i = 1; i < header.size(); i++){
            os_h << "\t" << header[i];
        }
    }
    os_h.close();
}

void TypedStrEncoder::import_model(const string &outdir){
    reset();
    ifstream is(outdir + "/encoder_id");
    string buffer;
    while (getline(is, buffer)){
        int i = stoi(buffer);
        ifstream ist(outdir + "/encoder_t" + std::to_string(i));

        string buf;
        getline(ist,buf);
#ifdef WSTRING
        wstring wbuf = str::decode(buf);
        assert(wbuf == decode(UNKNOWN,i));
        getline(ist, buf);
        wbuf = str::decode(buf);
        assert(wbuf == decode(UNDEF,i));
        while (getline(ist,buf)){
            wbuf = str::decode(buf);
            code(wbuf, i);
        }
#else
        assert(buf == decode(UNKNOWN,i));
        getline(ist,buf);
        assert(buf == decode(UNDEF,i));
        while (getline(ist,buf)){
            code(buf, i);
        }
#endif
        ist.close();
    }
    is.close();

    ifstream is_h(outdir + "/encoder_header");
    getline(is_h, buffer);
#ifdef DEBUG
    cerr << "Loaded header:" << endl;
    cerr << buffer << endl;
#endif
    str::split(buffer, "\t", "", this->header);
    is_h.close();
}

void TypedStrEncoder::set_header(vector<string> &header){
    this->header = header;
}

int TypedStrEncoder::get_dep_idx(){
    for (int i = 0; i < this->header.size(); i++){
        if (header[i] == "gdeprel")
            return i;
    }
    return -1;
}

void TypedStrEncoder::update_wordform_frequencies(unordered_map<String, int> &freqdict){
    for (auto &it : freqdict){
        STRCODE code = hodor.code(it.first, enc::TOK);
        freqs.update(code, it.second);
    }
}

double TypedStrEncoder::get_freq(STRCODE code){
    return freqs.freq(code);
}

string TypedStrEncoder::get_header(int i){
//    cerr << header.size() << i << endl;
//    for (int i = 0; i < header.size(); i++){
//        cerr << header[i] << " ";
//    }
//    cerr << endl;
    assert(i < header.size());
    return header[i];
}

}








Tokenizer::Tokenizer():Tokenizer(CHAR){}

Tokenizer::Tokenizer(int type): type(type){
#ifdef DEBUG
    cerr << "Tokenizer type : " << type << endl;
#endif
    // TODO: more information
}

void Tokenizer::operator()(String s, vector<String> &segments){
    switch(type){
    case CHAR : tokenize_on_chars(s, segments); break;
    case TOKEN :tokenize_on_tokens(s, segments); break;
    case SUFFIX: tokenize_suffix(s, segments);  break;
    case LAZY_CHAR :tokenize_lazy_chars(s, segments); break;
    default:
        assert(false);
    }
//    cerr << str::encode(s) << endl;
//    cerr << segments.size() << ":";
//    for (int i = 0; i < segments.size(); i++){
//        cerr << str::encode(segments[i]) << " ";
//    }cerr << endl;
}

void Tokenizer::tokenize_on_chars(String s, vector<String> &segments){
    segments = vector<String>(s.size());
    for (int i = 0; i < s.size(); i++){
        segments[i] += s[i];
    }
}

void Tokenizer::tokenize_on_tokens(String s, vector<String> &segments){
    str::split(s, "__", "", segments); // REMINDER -> data must be preprocessed accordingly
}

void Tokenizer::tokenize_suffix(String s, vector<String> &segments){
    segments.clear();
    segments.push_back(String(1, s[0]));

    int start = s.size() - 3;
    start = start < 0 ? 0 : start;
    segments.push_back(String(s.begin()+start, s.end()));

//    if (s.size() < 4){
//        segments.push_back(s);
//        return;
//    }
//    if (s.size() < 6){
//        segments.push_back(String(s.begin(), s.begin()+3));
//        for (int i = 3; i < s.size(); i++){
//            segments.push_back(String(1, s[i]));
//        }
//        return;
//    }
//    segments.push_back(String(s.begin(), s.end() - 3));
//    for (int i = s.size() - 3; i < s.size(); i++){
//        segments.push_back(String(1, s[i]));
//    }
}


void Tokenizer::tokenize_lazy_chars(String s, vector<String> &segments){
    if (s.size() < 8){
        tokenize_on_chars(s, segments);
    }else{
        segments = vector<String>(7);
        segments[0] += s[0];
        segments[1] = String(s.begin()+1, s.end()-5);
        int i = s.size() - 5;
        for (int j = 2; j < segments.size(); j++){
            segments[j] += s[i++];
        }
        assert( i == s.size() );
    }
}







SequenceEncoder::SequenceEncoder():SequenceEncoder(CHAR_LSTM){
#ifdef DEBUG
    cerr << "New SequenceEncoder created" << endl;
#endif
}
SequenceEncoder::SequenceEncoder(int i):tokenizer(i){}
SequenceEncoder::SequenceEncoder(const string &outdir){
    // TODO
}

void SequenceEncoder::init(){
    int vocsize = enc::hodor.size(enc::TOK);
    int from = dictionary.size();
#ifdef DEBUG
    cerr << "size of dictionary = " << dictionary.size() << endl;
    cerr << "encoding chars from " << from << " to " << vocsize << endl;
#endif
    //DBG("from = " << from << " dico.size " << dictionary.size())
    //dictionary = vector<vector<int>>(vocsize);
    for (int i = from; i < vocsize; i++){ // 2 : enc::UNDEF, enc::UNKNOWN
        //DBG("from = " << from << " dico.size " << dictionary.size() << "   i=" << i)
        assert(dictionary.size() == i);
        String s = enc::hodor.decode(i, enc::TOK);
        vector<String> tokens;
        vector<int> encoded_tokens;
        tokenizer(s, tokens);
        for (String &st : tokens){
            STRCODE char_code = encoder.code(st);
            //DBG("char code: " << char_code);
            encoded_tokens.push_back(char_code); // TODO: here handle unknown characters / unknown words -> no, handle this with lookup table
        }
        dictionary.push_back(encoded_tokens);
    }
}

void SequenceEncoder::export_model(const string &outdir){
    //TODO
}

vector<int>* SequenceEncoder::operator()(int code){
    //DBG("operator code = " << code)
    assert(code != enc::UNDEF && code != enc::UNKNOWN);
    if (code >= dictionary.size()){
        init();
    }
    assert(dictionary[code].size() > 0);
    return &dictionary[code];
}

int SequenceEncoder::char_voc_size(){
    return encoder.size();
}







