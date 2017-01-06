
#include "utils.h"


namespace enc{
    TypedStrEncoder hodor;

    StrDict::StrDict() : size_(0){
        code("UNKNOWN");
        code("UNDEF");
    }

    STRCODE StrDict::code(STRING s){
        auto it = encoder.find(s);
        if (it == encoder.end()){
            encoder[s] = size_ ++;
            decoder.push_back(s);
            return size_-1;
        }else{
            return it->second;
        }
    }

    STRCODE StrDict::code_unknown(STRING s){
        auto it = encoder.find(s);
        if (it == encoder.end()){
            return UNKNOWN;
        }else{
            return it->second;
        }
    }

    STRING StrDict::decode(STRCODE i){
        assert(i < encoder.size() && size_ == encoder.size() && "hodor error: decoding unknown code");
        return decoder.at(i);
    }
    int StrDict::size(){
        return size_;
    }

    ostream & operator<<(ostream &os, StrDict &ts){
        for (int i = 0; i < ts.decoder.size(); i++){
            os << ts.decoder[i] << endl;
        }
        return os;
    }






    TypedStrEncoder::TypedStrEncoder() : encoders(MAX_FIELDS){
        #ifdef DEBUG
        cerr << "Hold the door!" << endl;
        #endif
    }

    STRCODE TypedStrEncoder::code(STRING s, int type){
        return encoders[type].code(s);
    }
    STRCODE TypedStrEncoder::code_unknown(STRING s, int type){
        return encoders[type].code_unknown(s);
    }
    STRING TypedStrEncoder::decode(STRCODE i, int type){
        assert(type < encoders.size() && "hodor error: type unknown");
        return encoders[type].decode(i);
    }
    int TypedStrEncoder::size(int type){
        return encoders[type].size();
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
            assert(buf == decode(UNKNOWN,i));
            getline(ist,buf);
            assert(buf == decode(UNDEF,i));
            while (getline(ist,buf)){
                code(buf, i);
            }
            ist.close();
        }
        is.close();
    }
}



