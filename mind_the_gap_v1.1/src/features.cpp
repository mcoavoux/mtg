
#include "features.h"

namespace feature{
    const unordered_map<std::string, int> reader{
        {"top", TOP},
        {"left", LEFT},
        {"right", RIGHT},
        {"left_corner", LEFT_CORNER},
        {"right_corner", RIGHT_CORNER},
        {"left_corner_out", LEFT_CORNER_OUT},
        {"right_corner_out", RIGHT_CORNER_OUT},
        {"B", BUFFER},
        {"S", STACK},
        {"W", QUEUE},
        {"cat", enc::CAT},
        {"tag", enc::TAG},
        {"form", enc::TOK}
    };

    const vector<std::string> writer_datastruct {"B", "W", "S"};
    const vector<std::string> writer_type {"cat", "form", "tag"};
    const vector<std::string> writer_depth {"top", "left", "right", "left_corner", "right_corner", "left_corner_out", "right_corner_out"};
    //enum {TOP, LEFT, RIGHT, LEFT_CORNER, RIGHT_CORNER, LEFT_CORNER_OUT, RIGHT_CORNER_OUT, LEFT_CORNER_2, RIGHT_CORNER_2, LEFT_CORNER_OUT_2, RIGHT_CORNER_OUT_2};

}

//////////////////////////////////////////////////////////////////
///
///
///    FeatureTemplate
///
///
/// //////////////////////////////////////////////////////////////


int FeatureTemplate::N = 1;


FeatureTemplate::FeatureTemplate(int data_struct, int index, int depth, int type) : data_struct(data_struct), index(index), depth(depth), type(type), unique_id(FeatureTemplate::N++){
    assert (FeatureTemplate::N < 256);
}

Int FeatureTemplate::feature_id(STRCODE code){
    return unique_id + (code << 8);
    //return data_struct + (index << 3) + (depth << 6) + (type << 9) + (code << 15);
}

STRCODE FeatureTemplate::decode(Int f_id, int &ds, int &idx, int &dep, int &typ){
    assert (false && "Code is not up to date, check feature encoding");
    // TODO: udpate this
//    ds = f_id % 8;
//    f_id >>= 3;
//    idx = f_id % 8;
//    f_id >>= 3;
//    dep = f_id % 8;
//    f_id >>= 3;
//    typ = f_id % 64;
//    return f_id >> 6;
    return 0;
}

STRCODE FeatureTemplate::get_val(ParseState &state, vector<shared_ptr<Node>> &buffer){
    switch(data_struct){
        case feature::BUFFER: return get_val_buffer(state.buffer_j_, buffer);
        case feature::QUEUE : return get_val_queue(state, buffer);
        case feature::STACK : return get_val_stack(state, buffer);
        default:
        assert(false && "Ill-defined template");
    }
    return 0;
}

STRCODE FeatureTemplate::get_val_buffer(int j, vector<shared_ptr<Node>> &buffer){
    int i = j + index;
    if (i < buffer.size()){
        switch(type){
            case enc::TAG: return buffer[i]->get_field(Leaf::FIELD_TAG); // enc::TAG == 1 -> FIELD_TAG = 1
            case enc::TOK: return buffer[i]->get_field(Leaf::FIELD_TOK); // enc::TOK == 2 -> FIELD_TOK = 0
            default : return buffer[i]->get_field(type-1); // enc::TYPE -> field -1     // TODO: check logic on this one
            // TODO: default case should work in all cases now
        }
    }
    return enc::UNDEF;
}

STRCODE FeatureTemplate::get_val_queue(ParseState &state, vector<shared_ptr<Node>> &buffer){
    shared_ptr<StackItem> item = state.top_;
    int i = index;
    while (item != nullptr && item != state.mid_ && i > 0){
        item = item->predecessor;
        i--;
    }
    if (item != nullptr && item != state.mid_){
        return get_val_(item->n, buffer);
    }
    return enc::UNDEF;
}

STRCODE FeatureTemplate::get_val_stack(ParseState &state, vector<shared_ptr<Node>> &buffer){
    shared_ptr<StackItem> item = state.mid_;
    int i = index;
    while (item != nullptr && i > 0){
        item = item->predecessor;
        i--;
    }
    if (item != nullptr){
        return get_val_(item->n, buffer);
    }
    return enc::UNDEF;
}

STRCODE FeatureTemplate::get_val_(shared_ptr<Node> &node, vector<shared_ptr<Node>> &buffer){
    shared_ptr<Node> tmp;
    switch (depth){
        case feature::TOP: tmp = node; break;
        case feature::LEFT: if (node->arity() > 0){ node->get(0, tmp);}else return enc::UNDEF;   break;
        case feature::RIGHT: if (node->arity() > 1){ node->get(1, tmp);}else return enc::UNDEF;  break;
        case feature::LEFT_CORNER: tmp = buffer[node->left_corner()];      break;
        case feature::RIGHT_CORNER: tmp = buffer[node->right_corner()];    break;
        // Check if leftcorner2 is inside constituent, otherwise return undef
//        case feature::LEFT_CORNER_2:
//            tmp = buffer[node->left_corner()];      break;
//        case feature::RIGHT_CORNER_2:
//            tmp = buffer[node->right_corner()];    break;
        case feature::LEFT_CORNER_OUT:{
            int i = node->left_corner()-1;
            if (i >= 0)
                tmp = buffer[i];
            else return enc::UNDEF;
            break;
        }
        case feature::RIGHT_CORNER_OUT:{
            int i = node->right_corner()+1;
            if (i < buffer.size())
                tmp = buffer[i];
            else return enc::UNDEF;
            break;
        }
        default:// TODO : corner features
            assert(false && "Unknown type of features");
    }
    switch(type){
        case enc::CAT: return tmp->label();
        // TODO: merge tok and tag cases
        case enc::TOK:
            if (! tmp->is_preterminal()){
                shared_ptr<Node> tmp2;
                tmp->head(tmp2);
                tmp = tmp2;
            }
            return tmp->get_field(Leaf::FIELD_TOK);
        case enc::TAG:
            if (! tmp->is_preterminal()){
                shared_ptr<Node> tmp2;
                tmp->head(tmp2);
                tmp = tmp2;
            }
            return tmp->get_field(Leaf::FIELD_TAG);
        default:
            assert(false && "Unknown field / not implemented yet");
    }
    assert(false && "Major problem in FeatureTemplate class");
    return -1;
}

int FeatureTemplate::get_index_in_sent(shared_ptr<Node> &node, vector<shared_ptr<Node>> &buffer){
    assert(type != enc::CAT);
    shared_ptr<Node> tmp;
    switch (depth){
    case feature::TOP: tmp = node; break;
    case feature::LEFT: if (node->arity() > 0){ node->get(0, tmp);}else return feature::UNDEFINED_POSITION;   break;
    case feature::RIGHT: if (node->arity() > 1){ node->get(1, tmp);}else return feature::UNDEFINED_POSITION;  break;
    case feature::LEFT_CORNER: tmp = buffer[node->left_corner()];      break;
    case feature::RIGHT_CORNER: tmp = buffer[node->right_corner()];    break;
    case feature::LEFT_CORNER_OUT:{
        int i = node->left_corner()-1;
        if (i >= 0)
            tmp = buffer[i];
        else return feature::UNDEFINED_POSITION;
        break;
    }
    case feature::RIGHT_CORNER_OUT:{
        int i = node->right_corner()+1;
        if (i < buffer.size())
            tmp = buffer[i];
        else return feature::UNDEFINED_POSITION;
        break;
    }
    default:
        assert(false && "Unknown type of features");
    }
    if (! tmp->is_preterminal()){
        shared_ptr<Node> tmp2;
        tmp->head(tmp2);
        tmp = tmp2;
    }
    return tmp->index();
}

Int FeatureTemplate::operator()(ParseState &state, vector<shared_ptr<Node>> &buffer){
    return feature_id(get_val(state, buffer));
}

Int FeatureTemplate::get_raw_feature(ParseState &state, vector<shared_ptr<Node>> &buffer){
    return get_val(state, buffer);
}

Int FeatureTemplate::get_raw_rnn(ParseState &state, vector<shared_ptr<Node>> &buffer){

    if (type == enc::CAT){
        return get_raw_feature(state, buffer);
    }
    if (data_struct == feature::BUFFER){
        return state.buffer_j_ + index;
    }
    if (data_struct == feature::QUEUE){
        shared_ptr<StackItem> item = state.top_;
        int i = index;
        while (item != nullptr && item != state.mid_ && i > 0){
            item = item->predecessor;
            i--;
        }
        if (item != nullptr && item != state.mid_){
            return get_index_in_sent(item->n, buffer);
        }
        return feature::UNDEFINED_POSITION;
    }
    if (data_struct == feature::STACK){
        shared_ptr<StackItem> item = state.mid_;
        int i = index;
        while (item != nullptr && i > 0){
            item = item->predecessor;
            i--;
        }
        if (item != nullptr){
            return get_index_in_sent(item->n, buffer);
        }
        return feature::UNDEFINED_POSITION;
    }
    assert(false && "Error in feature extractor for rnn");
    return feature::UNDEFINED_POSITION;
}

ostream & operator<<(ostream & os, const FeatureTemplate &ft){
    if (ft.data_struct == feature::BUFFER){
            return os << feature::writer_datastruct.at(ft.data_struct)
               << " " << ft.index
               << " " << feature::writer_type.at(ft.type);
    }
    return os << feature::writer_datastruct.at(ft.data_struct)
       << " " << ft.index
       << " " << feature::writer_depth.at(ft.depth)
       << " " << feature::writer_type.at(ft.type);
}

void FeatureTemplate::test(){
    // TODO udpate this
    assert (false && "You should come and update this function");
    long long I = 0;
    for (int i = 0; i < 8; i++){
        for (int j = 0; j < 8; j++){
            for (int k = 0; k < 8; k++){
                for (int p = 0; p < 64; p += 10){
                    FeatureTemplate ft(i,j,k,p);
                    for (STRCODE l = 0; l < (2 << 30); l *= 1 + i + j + k + p){
                        I++;
                        if (I % 100000 == 0){ cerr << "\r" << I; }
                        Int id = ft.feature_id(l);
                        int ii;
                        int jj;
                        int kk;
                        int pp;
                        STRCODE m = ft.decode(id, ii,jj,kk,pp);
                        if (m != l){
                            cerr << "m=" << m << "  l=" << l << endl;
                        }
                        assert(m == l);
                        assert(ii == i);
                        assert(jj == j);
                        assert(kk == k);
                        assert(pp == p);
                    }
                }
            }
        }
    }
    cerr << endl;
    cerr << "Done " << I << " tests" << endl;
}




//////////////////////////////////////////////////////////////////
///
///
///    FeatureExtractor
///
///
/// //////////////////////////////////////////////////////////////






FeatureExtractor::~FeatureExtractor(){}


void FeatureExtractor::read_templates(const std::string &filename, vector<vector<FeatureTemplate>> &fts){
    std::ifstream is(filename);
    std::string buffer;
    while(getline(is, buffer)){
        if (buffer.size() > 0 && buffer[0] != '#'){
            vector<FeatureTemplate> tpl;
            read_template(buffer, tpl);
            if (tpl.size() > 0)
                fts.push_back(tpl);
        }
    }
}

void FeatureExtractor::read_template(const std::string &line, vector<FeatureTemplate> &ft){
    boost::char_separator<char> sep("&");
    boost::tokenizer<boost::char_separator<char>> toks(line, sep);
    vector<std::string> tokens(toks.begin(), toks.end());
#if DEBUG
    cerr << tokens.size() << "-ary template:" << line << endl;
#endif
    for (int i = 0; i < tokens.size(); i++){
        sep = boost::char_separator<char>(" ");
        toks = boost::tokenizer<boost::char_separator<char>>(tokens[i], sep);
        vector<std::string> fields(toks.begin(), toks.end());

        if (fields.size() < 3 || fields.size() > 4){
#if DEBUG
            cerr << "Ignoring tpl :" << tokens[i] << endl;
#endif
            continue;
        }
        int data_struct = feature::reader.at(fields[0]);
        if (data_struct == feature::BUFFER){
            assert(fields.size() == 3);
            int index = stoi(fields[1]);
            int type = feature::reader.at(fields[2]);
            ft.push_back(FeatureTemplate(data_struct, index, 0, type));
        }else if (data_struct == feature::QUEUE || data_struct == feature::STACK){
            assert(fields.size() == 4);
            int index = stoi(fields[1]);
            int depth = feature::reader.at(fields[2]);
            int type = feature::reader.at(fields[3]);
            ft.push_back(FeatureTemplate(data_struct, index, depth, type));
        }
    }
}




void FeatureExtractor::export_model(const string &outdir){
    assert(false && "Not implemented yet in this class");
}

FeatureExtractor* FeatureExtractor::import_model(const string &outdir){

    ifstream is1(outdir + "/feature_id");
    int id;
    is1 >> id;
    is1.close();
    switch(id){
    case feature::FAST_FEATURE_EXTRACTOR:
        return new FastFeatureExtractor(outdir + "/feature_templates");
    case feature::DENSE_FEATURE_EXTRACTOR:
        return new DenseFeatureExtractor(outdir + "/feature_templates");
    case feature::RNN_FEATURE_EXTRACTOR:
        return new RnnFeatureExtractor(outdir + "/feature_templates");
    default:
        assert(false && "Error ");
    }
    return nullptr;
}



//////////////////////////////////////////////////////////////////
///
///
///    Feature              FeatureHasher
///
///
/// //////////////////////////////////////////////////////////////




Feature::Feature(){}
Feature::Feature(int arity):vals(vector<Int>(arity,0)){}

// TODO: bool operator ==     -> collision wise -> tester avec return true ? probably won't work less you set the hashtable size on creation
bool Feature::operator==(const Feature &f) const{
    if (vals.size() != f.vals.size()){
        return false;
    }
    for (int i = 0; i < vals.size(); i++){
        if (vals[i] != f.vals[i])
            return false;
    }
    return true;
}


std::size_t FeatureHasher::operator()(const Feature & f) const{
    std::size_t seed = 0;
    for (int i = 0; i < f.vals.size(); i++){
        boost::hash_combine(seed, f.vals[i]);
    }
    return seed;
}




//////////////////////////////////////////////////////////////////
///
///
///    StdFeatureExtractor
///
///
/// //////////////////////////////////////////////////////////////



StdFeatureExtractor::StdFeatureExtractor(const std::string &filename){
    size = 0;
    FeatureExtractor::read_templates(filename, templates);
    features.resize(templates.size());
    for (int i = 0; i < templates.size(); i++){
        features[i] = Feature(templates[i].size());
    }
}

int StdFeatureExtractor::n_templates(){
    return templates.size();
}

// REMINDER: currently this is the speed bottleneck (49% of computation time)
void StdFeatureExtractor::operator()(ParseState &state, vector<shared_ptr<Node>> &buffer, vector<int> &feats){
    for (int i = 0; i < templates.size(); i++){
        for (int j = 0; j < templates[i].size(); j++){
            this->features[i].vals[j] = templates[i][j](state, buffer);
        }
        auto it = dict.find(features[i]);
        if (it != dict.end()){
            feats[i] = it->second;
        }else{
            dict[features[i]] = size;
            feats[i] = size++;
        }
    }
}

ostream & operator<<(ostream &os, const StdFeatureExtractor &fe){
    for (int i = 0; i < fe.templates.size(); i++){
        os << fe.templates[i][0];
        for (int j = 1; j < fe.templates[i].size(); j++){
            os << " & " << fe.templates[i][j];
        }os << endl;
    }
    return os;
}

void StdFeatureExtractor::test(const std::string &tpls_filename){
    StdFeatureExtractor fe(tpls_filename);
    ofstream os(tpls_filename+".new");
    os << fe << endl;
    os.close();
    StdFeatureExtractor fe2(tpls_filename + ".new");
    ofstream os2(tpls_filename + ".new.new");
    os2 << fe << endl;
    os2.close();
}

//////////////////////////////////////////////////////////////////
///
///
///    FastFeatureExtractor
///
///
/// //////////////////////////////////////////////////////////////



FastFeatureExtractor::FastFeatureExtractor(const std::string &filename){
    //size = 0;
    FeatureExtractor::read_templates(filename, templates);
    features.resize(templates.size());
    for (int i = 0; i < templates.size(); i++){
        //features[i] = Feature(templates[i].size());
        features[i] = Feature(4);
    }
}

int FastFeatureExtractor::n_templates(){
    return templates.size();
}

void FastFeatureExtractor::operator()(ParseState &state, vector<shared_ptr<Node>> &buffer, vector<int> &feats){
    for (int i = 0; i < templates.size(); i++){
        for (int j = 0; j < templates[i].size(); j++){
            this->features[i].vals[j] = templates[i][j](state, buffer);
        }

        feats[i] = indexer(features[i].vals[0],
                           features[i].vals[1],
                           features[i].vals[2],
                           features[i].vals[3]);
    }
}

ostream & operator<<(ostream &os, const FastFeatureExtractor &fe){
    for (int i = 0; i < fe.templates.size(); i++){
        assert (fe.templates[i].size() != 0);
        os << fe.templates[i][0];
        for (int j = 1; j < fe.templates[i].size(); j++){
            os << " & " << fe.templates[i][j];
        }os << endl;
    }
    return os;
}

void FastFeatureExtractor::export_model(const string &outdir){
    ofstream os1(outdir + "/feature_id");
    os1 << feature::FAST_FEATURE_EXTRACTOR << endl;
    os1.close();

    ofstream os(outdir + "/feature_templates");
    os << *this << endl;
    os.close();
}


//////////////////////////////////////////////////////////////////
///
///
///    DenseFeatureExtractor
///
///
/// //////////////////////////////////////////////////////////////



DenseFeatureExtractor::DenseFeatureExtractor(const std::string &filename){
    //size = 0;
    vector<vector<FeatureTemplate>> tpls;
    FeatureExtractor::read_templates(filename, tpls);
    for (int i = 0; i < tpls.size(); i++){
        assert(tpls[i].size() == 1);
        templates.push_back(tpls[i][0]);
    }
//    features.resize(templates.size());
//    for (int i = 0; i < templates.size(); i++){
//        //features[i] = Feature(templates[i].size());
//        features[i] = Feature(4);
//    }
}

int DenseFeatureExtractor::n_templates(){
    return templates.size();
}

void DenseFeatureExtractor::operator()(ParseState &state, vector<shared_ptr<Node>> &buffer, vector<int> &feats){
    for (int i = 0; i < templates.size(); i++){
        feats[i] = templates[i].get_raw_feature(state, buffer);
    }
}

ostream & operator<<(ostream &os, const DenseFeatureExtractor &fe){
    for (int i = 0; i < fe.templates.size(); i++){
        os << fe.templates[i] << endl;
    }
    return os;
}

void DenseFeatureExtractor::export_model(const string &outdir){
    ofstream os1(outdir + "/feature_id");
    os1 << feature::DENSE_FEATURE_EXTRACTOR << endl;
    os1.close();

    ofstream os(outdir + "/feature_templates");
    os << *this << endl;
    os.close();
}

void DenseFeatureExtractor::get_template_type(vector<int> &types){
    types.clear();
    for (int i = 0; i < templates.size(); i++){
        types.push_back(templates[i].type);
    }
}

////////////////////////////////
///
///     feature extractor or bi-rnn model
///
///
RnnFeatureExtractor::RnnFeatureExtractor(const std::string &filename):DenseFeatureExtractor(filename){}
void RnnFeatureExtractor::operator()(ParseState &state, vector<shared_ptr<Node>> &buffer, vector<int> &feats){
    for (int i = 0; i < templates.size(); i++){
        feats[i] = templates[i].get_raw_rnn(state, buffer);
    }
}

void RnnFeatureExtractor::export_model(const string &outdir){
    ofstream os1(outdir + "/feature_id");
    os1 << feature::RNN_FEATURE_EXTRACTOR << endl;
    os1.close();

    ofstream os(outdir + "/feature_templates");
    os << *this << endl;
    os.close();
}



