#include "treebank.h"


////////////////////////////////////////////////////////////////
///
///
///     Treebank
///
///
/// ////////////////////////////////////////////////////////////


//std::default_random_engine Treebank::random = std::default_random_engine(SEED);

#ifdef WSTRING
const String Treebank::LBRACKET = L"(";
const String Treebank::RBRACKET = L")";
const String Treebank::EQUAL = L"=";
const String Treebank::HEAD = L"head";
#else
const String Treebank::LBRACKET = "(";
const String Treebank::RBRACKET = ")";
const String Treebank::EQUAL = "=";
const String Treebank::HEAD = "head";
#endif


Treebank::~Treebank(){}
Treebank::Treebank(){}
Treebank::Treebank(const string & filename, int unknown, int format, bool train)
    : unknown(unknown), train(train){
    switch(format){
    case DISCBRACKET:
        read_discbracket_treebank(filename);
        break;
    case TBK:
        read_tbk_treebank(filename, train);
        break;
    case OLD_TBK:
        read_old_tbk_treebank(filename, train);
        break;
    default:assert(false && "Unknown format");
    }
}

Treebank::Treebank(const string & filename, int unknown, int format, vector<string> &header, bool train)
    : Treebank(filename, unknown, format, train){
    if (format == TBK || format == OLD_TBK){
        header = this->header;
    }
}

Treebank::Treebank(Treebank &tbk){
    this->field_freqs = tbk.field_freqs;
    this->header = tbk.header;
    this->voc_sizes = tbk.voc_sizes;

    for (int i = 0; i < tbk.size(); i++){
        Tree t;
        tbk[i]->copy(t);
        add_tree(t);
    }
}
Treebank& Treebank::operator=(Treebank &tbk){
    this->field_freqs = tbk.field_freqs;
    this->header = tbk.header;
    this->voc_sizes = tbk.voc_sizes;

    clear();
    for (int i = 0; i < tbk.size(); i++){
        Tree t;
        tbk[i]->copy(t);
        add_tree(t);
    }

    return *this;
}

void Treebank::get_raw_inputs(vector<vector<shared_ptr<Node>>> &v){
    v.resize(size());
    for (int i = 0; i < size(); i++){
        trees_[i].get_buffer(v[i]);
    }
}

Tree* Treebank::operator[](int i){
    return &trees_[i];
}

int Treebank::size()const{
    return trees_.size();
}

void Treebank::add_tree(const Tree &t){
    trees_.push_back(t);
}
void Treebank::clear(){
#if DEBUG
    cerr << "I'm clearing a treebank" << endl;
#endif
    trees_.clear();
}

int Treebank::rank(){
    int g = 0;
    for (auto &t : trees_){
        int r = t.rank();
        if (r > g) g = r;
    }
    return g;
}

int Treebank::gap_degree(){
    int g = 0;
    for (auto &t : trees_){
        int r = t.gap_degree();
        if (r > g) g = r;
    }
    return g;
}

void Treebank::shuffle(){
    //std::shuffle(trees_.begin(), trees_.end(), std::default_random_engine(SEED)); // REMINDER: REPRODUCIBILITY -> this has changed on 19 august
    //std::shuffle(trees_.begin(), trees_.end(), random);
    std::shuffle(trees_.begin(), trees_.end(), rd::Random::re);
}

void Treebank::subset(int n_sentences, Treebank &tbk){
    for (int i = 0; i < n_sentences; i++){
        tbk.add_tree(trees_[i]);
    }
}

void Treebank::encode_all_from_freqs(){
    for (int i = 0; i < field_freqs.size(); i++){
        for (auto &it : field_freqs[i]){
            enc::hodor.code(it.first, i+1);
        }
    }
    enc::hodor.vocsizes(voc_sizes);
}

void Treebank::encode_known_from_freqs(){
    for (int i = 0; i < field_freqs.size(); i++){
        for (auto &it : field_freqs[i]){
            if (it.second >= MIN_FREQUENCY){
                enc::hodor.code(it.first, i+1);
            }
        }
    }
}
void Treebank::get_vocsizes(vector<int> &vocsizes){
    vocsizes = this->voc_sizes;
}


void Treebank::annotate_heads(const HeadRules &hr){
    for (int i = 0; i < trees_.size(); i++){
        trees_[i].annotate_heads(hr);
    }
}

void Treebank::annotate_parent_ptr(){
    for (int i = 0; i < size(); i++){
        trees_[i].annotate_parent_ptr();
    }
}

void Treebank::transform(Grammar &grammar){
    for (int i = 0; i < trees_.size(); i++){
        trees_[i].binarize(grammar);
    }
}

void Treebank::detransform(Grammar &grammar){
    for (int i = 0; i < trees_.size(); i++){
        trees_[i].unbinarize(grammar);
    }
}


void Treebank::read_discbracket_treebank(const string &filename){
    field_freqs.resize(2);
    std::ifstream instream(filename);
    string buffer;
    vector<String> full_corpus;
    while(getline(instream,buffer)){
#ifdef WSTRING
        wstring wbuffer = str::decode(buffer);
        full_corpus.push_back(wbuffer);
#else
        full_corpus.push_back(buffer);
        //parse_discbracket_tree(buffer);
#endif
    }
    instream.close();

    if (train){
        for (int i = 0; i < full_corpus.size(); i++){
            update_frequencies(full_corpus[i]);
        }
#if DEBUG
        cerr << "{";
        for (auto &it : field_freqs[0]){
#ifdef WSTRING
            cerr << str::encode(it.first) << ":" << it.second << ", ";
#else
            cerr << it.first << ":" << it.second << ", ";
#endif
        }cerr << "}" << endl;
#endif
        encode_known_from_freqs();
        enc::hodor.vocsizes(voc_sizes);
    }

    for (int i = 0; i < full_corpus.size(); i++){
        parse_discbracket_tree(full_corpus[i]);
    }
}

void Treebank::parse_discbracket_tree(const String &line){

    vector<String> tokens;
    str::split(line, " ", ")(", tokens);

    shared_ptr<Node> node;
    parse_tokens_disco(tokens, 0, tokens.size() -1, node);
    trees_.push_back(Tree(node));
}


void Treebank::parse_tokens_disco(const vector<String> &tokens, int d, int f, shared_ptr<Node> &res){
    assert(f-d >= 3 && "Error reading treebank");
    if (f - d == 3 && tokens[d] == LBRACKET
                   && tokens[f] == RBRACKET){
        int delimiter_idx = tokens[d+2].find_first_of(EQUAL);
        String id(tokens[d+2].substr(0, delimiter_idx));
        String terminal = tokens[d+2].substr(delimiter_idx+1);
        int tok_idx = stoi(id);

        //vector<STRCODE> fields;
        vector<String> fields{terminal, tokens[d+1]};
        vector<STRCODE> enc_fields;
        encode_fields(fields, enc_fields);

        // REMINDER:this has changed results a bit, encode tag separately to reproduce eacl results

        //if (field_freqs[1][fields[1]] < MIN_FREQUENCY){
//        for (auto it: field_freqs[1]){
//            cerr << it.first << "   " << it.second << endl;
//        }
//        exit(0);

        // TODO: plug encode_fields function, beware : tag dict is not updated (special coding case ?)

//        switch(unknown){
//        case ENCODE_EVERYTHING:{
//            enc_fields.push_back(enc::hodor.code(fields[0], enc::TOK)); break;
//        }
//        case CUTOFF:{
//            if (field_freqs[0][terminal] < MIN_FREQUENCY)
//                enc_fields.push_back(enc::UNKNOWN);
//            else
//                enc_fields.push_back(enc::hodor.code(fields[0], enc::TOK));
//            break;
//        }
//        case UNKNOWN_CODING:{
//            enc_fields.push_back(enc::hodor.code_unknown(fields[0], enc::TOK));
//            break;
//        }
//        }

//        assert(fields.size() == 1);
//        enc_fields.push_back(enc::hodor.code(tokens[d+1], enc::TAG));


        res = shared_ptr<Node>(new Leaf(enc::hodor.code(tokens[d+1], enc::CAT), tok_idx, enc_fields));
        //res = shared_ptr<Node>(new Leaf(enc::hodor.code(tokens[d+1], enc::CAT), tok_idx, fields));
    }
    else if (tokens[d] == LBRACKET){
        assert(tokens[f] == RBRACKET && "Error reading treebank");
        String label = tokens[d+1];
        vector<shared_ptr<Node>> children;
        int m = aux(tokens, d+2, f);
        shared_ptr<Node> child;
        parse_tokens_disco(tokens, d+2, m, child);
        children.push_back(child);
        while (m != f-1){
            int n = m + 1;
            m = aux(tokens, n, f);
            parse_tokens_disco(tokens, n, m, child);
            children.push_back(child);
        }

        res = shared_ptr<Node>(new Node(enc::hodor.code(label, enc::CAT),
                                         children
                                         ));
    }else{
        assert(false && "This should not have happened, check code and treebank");
    }
}

void Treebank::update_frequencies(const String &line){
    vector<String> tokens;
    str::split(line, " ", ")(", tokens);
    update_frequencies_tokens(tokens, 0, tokens.size() -1);
}

void Treebank::update_frequencies_tokens(const vector<String> &tokens, int d, int f){
    assert(f-d >= 3 && "Error reading treebank");
    if (f - d == 3 && tokens[d] == LBRACKET
                   && tokens[f] == RBRACKET){
        int delimiter_idx = tokens[d+2].find_first_of(EQUAL);
        String terminal = tokens[d+2].substr(delimiter_idx+1);
        String tag  = tokens[d+1];
        vector<String> fields{terminal, tag};
        for (int i = 0; i < fields.size(); i++){
            if (field_freqs[i].find(fields[i]) != field_freqs[i].end()){
                field_freqs[i][fields[i]] += 1;
            }else{
                field_freqs[i][fields[i]] = 1;
            }
        }

    }else if (tokens[d] == LBRACKET){
        assert(tokens[f] == RBRACKET && "Error reading treebank");
        int m = aux(tokens, d+2, f);
        update_frequencies_tokens(tokens, d+2, m);
        while (m != f-1){
            int n = m + 1;
            m = aux(tokens, n, f);
            update_frequencies_tokens(tokens, n, m);
        }
    }else{
        assert(false && "This should not have happened, check code and treebank");
    }
}


void Treebank::write(const string& filename){
    ofstream os(filename);
    for (int i = 0; i < trees_.size(); i++){
        os << trees_[i] << endl;
    }
    os.close();
}

void Treebank::write(const string& filename, vector<vector<std::pair<String, String> > > &str_sentences, bool prob){
    ofstream os(filename);
    for (int i = 0; i < trees_.size(); i++){
        trees_[i].write(os, str_sentences[i], prob);
        os << endl;
    }
    os.close();
}

void Treebank::write_conll(const string& filename, vector<vector<std::pair<String, String> > > &str_sentences, bool unlex){
    int depfield_id = enc::hodor.get_dep_idx();
    ofstream os(filename);
    for (int i = 0; i < trees_.size(); i++){
        trees_[i].write_conll(os, str_sentences[i], depfield_id, unlex);
        os << endl;
    }
    os.close();
}



void Treebank::read_tbk_treebank(const string &filename, bool train_set){
    std::ifstream instream(filename);
    string buffer;

    vector<vector<String>> sentences;
    vector<String> sentence;
    vector<String> tokens;

    getline(instream, buffer);
    str::split(buffer, "\t ", "", header);
    field_freqs.resize(header.size());

    vector<String> all_sentences;
#ifdef WSTRING
    while(getline(instream, buffer)){
        wstring wbuffer = str::decode(buffer);
        boost::trim(wbuffer);
        if (buffer.size() > 0){
            all_sentences.push_back(wbuffer);
        }
    }
#else
    while(getline(instream, buffer)){
        boost::trim(buffer);
        if (buffer.size() > 0){
            all_sentences.push_back(buffer);
        }
    }
#endif
    int idx = 0;
    while (idx < all_sentences.size()){
        int end = find_closing_bracket(idx, all_sentences);
        sentence = vector<String>(all_sentences.begin()+idx, all_sentences.begin()+end+1);
        sentences.push_back(sentence);
        idx = end + 1;
    }

    instream.close();


    for (int i = 0; i < sentences.size(); i++){
        for (int j = 0; j < sentences[i].size(); j++){
            if (! is_xml_markup(sentences[i][j])){
                str::split(sentences[i][j], "\t", "", tokens);
                assert(tokens.size() == header.size() + 1);
                // IMPORTANT: when counting frequencies, ignore -head
                //is_head(tokens[1]);  // head no longer on tag
                //tokens.pop_front();
                tokens.erase(tokens.begin());

                for (int i = 0; i < tokens.size(); i++){
                    if (field_freqs[i].find(tokens[i]) != field_freqs[i].end()){
                        field_freqs[i][tokens[i]] += 1;
                    }else{
                        field_freqs[i][tokens[i]] = 1;
                    }
                }
            }
        }
    }
    encode_known_from_freqs();
    enc::hodor.vocsizes(voc_sizes);

    if (train_set){
        enc::hodor.update_wordform_frequencies(field_freqs[0]);
    }

    for (int i = 0; i < sentences.size(); i++){
        parse_tbk_tree(sentences[i]);
    }
}

void Treebank::parse_tbk_tree(const vector<String> &sentence){
    shared_ptr<Node> node;
    //int idx = 0;
    get_tbk_tree(sentence, 0, sentence.size()-1, node);
    trees_.push_back(Tree(node));
}

bool Treebank::get_tbk_tree(const vector<String> &sentence, int d, int f, shared_ptr<Node> &res){
    if (d == f){ // Gérer les -head
        vector<String> fields;
        vector<STRCODE> enc_fields;
        str::split(sentence[d], "\t", "", fields);
        assert(fields.size() == header.size() + 1);

        bool is_a_head = is_head(fields[0]);
        int idx = stoi(fields[0]) - 1;
        fields.erase(fields.begin());

        STRCODE label = enc::hodor.code(fields[1], enc::CAT);
        encode_fields(fields, enc_fields);
        res = shared_ptr<Node>(new Leaf(label, idx, enc_fields));
        return is_a_head;
    }else{
        assert(is_xml_beg(sentence[d]) && is_xml_end(sentence[f]));
        String slabel;
        bool is_a_head = get_xml_label(sentence[d], slabel);
        vector<shared_ptr<Node>> stack;
        int head = -1;

        shared_ptr<Node> node;
        int i = d+1;
        int end;
        while (i < f){
            end = find_closing_bracket(i, sentence);
            if (get_tbk_tree(sentence, i, end, node)){
                head = stack.size();
            }
            stack.push_back(node);
            i = end + 1;
        }
        assert(i == f);
        res = shared_ptr<Node>(new Node(enc::hodor.code(slabel, enc::CAT),stack));
        if (head == -1){
            assert(stack.size() == 1);
            head = 0;
        }
//#ifdef DEBUG
//        if (head == -1){
//            cerr << "d="<< d << " f=" << f << endl;
//            for (int i = 0; i < sentence.size(); i++){
//                cerr << "i=" << i << "   " << sentence[i] << endl;
//            }
//        }
//#endif
        res->set_h(head);
        return is_a_head;
    }
}



/// LEGACY CODE: for old tbk format, do not maintain this

void Treebank::read_old_tbk_treebank(const string &filename, bool train_set){
    std::ifstream instream(filename);
    string buffer;

    vector<vector<String>> sentences;
    vector<String> sentence;
    vector<String> tokens;

    getline(instream, buffer);
    str::split(buffer, "\t ", "", header);
    field_freqs.resize(header.size());

    vector<String> all_sentences;
#ifdef WSTRING
    while(getline(instream, buffer)){
        wstring wbuffer = str::decode(buffer);
        boost::trim(wbuffer);
        if (buffer.size() > 0){
            all_sentences.push_back(wbuffer);
        }
    }
#else
    while(getline(instream, buffer)){
        boost::trim(buffer);
        if (buffer.size() > 0){
            all_sentences.push_back(buffer);
        }
    }
#endif
    int idx = 0;
    while (idx < all_sentences.size()){
        int end = find_closing_bracket(idx, all_sentences, "-");
        sentence = vector<String>(all_sentences.begin()+idx, all_sentences.begin()+end+1);
        sentences.push_back(sentence);
        idx = end + 1;
    }

    instream.close();

    for (int i = 0; i < sentences.size(); i++){
        for (int j = 0; j < sentences[i].size(); j++){
            if (! is_xml_markup(sentences[i][j])){
                str::split(sentences[i][j], "\t", "", tokens);
//                for (int k = 0; k < tokens.size(); k++){
//                    cerr << tokens[k] << "  ";
//                }cerr << endl;
//                cerr << tokens.size() << "  " << header.size() << endl;
                assert(tokens.size() == header.size());
                // IMPORTANT: when counting frequencies, ignore -head
                //cerr << str::encode(tokens[1]) << endl;
                is_head(tokens[1], "-");  // head no longer on tag
                //cerr << str::encode(tokens[1]) << endl;
                //tokens.pop_front();
                //tokens.erase(tokens.begin());

                for (int i = 0; i < tokens.size(); i++){
                    if (field_freqs[i].find(tokens[i]) != field_freqs[i].end()){
                        field_freqs[i][tokens[i]] += 1;
                    }else{
                        field_freqs[i][tokens[i]] = 1;
                    }
                }
            }
        }
    }

    encode_known_from_freqs();
    enc::hodor.vocsizes(voc_sizes);

    if (train_set){
        enc::hodor.update_wordform_frequencies(field_freqs[0]);
    }

    for (int i = 0; i < sentences.size(); i++){
        parse_old_tbk_tree(sentences[i]);
    }
}

void Treebank::parse_old_tbk_tree(const vector<String> &sentence){
    shared_ptr<Node> node;
    int idx = 0;
    get_old_tbk_tree(sentence, 0, sentence.size()-1, idx, node);
    trees_.push_back(Tree(node));
}

bool Treebank::get_old_tbk_tree(const vector<String> &sentence, int d, int f, int &idx, shared_ptr<Node> &res){

    if (d == f){ // Gérer les -head
        vector<String> fields;
        vector<STRCODE> enc_fields;
        str::split(sentence[d], "\t", "", fields);
        assert(fields.size() == header.size());

        bool is_a_head = is_head(fields[1], "-");
//        int idx = stoi(fields[0]) - 1;
//        fields.erase(fields.begin());

        STRCODE label = enc::hodor.code(fields[1], enc::CAT);
        encode_fields(fields, enc_fields);
        res = shared_ptr<Node>(new Leaf(label, idx++, enc_fields));
        return is_a_head;
    }else{
        assert(is_xml_beg(sentence[d]) && is_xml_end(sentence[f]));
        String slabel;
        bool is_a_head = get_xml_label(sentence[d], slabel, "-");
        vector<shared_ptr<Node>> stack;
        int head = -1;

        shared_ptr<Node> node;
        int i = d+1;
        int end;
        while (i < f){
            end = find_closing_bracket(i, sentence, "-");
            if (get_old_tbk_tree(sentence, i, end, idx, node)){
                head = stack.size();
            }
            stack.push_back(node);
            i = end + 1;
        }
        assert(i == f);
        res = shared_ptr<Node>(new Node(enc::hodor.code(slabel, enc::CAT),stack));
        if (head == -1){
            assert(stack.size() == 1);
            head = 0;
        }
        res->set_h(head);
        return is_a_head;
    }
}



/// END






int Treebank::find_closing_bracket(int i, const vector<String> &sentence, string head_sep){
    if (! is_xml_markup(sentence[i]))
        return i;
    assert(is_xml_beg(sentence[i]));
    String label;
    get_xml_label(sentence[i], label, head_sep);
    int n = 1;
    while (n != 0){
        i++;
        assert(i < sentence.size());
        if (is_xml_beg(sentence[i]))
            n++;
        if (is_xml_end(sentence[i]))
            n--;

    }
    String close_label;
    get_xml_label(sentence[i], close_label, head_sep);
#ifdef DEBUG
    if (label != close_label){
#ifdef WSTRING
        cerr << "Sent: " << endl;
        for (int i = 0; i < sentence.size(); i++){
            cerr << str::encode(sentence[i]) << endl;
        }cerr << endl;
        cerr << "      label= "
             << str::encode(label) << "   "
             << str::encode(close_label) << endl;
    }
#else
        cerr << "Sent: " << endl;
        for (int i = 0; i < sentence.size(); i++){
            cerr << sentence[i]<< endl;
        }cerr << endl;
        cerr << "      label= " << label << "   "  << close_label << endl;
    }
#endif
#endif
    assert(label == close_label);
    return i;
}


void Treebank::get_header(vector<string> &header){
    header = this->header;
}






int Treebank::aux(const vector<String> &tokens, int d, int f){
    int count = 0;
    if (! (tokens[d] == LBRACKET)){
        cerr << "should not happen" << endl;
        return d;
    }
    while (d < f){
        if (tokens[d] == LBRACKET){
            count ++;
        }
        else if (tokens[d] == RBRACKET){
            count --;
            if (count == 0)
                return d;
        }
        d++;
    }
    return d;
}






void Treebank::test_binarization(const string &tbk_filename, const string &hr_filename){
    cerr << "Reading treebank" << endl;
    Treebank tbk(tbk_filename, Treebank::ENCODE_EVERYTHING, Treebank::DISCBRACKET, true);
    cerr << "Writing treebank" << endl;
    tbk.write(tbk_filename+"_rewritten");
    cerr << "Done" << endl;

    Grammar grammar;
    HeadRules hr(hr_filename, &grammar);


    tbk.annotate_heads(hr);
    tbk.transform(grammar);

    tbk.write(tbk_filename+"_binarized");
    tbk.detransform(grammar);
    tbk.write(tbk_filename+"_unbinarized");
}





























void Treebank::read_raw_input_sentences(
        const string &filename,
        vector<vector<shared_ptr<Node>>> &raw_test,
        vector<vector<std::pair<String, String>>> &str_sentences,
        int format,
        int unknown){
    switch(format){
    case DISCBRACKET:
        read_raw_input_sentences_discbracket(filename, raw_test, str_sentences, unknown);
        break;
    case TBK:
        read_raw_input_sentences_tbk(filename, raw_test, str_sentences, unknown);
        break;
    default:
        assert(false && "Unknown input format");
    }

}



void Treebank::read_raw_input_sentences_tbk(
        const string &filename,
        vector<vector<shared_ptr<Node>>> &raw_test,
        vector<vector<std::pair<String, String> > > &str_sentences,
        int unknown){
    std::ifstream is(filename);
    string buffer;
    getline(is, buffer);// header
    vector<string> header;
    boost::trim(buffer);
    str::split(buffer, " \t", "", header);

    if (header.size() == 1){
        header.push_back("tag");
    }

    vector<vector<String>> sentence;

    vector<String> tokens;
    while (getline(is, buffer)){
#ifdef WSTRING
        wstring wbuffer = str::decode(buffer);
        boost::trim(wbuffer);
        str::split(wbuffer, " \t", "", tokens);
        if (wbuffer.size() == 0 && sentence.size() > 0){
            read_raw_sentence_tbk(sentence, raw_test, str_sentences, unknown);
            sentence.clear();
        }else{
            if (tokens.size() > 0){
                if (tokens.size() == 1){
                    tokens.push_back(L"UNKNOWN");
                }
                assert(tokens.size() == header.size());
                sentence.push_back(tokens);
            }
        }
#else
        boost::trim(buffer);
        str::split(buffer, " \t", "", tokens);
        if (buffer.size() == 0 && sentence.size() > 0){
            read_raw_sentence_tbk(sentence, raw_test, str_sentences, unknown);
            sentence.clear();
        }else{
            if (tokens.size() > 0){
                if (tokens.size() == 1){
                    tokens.push_back("UNKNOWN");
                }
                assert(tokens.size() == header.size());
                sentence.push_back(tokens);
            }
        }
#endif
    }
}

void Treebank::read_raw_sentence_tbk(
        vector<vector<String>> &sentence,
        vector<vector<shared_ptr<Node>>> &sentences,
        vector<vector<std::pair<String,String>>> &str_sentences,
        int unknown){

    vector<shared_ptr<Node>> node_res;
    vector<std::pair<String,String>> str_res;

    for (int i = 0; i < sentence.size(); i++){
        str_res.push_back(std::make_pair(sentence[i][0], sentence[i][1]));
        vector<STRCODE> fields(sentence[i].size());
        STRCODE label = enc::hodor.code_unknown(sentence[i][Leaf::FIELD_TAG], enc::CAT);

        if (unknown == Treebank::UNKNOWN_CODING){
            for (int j = 0; j < sentence[i].size(); j++){
                fields[j] = enc::hodor.code_unknown(sentence[i][j], j+1);
            }
        }else{
            assert(unknown == Treebank::ENCODE_EVERYTHING);
            for (int j = 0; j < sentence[i].size(); j++){
                fields[j] = enc::hodor.code(sentence[i][j], j+1);
            }
        }
        shared_ptr<Node> node(new Leaf(label, i, fields));
        node_res.push_back(node);
    }
    sentences.push_back(node_res);
    str_sentences.push_back(str_res);
}



void Treebank::read_raw_input_sentences_discbracket(
        const string &filename,
        vector<vector<shared_ptr<Node>>> &raw_test,
        vector<vector<std::pair<String,String>>> &str_sentences,
        int unknown){
    std::ifstream is(filename);
    string buffer;
    while (getline(is, buffer)){
        vector<shared_ptr<Node>> sent;
        vector<pair<String,String>> str_sent;
#ifdef WSTRING
        wstring wbuffer = str::decode(buffer);
        read_raw_sentence_discbracket(wbuffer, sent, str_sent, unknown);
#else
        read_raw_sentence_discbracket(buffer, sent, str_sent, unknown);
#endif
        raw_test.push_back(sent);
        str_sentences.push_back(str_sent);
    }
}

void Treebank::read_raw_sentence_discbracket(
        const String &line,
        vector<shared_ptr<Node>> &sentence,
        vector<std::pair<String, String> > &str_sentence,
        int unknown){
    assert(sentence.empty() && str_sentence.empty());
//    boost::char_separator<char> sep(" ");
//    boost::tokenizer<boost::char_separator<char>> toks(line, sep);
//    vector<string> tokens(toks.begin(), toks.end());
    vector<String> tokens;
    str::split(line, " ", "", tokens);

    //boost::char_separator<char> sep2("/");
    for (int i = 0; i < tokens.size(); i++){
        vector<String> tok_tag;
        int idx = tokens[i].size() -1;
        while (tokens[i][idx] != '/'){
            idx --;
        }
        assert(idx > 0);

        String token = tokens[i].substr(0, idx);
        String postag = tokens[i].substr(idx+1);

        tok_tag = {token, postag};

        assert(tok_tag.size() == 2);
        vector<STRCODE> fields;
        // TODO: update this
        if (unknown == UNKNOWN_CODING){
            fields.push_back(enc::hodor.code_unknown(tok_tag[0], enc::TOK));   /// REMINDER: this could cause problem for certain languages (unknown tags)
            fields.push_back(enc::hodor.code_unknown(tok_tag[1], enc::TAG));    // replace by UNKNOWN code (0) when unknown
        }else{
            assert(unknown == ENCODE_EVERYTHING);
            fields.push_back(enc::hodor.code(tok_tag[0], enc::TOK));   /// REMINDER: this could cause problem for certain languages (unknown tags)
            fields.push_back(enc::hodor.code(tok_tag[1], enc::TAG));    // replace by UNKNOWN code (0) when unknown
        }
        sentence.push_back(
                    shared_ptr<Node>(
                        new Leaf(enc::hodor.code(tok_tag[1], enc::CAT), i, fields)
                        )
                    );
        str_sentence.push_back(std::make_pair(tok_tag[0], tok_tag[1]));
    }
}

void Treebank::test_raw_sentence_reading(const string &filename){
    vector<vector<shared_ptr<Node>>> raw_test;
    vector<vector<std::pair<String,String>>> str_sentences;

    read_raw_input_sentences_discbracket(filename, raw_test, str_sentences, UNKNOWN_CODING);

    ofstream out1(filename+"_foo1");
    for (int i = 0; i < raw_test.size(); i++){
        for (int j = 0; j < raw_test[i].size(); j++){
            STRCODE tag = raw_test[i][j]->label();          // REMINDER : some languages -> this might be unknown
            STRCODE tok = raw_test[i][j]->get_field(Leaf::FIELD_TOK);
            string stag = enc::hodor.decode_to_str(tag, enc::CAT);
            string stok = enc::hodor.decode_to_str(tok, enc::TOK);
            if (tok == enc::UNKNOWN){
#ifdef WSTRING
                stok = str::encode(str_sentences[i][j].first);
#else
                stok = str_sentences[i][j].first;
#endif
            }
            out1 << stok << "/" << stag << " ";
        }out1 << endl;
    }
    out1.close();
    ofstream out2(filename+"_foo2");
    for (int i = 0; i < str_sentences.size(); i++){
        for (int j = 0; j < str_sentences[i].size(); j++){
#ifdef WSTRING
            out2 << str::encode(str_sentences[i][j].first) << "/"
                 << str::encode(str_sentences[i][j].second) << " ";
#else
            out2 << str_sentences[i][j].first << "/" << str_sentences[i][j].second << " ";
#endif
        }out2 << endl;
    }
    out2.close();
}















bool Treebank::is_xml_markup(const String &s){
    return s.size() > 2 && s[0] == LANGLE && s.back() == RANGLE;
}
bool Treebank::is_xml_beg(const String &s){
    return is_xml_markup(s) && s[1] != SLASH;
}
bool Treebank::is_xml_end(const String &s){
    return is_xml_markup(s) && s[1] == SLASH;
}
bool Treebank::get_xml_label(const String &s, String &label, string head_sep){ // return true if is head
    int beg = 1;
    if (is_xml_end(s)){
        beg = 2;
    }
    label = String(s.begin()+beg, s.end()-1);
    vector<String> fields;
    str::split(label, head_sep, "", fields);
    assert(fields.size() == 1 || fields.size() == 2);
    if (fields.size() == 2){
        assert(fields[1] == HEAD);
        label = fields[0];
        return true;
    }
    return false;
}

bool Treebank::is_head(String &s, string sep){
    vector<String> splits;
    str::split(s, sep, "", splits);
    if (splits.size() <= 1){
        return false;
    }
//    cerr << str::encode(s) << endl;
//    cerr << splits.size() << endl;
//    assert(splits.size() == 2);
//    assert(splits[1] == HEAD);
    if (splits.back() == HEAD){
        if (splits.size() == 3){
            s = splits[0] + splits[1];
#ifdef WSTRING
            cerr << "Warning. Encountered: " << str::encode(s) << " while reading treebank" << endl;
#else
            cerr << "Warning. Encountered: " << s << " while reading treebank" << endl;
#endif
        }else{
            assert(splits.size() == 2);
            s = splits[0];
        }
    }else{
        cerr << "Error, aborting" << endl;
        exit(1);
    }


    return true;
}


void Treebank::encode_fields(const vector<String> &fields, vector<STRCODE> &enc_fields){
    enc_fields.resize(fields.size());
    switch(unknown){
    case ENCODE_EVERYTHING:{
        for (int i = 0; i < fields.size(); i++){
            enc_fields[i] = enc::hodor.code(fields[i], i+1); // i+1 because 0 is type for CAT
        }
        break;
    }
    case CUTOFF:{
        for (int i = 0; i < fields.size(); i++){
            if (field_freqs[i][fields[i]] < MIN_FREQUENCY){
                enc_fields[i] = enc::UNKNOWN;
            }else{
                enc_fields[i] = enc::hodor.code(fields[i], i+1);
            }
        }
        break;
    }
    case UNKNOWN_CODING:{
        for (int i = 0; i < fields.size(); i++){
            enc_fields[i] = enc::hodor.code_unknown(fields[i], i+1);
        }
        break;
    }
    }
    enc_fields[1] = enc::hodor.code(fields[1], 2);
}





////////////////////////////////////////////////////////////////
///
///
///     Treebank Stats
///
///
/// ////////////////////////////////////////////////////////////

TreebankStats::TreebankStats(Treebank &tbk): rank(0), gap_degree(0), n_tokens(0), n_sentences(tbk.size()), longest_sent(0), num_constituents(0), num_disco_constituents(0){
    sentence_lengths = vector<int>(10, 0);
    gap_degrees = vector<int>(60, 0); // 60 -> we should be safe
    for (int i = 0; i < n_sentences; i++){
        Tree *t = tbk[i];

        num_constituents += t->num_constituents();
        num_disco_constituents += t->num_disco_constituents();

        int r = t->rank();
        int g = t->gap_degree();
        int l = t->length();

        longest_sent = std::max(longest_sent, l);
        n_tokens += l;

        int idx_len = (l-1)/10;
        if (idx_len > 9) idx_len = 9;
        sentence_lengths[idx_len] ++;

        if (r > rank) rank = r;
        if (g > gap_degree) gap_degree = g;
        gap_degrees[g] ++;
    }
}


ostream & operator<<(ostream &os, TreebankStats &ts){
    os << "- Number of sentences: " << ts.n_sentences << endl;
    os << "- Number of tokens: " << ts.n_tokens << endl;
    os << "- Average length: " << (double)ts.n_tokens / (double)ts.n_sentences;
    os << "- Rank of treebank: " << ts.rank << endl;
    os << "- Gap degree of treebank: " << ts.gap_degree << endl;
    os << "- Number of constituents: " << ts.num_constituents << " (disco: " << ts.num_disco_constituents << ", " << (float)(ts.num_disco_constituents) / ts.num_constituents << ")" << endl;
    os << "- Sentence lengths:" << endl;
    for (int i = 0; i < 9; i++){
        os << "\t["<<i<<"1-"<<(i+1)<<"0] : " << ts.sentence_lengths[i] << " (" << (ts.sentence_lengths[i] * 100.0) / ts.n_sentences << "%)" << endl;
    }
    os << "\t>90 : " << ts.sentence_lengths[9] << endl;
    os << "- Gap degree of sentences (frequencies):" << endl;
    for (int i = 0; i < ts.gap_degrees.size(); i++){
        if (ts.gap_degrees[i] > 0){
            os << "\t" << i << " : " << ts.gap_degrees[i] << " sentences (" << (ts.gap_degrees[i] * 100.0) / ts.n_sentences << "%)" << endl;
        }
    }
    return os;
}
