#include "treebank.h"


////////////////////////////////////////////////////////////////
///
///
///     Treebank
///
///
/// ////////////////////////////////////////////////////////////


std::default_random_engine Treebank::random = std::default_random_engine(SEED);

Treebank::~Treebank(){}
Treebank::Treebank(){}
Treebank::Treebank(const string & filename, int unknown) : unknown(unknown){
    read_discbracket_treebank(filename);
}

Treebank::Treebank(Treebank &tbk){
    for (int i = 0; i < tbk.size(); i++){
        Tree t;
        tbk[i]->copy(t);
        add_tree(t);
    }
}
Treebank& Treebank::operator=(Treebank &tbk){
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
    std::shuffle(trees_.begin(), trees_.end(), random);
}

void Treebank::subset(int n_sentences, Treebank &tbk){
    for (int i = 0; i < n_sentences; i++){
        tbk.add_tree(trees_[i]);
    }
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
    std::ifstream instream(filename);
    string buffer;
    vector<string> full_corpus;
    while(getline(instream,buffer)){
        full_corpus.push_back(buffer);
        //parse_discbracket_tree(buffer);
    }
    instream.close();
    if (unknown == CUTOFF){
        for (int i = 0; i < full_corpus.size(); i++){
            update_frequencies(full_corpus[i]);
        }
        #if DEBUG
            cerr << "{";
            for (auto &it : frequencies){
                cerr << it.first << ":" << it.second << ", ";
            }cerr << "}" << endl;
        #endif
    }
    for (int i = 0; i < full_corpus.size(); i++){
        parse_discbracket_tree(full_corpus[i]);
    }
}

void Treebank::parse_discbracket_tree(const string &line){
    boost::char_separator<char> sep(" ", ")(");
    boost::tokenizer<boost::char_separator<char>> toks(line, sep);
    vector<string> tokens(toks.begin(), toks.end());

    shared_ptr<Node> node;
    parse_tokens_disco(tokens, 0, tokens.size() -1, node);
    trees_.push_back(Tree(node));
}


void Treebank::parse_tokens_disco(const vector<string> &tokens, int d, int f, shared_ptr<Node> &res){
    assert(f-d >= 3 && "Error reading treebank");
    if (f - d == 3 && tokens[d].compare("(") == 0
                   && tokens[f].compare(")") == 0){
        int delimiter_idx = tokens[d+2].find_first_of("=");
        string id(tokens[d+2].substr(0, delimiter_idx));
        STRING terminal = tokens[d+2].substr(delimiter_idx+1);
        int tok_idx = stoi(id);

        vector<STRCODE> fields;
        switch(unknown){
            case ENCODE_EVERYTHING:{
                fields.push_back(enc::hodor.code(terminal, enc::TOK)); break;
            }
            case CUTOFF:{
                if (frequencies[terminal] < MIN_FREQUENCY)
                    fields.push_back(enc::UNKNOWN);
                else
                    fields.push_back(enc::hodor.code(terminal, enc::TOK));
                break;
            }
            case UNKNOWN_CODING:{
                fields.push_back(enc::hodor.code_unknown(terminal, enc::TOK));
                break;
            }
        }

//        if (unknown == CUTOFF && frequencies[terminal] < MIN_FREQUENCY){
//            fields.push_back(enc::UNKNOWN);
//        }else if {
//            fields.push_back(enc::hodor.code(terminal, enc::TOK));
//        }
        assert(fields.size() == 1);
        fields.push_back(enc::hodor.code(tokens[d+1], enc::TAG));

        res = shared_ptr<Node>(new Leaf(enc::hodor.code(tokens[d+1], enc::CAT), tok_idx, fields));
    }
    else if (tokens[d].compare("(") == 0){
        assert(tokens[f].compare(")") == 0 && "Error reading treebank");
        string label = tokens[d+1];
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

void Treebank::update_frequencies(const string &line){
    boost::char_separator<char> sep(" ", ")(");
    boost::tokenizer<boost::char_separator<char>> toks(line, sep);
    vector<string> tokens(toks.begin(), toks.end());
    update_frequencies_tokens(tokens, 0, tokens.size() -1);
}

void Treebank::update_frequencies_tokens(const vector<string> &tokens, int d, int f){
    assert(f-d >= 3 && "Error reading treebank");
    if (f - d == 3 && tokens[d].compare("(") == 0
                   && tokens[f].compare(")") == 0){
        int delimiter_idx = tokens[d+2].find_first_of("=");
        STRING terminal = tokens[d+2].substr(delimiter_idx+1);

        if (frequencies.find(terminal) != frequencies.end()){
            frequencies[terminal] += 1;
        }else{
            frequencies[terminal] = 1;
        }
    }else if (tokens[d].compare("(") == 0){
        assert(tokens[f].compare(")") == 0 && "Error reading treebank");
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

void Treebank::write(const string& filename, vector<vector<std::pair<string,string>>> &str_sentences){
    ofstream os(filename);
    for (int i = 0; i < trees_.size(); i++){
        trees_[i].write(os, str_sentences[i]);
        os << endl;
    }
    os.close();
}



int Treebank::aux(const vector<string> &tokens, int d, int f){
    int count = 0;
    if (! (tokens[d].compare("(") == 0)){
        cerr << "should not happen" << endl;
        return d;
    }
    while (d < f){
        if (tokens[d].compare("(") == 0){
            count ++;
        }
        else if (tokens[d].compare(")") == 0){
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
    Treebank tbk(tbk_filename, Treebank::ENCODE_EVERYTHING);
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
































void Treebank::read_raw_input_sentences(const string &filename, vector<vector<shared_ptr<Node>>> &raw_test, vector<vector<std::pair<string,string>>> &str_sentences){
    std::ifstream is(filename);
    string buffer;
    while (getline(is, buffer)){
        vector<shared_ptr<Node>> sent;
        vector<pair<string,string>> str_sent;
        read_raw_sentence(buffer, sent, str_sent);
        raw_test.push_back(sent);
        str_sentences.push_back(str_sent);
    }
}

void Treebank::read_raw_sentence(const string &line, vector<shared_ptr<Node>> &sentence, vector<std::pair<string,string>> &str_sentence){
    assert(sentence.empty() && str_sentence.empty());
    boost::char_separator<char> sep(" ");
    boost::tokenizer<boost::char_separator<char>> toks(line, sep);
    vector<string> tokens(toks.begin(), toks.end());

    boost::char_separator<char> sep2("/");
    for (int i = 0; i < tokens.size(); i++){
        vector<string> tok_tag;
        int idx = tokens[i].size() -1;
        while (tokens[i][idx] != '/'){
            idx --;
        }
        assert(idx > 0);
        string token = tokens[i].substr(0, idx);
        string postag = tokens[i].substr(idx+1);


        tok_tag.push_back(token);
        tok_tag.push_back(postag);

        assert(tok_tag.size() == 2);
        vector<STRCODE> fields;
        fields.push_back(enc::hodor.code_unknown(tok_tag[0], enc::TOK));   /// REMINDER: this could cause problem for certain languages (unknown tags)
        fields.push_back(enc::hodor.code_unknown(tok_tag[1], enc::TAG));    // replace by UNKNOWN code (0) when unknown
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
    vector<vector<std::pair<string,string>>> str_sentences;

    read_raw_input_sentences(filename, raw_test, str_sentences);

    ofstream out1(filename+"_foo1");
    for (int i = 0; i < raw_test.size(); i++){
        for (int j = 0; j < raw_test[i].size(); j++){
            STRCODE tag = raw_test[i][j]->label();          // REMINDER : some languages -> this might be unknown
            STRCODE tok = raw_test[i][j]->get_field(Leaf::FIELD_TOK);
            string stag = enc::hodor.decode(tag, enc::CAT);
            string stok = enc::hodor.decode(tok, enc::TOK);
            if (tok == enc::UNKNOWN){
                stok = str_sentences[i][j].first;
            }
            out1 << stok << "/" << stag << " ";
        }out1 << endl;
    }
    out1.close();
    ofstream out2(filename+"_foo2");
    for (int i = 0; i < str_sentences.size(); i++){
        for (int j = 0; j < str_sentences[i].size(); j++){
            out2 << str_sentences[i][j].first << "/" << str_sentences[i][j].second << " ";
        }out2 << endl;
    }
    out2.close();
}







////////////////////////////////////////////////////////////////
///
///
///     Treebank Stats
///
///
/// ////////////////////////////////////////////////////////////

TreebankStats::TreebankStats(Treebank &tbk): rank(0), gap_degree(0), n_tokens(0), n_sentences(tbk.size()), longest_sent(0){
    sentence_lengths = vector<int>(10, 0);
    gap_degrees = vector<int>(60, 0); // 60 -> we should be safe
    for (int i = 0; i < n_sentences; i++){
        Tree *t = tbk[i];

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
