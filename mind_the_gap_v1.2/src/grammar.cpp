#include "grammar.h"



Grammar::Grammar(){
    axioms = vector<bool>(MAX_SIZE, false);
    tmps   = vector<bool>(MAX_SIZE, false);
    tmps[enc::UNDEF] = true;   // All undefs are considered tmps for the purpose of Merge Label systems. 14/03/2017
    unary  = vector<bool>(MAX_SIZE, false);
    punct  = vector<bool>(MAX_SIZE, false);
    unary_splits = vector<pair<STRCODE, STRCODE>>(MAX_SIZE, pair<STRCODE,STRCODE>(enc::UNDEF, enc::UNDEF));
    tmps2nontmps = vector<unordered_set<STRCODE>>(MAX_SIZE, unordered_set<STRCODE>());
    nontmps2tmps = vector<STRCODE>(MAX_SIZE, enc::UNDEF);

    binarise_ = true;
    //merge_unary_chains_ = true;
}

Grammar::Grammar(const string &outdir):Grammar(){
    import_bit_vec(outdir + "/grammar_axioms", axioms);
    import_bit_vec(outdir + "/grammar_tmps", tmps);
    tmps[enc::UNDEF] = true;   // All undefs are considered tmps for the purpose of Merge Label systems. 14/03/2017
    import_bit_vec(outdir + "/grammar_unary", unary);
    import_bit_vec(outdir + "/grammar_punct", punct);

    ifstream is_b(outdir + "/grammar_binarize");
    int bin;
    is_b >> bin;
    binarise_ = (bin != 0);
    is_b.close();

    ifstream is_un(outdir + "/grammar_unary_splits");
    string buffer;
    while (getline(is_un, buffer)){
        boost::char_separator<char> sep(" ");
        boost::tokenizer<boost::char_separator<char>> toks(buffer, sep);
        vector<string> tokens(toks.begin(), toks.end());
        assert (tokens.size() == 3 && "Error reading grammar");
        unary_splits[stoi(tokens[0])] = make_pair(stoi(tokens[1]), stoi(tokens[2]));
    }
    is_un.close();

    ifstream is_tmp(outdir + "/grammar_ntmps2tmps");
    while(getline(is_tmp, buffer)){
        boost::char_separator<char> sep(" ");
        boost::tokenizer<boost::char_separator<char>> toks(buffer, sep);
        vector<string> tokens(toks.begin(), toks.end());
        assert (tokens.size() == 2 && "Error reading grammar");
        nontmps2tmps[stoi(tokens[0])] = stoi(tokens[1]);
    }
    is_tmp.close();

    for (int i = 0; i < nontmps2tmps.size(); i++){
        tmps2nontmps[nontmps2tmps[i]].insert(i);
    }
    //        vector<unordered_set<STRCODE>> tmps2nontmps; // this one can be retrieved with nontmps2tmps
}


bool Grammar::binarise(){return binarise_;}
void Grammar::do_not_binarise(){ binarise_ = false; }

STRCODE Grammar::merge_unary_chain(STRCODE nt1, STRCODE nt2){
    String tmp = enc::hodor.decode(nt1, enc::CAT) + UNARY_CODE + enc::hodor.decode(nt2, enc::CAT);
    STRCODE code = enc::hodor.code(tmp, enc::CAT);
    unary[code] = true;
    unary_splits[code] = make_pair(nt1, nt2);
    return code;
}

pair<STRCODE,STRCODE> Grammar::get_unary_split(STRCODE nt){
    assert(unary[nt] && "Grammar:: not a unary chain symbol");
    return unary_splits[nt];
}

STRCODE Grammar::get_tmp(STRCODE non_tmp){
    if (nontmps2tmps[non_tmp] != enc::UNDEF){
        return nontmps2tmps[non_tmp];
    }
    String tmp_symbol = enc::hodor.decode(non_tmp, enc::CAT) + TMP_CODE;
    if (unary[non_tmp]){
        int idx = tmp_symbol.find_last_of(UNARY_CODE);
        tmp_symbol = tmp_symbol.substr(idx+1);
    }
    STRCODE code = enc::hodor.code(tmp_symbol, enc::CAT);
    tmps[code] = true;
    tmps2nontmps[code].insert(non_tmp);
    nontmps2tmps[non_tmp] = code;
    return code;
}

void Grammar::add_axiom(STRCODE code){   axioms[code] = true; }
bool Grammar::is_axiom(STRCODE code)const{    return axioms[code]; }
bool Grammar::is_tmp(STRCODE code)const{      return tmps[code];   }
bool Grammar::is_unary_chain(STRCODE code)const{ return unary[code]; }
bool Grammar::is_punct(STRCODE code)const{ return punct[code]; }

void Grammar::mark_punct(){
    for (int i = 0; i < enc::hodor.size(enc::CAT); i++){
        if (enc::hodor.decode(i, enc::CAT)[0] == DOLLAR){
            punct[i] = true;
        }
    }
    #ifdef DEBUG
    cerr << "Punctuations:" << endl;
    for (int i = 0; i < punct.size(); i++){
        if (punct[i])
            cerr << enc::hodor.decode_to_str(i, enc::CAT) << endl;
    }
    #endif
}















void Grammar::print_bit_vec(const string &outfile, vector<bool> &vec){
    ofstream out(outfile);
    for (int i = 0; i < vec.size(); i++){
        if (vec[i])
            out << i << endl;
    }
    out.close();
}

void Grammar::import_bit_vec(const string &outfile, vector<bool> &vec){
    ifstream is(outfile);
    string buffer;
    while (getline(is, buffer)){
        vec[stoi(buffer)] = true;
    }
    is.close();
}
void Grammar::export_model(const string &outdir){
    print_bit_vec(outdir + "/grammar_axioms", axioms);
    print_bit_vec(outdir + "/grammar_tmps", tmps);
    print_bit_vec(outdir + "/grammar_unary", unary);
    print_bit_vec(outdir + "/grammar_punct", punct);

    ofstream out_b(outdir + "/grammar_binarize");
    out_b << binarise_ << endl;
    out_b.close();

    ofstream out_un(outdir + "/grammar_unary_splits");
    for (int i = 0; i < unary_splits.size(); i++){
        out_un << i << " " << unary_splits[i].first << " " << unary_splits[i].second << endl;
    }
    out_un.close();

    ofstream out_tmp(outdir + "/grammar_ntmps2tmps");
    for (int i = 0; i < nontmps2tmps.size(); i++){
        out_tmp << i << " " << nontmps2tmps[i] << endl;
    }
    out_tmp.close();

//        vector<unordered_set<STRCODE>> tmps2nontmps; // this one can be retrieved with nontmps2tmps
}














ostream& operator<<(ostream &os, const Grammar &grammar){
    os << endl << "Axioms" << endl;
    os << "------" << endl << endl;
    for (int i = 0; i < enc::hodor.size(enc::CAT); i++){
        if (grammar.axioms[i])
            os << enc::hodor.decode_to_str(i, enc::CAT) << endl;
    }
    os << endl << "Temporaries" << endl;
    os << "-----------" << endl << endl;
    for (int i = 0; i < enc::hodor.size(enc::CAT); i++){
        if (grammar.tmps[i])
            os << enc::hodor.decode_to_str(i, enc::CAT) << endl;
    }
    os << endl << "Unary chains" << endl;
    os << "------------" << endl << endl;
    for (int i = 0; i < enc::hodor.size(enc::CAT); i++){
        if (grammar.unary[i])
            os << enc::hodor.decode_to_str(i, enc::CAT) << endl;
    }
    os << endl << "Others" << endl;
    os << "------" << endl << endl;
    for (int i = 0; i < enc::hodor.size(enc::CAT); i++){
        if (! grammar.unary[i] && ! grammar.tmps[i] && ! grammar.axioms[i])
            os << enc::hodor.decode_to_str(i, enc::CAT) << endl;
    }return os << endl;
}





RulePriority::RulePriority(int direction, const vector<STRCODE> &priority) : direction(direction), priority(priority){}


ostream& operator<<(ostream& os, const RulePriority &rp){
    if (rp.direction == LEFT_TO_RIGHT) os << "left-to-right";
    else  os << "right-to-left";
    for (const STRCODE &code : rp.priority){
        string nt = enc::hodor.decode_to_str(code, enc::CAT);
        std::transform(nt.begin(), nt.end(), nt.begin(), ::tolower);
        os << " " << nt;
    }
    return os;
}




HeadRules::HeadRules(const string &filename, Grammar *grammar){
    this->grammar_ = grammar;
    rules = vector<vector<RulePriority>>(Grammar::MAX_SIZE);
    read_from_file(filename);
}

HeadRules::~HeadRules(){}

void HeadRules::add(STRCODE nt, const RulePriority &rp){
    rules[nt].push_back(rp);
}

/*int HeadRules::find_head(Node &n) const{
    int head = -1;
    STRCODE label = n.label();
    for (const RulePriority &rp : rules[label]){
        if (rp.direction == LEFT_TO_RIGHT){
            for (const STRCODE &nt : rp.priority){
                for (int i = 0; i < n.arity(); i++){
                    shared_ptr<Node> child;
                    n.get(i, child);
                    if (child->label() == nt){
                        return i;
                    }
                }
            }
            if (head == -1) head = 0;
        }
        if (rp.direction == RIGHT_TO_LEFT){
            for (const STRCODE &nt : rp.priority){
                for (int i = n.arity()-1; i >= 0; i--){
                    shared_ptr<Node> child;
                    n.get(i, child);
                    if (child->label() == nt){
                        return i;
                    }
                }
            }
            if (head == -1) head = n.arity()-1;
        }
    }
    if (head == -1){
        cerr << "Warning : no head found for " << enc::hodor.decode(label, enc::CAT) << endl;
    }
    return head;
}*/

// new headfinder
int HeadRules::find_head(Node &n) const{
    STRCODE label = n.label();
    for (const RulePriority &rp : rules[label]){
        if (rp.direction == LEFT_TO_RIGHT){
            for (const STRCODE &nt : rp.priority){
                for (int i = 0; i < n.arity(); i++){
                    shared_ptr<Node> child;
                    n.get(i, child);
                    if (child->label() == nt){
                        return i;
                    }
                }
            }
        }
        else if (rp.direction == RIGHT_TO_LEFT){
            for (const STRCODE &nt : rp.priority){
                for (int i = n.arity()-1; i >= 0; i--){
                    shared_ptr<Node> child;
                    n.get(i, child);
                    if (child->label() == nt){
                        return i;
                    }
                }
            }
        }else{
            assert(false && "HeadRules::find_head : Error : no direction");
        }
    }
    // find with first direction, exclude punct
    for (const RulePriority &rp : rules[label]){
        if (rp.direction == LEFT_TO_RIGHT){
            for (int i = 0; i < n.arity(); i++){
                shared_ptr<Node> child;
                n.get(i, child);
                if (! grammar_->is_punct(child->label())){
                    return i;
                }
            }
        }
        if (rp.direction == RIGHT_TO_LEFT){
            for (int i = n.arity()-1; i >= 0; i--){
                shared_ptr<Node> child;
                n.get(i, child);
                if (! grammar_->is_punct(child->label())){
                    return i;
                }
            }
        }
    }
    // third pass : if nothing found so far, include punctuation
    for (const RulePriority &rp : rules[label]){
        if (rp.direction == LEFT_TO_RIGHT){
            return 0;
        }
        if (rp.direction == RIGHT_TO_LEFT){
            return n.arity() -1;
        }
    }
    cerr << "Warning : no head found for " << enc::hodor.decode_to_str(label, enc::CAT) << "  --->";
    for (int i = 0; i < n.arity(); i++){
        shared_ptr<Node> child;
        n.get(i, child);
        cerr << "  " << enc::hodor.decode_to_str(child->label(), enc::CAT);
    }
    cerr << endl;
    return 0;
}


void HeadRules::read_from_file(const string &filename){
    std::ifstream is(filename);
    string buffer;
    while(getline(is, buffer)){
        if (buffer.length() == 3 || buffer[0] == '%')
            continue;
        if (filename == "../data/negra.headrules"){
            std::transform(buffer.begin(), buffer.end(), buffer.begin(), ::toupper);
        }
        parse_line(buffer);
    }
    is.close();
    parse_line("PN LEFT-TO-RIGHT"); // TODO / REMINDER: ad hoc line for tiger corpus
}

void HeadRules::parse_line(const string &buffer){
    vector<string> tokens;
    str::split(buffer, " ", "", tokens);
    if (tokens.size() < 2) return;
#ifdef WSTRING
    STRCODE code = enc::hodor.code(str::decode(tokens[0]), enc::CAT);
#else
    STRCODE code = enc::hodor.code(tokens[0], enc::CAT);
#endif
    int direction = -1;
    if (tokens[1].compare("LEFT-TO-RIGHT") == 0){
        direction = LEFT_TO_RIGHT;
    }else if (tokens[1].compare("RIGHT-TO-LEFT") == 0){
        direction = RIGHT_TO_LEFT;
    }else{
        assert(false && "Head rule invalid or (more likely) bug");
    }
    vector<STRCODE> priority;
    for (int i = 2; i < tokens.size(); i++){
#ifdef WSTRING
        priority.push_back(enc::hodor.code(str::decode(tokens[i]), enc::CAT));
#else
        priority.push_back(enc::hodor.code(tokens[i], enc::CAT));
#endif
    }
    RulePriority rp(direction, priority);
    add(code, rp);
}

const Grammar * HeadRules::grammar()const{
    return grammar_;
}


ostream& operator<<(ostream &os, const HeadRules &hr){
    for (int i = 0; i < hr.rules.size(); i++){
        for (int j = 0; j < hr.rules[i].size(); j++){
            string nt = enc::hodor.decode_to_str(i, enc::CAT);
            std::transform(nt.begin(), nt.end(), nt.begin(), ::tolower);
            os << nt << " " << hr.rules[i][j] << endl;
        }
    }
    return os;
}

void HeadRules::test(const string &filename){
    HeadRules hr(filename, nullptr);
    std::ofstream os("hr_test");
    os << hr << endl;
    os.close();
}

