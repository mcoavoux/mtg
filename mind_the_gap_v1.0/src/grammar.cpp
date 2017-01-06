#include "grammar.h"



Grammar::Grammar(){
    axioms = vector<bool>(MAX_SIZE, false);
    tmps   = vector<bool>(MAX_SIZE, false);
    unary  = vector<bool>(MAX_SIZE, false);
    punct  = vector<bool>(MAX_SIZE, false);
    unary_splits = vector<pair<STRCODE, STRCODE>>(MAX_SIZE, pair<STRCODE,STRCODE>(enc::UNDEF, enc::UNDEF));
    tmps2nontmps = vector<unordered_set<STRCODE>>(MAX_SIZE, unordered_set<STRCODE>());
    nontmps2tmps = vector<STRCODE>(MAX_SIZE, enc::UNDEF);

    binarise_ = true;
    //merge_unary_chains_ = true;
}

bool Grammar::binarise(){return binarise_;}
void Grammar::do_not_binarise(){ binarise_ = false; }

STRCODE Grammar::merge_unary_chain(STRCODE nt1, STRCODE nt2){
    STRING tmp = enc::hodor.decode(nt1, enc::CAT) + UNARY_CODE + enc::hodor.decode(nt2, enc::CAT);
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
    STRING tmp_symbol = enc::hodor.decode(non_tmp, enc::CAT) + TMP_CODE;
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
        if (enc::hodor.decode(i, enc::CAT)[0] == '$'){
            punct[i] = true;
        }
    }
    #ifdef DEBUG
    cerr << "Punctuations:" << endl;
    for (int i = 0; i < punct.size(); i++){
        if (punct[i])
            cerr << enc::hodor.decode(i, enc::CAT) << endl;
    }
    #endif
}



ostream& operator<<(ostream &os, const Grammar &grammar){
    os << endl << "Axioms" << endl;
    os << "------" << endl << endl;
    for (int i = 0; i < enc::hodor.size(enc::CAT); i++){
        if (grammar.axioms[i])
            os << enc::hodor.decode(i, enc::CAT) << endl;
    }
    os << endl << "Temporaries" << endl;
    os << "-----------" << endl << endl;
    for (int i = 0; i < enc::hodor.size(enc::CAT); i++){
        if (grammar.tmps[i])
            os << enc::hodor.decode(i, enc::CAT) << endl;
    }
    os << endl << "Unary chains" << endl;
    os << "------------" << endl << endl;
    for (int i = 0; i < enc::hodor.size(enc::CAT); i++){
        if (grammar.unary[i])
            os << enc::hodor.decode(i, enc::CAT) << endl;
    }
    os << endl << "Others" << endl;
    os << "------" << endl << endl;
    for (int i = 0; i < enc::hodor.size(enc::CAT); i++){
        if (! grammar.unary[i] && ! grammar.tmps[i] && ! grammar.axioms[i])
            os << enc::hodor.decode(i, enc::CAT) << endl;
    }return os << endl;
}





RulePriority::RulePriority(int direction, const vector<STRCODE> &priority) : direction(direction), priority(priority){}


ostream& operator<<(ostream& os, const RulePriority &rp){
    if (rp.direction == LEFT_TO_RIGHT) os << "left-to-right";
    else  os << "right-to-left";
    for (const STRCODE &code : rp.priority){
        STRING nt = enc::hodor.decode(code, enc::CAT);
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
    cerr << "Warning : no head found for " << enc::hodor.decode(label, enc::CAT) << endl;
    return 0;
}


void HeadRules::read_from_file(const string &filename){
    std::ifstream is(filename);
    string buffer;
    while(getline(is, buffer)){
        if (buffer.length() == 3 || buffer[0] == '%')
            continue;
        std::transform(buffer.begin(), buffer.end(), buffer.begin(), ::toupper);
        parse_line(buffer);
    }
    is.close();
    parse_line("PN LEFT-TO-RIGHT"); // TODO : line complètement ad hoc (pour tiger corpus)
}

void HeadRules::parse_line(const string &buffer){
    boost::char_separator<char> sep(" ");
    boost::tokenizer<boost::char_separator<char>> toks(buffer, sep);
    vector<string> tokens(toks.begin(), toks.end());
    if (tokens.size() < 2) return;

    STRCODE code = enc::hodor.code(tokens[0], enc::CAT);
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
        priority.push_back(
                    enc::hodor.code(tokens[i], enc::CAT)
                  );
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
            STRING nt = enc::hodor.decode(i, enc::CAT);
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

