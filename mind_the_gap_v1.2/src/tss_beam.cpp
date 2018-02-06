#include "tss_beam.h"


////////////////////////////////////////////////////////
///
///
///         Action
///
/// /////////////////////////////////////////////////////


Action::Action():Action(NULL_ACTION, 0,-1){}

Action::Action(int type, STRCODE label, int code)
    : type_(type), label_(label), code_(code){}
Action::Action(int type, int code) : Action(type, enc::UNDEF,-1){}

Action::Action(const Action &a, int code){
    this->type_ = a.type_;
    this->label_ = a.label_;
    this->code_ = code;
}
Action::Action(const Action &a)
    : type_(a.type_),
      label_(a.label_),
      code_(a.code_){}
Action& Action::operator=(const Action &a){
    this->type_ = a.type_;
    this->label_ = a.label_;
    this->code_ = a.code_;
    return *this;
}
Action::~Action(){}

int Action::code()const{ return code_;}
int Action::type()const{ return type_;}
int Action::order()const{
    assert(type_ == COMPOUND_GAP && "Cannot get order of action");
    return label_;
}
STRCODE Action::label()const{
    //assert((type_ == REDUCE_L || type_ == REDUCE_R || type_ == REDUCE_U || type_ == LABEL) && "Cannot get label of action");
    return label_;
}


bool Action::operator==(const Action &o)const{
    // use code ?
    if (type_ != o.type_){ return false;}
    switch (type_){
        case REDUCE_L:
        case REDUCE_R:
        case REDUCE_U:
        case LABEL:
            return label_ == o.label_;
        case SHIFT:
        case GAP:
        case IDLE:
        case GHOST_REDUCE:
        case MERGE:
            return true;
        case COMPOUND_GAP:
            //return order_ == o.order_;
            return label_ == o.label_;
    }
    assert(false);
    return false;
}

bool Action::operator!=(const Action &o)const{
    return ! (*this == o);
}

bool Action::operator<(const Action &o)const{
    if (type_ != o.type_) return type_ < o.type_;
    switch (type_){
        case SHIFT: return false;
        case REDUCE_L:
        case REDUCE_R:
        case REDUCE_U:
        case LABEL: return label_ < o.label_;
        case LEFT:
        case RIGHT:
        case GAP:
        case IDLE:
        case GHOST_REDUCE:
        case MERGE: return false;
        //case COMPOUND_GAP : return order_ < o.order_;
        case COMPOUND_GAP : return label_ < o.label_;
        case NULL_ACTION: return false;
    }
    assert(false && "Action::operator< should not be here");
    return false;
}

bool Action::is_null()const{
    return type_ == NULL_ACTION;
}

ostream& operator<< (ostream &os, const Action &action){
    switch (action.type_){
        case Action::SHIFT: return os << "sh" << action.code_;
        case Action::REDUCE_L: return os << "rl(" << action.label_ << "=" << enc::hodor.decode_to_str(action.label_,enc::CAT) << ")" << action.code_;
        case Action::REDUCE_R: return os << "rr(" << action.label_ << "=" << enc::hodor.decode_to_str(action.label_,enc::CAT) << ")" << action.code_;
        case Action::REDUCE_U: return os << "ru(" << action.label_ << "=" << enc::hodor.decode_to_str(action.label_,enc::CAT) << ")" << action.code_;
        case Action::GAP: return os << "gap" << action.code_;
        case Action::IDLE: return os << "idle" << action.code_;
        case Action::GHOST_REDUCE: return os << "gr" << action.code_;
        case Action::COMPOUND_GAP : return os << "cgap(" << action.label_ << ")" << action.code_;
        case Action::NULL_ACTION : return os << "NULL_ACTION" << action.code_;
        case Action::LABEL : return os << "Label("<< action.label_ << "=" << enc::hodor.decode_to_str(action.label_,enc::CAT) << ")" << action.code_;
        case Action::MERGE : return os << "merge" << action.code_;
        case Action::LEFT : return os << "merge left" << action.code_;
        case Action::RIGHT: return os << "merge right" << action.code_;
    }
    assert(false);
    return os;
}




///////////////////////////////////////////////////////////
///
///     StackItem
///
/// /////////////////////////////////////////


StackItem::StackItem(const shared_ptr<StackItem> &predecessor, const shared_ptr<Node> &n){
    this->predecessor = predecessor;
    this->n = n;
}
StackItem::StackItem(const shared_ptr<Node> &n){
    this->n = n;
}
StackItem::~StackItem(){}

void StackItem::get(shared_ptr<Node> &res){
    res = n;
}

void StackItem::get(int i, shared_ptr<Node> &res){
    if (i == 0){
        get(res);
        return;
    }
    if (is_bottom()){
        return;
    }
    predecessor->get(i-1, res);
}

void StackItem::get(int i, const shared_ptr<StackItem> &mid, shared_ptr<Node> &res){
    if (this == mid.get()) return;
    if (i == 0){
        get(res);
        return;
    }
    if (is_bottom()){
        return;
    }
    return predecessor->get(i-1, mid, res);
}

bool StackItem::is_bottom()const{ return predecessor == nullptr;}


ostream& operator<< (ostream &os, const StackItem &stack_item){
    if (stack_item.is_bottom())
        return os << *(stack_item.n);
    return os << *(stack_item.predecessor) << " || " << *(stack_item.n);
}



////////////////////////////////////////////////////
///
///    ParseState
///
/// //////////////////////////////////////////


ParseState::ParseState():
    buffer_j_(0),
    time_step_(0),
    score_(0){}

ParseState::ParseState(const shared_ptr<ParseState> &predecessor,
           const shared_ptr<StackItem> &top,
           const shared_ptr<StackItem> &mid,
           int buffer_j,
           const Action &last_action) :
    predecessor_(predecessor), top_(top), mid_(mid),
    buffer_j_(buffer_j), last_action_(last_action), score_(0.0){
    time_step_ = predecessor_->time_step_+1;
}

bool ParseState::is_init()const{
    return buffer_j_ == 0 && last_action_.is_null();
}

bool ParseState::is_final(const Grammar &grammar, int buffer_size)const{
    return buffer_j_ == buffer_size
            && top_->is_bottom()
            && grammar.is_axiom(top_->n->label());
}

void ParseState::set_score(double d){
    score_ = d;
}

double ParseState::score(){
    return score_;
}

int ParseState::time_step(){
    return time_step_;
}

bool ParseState::structure_action(){
    int a = last_action_.type();
    return a == Action::LABEL || a == Action::NULL_ACTION || a == Action::GAP;
}

bool ParseState::label_action(){
    int a = last_action_.type();
    return a == Action::SHIFT || a == Action::MERGE || a == Action::LEFT || a == Action::RIGHT;
}


///// Fork in tree-structured stack before reduction
/// if s0 and w0 are contiguous, return s0.predecessor
/// else return a the head of new branch containing gapped elements
/// and going back to s0.predecessor
/// exemple  [...s1,s0] [w2,w1,w0]     Reduce(s0,w0) = X
/// grow this branh from s1: [...s1, w2, w1]
/// Reduction will add X
/// If there has been gap actions, the gapped stack items are copied (but the nodes
/// they contains are shared (pointers)
void ParseState::grow_new_branch(shared_ptr<StackItem> &res){
    shared_ptr<StackItem> tmp = top_->predecessor;
    grow_rec(tmp, res);
}

bool ParseState::prefix_equals(Derivation &d){
    if (is_init()) return true;
    if (time_step_ <= d.size()){
        return last_action_ == *(d[time_step_-1]) && predecessor_->prefix_equals(d);
    }else{
        return last_action_.type() == Action::IDLE && predecessor_->prefix_equals(d);
    }
}


void ParseState::grow_rec(const shared_ptr<StackItem> &tmp, shared_ptr<StackItem> &res){
    if (tmp == mid_){
        res = mid_->predecessor;
    }else{
        grow_rec(tmp->predecessor, res);
        res = shared_ptr<StackItem>(new StackItem(res, tmp->n));
    }
}

void ParseState::get_derivation(vector<Action> &actions){
    if (is_init()) return;
    predecessor_->get_derivation(actions);
    actions.push_back(last_action_);
}

void ParseState::get_top_node(shared_ptr<Node> &node){
    node = top_->n;
}

ostream& operator<<(ostream &os, const ParseState &ps){
    if (ps.top_ != nullptr){
        os << "top: " << *ps.top_ << endl;
    }else{
        os << "top: null " << endl;
    }
    if (ps.mid_ != nullptr){
        os << "mid: " << *ps.mid_ << endl;
    }else{
        os << "mid: null" << endl;
    }
    os << "t=" << ps.time_step_ << " j=" << ps.buffer_j_ << " score=" << ps.score_ << " last_action=" << ps.last_action_;
    return os;
}



//////////////////////////////////////////////////////////
///
///
///         Candidate
///
/// /////////////////////////////////////////////////////




Candidate::Candidate(int predecessor, double score, const Action &a){
    this->predecessor = predecessor;
    this->score = score;
    this->action = a;
}

bool Candidate::operator<(const Candidate &c)const{
    return score > c.score;
}

ostream & operator<<(ostream &os, const Candidate &c){
    return os << "{p=" << c.predecessor << ",score=" << c.score << ",action=" << c.action << "}";
}


//////////////////////////////////////////////////////////
///
///
///         TSS beam
///
/// /////////////////////////////////////////////////////

TssBeam::TssBeam(int size, const vector<shared_ptr<Node>> &buffer){
    this->size_ = size;
    this->buffer = buffer;
    this->beam.push_back(shared_ptr<ParseState>(new ParseState()));
}

int TssBeam::size()const{ return beam.size(); }

void TssBeam::add_candidate(const Candidate &c){
    assert( ! std::isnan(c.score ) && ! std::isinf(c.score) && "Candidate score is nan or inf");
    candidates.push_back(c);
}

void TssBeam::next_step(TransitionSystem *ts){
    std::sort(candidates.begin(), candidates.end());

    vector<shared_ptr<ParseState>> newbeam;
    int current_size = candidates.size() < size_ ? candidates.size() : size_;
    for (int i = 0; i < current_size; i++){
        Candidate cand = candidates[i];
        shared_ptr<ParseState> next_state;
        ts->next(beam[cand.predecessor], cand.action, buffer, next_state);
        next_state->set_score(cand.score);
        newbeam.push_back(next_state);
    }

    if (candidates.size() == 0){
        cerr << "Dead end, no candidates, dumping beam" << endl;
        cerr << *this << endl;
        cerr << "Aborting" << endl;
        exit(1);
    }
    beam = newbeam;
    candidates.clear();
}

void TssBeam::get(int i, shared_ptr<ParseState> &ps){
    ps = beam[i];
}

bool TssBeam::gold_in_beam(Derivation &gold){
    for (int i = 0; i < size(); i++){
        if (beam[i]->prefix_equals(gold)){
            return true;
        }
    }
    return false;
}


void TssBeam::best_derivation(Derivation &d){
    double max_score = -std::numeric_limits<double>::infinity();
    shared_ptr<ParseState> best;
    for (int i = 0; i < beam.size(); i++){
        if (beam[i]->score() > max_score){
            best = beam[i];
            max_score = best->score();
        }
    }
    vector<Action> derivation;
    best->get_derivation(derivation);
    d = Derivation(derivation);
}

void TssBeam::best_tree(Tree &t){
    //assert(finished() && "No tree available, beam still has active parse states");
    shared_ptr<ParseState> best;
    double max_score = -std::numeric_limits<double>::infinity();
    if (beam.size() == 0){
        cerr << *this << endl;
        cerr << "Parsing failure" << endl;
        cerr << "probable NaN in weights, aborting" << endl;
        exit(1);
    }
    for (int i = 0; i < beam.size(); i++){
        if (beam[i]->score() > max_score){
            best = beam[i];
            max_score = best->score();
        }
    }
    if (best == nullptr){
        cerr << *this << endl;
        best = beam[0];
        cerr << "Parsing failure: Warning probable NaN in weights, you might wanna abort" << endl;
    }
    //assert(best->top_->is_bottom());
    shared_ptr<Node> node;
    best->get_top_node(node);
    t = Tree(node);
    t.score = max_score;
}

bool TssBeam::finished(const Grammar &grammar){
    for (int i = 0; i < beam.size(); i++){
        if (! beam[i]->is_final(grammar, buffer.size()))
            return false;
    }
    return true;
}

ostream& operator<<(ostream &os, const TssBeam &beam){
    os << "buffer=";
    for (int i = 0; i < beam.buffer.size(); i++)
        os << " | " << *(beam.buffer[i]);
    os << endl << "states=" << endl;
    for (int i = 0; i < beam.beam.size(); i++)
        os << *(beam.beam[i]) << endl;

    return os;
}
