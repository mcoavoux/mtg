#include "tree.h"

////////////////////////////////////
///
/// Constituent
///
///////////////////////////////////////



Constituent::Constituent(STRCODE label, const vector<pair<int,int>> &spans)
    : label_(label), spans_(spans){}

bool Constituent::operator==(const Constituent &c){
    if (label_ != c.label_) return false;
    if (spans_.size() != c.spans_.size()) return false;
    for (int i = 0; i < spans_.size(); i++){
        if (spans_[i] != c.spans_[i]){
            return false;
        }
    }
    return true;
}
bool Constituent::operator!=(const Constituent &c){
    return ! ((*this) == c);
}
// REMINDER: in case of evaluation problem, double check this function
bool Constituent::operator<(const Constituent &c)const{
    if (label_ < c.label_) return true;
    if (label_ > c.label_) return false;
    if (spans_.size() < c.spans_.size()) return true;
    if (spans_.size() > c.spans_.size()) return false;
    for (int i = 0; i < spans_.size(); i++){
        if (spans_[i].first < c.spans_[i].first) return true;
        if (spans_[i].first > c.spans_[i].first) return false;
        if (spans_[i].second < c.spans_[i].second) return true;
        if (spans_[i].second > c.spans_[i].second) return false;
    }
    return false;
}




////////////////////////////////////
///
/// Node
///
///////////////////////////////////////
Node::Node(STRCODE label, int j) :label_(label){
    spans_.push_back(make_pair(j,j+1));
    h_ = -1;
}


Node::Node(STRCODE label, const vector<shared_ptr<Node>> &children) : label_(label), children_(children){
    h_ = -1;
    std::sort(children_.begin(), children_.end(),
              [](const shared_ptr<Node> &a, const shared_ptr<Node> &b) -> bool{
                    return a->index() < b->index();
                });
    compute_spans();
}

Node::Node(const vector<shared_ptr<Node>> &children): Node(enc::UNDEF, children){}

Node::~Node(){}

void Node::compute_spans(){
    for (auto& it : children_){
        for (auto& sp : it->spans_){
            spans_.push_back(sp);
        }
    }

    std::sort(spans_.begin(), spans_.end(),
              [](const pair<int,int> &a, const pair<int,int> &b) -> bool{
                  if (a.first == b.first) return a.second > b.second;
                  return a.first < b.first;
                }
              );


//    cerr << "spansize : " << spans_.size() <<  "   "  << children_.size() << endl;
    assert(spans_.size() > 0);
    for (int i = 0; i < spans_.size() - 1; i++){
//        cerr << "    " << endl;
//        cerr << spans_.size() << " " << spans_.size() -1 << endl;
//        cerr << "conditions " << (i < spans_.size() - 1) << endl;
//        cerr << spans_[i].second << endl;
//        cerr << spans_[i+1].first << endl;
        if (spans_[i].second == spans_[i+1].first){
            pair<int,int> span = make_pair(spans_[i].first, spans_[i+1].second);
            spans_[i] = span;
            spans_.erase(spans_.begin() + i+1);
            i--;
        }
    }
}

int Node::arity() const { return children_.size();}
STRCODE Node::label()const { return label_;}
int Node::index() const{ return std::get<0>(spans_[0]); }
int Node::left_corner()const{  return index();  }
int Node::right_corner()const{
    int max = index();
    for (int i = 0; i < spans_.size(); i++){
        if (std::get<1>(spans_[i]) > max){
            max = std::get<1>(spans_[i]);
        }
    }
    return max -1;
}


void Node::get(int i, shared_ptr<Node> &ptr){
    assert(i < children_.size() && "Trying to access inexisting descendant");
    ptr = children_[i];
}

int Node::rank()const{
    int g = arity();
    for (int i = 0; i < children_.size(); i++){
        int r = children_[i]->rank();
        if (r > g)
            g = r;
    }
    return g;
}

int Node::gap_degree()const{
    int g = spans_.size() -1;
    for (int i = 0; i < children_.size(); i++){
        int r = children_[i]->gap_degree();
        if (r > g)
            g = r;
    }
    return g;
}

int Node::num_constituents(){
    if (this->is_preterminal()){
        return 0;
    }
    int n = 1;
    for (shared_ptr<Node> &child : children_){
        n += child->num_constituents();
    }
    return n;
}

int Node::num_disco_constituents(){
    if (this->is_preterminal()){
        return 0;
    }
    int n = 0;
    if (spans_.size() > 1){ n++; }
    for (shared_ptr<Node> &child : children_){
        n += child->num_disco_constituents();
    }
    return n;
}


void Node::set_label(STRCODE label){
    this->label_ = label;
}

bool Node::has_label(){
    return this->label_ > enc::UNDEF;
}

void Node::get_children(vector<shared_ptr<Node>> &newchildren){
    for (int i = 0; i < children_.size(); i++){
        newchildren.push_back(children_[i]);
    }
}

int Node::h()const{
    //assert(h_ != -1);
    return h_;
}

void Node::head(shared_ptr<Node> &ptr){
    assert(head_.get() != nullptr);
    ptr = head_;
}

void Node::set_h(int i){
    h_ = i;
    if (children_[i]->is_preterminal()){
        head_ = children_[i];
    }else{
        head_ = children_[i]->head_;
    }
}


STRCODE Node::dlabel(){
    assert(false);
    return enc::UNKNOWN;
}
void Node::set_dlabel(STRCODE label){
    assert(false);
}

void Node::set_pred_field(int i, STRCODE val){
    assert(false);
}

string Node::morpho_repr(){
    assert(false);
    return "_";
}

bool Node::is_preterminal()const{ return false;}

void Node::yield(vector<shared_ptr<Node>> &res){
    for (int i = 0; i < children_.size(); i++){
        if (children_[i]->is_preterminal()){
            res.push_back(children_[i]);
        }else{
            children_[i]->yield(res);
        }
    }
}

STRCODE Node::get_field(int i)const{
    assert(false && "Not implemented in this class");
    return 0;
}

int Node::n_fields(){
    return 0;
}



void Node::annotate_heads(const HeadRules &hr){
    if (is_preterminal()){      // preterminal do not point to themselves (avoid circular references)
        return;
    }
    for (int i = 0; i < children_.size(); i++){
        children_[i]->annotate_heads(hr);
    }
    h_ = hr.find_head(*this);
    assert(h_ < children_.size() && h_ >= 0 && "Node::annotate_head : head index is too high");
    if (children_[h_]->is_preterminal()){
        head_ = children_[h_];
    }else{
        children_[h_]->head(head_);
    }
    #ifdef DEBUG
        // Check if rules causes heads to be punctuation tokens (should not happen, except for ROOT, VROOT)
        if (hr.grammar()->is_punct(head_->label()) && enc::hodor.decode_to_str(label_, enc::CAT) != "ROOT"){
            cerr << "head of " << enc::hodor.decode_to_str(label_, enc::CAT)
                 << " is "     << enc::hodor.decode_to_str(head_->label(), enc::CAT) << endl;
        }
    #endif
}




void Node::binarize(Grammar &grammar){
    if (is_preterminal()){
        return;
    }
    for (int i = 0; i < children_.size(); i++){
        children_[i]->binarize(grammar);
    }
    if (arity() < 3) return;
    STRCODE tmp_code = grammar.get_tmp(label_);
    assert(h_ != -1 && "Heads have not been assigned, cannot binarize");

    while (arity() > 2 && h_ > 0){ // binarize left of head
        vector<shared_ptr<Node>> newch(children_.begin()+h_-1, children_.begin()+h_+1);
        assert (newch.size() == 2);
        shared_ptr<Node> n = shared_ptr<Node>(new Node(tmp_code, newch));
        n->h_ = 1;
        n->head_ = head_;
        children_[h_] = n;
        children_.erase(children_.begin()+h_-1, children_.begin()+h_);
        h_--;
    }
    while (arity() > 2){ // binarize left of head
        vector<shared_ptr<Node>> newch(children_.begin()+h_, children_.begin()+h_+2);
        assert (newch.size() == 2);
        shared_ptr<Node> n(new Node(tmp_code, newch));
        n->h_ = 0;
        n->head_ = head_;
        children_[h_] = n;
        children_.erase(children_.begin()+h_+1, children_.begin()+h_+2);
    }
    /// TODO: (or not) recompute spans ?
}


void Node::unbinarize(Grammar &grammar){
    if (is_preterminal()) return;
    vector<shared_ptr<Node>> frontier;
    get_frontier(grammar, frontier);
    children_ = frontier;
    for (int i = 0; i < children_.size(); i++){
        children_[i]->unbinarize(grammar);
    }
    // TODO: heads not updated
    // spans not updated but should be ok
}

void Node::merge_unaries(Grammar &grammar){
    for (int i = 0; i < children_.size(); i++){
        children_[i]->merge_unaries(grammar);
    }
    if (arity() == 1 && ! children_[0]->is_preterminal()){
        label_ = grammar.merge_unary_chain(label_, children_[0]->label_);
        h_ = children_[0]->h_;
        head_ = children_[0]->head_;
        children_ = children_[0]->children_;
    }
}

void Node::split_unaries(Grammar &grammar){
    if (is_preterminal()) return;
    if (grammar.is_unary_chain(label_)){
        pair<STRCODE, STRCODE> splits = grammar.get_unary_split(label_);
        shared_ptr<Node> child(new Node(splits.second, children_));
        child->h_ = h_;
        child->head_ = head_;
        children_.clear();
        children_.push_back(child);
        label_ = splits.first;
        h_ = 0;
    }
    for (int i = 0; i < children_.size(); i++){
        children_[i]->split_unaries(grammar);
    }
}

void Node::get_frontier(Grammar &grammar, vector<shared_ptr<Node>> &frontier){
    for (int i = 0; i < children_.size(); i++){
        if (grammar.is_tmp(children_[i]->label())){
            children_[i]->get_frontier(grammar, frontier);
        }else{
            frontier.push_back(children_[i]);
        }
    }
}

void Node::get_parent(shared_ptr<Node> &n){
    n = shared_ptr<Node>(parent_);
}

void Node::set_parent(shared_ptr<Node> &n){
    parent_ = weak_ptr<Node>(n);
}

void Node::annotate_parent_ptr(shared_ptr<Node> &n){
    for (int i = 0; i < n->children_.size(); i++){
        annotate_parent_ptr(n->children_[i]);
        n->children_[i]->parent_ = weak_ptr<Node>(n);
    }
}

void Node::extract_constituents(set<Constituent> &cset){
    // start directly with children_ -> ignore root of tree
    for (int i = 0; i < children_.size(); i++){
        if (! children_[i]->is_preterminal()){
            cset.insert(Constituent(children_[i]->label_, children_[i]->spans_));
            children_[i]->extract_constituents(cset);
        }
    }
}

void Node::copy(shared_ptr<Node> &result)const{
    vector<shared_ptr<Node>> newchildren(children_.size());
    for (int i = 0; i < children_.size(); i++){
        children_[i]->copy(newchildren[i]);
    }
    result = shared_ptr<Node>(new Node(this->label(), newchildren));
    if (h_ != -1){ // Important to handle tbk/discbracket (head-annotation wise)
        result->set_h(h_);
    }
}


void Node::write(ostream &os, vector<std::pair<String, String>> &str_sentences){
    if (is_preterminal()){
        String postag;
        String token;
        //if (get_field(Leaf::FIELD_TAG) == enc::UNKNOWN){
        if (label() == enc::UNKNOWN){
            postag = str_sentences[index()].second;
        }else{
            //postag = enc::hodor.decode(get_field(Leaf::FIELD_TAG), enc::TAG);
            postag = enc::hodor.decode(label(), enc::CAT);
        }
        if (get_field(Leaf::FIELD_TOK) == enc::UNKNOWN){
            token = str_sentences[index()].first;
        }else{
            token = enc::hodor.decode(get_field(Leaf::FIELD_TOK), enc::TOK);
        }
#ifdef WSTRING
        os << "(" << str::encode(postag)
           << " " << index()
           << "=" << str::encode(token)
           << ")";
#else
        os << "(" << postag
           << " " << index()
           << "=" << token
           << ")";
#endif
    }else{
        os << "(" << enc::hodor.decode_to_str(label(), enc::CAT);
        for (auto &it : children_){
            os << " ";
            it->write(os, str_sentences);
        }
        os << ")";
    }
}



void Node::update_conll_dep(vector<int> &dtree){
    if (is_preterminal()){ return;}
    assert(h_ >= 0 && h_ < children_.size());
    shared_ptr<Node> head_node;

    int j = -1;
    if (children_[h_]->is_preterminal()){
        j = children_[h_]->index();
    }else{
        children_[h_]->head(head_node);
        j = head_node->index();
    }

    for (int i = 0; i < children_.size(); i++){
        if (i != h_){
            if (children_[i]->is_preterminal()){
                dtree[children_[i]->index()] = j;
            }else{
                shared_ptr<Node> c;
                children_[i]->head(c);
                dtree[c->index()] = j;
            }
        }
    }
    for (int i = 0; i < children_.size(); i++){
        children_[i]->update_conll_dep(dtree);
    }
}

ostream& operator<<(ostream &os, const Node &node){
    if (node.is_preterminal()){
        os << "(" << enc::hodor.decode_to_str(node.get_field(Leaf::FIELD_TAG), enc::TAG)
           << " " << node.index()
           << "=" << enc::hodor.decode_to_str(node.get_field(Leaf::FIELD_TOK), enc::TOK);
    }else{
        os << "(" << enc::hodor.decode_to_str(node.label(), enc::CAT);
        for (auto &it : node.children_)
            os << " " << *it;
    }
    return os << ")";
}


/******************************
 *
 *
 *  Leaf
 *
 *
 ********************************/


Leaf::Leaf(STRCODE label, int j, const vector<STRCODE> &fields) : Node(label, j), fields_(fields), dlabel_(enc::UNKNOWN){}
Leaf::~Leaf(){}
bool Leaf::is_preterminal()const{ return true;}

STRCODE Leaf::get_field(int i)const{
    assert (i < fields_.size() && "Leaf::get_field: error");
    return fields_.at(i);
}
int Leaf::n_fields(){
    return fields_.size();
}

STRCODE Leaf::dlabel(){
    return dlabel_;
}

void Leaf::set_dlabel(STRCODE label){
    dlabel_ = label;
}

void Leaf::set_pred_field(int i, STRCODE val){
    if (pred_fields_.size() <= i){
        pred_fields_.resize(i+1);
    }
    pred_fields_[i] = val;
}

string Leaf::morpho_repr(){
    vector<string> m;
    for (int i = 1; i < pred_fields_.size(); i++){
        if (pred_fields_[i] > enc::UNDEF){
            string attribute = enc::hodor.get_header(i+1);
            if (attribute != "gdeprel"){
                string s = attribute + "=" + enc::hodor.decode_to_str(pred_fields_[i], i+2);
                m.push_back(s);
            }
        }
    }
    if (m.size() == 0){
        return "_";
    }
    string all = m[0];
    for (int i = 1; i < m.size(); i++){
        all += "|" + m[i];
    }
    return all;
}

void Leaf::copy(shared_ptr<Node> &result)const{
    result = shared_ptr<Node>(new Leaf(label_, index(), fields_));
}


/******************************
 *
 *
 *  Tree
 *
 *
 ********************************/

Tree::Tree(){}

Tree::Tree(const shared_ptr<Node> &root) : root_(root){
    root->yield(leafs);
    std::sort(leafs.begin(), leafs.end(),
              [](const shared_ptr<Node> &a, const shared_ptr<Node> &b) -> bool{
                    return a->index() < b->index();
                });
}

//Tree::Tree(const Tree &t){
//    t.root_->copy(root_);
//    root_->yield(leafs);
//    std::sort(leafs.begin(), leafs.end(),
//              [](const shared_ptr<Node> &a, const shared_ptr<Node> &b) -> bool{
//                    return a->index() < b->index();
//                });
//}

//Tree& Tree::operator=(const Tree &t){
//    t.root_->copy(root_);
//    root_->yield(leafs);
//    std::sort(leafs.begin(), leafs.end(),
//              [](const shared_ptr<Node> &a, const shared_ptr<Node> &b) -> bool{
//                    return a->index() < b->index();
//                });
//    return *this;
//}

void Tree::copy(Tree &t){
    shared_ptr<Node> node;
    root_->copy(node);
    t = Tree(node);
}

int Tree::length(){ return leafs.size();}
int Tree::rank(){ return root_->rank();}
int Tree::gap_degree(){ return root_->gap_degree();}

int Tree::num_constituents(){return root_->num_constituents();}
int Tree::num_disco_constituents(){return root_->num_disco_constituents();}


void Tree::annotate_heads(const HeadRules &hr){
    root_->annotate_heads(hr);
}

void Tree::binarize(Grammar &grammar){
    if (grammar.binarise()){
        root_->binarize(grammar);
    }
    root_->merge_unaries(grammar);
    grammar.add_axiom(root_->label());
}
void Tree::unbinarize(Grammar &grammar){
    root_->split_unaries(grammar);
    if (grammar.binarise()){
        root_->unbinarize(grammar);
    }
}

void Tree:: get_buffer(vector<shared_ptr<Node>> &buffer){
    buffer = leafs;
}

void Tree::get_root(shared_ptr<Node> &root){
    root = root_;
}

void Tree::annotate_parent_ptr(){
    Node::annotate_parent_ptr(root_);
}

void Tree::extract_constituents(set<Constituent> &cset){
    root_->extract_constituents(cset);
}

void Tree::write(ostream &os, vector<std::pair<String,String>> &str_sentences, bool prob){
    if (prob){
        os << this->score << "\t";
    }
    root_->write(os, str_sentences);
}

void Tree::write_conll(ostream &os, vector<std::pair<String, String>> &str_sentences, int deprel_idx, bool unlex){
    const string t = "\t";
    vector<int> dtree(str_sentences.size(), -1);

    if (! unlex){
        root_->update_conll_dep(dtree);
    }

    for (int i = 0; i < leafs.size(); i++){
        //1	Joan	joan	ADI	SIN	ADM=PART|ASP=BURU	3	cmod	_	_
#ifdef WSTRING
        os << (i+1)
           << t << str::encode(str_sentences[i].first)
           << t << "_"
           << t << enc::hodor.decode_to_str(leafs[i]->label(), enc::CAT)
           << t << enc::hodor.decode_to_str(leafs[i]->label(), enc::CAT)
           << t << leafs[i]->morpho_repr()
           << t << (dtree[i] + 1)
           << t << enc::hodor.decode_to_str(leafs[i]->dlabel(), deprel_idx+1) // +1 because first encoder is for non terminal
           << t << "_"
           << t << "_"
           << endl;
#else
        os << (i+1)
           << t << str_sentences[i].first
           << t << "_"
           << t << enc::hodor.decode_to_str(leafs[i]->label(), enc::CAT)
           << t << enc::hodor.decode_to_str(leafs[i]->label(), enc::CAT)
           << t << leafs[i]->morpho_repr()
           << t << (dtree[i] + 1)
           << t << enc::hodor.decode(leafs[i]->dlabel(), deprel_idx+1) // +1 because first encoder is for non terminal
           << t << "_"
           << t << "_"
           << endl;
#endif
    }
}

ostream& operator<<(ostream &os, const Tree &tree){
    return os << *(tree.root_);
}





/////////////////////////////////////////////////////////////////
///
///
///     evaluation
///
///
/// /////////////////////////////////////////////////////////////


namespace eval{

    EvalTriple::EvalTriple() : true_positive(0.0), predicted(0.0), gold(0.0){}

    double EvalTriple::precision()const{
        return 100.0 * (true_positive / predicted);
    }

    double EvalTriple::recall()const{
        return 100.0 * (true_positive / gold);
    }

    double EvalTriple::fscore()const{
        double p = precision();
        double r = recall();
        return 2 * p * r / (p+r);
    }


    void compare(Tree &gold, Tree &pred, EvalTriple &triple){
        set<Constituent> gs;
        gold.extract_constituents(gs);
        set<Constituent> ps;
        pred.extract_constituents(ps);

        set<Constituent> intersection;
        for (const Constituent &c : gs){
            if (ps.find(c) != ps.end()){
                intersection.insert(c);
            }
        }
        triple.true_positive += intersection.size();
        triple.predicted += ps.size();
        triple.gold += gs.size();
    }
}
