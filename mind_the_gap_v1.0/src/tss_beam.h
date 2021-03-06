#ifndef TSS_BEAM_H
#define TSS_BEAM_H

#include <memory>
#include <assert.h>
#include <ostream>
#include <iostream>
#include <limits>

#include "utils.h"
#include "tree.h"
#include "transition_system.h"


using std::shared_ptr;
using std::ostream;

//http://users.cms.caltech.edu/~donnie/cs11/cpp/cpp-ops.html   -> check for selfassignement when implementing assignment operator


class Derivation;
class TransitionSystem;

// An action, versatile, could be used in any transition system
struct Action{
    enum {SHIFT, REDUCE_L, REDUCE_R, REDUCE_U, GAP, IDLE, GHOST_REDUCE, COMPOUND_GAP, NULL_ACTION, MERGE, LABEL};

private:
    int type_;         // shift, etc
    STRCODE label_;    // label for reductions
    int code_;         // keep ?


public:
    Action();//:Action(NULL_ACTION, 0,-1){}

    Action(int type, STRCODE label, int code);
    Action(int type, int code);

    Action(const Action &a, int code);
    Action(const Action &a);
    Action& operator=(const Action &a);
    ~Action();

    int code()const;
    int type()const;
    int order()const;
    STRCODE label()const;


    bool operator==(const Action &o)const;

    bool operator!=(const Action &o)const;

    bool operator<(const Action &o)const;

    bool is_null()const;

    friend ostream& operator<< (ostream &os, const Action &action);
};

// StackItem for a TSS, usable for std-sr (although not optimal) and gap-sr
struct StackItem{
    shared_ptr<StackItem> predecessor;  // parent in TSS
    shared_ptr<Node> n;                            // content of stackItem

    StackItem(const shared_ptr<StackItem> &predecessor, const shared_ptr<Node> &n);
    StackItem(const shared_ptr<Node> &n);
    ~StackItem();

    void get(shared_ptr<Node> &res);

    void get(int i, shared_ptr<Node> &res);

    void get(int i, const shared_ptr<StackItem> &mid, shared_ptr<Node> &res);

    bool is_bottom()const;

    friend ostream& operator<< (ostream &os, const StackItem &stack_item);
};

class ParseState{
    shared_ptr<ParseState> predecessor_;
    shared_ptr<StackItem> top_;
    shared_ptr<StackItem> mid_;
    int buffer_j_;
    int time_step_;
    Action last_action_;
    double score_;
public:
    friend class TransitionSystem;
    friend class GapTS;
    friend class CompoundGapTS;
    friend class MergeLabelTS;
    friend struct FeatureTemplate;

    ParseState();

    ParseState(const shared_ptr<ParseState> &predecessor,
               const shared_ptr<StackItem> &top,
               const shared_ptr<StackItem> &mid,
               int buffer_j,
               const Action &last_action);

    bool is_init()const;

    bool is_final(const Grammar &grammar, int buffer_size)const;

    void set_score(double d);

    double score();

    int time_step();


    ///// Fork in tree-structured stack before reduction
    /// if s0 and w0 are contiguous, return s0.predecessor
    /// else return a the head of new branch containing gapped elements
    /// and going back to s0.predecessor
    /// exemple  [...s1,s0] [w2,w1,w0]     Reduce(s0,w0) = X
    /// grow this branh from s1: [...s1, w2, w1]
    /// Reduction will add X
    /// If there has been gap actions, the gapped stack items are copied (but the nodes
    /// they contains are shared (pointers)
    void grow_new_branch(shared_ptr<StackItem> &res);

    bool prefix_equals(Derivation &d);

private:
    void grow_rec(const shared_ptr<StackItem> &tmp, shared_ptr<StackItem> &res);

public:
    void get_derivation(vector<Action> &actions);

    void get_top_node(shared_ptr<Node> &node);


    friend ostream& operator<<(ostream &os, const ParseState &ps);
};


struct Candidate{
    int predecessor;
    double score;
    Action action;
    Candidate(int predecessor, double score, const Action &a);
    bool operator<(const Candidate &c)const;
    friend ostream & operator<<(ostream &os, const Candidate &c);
};

class TssBeam{
    int size_;
    vector<shared_ptr<ParseState>> beam;
    vector<Candidate> candidates;
    vector<shared_ptr<Node>> buffer;
public:

    TssBeam(int size, const vector<shared_ptr<Node>> &buffer);

    int size()const;

    void add_candidate(const Candidate &c);

    void next_step(TransitionSystem *ts);

    void get(int i, shared_ptr<ParseState> &ps);

    bool gold_in_beam(Derivation &gold);

    void best_derivation(Derivation &d);

    void best_tree(Tree &t);

    bool finished(const Grammar &grammar);

    friend ostream& operator<<(ostream &os, const TssBeam &beam);
};


#endif // TSS_BEAM_H
