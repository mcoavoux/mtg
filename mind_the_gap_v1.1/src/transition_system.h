#ifndef TRANSITION_SYSTEM_H
#define TRANSITION_SYSTEM_H

#include <vector>
#include <map>
#include <deque>
#include <fstream>
#include <boost/tokenizer.hpp>

#include "tree.h"
#include "tss_beam.h"
#include "treebank.h"

using std::map;
using std::ofstream;
using std::ifstream;

struct Action;
class ParseState;
struct StackItem;


class Derivation{
    vector<Action> derivation_;
public:
    Derivation();
    Derivation(const vector<Action> &derivation);

    void add(const Action &a);

    int size()const;

    void pad(int i, const Action &a);

    void clear();

    const Action* operator[](int i)const;

    bool operator==(const Derivation &d);

    bool operator!=(const Derivation &d);

    bool prefix_equals(const Derivation &d);

    friend ostream& operator<<(ostream &os, const Derivation &d);
};


// Base abstract class for all transition systems
//  Any TS should extend this one
//   All transition systems have a list of actions
//     and an access to a Grammar
// [Grammar class stores list of non-terminals
//  and methods to test efficiently if a non-terminal
//  is temporary, is an axiom, is a merge of unary symbols ...]
class TransitionSystem{
protected:
    vector<Action> actions;
    map<Action, int> encoder;
    Grammar grammar;
    int system_id;
public:
    enum {GAP_TS, CGAP_TS, MERGE_LABEL_TS, SHIFT_REDUCE};// transition system identifiers

    TransitionSystem(int s_id);
    virtual ~TransitionSystem();

    virtual void compute_derivation(Tree &tree, Derivation &derivation)=0;
    virtual bool allowed(ParseState &state, int buffer_size, int i) = 0;

    virtual Action* get_idle()=0;

    void allowed(ParseState &state, int buffer_size, vector<bool> &allowed_actions);

    void add_action(const Action &a);

    Action* operator[](int i);

    int num_actions();

    void next(const shared_ptr<ParseState> &state,
              const Action &a,
              const vector<shared_ptr<Node>> &buffer,
              shared_ptr<ParseState> &newstate);

    Grammar* grammar_ptr();

    void print_transitions(ostream &os) const;

    void export_model(const string &outdir);
    static TransitionSystem* import_model(const string &outdir);

    static void shift(vector<shared_ptr<Node>> &stack, std::deque<shared_ptr<Node>> &deque, vector<shared_ptr<Node>> &buffer, int &j);

    static void test_derivation_extraction(const string &tbk_filename, const string &hr_filename, int transition_system);

    static bool has_head(shared_ptr<Node> &parent, shared_ptr<Node> child);

    static void print_stats(const string &train_file, const string &outfile);
};


// Shift, Gap, Idle, RU, RL, RR
class GapTS : public TransitionSystem{
    enum {SHIFT_I, GAP_I, IDLE_I};
public:
    GapTS(const Grammar &g);

    Action* get_idle();

    void compute_derivation(Tree &tree, Derivation &derivation);

    bool allowed(ParseState &state, int buffer_size, int i);

    virtual bool allowed_reduce(ParseState &state, const Action &a, int buffer_size);

    bool allowed_reduce_u(ParseState &state, const Action &a, int buffer_size);

    virtual bool allowed_gap(ParseState &state);

};


// Shift, Gap_i, Idle, RU, Ghost-Reduce, RL, RR
class CompoundGapTS : public TransitionSystem{
    enum {SHIFT_I, GHOST_REDUCE_I, IDLE_I};
public:
    CompoundGapTS(const Grammar &g);

    Action* get_idle();

    void compute_derivation(Tree &tree, Derivation &derivation);

    bool allowed(ParseState &state, int buffer_size, int i);

    bool allowed_reduce(ParseState &state, const Action &a, int buffer_size);

    bool allowed_reduce_u(ParseState &state, const Action &a, int buffer_size);

    bool allowed_cgap(ParseState &state, const Action &action, int buffer_size);

};

// Shift, Merge, Gap, Label, Idle
// Unlexicalised transition system
class MergeLabelTS : public TransitionSystem{
    enum {SHIFT_I, MERGE_I, IDLE_I, GAP_I};
public:

    MergeLabelTS(const Grammar &g);

    Action* get_idle();

    void compute_derivation(Tree &tree, Derivation &derivation);

    bool allowed(ParseState &state, int buffer_size, int i);

    bool allowed_reduce_u(ParseState &state, const Action &a, int buffer_size);

};

// Shift, RR, RL, RU, Idle
class ShiftReduce : public GapTS{
public:
    ShiftReduce(const Grammar &g);
    bool allowed_reduce(ParseState &state, const Action &a, int buffer_size);
    bool allowed_gap(ParseState &state);
};





#endif // TRANSITION_SYSTEM_H
