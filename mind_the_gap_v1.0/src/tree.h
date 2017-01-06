#ifndef TREE_H
#define TREE_H

#include <vector>
#include <tuple>
#include <set>
#include <memory>
#include <algorithm>
#include <assert.h>
#include <ostream>
#include "utils.h"
#include "grammar.h"

using std::vector;
using std::pair;
using std::set;
using std::ostream;
using std::make_pair;
using std::shared_ptr;
using std::weak_ptr;
using std::endl;

class Grammar;
class HeadRules;
class Leaf;
class Tree;


struct Constituent{
    STRCODE label_;
    vector<pair<int,int>> spans_;

    Constituent(STRCODE label, const vector<pair<int,int>> &spans);

    bool operator==(const Constituent &c);

    bool operator!=(const Constituent &c);

    // REMINDER: in case of evaluation problem, double check this function
    bool operator<(const Constituent &c)const;
};

class Node{
protected:
    STRCODE label_;
    vector<shared_ptr<Node>> children_;
    vector<pair<int,int>> spans_;
    int h_;                         // index of head
    shared_ptr<Node> head_;         // pointer to head

    weak_ptr<Node> parent_;         // weak ptr (non owner) to parent node
                                    // use with great caution, normally only used for computing the gold derivation

    Node(STRCODE label, int j);

public:

    Node(STRCODE label, const vector<shared_ptr<Node>> &children);
    Node(const vector<shared_ptr<Node>> &children);
    virtual ~Node();
    void compute_spans();       // update spans_ from children's spans (concatenation / sort / merge)

    int arity() const;          // arity of corresponding grammar rule
    STRCODE label()const;
    int index() const;          // index of leftmost element in span
    int left_corner()const;     // index of leftmost element in span
    int right_corner()const;    // index of rightmost element in span (span=(0,4) -> return 3)

    void get(int i, shared_ptr<Node> &ptr);     // accesor for ieth child

    int rank()const;                // get rank of (sub)tree
    int gap_degree()const;          // get gap degree of (sub)tree

    void set_label(STRCODE label);
    bool has_label();
    void get_children(vector<shared_ptr<Node>> &newchildren);

    int h()const;
    void head(shared_ptr<Node> &ptr);
    void set_h(int i);

    virtual bool is_preterminal()const;

    void yield(vector<shared_ptr<Node>> &res);

    virtual STRCODE get_field(int i)const;

    void annotate_heads(const HeadRules &hr);
    void binarize(Grammar &grammar);
    void unbinarize(Grammar &grammar);
    void merge_unaries(Grammar &grammar);
    void split_unaries(Grammar &grammar);
    void get_frontier(Grammar &grammar, vector<shared_ptr<Node>> &frontier);

    void get_parent(shared_ptr<Node> &n);
    void set_parent(shared_ptr<Node> &n);
    static void annotate_parent_ptr(shared_ptr<Node> &n);

    void extract_constituents(set<Constituent> &cset);

    virtual void copy(shared_ptr<Node> &result)const;

    void write(ostream &os, vector<std::pair<string,string>> &str_sentences);
    friend ostream& operator<<(ostream &os, const Node &node);

};

class Leaf : public Node{
    vector<STRCODE> fields_;
public:
    enum {FIELD_TOK, FIELD_TAG};
    Leaf(STRCODE label, int j, const vector<STRCODE> &fields);
    ~Leaf();
    bool is_preterminal()const;

    STRCODE get_field(int i)const;

    void copy(shared_ptr<Node> &result)const;
};


class Tree{
    shared_ptr<Node> root_;
    vector<shared_ptr<Node>> leafs;
public:
    Tree();
    Tree(const shared_ptr<Node> &root);
//    Tree(const Tree &t);
//    Tree& operator=(const Tree &t);       // no -> default is shallow copy, use copy() for deep copy
                                            // tree vectors use shallow copy (shared_ptr
                                            // treebank cloning use copy()

    void copy(Tree &t);

    int length();
    int rank();
    int gap_degree();

    void annotate_heads(const HeadRules & hr);

    void binarize(Grammar &grammar);
    void unbinarize(Grammar &grammar);

    void get_buffer(vector<shared_ptr<Node>> &buffer);
    void get_root(shared_ptr<Node> &root);

    void annotate_parent_ptr();

    void extract_constituents(set<Constituent> &cset);

    void write(ostream &os, vector<std::pair<string,string>> &str_sentences);

    friend ostream& operator<<(ostream &os, const Tree &tree);
};




namespace eval{
    struct EvalTriple{
        double true_positive;
        double predicted;
        double gold;

        EvalTriple();
        double precision()const;
        double recall()const;
        double fscore()const;
    };

    void compare(Tree &gold, Tree &pred, EvalTriple &triple);
}



#endif // TREE_H
