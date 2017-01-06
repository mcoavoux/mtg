#ifndef PARSER_H
#define PARSER_H

#include <vector>
#include <memory>

#include "tss_beam.h"
#include "tree.h"
#include "treebank.h"
#include "features.h"
#include "classifier.h"


class Parser{

    vector<bool> allowed;
    vector<bool> gold_actions;
    vector<int> features;

    const int transition_system_id;
    const int classifier_id;
    const int feature_extractor_id;

public:
    TransitionSystem *ts;
    Classifier *cf;
    FeatureExtractor *fe;
    NeuralNet *nn;


    Parser(int tsi, int cfi, int fei,
           Treebank &train, Treebank &dev, const Grammar &grammar, const string &tpl_filename, NeuralNetParameters &params);

    Parser(const string &outdir);


    ~Parser();

    void summary(int beamsize, int epochs, const string &outdir, int train_size, int dev_size, int test_size);

    void train_global(Treebank &train_tbk,
                      Treebank &dev_tbk,
                      Treebank &train_nary,
                      Treebank &dev_nary,
                      int epochs,
                      int beamsize,
                      const std::string &outdir,
                      vector<vector<shared_ptr<Node>>> &raw_test,
                      vector<vector<std::pair<string, string> > > &str_raw_test);

    void train_global_one(Tree &tree, int beamsize);

    void train_neural_one(Tree &tree, int beamsize);

    static bool check(const vector<bool> &v);

    void update_global(const Derivation &gold, const Derivation &best, const Tree &tree, vector<shared_ptr<Node>> &buffer);

    void evaluate_global(Treebank &tbk, Treebank &nary, int beamsize, eval::EvalTriple &fscore);

    void predict_tree(vector<shared_ptr<Node>> &buffer, int beamsize, Tree &pred);

    void predict_treebank(vector<vector<shared_ptr<Node>>> &raw_test, int beamsize, Treebank &tbk);

    void export_model(const string &outdir);

    static Classifier* classifier_factory(int classifier_id, int num_classes, FeatureExtractor *fe, NeuralNetParameters &params);

    static TransitionSystem* transition_system_factory(int ts_id, const Grammar &grammar);

    static FeatureExtractor* feature_extractor_factory(int feat_extractor_id, const string &filename);

};




#endif // PARSER_H
