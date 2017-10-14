#ifndef PARSER_H
#define PARSER_H

#include <vector>
#include <memory>

#include "tss_beam.h"
#include "tree.h"
#include "treebank.h"
#include "features.h"
#include "classifier.h"


/******************************************
 *
 * This is a modular parser, pick and choose:
 *   - a transition system defining possible actions,
 *      grammar symbols, constraints on actions
 *   - a feature extractor (function configuration -> int vector)
 *   - a classifier (function int vector -> double vector)
 *      - rnn classifiers are treated as a special case
 ******************************************/
class Parser{

    vector<bool> allowed;
    vector<bool> gold_actions;
    vector<int> features;

    const int transition_system_id;
    const int classifier_id;
    const int feature_extractor_id;

public:
    TransitionSystem *ts;
    FeatureExtractor *fe;
    Classifier *cf;

    Parser(int tsi, int cfi, int fei,
           Treebank &train, Treebank &dev, const Grammar &grammar, const string &tpl_filename, NeuralNetParameters &params);

    Parser(const string &outdir);


    ~Parser();

    void precompute_char_lstm();

    void summary(int beamsize, int epochs, const string &outdir, int train_size, int dev_size);

    void train_global(Treebank &train_tbk,
                      Treebank &dev_tbk,
                      Treebank &train_nary,
                      Treebank &dev_nary,
                      int epochs,
                      int beamsize,
                      const std::string &outdir);

    void train_global_one(Tree &tree, int beamsize);

    void train_neural_one(Tree &tree, int beamsize);

    void train_rnn_one(Tree &tree, int beamsize);

    void gradient_check_one(Tree &tree, int beamsize, double epsilon);

    static bool check(const vector<bool> &v);

    void update_global(const Derivation &gold, const Derivation &best, const Tree &tree, vector<shared_ptr<Node>> &buffer);

    void evaluate_global(Treebank &tbk, Treebank &nary, int beamsize, eval::EvalTriple &fscore);
    //void eval_aux_one(vector<shared_ptr<Node>> &buffer, AuxiliaryTaskEvaluator &evaluator);

    void evaluate_auxiliary_tasks(Treebank &tbk, AuxiliaryTaskEvaluator &evaluator);
    void train_aux_task(Treebank &train_tbk, Treebank &sample_train, Treebank &dev_tbk, int epochs);
    void train_aux_task_one(Rnn *rnn, Tree &tree);

    void predict_tree(vector<shared_ptr<Node>> &buffer, int beamsize, Tree &pred);

    void predict_treebank(vector<vector<shared_ptr<Node>>> &raw_test, int beamsize, Treebank &tbk);

    void export_model(const string &outdir);

    int get_classifier_id();
    int get_transition_system_id();

    static Classifier* classifier_factory(int classifier_id, int num_classes, FeatureExtractor *fe, NeuralNetParameters &params);

    static TransitionSystem* transition_system_factory(int ts_id, const Grammar &grammar);

    static FeatureExtractor* feature_extractor_factory(int feat_extractor_id, const string &filename);

};




#endif // PARSER_H
