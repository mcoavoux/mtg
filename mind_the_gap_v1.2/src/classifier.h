#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <limits>
#include <unordered_map>
#include <fstream>
#include <math.h>
#include <memory>
#include <iomanip>

#include "features.h"
#include "layers.h"
#include "str_utils.h"

using std::ifstream;
using std::ofstream;
using std::cerr;
using std::endl;
using std::unordered_map;
using std::unique_ptr;
using std::shared_ptr;


const double MINUS_INFINITY = - std::numeric_limits<double>::infinity();



////////////////////////////////////////
/// Abstract classifier class
///////////////////////////////////////
class Classifier{

protected:
    vector<double> scores_;
    int n_updates_;
    int T_;
    const int N_;

public:

    enum {PERCEPTRON, FAST_PER, FASTER_PER, FASTEST_PER, FFNN, RNN, RNN_LABEL_STRUCTURE, RNN_LABEL_STRUCTURE_LEX};

    Classifier(int n_classes);


    virtual ~Classifier();

    virtual void score(const vector<int> &features,
                       const vector<bool> &allowed)=0;

    virtual void average_weights()=0;
    virtual void global_update_one(const vector<int> &feats,
                                   int action_num,
                                   int increment)=0;

    virtual int get_id()=0;
    virtual Classifier* copy()=0;

    virtual void print_stats(ostream &os);

    virtual void precompute_char_lstm(){} // do nothing if I am not an rnn with char rnn

    double operator[](int i);

    int get_n_updates();

    void reset_updates();

    int argmax();

    void increment_T();

    void increment_updates();

    void export_classifier_id(const string &outdir);

    virtual void export_model(const string &outdir)=0;

    static Classifier* import_model(const string &outdir);
};



/////////////////////////////////////:
///
/// Perceptron (deprecated)
///
class Perceptron : public Classifier{

    vector<unordered_map<int, double>> weights;
    vector<unordered_map<int, double>> cached;

public:
    Perceptron(int n_classes);

    int get_id();

    Classifier* copy();

    void average_weights();

    void score(const vector<int> &features, const vector<bool> &allowed);

    void global_update_one(const vector<int> &feats,
                           int action_num,
                           int increment);

    void export_model(const string &outdir);
};

// SparseVector class, random access is slow, summing lots of SparseVector (dot product) is very fast
struct SparseVector{
    vector<int>loc;         // locations of non-zero coefs
    vector<double> val;     // values of non zero coefs

    double& operator[](int i);

    void accumulate(vector<double> &s);
    SparseVector& operator-=(const SparseVector &v);
    SparseVector& operator+=(const SparseVector &v);
    SparseVector& operator*=(const double d);
    SparseVector& operator/=(const double d);

    int size();

    friend ostream & operator<<(ostream &os, const SparseVector &v);
    static void test();
};


////////////////////////////////////
///
/// Another Perceptron (deprecated)
///
class FastPerceptron : public Classifier{
    //                           weights        cached weights (averaging)
    unordered_map<int, std::pair<SparseVector, SparseVector>> weights;

public:
    FastPerceptron(int n_classes);

    int get_id();

    Classifier* copy();

    void average_weights();

    void score(const vector<int> &features, const vector<bool> &allowed);

    void global_update_one(const vector<int> &feats,
                           int action_num,
                           int increment);

    virtual void print_stats(ostream &os);

    void export_model(const string &outdir);
};

// Uses a table instead of unordered map
class FasterPerceptron : public Classifier{
    //                           weights        cached weights (averaging)
    //unordered_map<int, std::pair<SparseVector, SparseVector>> weights;
    vector<std::pair<SparseVector, SparseVector>> weights;

public:
    FasterPerceptron(int n_classes);

    int get_id();

    Classifier* copy();

    void average_weights();

    void score(const vector<int> &features, const vector<bool> &allowed);

    void global_update_one(const vector<int> &feats,
                           int action_num,
                           int increment);

    virtual void print_stats(ostream &os);

    void export_model(const string &outdir);
};

// Kernel trick + no resizing for feature table
class FastestPerceptron : public Classifier{
    vector<std::pair<SparseVector, SparseVector>> weights;

public:
    FastestPerceptron(int n_classes);

    int get_id();

    FastestPerceptron(int n_classes, const string &outdir);

    Classifier* copy();

    void average_weights();

    void score(const vector<int> &features, const vector<bool> &allowed);

    void global_update_one(const vector<int> &feats,
                           int action_num,
                           int increment);

    virtual void print_stats(ostream &os);

    void export_model(const string &outdir);
};










////////////////////////////////////////////////////////////////////
///
///     Below are neural nets classifiers
///
///

struct AuxiliaryTaskEvaluator{
    vector<int> good;
    float total;
    float complete_match;

    friend ostream& operator<<(ostream &os, AuxiliaryTaskEvaluator &ev){
        os << "{";
        for (int i = 0; i < ev.good.size(); i++){
            os << " " << i << "="<< std::setprecision(4) << 100.0*(ev.good[i]/ev.total);
        }
        os << " cm=" << std::setprecision(4) << 100.0*(ev.complete_match/ev.total) << " }";
        return os;
    }
};


struct NetTopology{
    int n_hidden_layers;
    int size_hidden_layers;
    vector<int> embedding_size_type;
    NetTopology();
};

struct CharRnnParameters{
    int dim_char;
    int dim_char_based_embeddings;
    int crnn;
    CharRnnParameters();
};

struct RnnParameters{
    int cell_type;
    int depth; // 1 forward rnn, 2, bi-rnn, etc
    int hidden_size;
    int features; // number of features to consider for bi-rnn: if 2 -> (word,tag) if 3 -> (word,tag,morph1) etc..

    CharRnnParameters crnn;
    //int char_rnn_feature_extractor;  // make this an int ?

    bool auxiliary_task;
    int auxiliary_task_max_target;  // predict from features + 1 to auxiliary_task_max

    RnnParameters();
};

struct NeuralNetParameters{
    NetTopology topology;
    RnnParameters rnn;
    double learning_rate;
    double decrease_constant;
    double clip_value;
    double gaussian_noise_eta;

    bool gaussian_noise;
    bool gradient_clipping;
    bool soft_clipping;
    bool rnn_feature_extractor;

    vector<string> header;
    vector<int> voc_sizes;
    NeuralNetParameters();

    void print(ostream &os);

    static void read_option_file(const string &filename, NeuralNetParameters &p);
};


class NeuralNet : public Classifier{
    int n_features;

    NeuralNetParameters params_;
    vector<int> feature_types;


    vector<LookupTable> lu;

    vector<Layer*> layers;
    vector<shared_ptr<Parameter>> parameters;

    vector<Vec*> edata;
    vector<Vec*> edata_grad;

    vector<Vec> states;
    vector<Vec> dstates;

    Vec softmax_filter;

public:
    NeuralNet(int n_classes, const string &outdir);

    NeuralNet(int n_classes);

    NeuralNet(int n_classes, FeatureExtractor *fe, NeuralNetParameters &params);

    void init_feature_types_and_lu(FeatureExtractor *fe);

    int get_id();

    ~NeuralNet();

    void initialize_network();

    void average_weights();

    double fprop(const vector<int> &features, const vector<bool> &allowed, int target);

    void bprop(const vector<int> &features, const vector<bool> &allowed, int target);

    void update();

    void gradient_check(const vector<int> &features, const vector<bool> &allowed, int target, double epsilon);

    double get_learning_rate();

    Classifier* copy();

    void score(const vector<int> &features, const vector<bool> &allowed);

    void global_update_one(const vector<int> &feats,
                           int action_num,
                           int increment);

    void print_stats(ostream &os);

    void export_model(const string &outdir);

    void print_parameters(ostream &os);

    static void check_gradient();
};


typedef vector<vector<shared_ptr<AbstractNeuralNode>>> NodeMatrix;


class CharBiRnnFeatureExtractor{
    vector<shared_ptr<RecurrentLayerWrapper>> layers;// 0: forward, 1: backward, 2: forward, 3:backward, etc...

    // Computation nodes
    vector<shared_ptr<AbstractNeuralNode>> init_nodes;
    vector<NodeMatrix> states; // states[word][depth][char]

    // input (lookup) nodes
    vector<NodeMatrix> input; // input[word][depth][char]

    // hyperparameters and lookup tables
    LookupTable lu;
    CharRnnParameters *params;

    // parameters
    vector<shared_ptr<Parameter>> parameters;
    SequenceEncoder encoder;

    vector<vector<Vec>> precomputed_embeddings;

public:
    CharBiRnnFeatureExtractor();
    CharBiRnnFeatureExtractor(CharRnnParameters *nn_parameters);
    ~CharBiRnnFeatureExtractor();

    void precompute_lstm_char();
    bool has_precomputed();
    void init_encoders();
    void build_computation_graph(vector<shared_ptr<Node>> &buffer);
    void add_init_node(int depth);
    void fprop();
    void bprop();
    void update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta);
    double gradient_squared_norm();
    void scale_gradient(double scale);
    void operator()(int i, vector<shared_ptr<AbstractNeuralNode>> &output);
    int size();
    void copy_encoders(CharBiRnnFeatureExtractor &other);
    void assign_parameters(CharBiRnnFeatureExtractor &other);
    void average_weights(int T);
    void get_parameters(vector<shared_ptr<Parameter>> &weights);
    void export_model(const string &outdir);
    void load_parameters(const string &outdir);
    void reset_gradient_history();
};




class BiRnnFeatureExtractor{

    vector<shared_ptr<RecurrentLayerWrapper>> layers;// 0: forward, 1: backward, 2: forward, 3:backward, etc...

    // Computation nodes
    vector<shared_ptr<AbstractNeuralNode>> init_nodes;
    //vector<vector<shared_ptr<AbstractNeuralNode>>> states; // idem
    NodeMatrix states;

    // input (lookup) nodes
    //vector<vector<shared_ptr<AbstractNeuralNode>>> input;
    NodeMatrix input;

    // hyperparameters and lookup tables
    vector<LookupTable> *lu;
    NeuralNetParameters *params;

    // parameters
    vector<shared_ptr<Parameter>> parameters;

    Vec out_of_bounds;
    Vec out_of_bounds_d;

    CharBiRnnFeatureExtractor char_rnn;

    // Auxiliary task
    //static const int AUX_HIDDEN_LAYER_SIZE = 32;
    vector<vector<shared_ptr<Layer>>> auxiliary_layers;
    vector<shared_ptr<Parameter>> auxiliary_parameters;
    vector<int> aux_output_sizes;
    vector<vector<int>> aux_targets;
    vector<NodeMatrix> auxiliary_output_nodes;
    int aux_start;
    int aux_end;

    bool train_time;

    bool parse_time;

public:
    BiRnnFeatureExtractor();
    BiRnnFeatureExtractor(NeuralNetParameters *nn_parameters, vector<LookupTable> *lookup);

    ~BiRnnFeatureExtractor();

    void precompute_char_lstm();

    void build_computation_graph(vector<shared_ptr<Node>> &buffer, bool aux_task=false);

    void add_init_node(int depth);

    AbstractNeuralNode* get_recurrent_node(shared_ptr<AbstractNeuralNode> &pred,
                                           vector<shared_ptr<AbstractNeuralNode>> &input_nodes,
                                           RecurrentLayerWrapper &l);

    void fprop();

    void bprop();

    void update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta);
    double gradient_squared_norm();
    void scale_gradient(double scale);

    void operator()(int i, vector<Vec*> &data, vector<Vec*> &data_grad);

    int size();

    void assign_parameters(BiRnnFeatureExtractor &other);
    void copy_char_birnn(BiRnnFeatureExtractor &other);

    void average_weights(int T);

    void get_parameters(vector<shared_ptr<Parameter>> &weights);

    void export_model(const string &outdir);

    void load_parameters(const string &outdir);

    void auxiliary_task_summary(ostream &os);
    void add_aux_graph(vector<shared_ptr<Node>> &buffer, bool aux_only);
    void fprop_aux();
    void bprop_aux();
    void update_aux(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta);
    void eval_aux(AuxiliaryTaskEvaluator &evaluator);

    void assign_deplabels(vector<shared_ptr<Node>> &buffer, int deplabel_id);
    void assign_tags(vector<shared_ptr<Node>> &buffer);
    void assign_morphological_features(vector<shared_ptr<Node>> &buffer, int deplabel_id);

    void auxiliary_gradient_check(vector<shared_ptr<Node>> &buffer, double epsilon);
    double aux_loss();
    double full_fprop_aux(vector<shared_ptr<Node>> &buffer);
    //int n_aux_tasks();
    void aux_reset_gradient_history();

    void set_train_time(bool b);
};







class Rnn : public Classifier{

protected:
    int n_features;
    NeuralNetParameters params_;
    vector<int> feature_types;          // Réécrire DENseFeatureExtractor: tout ce qui renvoie des tokens -> index dans le buffer

    vector<LookupTable> lu;

    vector<Layer*> layers;                      // layers de sortie
    vector<shared_ptr<Parameter>> parameters;   // parametres de sortie

    vector<vector<Vec*>> t_edata;
    vector<vector<Vec*>> t_edata_grad;

    vector<vector<Vec>> t_states;
    vector<vector<Vec>> t_dstates;

    vector<Vec> softmax_filters;

    BiRnnFeatureExtractor rnn;

public:
    Rnn(int n_classes, const string &outdir, bool initialize=true);

    Rnn(int n_classes);

    Rnn(int n_classes, FeatureExtractor *fe, NeuralNetParameters &params, bool initialize=true);
    virtual ~Rnn();

    void precompute_char_lstm();

    void init_feature_types_and_lu(FeatureExtractor *fe);

    virtual int get_id();

    virtual void initialize_network();

    void run_rnn(vector<shared_ptr<Node>> &buffer);

    virtual void flush();

    void get_feature(int f, int type, vector<Vec*> &input, vector<Vec*> &dinput);

    virtual double fprop(const vector<int> &features, const vector<bool> &allowed, int target);

    virtual void bprop(const vector<int> &targets);

    virtual void score(const vector<int> &features, const vector<bool> &allowed);

    double get_learning_rate();

    void update();
    void soft_gradient_clipping();

    void train_auxiliary_task(vector<shared_ptr<Node>> &buffer);
    void predict_auxiliary_task(vector<shared_ptr<Node>> &buffer, bool aux_only);
    void compare_auxiliary_task(AuxiliaryTaskEvaluator &evaluator);
    void assign_deplabels(vector<shared_ptr<Node>> &buffer, int deplabel_id);
    void assign_tags(vector<shared_ptr<Node>> &buffer);
    void assign_morphological_features(vector<shared_ptr<Node>> &buffer, int deplabel_id);

    void average_weights();

    Classifier* copy();

    double full_fprop(vector<shared_ptr<Node>> &buffer,
                    const vector<vector<int>> &features,
                    const vector<vector<bool>> &allowed,
                    const vector<int> &targets);

    void gradient_check(vector<shared_ptr<Node>> &buffer,
                        const vector<vector<int>> &features,
                        const vector<vector<bool>> &allowed,
                        const vector<int> &targets,
                        double epsilon);


    void print_parameters(ostream &os);

    void export_model(const string &outdir);

    void print_stats(ostream &os);

    void global_update_one(const vector<int> &feats, int action_num, int increment);

    int n_aux_tasks();
    bool auxiliary_task();
    bool auxiliary_tags();
    void auxiliary_gradient_check(vector<shared_ptr<Node>> &buffer, double epsilon);
    void aux_reset_gradient_history();

    void set_train_time(bool b);

};

/**
 * Same as Rnn, except that it uses 2 distinct classifiers for label and structure actions.
 */
class RnnStructureLabel : public Rnn{
    int I = MergeLabelTS::GAP_I + 1;   // until I excluded-> structure actions, after that -> label actions

    vector<Layer*> layers_struct;                        // layers de sortie
    //vector<shared_ptr<Parameter>> parameters_struct;   // parametres de sortie

    vector<bool> struct_or_label;

    bool is_struct_action(const vector<bool> &allowed);
    //void sanity_check(const vector<bool> &allowed);

    bool lex;

public:
    RnnStructureLabel(int n_classes, const string &outdir, bool lex);

    RnnStructureLabel(int n_classes);

    RnnStructureLabel(int n_classes, FeatureExtractor *fe, NeuralNetParameters &params, bool lex);
    ~RnnStructureLabel();

    int get_id();
    void initialize_network();
    void flush();
    double fprop(const vector<int> &features, const vector<bool> &allowed, int target);
    void bprop(const vector<int> &targets);
    void score(const vector<int> &features, const vector<bool> &allowed);
    Classifier* copy();
};


#endif // CLASSIFIER_H
