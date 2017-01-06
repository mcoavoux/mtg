#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <limits>
#include <unordered_map>
#include <fstream>
#include <math.h>

#include "features.h"
#include "layers.h"
#include "str_utils.h"

using std::ifstream;
using std::ofstream;
using std::cerr;
using std::endl;
using std::unordered_map;


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

    enum {PERCEPTRON, FAST_PER, FASTER_PER, FASTEST_PER, FFNN};

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

    double operator[](int i);

    int get_n_updates();

    void reset_updates();

    int argmax();

    void increment_T();

    void increment_updates();

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



struct NetTopology{
    int n_hidden_layers;
    int size_hidden_layers;
    vector<int> embedding_size_type;
    NetTopology();
};

struct NeuralNetParameters{
    NetTopology topology;
    double learning_rate;
    double decrease_constant;
    double clip_value;
    bool gaussian_noise;
    bool gradient_clipping;
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


#endif // CLASSIFIER_H
