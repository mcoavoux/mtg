#ifndef LAYERS_H
#define LAYERS_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <cmath>
#include <random>
#include <memory>

#include "str_utils.h"
#include "random_utils.h"
#include "utils.h"

#define DBG(x) cerr << x << endl;
////
///
/// Building blocks for neural nets, using Eigen
///
///

using std::vector;
using std::string;
using std::unordered_map;
using std::unordered_set;
using std::ifstream;
using std::ofstream;
using std::ostream;
using std::cerr;
using std::cout;
using std::endl;
using std::shared_ptr;

typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;

Mat xavier(int insize, int outsize);

double rectifier(double x);



template <class M>
void load_matrix(std::string filename, M &res){

    std::ifstream input(filename);
    std::string line;
    double d;
    std::vector<double> v;
    int rows = 0;
    while(getline(input, line)){
        rows++;
        std::stringstream input_line(line);
        while (!input_line.eof()){
            input_line >> d;
            v.push_back(d);
        }
    }
    input.close();
    int cols = v.size() / rows;
    res = M::Zero(rows, cols);

    for (int i=0; i< rows; i++){
        for (int j=0; j< cols; j++){
            res(i,j) = v[i*cols + j];
        }
    }
}



/////////////////////////
/// Parameters objects store pointers to parameters
struct Parameter{
    bool avg;       // ASGD
    Parameter();
    virtual ~Parameter();
    virtual void update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta)=0;
    virtual void average(double T)=0;
    virtual void add_epsilon(int i, double epsilon)=0;
    virtual int size()=0;
    virtual void set_empirical_gradient(int i, double eg)=0;
    virtual void print_gradient_differences()=0;
    virtual void assign(shared_ptr<Parameter> &other)=0;
    virtual void print(ostream &os)=0;
    virtual void load(const string &outfile)=0;
    virtual void reset_gradient_history()=0;
    virtual void scale_gradient(double p) = 0;
    virtual double gradient_squared_norm()=0;
    void export_model(const string &outfile);
};

struct MatParam : public Parameter{
    Mat *w, *dw, *cw;
    MatParam(Mat *w1, Mat *w2, Mat *w3);
    ~MatParam();
    void update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta);
    void average(double T);
    int size();
    void add_epsilon(int i, double epsilon);
    void set_empirical_gradient(int i, double eg);
    void print_gradient_differences();
    void assign(shared_ptr<Parameter> &other);
    void print(ostream &os);
    void load(const string &outfile);
    void reset_gradient_history();
    void scale_gradient(double p);
    double gradient_squared_norm();
};

struct VecParam : public Parameter{
    Vec *b, *db, *cb;
    VecParam();
    VecParam(Vec *b1, Vec *b2, Vec *b3);
    ~VecParam();
    void update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta);
    void average(double T);
    int size();
    void add_epsilon(int i, double epsilon);
    void set_empirical_gradient(int i, double eg);
    void print_gradient_differences();
    void assign(shared_ptr<Parameter> &other);
    void print(ostream &os);
    void load(const string &outfile);
    void reset_gradient_history();
    void scale_gradient(double p);
    double gradient_squared_norm();
};

struct Layer{
    int target;
    virtual ~Layer();
    virtual void fprop(const vector<Vec*> &data, Vec& output)=0;
    virtual void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient)=0;
    virtual void get_params(vector<shared_ptr<Parameter>> &t);
};

struct AffineLayer : public Layer{
    Mat w, dw, cw;
    Vec b, db, cb;
    AffineLayer(int insize, int outsize);
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
    void get_params(vector<shared_ptr<Parameter>> &t);
};

struct LinearLayer : public Layer{
    Mat w, dw, cw;
    LinearLayer(int insize, int outsize);
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
    void get_params(vector<shared_ptr<Parameter>> &t);
};

struct MultipleLinearLayer : public Layer{
    vector<LinearLayer*> layers;
    Vec b, db, cb;
    Vec buffer;
    MultipleLinearLayer(int insize, vector<int> &insizes, int outsize);
    ~MultipleLinearLayer();
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
    void get_params(vector<shared_ptr<Parameter>> &t);
};


struct RecurrentLayer : public Layer{
    Mat w, dw, cw;
    Mat rw, drw, crw;
    Vec b, db, cb;
    RecurrentLayer(int insize, int outsize);
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
    void get_params(vector<shared_ptr<Parameter>> &t);
};

struct AddBias : public Layer{
    Vec b, db, cb;
    AddBias(int outsize);
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
    void get_params(vector<shared_ptr<Parameter>> &t);
};

struct ConstantLayer : public Layer{
    Vec b, db, cb;
    ConstantLayer(int outsize);
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
    void get_params(vector<shared_ptr<Parameter>> &t);
};


// Standard operations
struct Mult : public Layer{
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
};

struct Add : public Layer{
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
};

struct Mixture : public Layer{ // y = (1 - x1) * x2 + x1 * x3
    virtual void fprop(const vector<Vec*> &data, Vec& output);
    virtual void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
};

// Activation
struct Tanh : public Layer{
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
};

struct Sigmoid : public Layer{
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
};

struct ReLU : public Layer{
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
};

struct Softmax : public Layer{
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
};

struct SoftmaxFilter : public Layer{
    void fprop(const vector<Vec*> &data, Vec& output);
    void bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient);
};



struct LookupTable{
    vector<Vec> v;
    vector<Vec> dv;
    vector<Vec> cv;

    unordered_map<int, shared_ptr<VecParam>> active;

    int vocsize;
    int dimension;

    LookupTable();
    ~LookupTable();

    LookupTable(const LookupTable &other);

    LookupTable& operator=(const LookupTable &other);

    LookupTable(int vocsize, int dimension);

    void get(int i, shared_ptr<VecParam> &param);

    void update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta);

    double gradient_squared_norm();

    void scale_gradient(double scale);

    void get_active_params(vector<shared_ptr<Parameter>> &params);

    void average(int T);

    void export_model(const string &filename);

    void load(const string &filename);
    void clear();

    bool consistent_dimension();

    void reset_gradient_history();
};



//////////////////////////////////////////////////////
///
///
///
//////////////////////////////////////////////////////


/**
 * @brief The AbstractNeuralNode struct
 *   is an abstract node in a computation graph.
 * Nodes are connected to their input nodes
 *  and usually stores a state and its derivative.
 */
struct AbstractNeuralNode{
    virtual ~AbstractNeuralNode();
    virtual void fprop() = 0;   // forward propagation
    virtual void bprop() = 0;   // backward propagation
    virtual Vec* v()=0;         // get pointer to state
    virtual Vec* d()=0;         // get pointer to state derivative
};


struct ConstantNode:public AbstractNeuralNode{
    Vec *state;
    ConstantNode(Vec* vec):state(vec){}
    void fprop(){}
    void bprop(){}
    Vec* v(){ return state;}
    Vec* d(){ throw "Error"; }
};

/**
 * @brief The LookupNode struct is an input node
 *   (no predecessor nodes) serving as a place-holder
 *   for next layers.
 */
struct LookupNode : public AbstractNeuralNode{
    VecParam embedding;
    LookupNode();
    LookupNode(VecParam &e);
    void fprop();
    void bprop();
    Vec* v();
    Vec* d();
};

/**
 * @brief The NeuralNode struct is the base class
 * for subsequent nodes.
 */
struct NeuralNode : public AbstractNeuralNode{
    Vec state;
    Vec dstate;

    virtual ~NeuralNode();
    NeuralNode(int size);

    Vec* v();
    Vec* d();
};

/**
 * @brief The SimpleNode struct is a node with
 *   a single input node.
 */
struct SimpleNode : public NeuralNode{
    Layer *layer;
    shared_ptr<AbstractNeuralNode> input;

    SimpleNode(int size, Layer *layer, const shared_ptr<AbstractNeuralNode> &input);

    void fprop();
    void bprop();
};

/**
 * @brief The ComplexNode struct is a node with
 * several input nodes
 */
struct ComplexNode : public NeuralNode{
    Layer *layer;
    vector<shared_ptr<AbstractNeuralNode>> input;

    ComplexNode(int size, Layer *layer, vector<shared_ptr<AbstractNeuralNode>> &input);

    void fprop();
    void bprop();
};

/**
 * @brief The RecurrentLayerWrapper struct is a wrapper
 * for the vector of functions required for typical
 * rnn network internal computations.
 */
struct RecurrentLayerWrapper{
    enum {RNN, GRU, LSTM};
    vector<Layer*> layers;

    RecurrentLayerWrapper(int cell_type, vector<int> &input_sizes, int hidden_size);
    ~RecurrentLayerWrapper();

    void get_gru(vector<int> &input_sizes, int hidden_size);
    void get_vanilla_rnn(vector<int> &input_sizes, int hidden_size);
    void get_lstm(vector<int> &input_sizes, int hidden_size);

    Layer* operator[](int i);
    int size();
};


/**
 * @brief The AbstractMemoryNode struct is used as initial node
 * for recurrent cell which must access several memory cells
 * from previous step.
 */
struct AbstractMemoryNode{
    virtual void get_memory_node(shared_ptr<AbstractNeuralNode> &hnode)=0;
};

struct ParamNode : public NeuralNode{
    Layer* layer;
    vector<Vec*> place_holder;

    ParamNode(int size, Layer *layer);

    void fprop();
    void bprop();
};


struct MemoryNodeInitial : public NeuralNode, public AbstractMemoryNode{
    shared_ptr<ParamNode> h;
    Layer* layer;
    vector<Vec*> place_holder;
    MemoryNodeInitial(int size, Layer* layer, shared_ptr<ParamNode> &paramnode);
    void get_memory_node(shared_ptr<AbstractNeuralNode> &hnode);
    void fprop();   // forward propagation
    void bprop();   // backward propagation
};


/**
 * @brief The GruNode struct need to be used as an abstraction.
 * It encapsulates several nodes and performs
 * all the internal computations of a Gated Recurrent Unit.
 */
struct GruNode : public ComplexNode, public AbstractMemoryNode{

    enum {INIT1, INIT2, Z1, Z2, R1, R2, H1, H2, H3, S};

    shared_ptr<GruNode> pred;


    // internal registers
    shared_ptr<ComplexNode> pz;
    shared_ptr<SimpleNode> z;
    shared_ptr<ComplexNode> pr;
    shared_ptr<SimpleNode> r;
    shared_ptr<ComplexNode> hr;
    shared_ptr<ComplexNode> ph;
    shared_ptr<SimpleNode> h;

    vector<shared_ptr<AbstractNeuralNode>> internal_nodes;

    GruNode(int size,
            shared_ptr<AbstractNeuralNode> &predecessor,
            vector<shared_ptr<AbstractNeuralNode>> &input,
            RecurrentLayerWrapper &layers);

    ~GruNode();

    void fprop();

    void bprop();

    void get_memory_node(shared_ptr<AbstractNeuralNode> &hnode);
};

struct RnnNode : public ComplexNode{
    enum {INIT, REC, ACTIVATION};

    shared_ptr<AbstractNeuralNode> pred;

    shared_ptr<ComplexNode> h;

    RnnNode(int size,
            shared_ptr<AbstractNeuralNode> &predecessor,
            vector<shared_ptr<AbstractNeuralNode>> &input,
            RecurrentLayerWrapper &layers);

    ~RnnNode();

    void fprop();
    void bprop();

};

struct LstmNode : public ComplexNode, public AbstractMemoryNode{ // See Goldberg's manuel
    enum {INIT_C, INIT_H, I, IS, F, FS, O, OS, G, GT, CF, GI, C, CT, H};

    shared_ptr<LstmNode> pred;

    shared_ptr<ComplexNode> ia;
    shared_ptr<SimpleNode> ih;

    shared_ptr<ComplexNode> fa;
    shared_ptr<SimpleNode> fh;

    shared_ptr<ComplexNode> oa;
    shared_ptr<SimpleNode> oh;

    shared_ptr<ComplexNode> ga;
    shared_ptr<SimpleNode> gh;

    shared_ptr<ComplexNode> cf_mult;
    shared_ptr<ComplexNode> gi_mult;
    shared_ptr<ComplexNode> c;

    shared_ptr<SimpleNode> ch;

    vector<shared_ptr<AbstractNeuralNode>> internal_nodes;

    LstmNode(int size,
             shared_ptr<AbstractNeuralNode> &predecessor,
             vector<shared_ptr<AbstractNeuralNode>> &input,
             RecurrentLayerWrapper &layers);

    ~LstmNode();

    void fprop();

    void bprop();

    void get_memory_node(shared_ptr<AbstractNeuralNode> &hnode);

};



#endif // LAYERS_H
