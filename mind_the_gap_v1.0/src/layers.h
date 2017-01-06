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



/////////////////////////
/// Parameters objects store pointers to parameters
struct Parameter{
    bool avg;       // ASGD
    Parameter();
    virtual ~Parameter();
    virtual void update(double lr, double T, double clip, bool clipping, bool gaussian)=0;
    virtual void average(double T)=0;
    virtual void add_epsilon(int i, double epsilon)=0;
    virtual int size()=0;
    virtual void set_empirical_gradient(int i, double eg)=0;
    virtual void print_gradient_differences()=0;
    virtual void assign(shared_ptr<Parameter> &other)=0;
    virtual void print(ostream &os)=0;
};

struct MatParam : public Parameter{
    Mat *w, *dw, *cw;
    MatParam(Mat *w1, Mat *w2, Mat *w3);
    ~MatParam();
    void update(double lr, double T, double clip, bool clipping, bool gaussian);
    void average(double T);
    int size();
    void add_epsilon(int i, double epsilon);
    void set_empirical_gradient(int i, double eg);
    void print_gradient_differences();
    void assign(shared_ptr<Parameter> &other);
    void print(ostream &os);
};

struct VecParam : public Parameter{
    Vec *b, *db, *cb;
    VecParam();
    VecParam(Vec *b1, Vec *b2, Vec *b3);
//    VecParam(const VecParam & other);
//    VecParam& operator=(const VecParam & other);
    ~VecParam();
    void update(double lr, double T, double clip, bool clipping, bool gaussian);
    void average(double T);
    int size();
    void add_epsilon(int i, double epsilon);
    void set_empirical_gradient(int i, double eg);
    void print_gradient_differences();
    void assign(shared_ptr<Parameter> &other);
    void print(ostream &os);
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

    void update(double lr, double T, double clip, bool clipping, bool gaussian);

    void get_active_params(vector<shared_ptr<Parameter>> &params);

    void average(int T);

    void export_model(const string &filename);
};


void get_gru_layer(vector<shared_ptr<Layer>> &layers, int insize, int hiddensize);


#endif // LAYERS_H
