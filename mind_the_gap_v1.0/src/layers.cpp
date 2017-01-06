#include "layers.h"

Mat xavier(int insize, int outsize){
    return Mat::Random(outsize, insize) * sqrt(6.0 / (outsize + insize));
}
double rectifier(double x){
    if (x < 0) return 0.0;
    return x;
}

Parameter::Parameter(){
    avg = true;
}

Parameter::~Parameter(){}


MatParam::MatParam(Mat *w1, Mat *w2, Mat *w3){
    w = w1;
    dw = w2;
    cw = w3;
}

MatParam::~MatParam(){}

void MatParam::update(double lr, double T, double clip, bool clipping, bool gaussian){
    if (clipping){
        dw->array() = dw->unaryExpr([clip](double x) -> double { if (x > clip) return clip; if (x < -clip) return -clip; return x; });
    }
//    if (gaussian){
//        *dw +=
//    }
    *w -= lr * (*dw);
    if (avg){ *cw -= lr * T * (*dw); }
    (*dw).fill(0.0);
}

void MatParam::average(double T){
    *w -= (*cw) / T;
}

int MatParam::size(){return w->size();}

void MatParam::add_epsilon(int i, double epsilon){
    (*w)(i / w->cols(), i % w->cols()) += epsilon;
}

void MatParam::set_empirical_gradient(int i, double eg){
    (*cw)(i / w->cols(), i % w->cols()) = eg;
}

void MatParam::print_gradient_differences(){
    cerr << "Gradient differences " << (*dw - *cw).array().abs().sum() / w->size() << endl;
}

void MatParam::assign(shared_ptr<Parameter> &other){
    shared_ptr<MatParam> p = std::dynamic_pointer_cast<MatParam>(other);
    *w = *(p->w);
    *dw = *(p->dw);
    *cw = *(p->cw);
}

void MatParam::print(ostream &os){
    os << *w << endl;
}





/////////////////////////////////////////////////////////////////
///
///
///
///
///







VecParam::VecParam(){
    b = db = cb = nullptr;
}

VecParam::VecParam(Vec *b1, Vec *b2, Vec *b3){
    b = b1;
    db = b2;
    cb = b3;
}

VecParam::~VecParam(){}

void VecParam::update(double lr, double T, double clip, bool clipping, bool gaussian){
    if (clipping){
        db->array() = db->unaryExpr([clip](double x) -> double { if (x > clip) return clip; if (x < -clip) return -clip; return x; });
    }
    *b -= lr * (*db);
    if (avg){ *cb -= lr * T * (*db); }
    (*db).fill(0.0);
}

void VecParam::average(double T){
    *b -= (*cb) / T;
}

int VecParam::size(){return b->size();}

void VecParam::add_epsilon(int i, double epsilon){
    (*b)[i] += epsilon;
}

void VecParam::set_empirical_gradient(int i, double eg){
    (*cb)[i] = eg;
}

void VecParam::print_gradient_differences(){
    cerr << "Gradient differences " << (*db - *cb).array().abs().sum() / b->size() << endl;
}


void VecParam::assign(shared_ptr<Parameter> &other){
    shared_ptr<VecParam> p = std::dynamic_pointer_cast<VecParam>(other);
    *b = *(p->b);
    *db = *(p->db);
    *cb = *(p->cb);
}

void VecParam::print(ostream &os){
    os << *b << endl;
}


Layer::~Layer(){}

void Layer::get_params(vector<shared_ptr<Parameter>> &t){}


AffineLayer::AffineLayer(int insize, int outsize){
    w = xavier(insize, outsize);
    dw = cw = Mat::Zero(outsize, insize);
    b = db = cb = Vec::Zero(outsize);
}

void AffineLayer::fprop(const vector<Vec*> &data, Vec& output){
    output = w * *(data[0]) + b;
}

void AffineLayer::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    db += out_derivative;
    dw += out_derivative * (*(data[0])).transpose();
    *(gradient[0]) += w.transpose() * out_derivative;
}

void AffineLayer::get_params(vector<shared_ptr<Parameter>> &t){
    t.push_back(shared_ptr<MatParam>(new MatParam(&w, &dw, &cw)));
    t.push_back(shared_ptr<VecParam>(new VecParam(&b, &db, &cb)));
}




LinearLayer::LinearLayer(int insize, int outsize){
    w = xavier(insize, outsize);
    dw = cw = Mat::Zero(outsize, insize);
}
void LinearLayer::fprop(const vector<Vec*> &data, Vec& output){
    output = w * *(data[0]);
}
void LinearLayer::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    dw += out_derivative * (*(data[0])).transpose();
    *(gradient[0]) += w.transpose() * out_derivative;
}
void LinearLayer::get_params(vector<shared_ptr<Parameter>> &t){
    t.push_back(shared_ptr<MatParam>(new MatParam(&w, &dw, &cw)));
}






MultipleLinearLayer::MultipleLinearLayer(int insize, vector<int> &insizes, int outsize){
    layers.resize(insize);
    for (int i = 0; i < layers.size(); i++){
        layers[i] = new LinearLayer(insizes[i], outsize);
    }
}
MultipleLinearLayer::~MultipleLinearLayer(){
    for (int i = 0; i < layers.size(); i++){
        delete layers[i];
    }
}

void MultipleLinearLayer::fprop(const vector<Vec*> &data, Vec& output){
    output.fill(0.0);
    for (int i = 0; i < layers.size(); i++){
        vector<Vec*> datum{data[i]};
        layers[i]->fprop(datum, buffer);
        output += buffer;
    }
}

void MultipleLinearLayer::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    for (int i = 0; i < layers.size(); i++){
        vector<Vec*> datum{data[i]};
        vector<Vec*> grad{gradient[i]};
        layers[i]->bprop(datum, output, out_derivative, grad);
    }
}

void MultipleLinearLayer::get_params(vector<shared_ptr<Parameter>> &t){
    for (int i = 0; i < layers.size(); i++){
        layers[i]->get_params(t);
    }
}






RecurrentLayer::RecurrentLayer(int insize, int outsize){
    w = xavier(insize, outsize);
    rw = xavier(outsize, outsize);
    dw = cw = Mat::Zero(outsize, insize);
    drw = crw = Mat::Zero(outsize, outsize);
    b = db = cb = Vec::Zero(outsize);
}
void RecurrentLayer::fprop(const vector<Vec*> &data, Vec& output){
    output = w * *(data[0]) + rw * *(data[1]) + b;
}
void RecurrentLayer::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    db += out_derivative;
    dw += out_derivative * (*(data[0])).transpose();
    drw += out_derivative * (*(data[1])).transpose();
    *(gradient[0]) += w.transpose() * out_derivative;
    *(gradient[1]) += rw.transpose() * out_derivative;
}
void RecurrentLayer::get_params(vector<shared_ptr<Parameter>> &t){
    t.push_back(shared_ptr<MatParam>(new MatParam(&w, &dw, &cw)));
    t.push_back(shared_ptr<MatParam>(new MatParam(&rw, &drw, &crw)));
    t.push_back(shared_ptr<VecParam>(new VecParam(&b, &db, &cb)));
}







AddBias::AddBias(int outsize){
    b = db = cb = Vec::Zero(outsize);
}
void AddBias::fprop(const vector<Vec*> &data, Vec& output){
    output = *(data[0]) + b;
}
void AddBias::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    db += out_derivative;
    *(gradient[0]) += out_derivative;
}



// Standard operations
void Mult::fprop(const vector<Vec*> &data, Vec& output){
    output = (*(data[0])).cwiseProduct(*(data[1]));
}
void Mult::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    *(gradient[0]) += out_derivative.cwiseProduct(*(data[1]));
    *(gradient[1]) += out_derivative.cwiseProduct(*(data[0]));
}



void Add::fprop(const vector<Vec*> &data, Vec& output){
    output = *(data[0]);
    for (int i = 1; i < data.size(); i++){
        output += *(data[i]);
    }
}
void Add::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    for (int i = 0; i < gradient.size(); i++){
        *(gradient[i]) += out_derivative;
    }
}



void Mixture::fprop(const vector<Vec*> &data, Vec& output){
    output = (1.0 - data[0]->array()) * data[1]->array() + data[0]->array() * data[2]->array();
}
void Mixture::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    *(gradient[0]) += out_derivative.cwiseProduct(*(data[2]) - *(data[1]));
    (gradient[1])->array() += out_derivative.array() * (1.0 - data[0]->array());
    *(gradient[2]) += data[0]->cwiseProduct(out_derivative);
}





// Activation

void Tanh::fprop(const vector<Vec*> &data, Vec& output){
    output = (*(data[0])).unaryExpr(std::ptr_fun<double, double>(tanh));
}
void Tanh::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    (gradient[0])->array() += out_derivative.array() * (1.0 - output.array() * output.array());
}




void Sigmoid::fprop(const vector<Vec*> &data, Vec& output){
    output = 1.0 / (1.0 + (-(*(data[0]))).array().exp());
}
void Sigmoid::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    (gradient[0])->array() += out_derivative.array() * (output.array() * (1.0 - output.array()));
}




void ReLU::fprop(const vector<Vec*> &data, Vec& output){
    output = (*(data[0])).unaryExpr(std::ptr_fun<double, double>(rectifier));
}
void ReLU::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    (gradient[0])->array() += out_derivative.array() * ((*(data[0])).array() > 0.0).cast<double>();
}




void Softmax::fprop(const vector<Vec*> &data, Vec& output){
    output = ((data[0])->array() - (data[0])->maxCoeff()).exp();
    output /= output.sum();
}
void Softmax::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    *(gradient[0]) = output;
    (*(gradient[0]))[target] -= 1;
}


void SoftmaxFilter::fprop(const vector<Vec*> &data, Vec& output){
    output = ((data[0])->array() - (data[0])->maxCoeff()).exp() * data[1]->array();
    output /= output.sum();
}
void SoftmaxFilter::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    *(gradient[0]) = output;
    (*(gradient[0]))[target] -= 1;
}









/////////////////////////////////////////////////////////////////////////////////////////
///
///
///
///



LookupTable::LookupTable(){}
LookupTable::~LookupTable(){}

LookupTable::LookupTable(const LookupTable &other){
    v = other.v;
    dv = other.dv;
    cv = other.cv;
    vocsize = other.vocsize;
    dimension = other.dimension;
}

LookupTable& LookupTable::operator=(const LookupTable &other){
    v = other.v;
    dv = other.dv;
    cv = other.cv;
    vocsize = other.vocsize;
    dimension = other.dimension;
    return *this;
}

LookupTable::LookupTable(int vocsize, int dimension){
    this->vocsize = vocsize;
    this->dimension = dimension;
    v = vector<Vec>(vocsize);
    dv = vector<Vec>(vocsize);
    cv = vector<Vec>(vocsize);
    for (int i = 0; i < vocsize; i++){
        dv[i] = cv[i] = Vec::Zero(dimension);
        v[i] = Vec::Random(dimension) / 100.0;
    }
}

void LookupTable::get(int i, shared_ptr<VecParam> &param){
    assert(i < vocsize);
    if (active.find(i) == active.end()){
        active[i] = shared_ptr<VecParam>(new VecParam(&(v[i]), &(dv[i]), &(cv[i])));
    }
    param = active[i];
}

void LookupTable::update(double lr, double T, double clip, bool clipping, bool gaussian){
    for (auto it = active.begin(); it != active.end(); it++){
        it->second->update(lr, T, clip, clipping, gaussian);
    }
    active.clear();
}

void LookupTable::get_active_params(vector<shared_ptr<Parameter>> &params){
    for (auto &it : active){
        params.push_back(it.second);
    }
}

void LookupTable::average(int T){
    for (int i = 0; i < vocsize; i++){
        VecParam p(&v[i], &dv[i], &cv[i]);
        p.average(T);
    }
}

void LookupTable::export_model(const string &filename){
    ofstream os(filename);
    for (int i = 0; i < vocsize; i++){
        os << v[i].transpose() << endl;
    }
}





void get_gru_layer(vector<shared_ptr<Layer>> &layers, int insize, int hiddensize){
    layers.push_back(shared_ptr<RecurrentLayer>(new RecurrentLayer(insize, hiddensize)));
    layers.push_back(shared_ptr<Sigmoid>(new Sigmoid()));
    layers.push_back(shared_ptr<RecurrentLayer>(new RecurrentLayer(insize, hiddensize)));
    layers.push_back(shared_ptr<Sigmoid>(new Sigmoid()));
    layers.push_back(shared_ptr<Mult>(new Mult()));
    layers.push_back(shared_ptr<RecurrentLayer>(new RecurrentLayer(insize, hiddensize)));
    layers.push_back(shared_ptr<Tanh>(new Tanh()));
    layers.push_back(shared_ptr<Mixture>(new Mixture()));
}



