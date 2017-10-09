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





void Parameter::export_model(const string &outfile){
    ofstream os(outfile);
    print(os);
    os.close();
}




MatParam::MatParam(Mat *w1, Mat *w2, Mat *w3){
    w = w1;
    dw = w2;
    cw = w3;
}

MatParam::~MatParam(){}

void MatParam::update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta){
    if (gaussian){
        double sigma = pow(gaussian_eta / pow(1.0 + T, 0.55), 0.5);
        std::normal_distribution<double> distribution(0.0, sigma);
        //dw->array() = dw->unaryExpr([&distribution](double x) -> double { return x + distribution(random); });
        dw->array() = dw->unaryExpr([&distribution](double x) -> double { return x + distribution(rd::Random::re); });
    }
    if (clipping){
        dw->array() = dw->unaryExpr([clip](double x) -> double { if (x > clip) return clip; if (x < -clip) return -clip; return x; });
    }
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
    double grad_diff = (*dw - *cw).array().abs().sum() / w->size();
    cerr << "Gradient differences " << grad_diff
         << "  (size " << dw->rows() << " x " << dw->cols() << ")" << endl;
    if (grad_diff > 1e-5){
        cerr << "Empirical  " << cw->transpose() << endl;
        cerr << "Analytical " << dw->transpose() << endl;
    }
}

void MatParam::assign(shared_ptr<Parameter> &other){
    shared_ptr<MatParam> p = std::static_pointer_cast<MatParam>(other);
    *w = *(p->w);
    *dw = *(p->dw);
    *cw = *(p->cw);
}

void MatParam::print(ostream &os){
    os << *w << endl;
}


void MatParam::load(const string &file){
    Mat m;
    load_matrix<Mat>(file, m);
    assert(w->size() == m.size() && w->cols() == m.cols());
    *w = m;
}

void MatParam::reset_gradient_history(){
    cw->fill(0.0);
}

void MatParam::scale_gradient(double p){
    *dw *= p;
}

double MatParam::gradient_squared_norm(){
    return dw->squaredNorm();
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

void VecParam::update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta){
    if (gaussian){
        double sigma = pow(gaussian_eta / pow(1.0 + T, 0.55), 0.5);
        std::normal_distribution<double> distribution(0.0, sigma);
        db->array() = db->unaryExpr([&distribution](double x) -> double { return x + distribution(rd::Random::re); });
    }
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
    double grad_diff = (*db - *cb).array().abs().sum() / b->size();
    cerr << "Gradient differences " << grad_diff << " (size: " << b->size() << ")" << endl;
    if (grad_diff > 1e-5){
        cerr << "Empirical  " << cb->transpose() << endl;
        cerr << "Analytical " << db->transpose() << endl;
    }
}


void VecParam::assign(shared_ptr<Parameter> &other){
    shared_ptr<VecParam> p = std::static_pointer_cast<VecParam>(other);
    *b = *(p->b);
    *db = *(p->db);
    *cb = *(p->cb);
}

void VecParam::print(ostream &os){
    os << *b << endl;
}


void VecParam::load(const string &file){
    Vec v;
    load_matrix<Vec>(file, v);
    assert(b->size() == v.size());
    *b = v;
}

void VecParam::reset_gradient_history(){
    cb->fill(0.0);
}

void VecParam::scale_gradient(double p){
    *db *= p;
}

double VecParam::gradient_squared_norm(){
    return db->squaredNorm();
}





//////////////////////////////////////////////////////////

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
    b = db = cb = Vec::Zero(outsize);
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
    output = b;
    for (int i = 0; i < layers.size(); i++){
        vector<Vec*> datum{data[i]};
        layers[i]->fprop(datum, buffer);
        output += buffer;
    }
}

void MultipleLinearLayer::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    db += out_derivative;
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
    t.push_back(shared_ptr<Parameter>(new VecParam(&b, &db, &cb)));
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
void AddBias::get_params(vector<shared_ptr<Parameter>> &t){
    t.push_back(shared_ptr<VecParam>(new VecParam(&b, &db, &cb)));
}



ConstantLayer::ConstantLayer(int outsize){
    b = Vec::Random(outsize) / 100;
    db = cb = Vec::Zero(outsize);
}
void ConstantLayer::fprop(const vector<Vec*> &data, Vec& output){
    output = b;
}
void ConstantLayer::bprop(const vector<Vec*> &data, const Vec& output, const Vec & out_derivative, vector<Vec*> &gradient){
    db += out_derivative;
}
void ConstantLayer::get_params(vector<shared_ptr<Parameter>> &t){
    t.push_back(shared_ptr<VecParam>(new VecParam(&b, &db, &cb)));
}




// Standard operations
void Mult::fprop(const vector<Vec*> &data, Vec& output){
    assert(data.size() == 2);
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
    //assert(i < vocsize);
    if (i >= vocsize){
//#ifdef DEBUG
//        cerr << "Warning: unknown character accessed" << endl;
//#endif
        i = enc::UNKNOWN;
    }
    if (active.find(i) == active.end()){
        active[i] = shared_ptr<VecParam>(new VecParam(&(v[i]), &(dv[i]), &(cv[i])));
    }
    param = active[i];
}

void LookupTable::update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta){
    for (auto it = active.begin(); it != active.end(); it++){
        it->second->update(lr, T, clip, clipping, gaussian, gaussian_eta);
    }
    active.clear();
}

double LookupTable::gradient_squared_norm(){
    double gsn = 0;
    for (auto it = active.begin(); it != active.end(); it++){
        gsn += it->second->gradient_squared_norm();
    }
    return gsn;
}

void LookupTable::scale_gradient(double scale){
    for (auto it = active.begin(); it != active.end(); it++){
        it->second->scale_gradient(scale);
    }
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

void LookupTable::load(const string &filename){
    ifstream is(filename);
    string buffer;
    vector<string> tokens;
    while(getline(is, buffer)){
        str::split(buffer, " ", "", tokens);
        Vec vec(tokens.size());
        for (int i = 0; i < tokens.size(); i++){
            vec[i] = stod(tokens[i]);
        }
        v.push_back(vec);
    }
    vocsize = v.size();
    dimension = v[0].size();
    assert(consistent_dimension());
}

void LookupTable::clear(){
    v.clear();
    dv.clear();
    cv.clear();
}

bool LookupTable::consistent_dimension(){
    for (int i = 0; i < v.size(); i++){
        if (v[i].size() != this->dimension)
            return false;
    }
    return true;
}

void LookupTable::reset_gradient_history(){
    for (int i = 0; i < cv.size(); i++){
        cv[i].fill(0.0);
    }
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
/// ///////////////////////////////////////////////////////////////////////////////////
/// ///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

///What follows is RNN or computation graph related



AbstractNeuralNode::~AbstractNeuralNode(){}


LookupNode::LookupNode(){}
LookupNode::LookupNode(VecParam &e):embedding(e){}

void LookupNode::fprop(){}
void LookupNode::bprop(){}
Vec* LookupNode::v(){ return embedding.b;}
Vec* LookupNode::d(){ return embedding.db;}


NeuralNode::~NeuralNode(){}
NeuralNode::NeuralNode(int size){
    state = Vec::Zero(size);
    dstate = Vec::Zero(size);
}

Vec* NeuralNode::v(){return &state;}
Vec* NeuralNode::d(){return &dstate;}




SimpleNode::SimpleNode(int size, Layer *layer, const shared_ptr<AbstractNeuralNode> &input)
    : NeuralNode(size){
    this->layer = layer;
    this->input = input;
}

void SimpleNode::fprop(){
    vector<Vec*> data{input->v()};
    layer->fprop(data, state);
}
void SimpleNode::bprop(){
    vector<Vec*> data{input->v()};
    vector<Vec*> data_grad{input->d()};
    layer->bprop(data, state, dstate, data_grad);
}



ComplexNode::ComplexNode(int size, Layer *layer, vector<shared_ptr<AbstractNeuralNode> > &input)
    : NeuralNode(size),
      layer(layer),
      input(input){}


void ComplexNode::fprop(){
    vector<Vec*> data;
    for (int i = 0; i < input.size(); i++){
        data.push_back(input[i]->v());
    }
    layer->fprop(data, state);
}

void ComplexNode::bprop(){
    vector<Vec*> data;
    vector<Vec*> data_grad;
    for (int i = 0; i < input.size(); i++){
        data.push_back(input[i]->v());
        data_grad.push_back(input[i]->d());
    }
    layer->bprop(data, state, dstate, data_grad);
}



RecurrentLayerWrapper::RecurrentLayerWrapper(int cell_type, vector<int> &input_sizes, int hidden_size){
    switch (cell_type){
    case GRU: get_gru(input_sizes, hidden_size); break;
    case RNN: get_vanilla_rnn(input_sizes, hidden_size); break;
    case LSTM: get_lstm(input_sizes, hidden_size); break;
    default:
        assert(false && "Unknown recurrent cell type");
    }
}

RecurrentLayerWrapper::~RecurrentLayerWrapper(){
    for (int i = 0; i < layers.size(); i++){
        delete layers[i];
        layers[i] = nullptr;
    }
}

void RecurrentLayerWrapper::get_gru(vector<int> &input_sizes, int hidden_size){
    vector<int> input(input_sizes);
    input.push_back(hidden_size);
    layers.push_back(new ConstantLayer(hidden_size));
    layers.push_back(new ConstantLayer(hidden_size));
    layers.push_back(new MultipleLinearLayer(input.size(), input, hidden_size));
    layers.push_back(new Sigmoid());
    layers.push_back(new MultipleLinearLayer(input.size(), input, hidden_size));
    layers.push_back(new Sigmoid());
    layers.push_back(new Mult());
    layers.push_back(new MultipleLinearLayer(input.size(), input, hidden_size));
    layers.push_back(new Tanh());
    layers.push_back(new Mixture());
}
void RecurrentLayerWrapper::get_vanilla_rnn(vector<int> &input_sizes, int hidden_size){
    vector<int> input(input_sizes);
    input.push_back(hidden_size);
    layers.push_back(new ConstantLayer(hidden_size));
    layers.push_back(new MultipleLinearLayer(input.size(), input, hidden_size));
    layers.push_back(new ReLU());
}

void RecurrentLayerWrapper::get_lstm(vector<int> &input_sizes, int hidden_size){
//    vector<int> input(input_sizes);
//    input.push_back(hidden_size);
    vector<int> input(input_sizes);
    input.push_back(hidden_size);
    layers.push_back(new ConstantLayer(hidden_size)); // c0
    layers.push_back(new ConstantLayer(hidden_size)); // h0
    layers.push_back(new MultipleLinearLayer(input.size(), input, hidden_size)); // i
    layers.push_back(new Sigmoid());                                             // i act
    layers.push_back(new MultipleLinearLayer(input.size(), input, hidden_size)); // f
    layers.push_back(new Sigmoid());                                             // f act
    layers.push_back(new MultipleLinearLayer(input.size(), input, hidden_size)); // o
    layers.push_back(new Sigmoid());                                             // o act
    layers.push_back(new MultipleLinearLayer(input.size(), input, hidden_size)); // g
    layers.push_back(new Tanh());                                                // g act
    layers.push_back(new Mult());           // cf
    layers.push_back(new Mult());           // gi
    layers.push_back(new Add());            // c
    layers.push_back(new Tanh());           // c act
    layers.push_back(new Mult);             // h
}



Layer* RecurrentLayerWrapper::operator[](int i){
    return layers[i];
}

int RecurrentLayerWrapper::size(){
    return layers.size();
}


ParamNode::ParamNode(int size, Layer *layer):NeuralNode(size),layer(layer){}

void ParamNode::fprop(){
    layer->fprop(place_holder, state);
}
void ParamNode::bprop(){
    layer->bprop(place_holder, state, dstate, place_holder);
}


MemoryNodeInitial::MemoryNodeInitial(int size, Layer* layer, shared_ptr<ParamNode> &paramnode):NeuralNode(size),h(paramnode), layer(layer){}
void MemoryNodeInitial::get_memory_node(shared_ptr<AbstractNeuralNode> &hnode){
    hnode = h;
}
void MemoryNodeInitial::fprop(){
    layer->fprop(place_holder, state);
    h->fprop();
}
void MemoryNodeInitial::bprop(){
    layer->bprop(place_holder, state, dstate, place_holder);
    h->bprop();
}









GruNode::GruNode(int size, shared_ptr<AbstractNeuralNode> &predecessor, vector<shared_ptr<AbstractNeuralNode> > &input, RecurrentLayerWrapper &layers)
    : ComplexNode(size,nullptr,input){

    this->pred = std::static_pointer_cast<GruNode>(predecessor);

    layer = layers[S];

    vector<shared_ptr<AbstractNeuralNode>> z_in(input);
    shared_ptr<AbstractNeuralNode> h_node;
    pred->get_memory_node(h_node);
    z_in.push_back(h_node);
    pz = shared_ptr<ComplexNode>(new ComplexNode(size, layers[Z1], z_in));
    z  = shared_ptr<SimpleNode>(new SimpleNode(size, layers[Z2], pz));

    pr = shared_ptr<ComplexNode>(new ComplexNode(size, layers[R1], z_in));
    r  = shared_ptr<SimpleNode>(new SimpleNode(size, layers[R2], pr));


    vector<shared_ptr<AbstractNeuralNode>> h_in{h_node, r};
    hr = shared_ptr<ComplexNode>(new ComplexNode(size, layers[H1], h_in));
    vector<shared_ptr<AbstractNeuralNode>> h2_in(input);
    h2_in.push_back(hr);
    ph = shared_ptr<ComplexNode>(new ComplexNode(size, layers[H2], h2_in));
    h = shared_ptr<SimpleNode>(new SimpleNode(size, layers[H3], ph));

    internal_nodes = {pz, z, pr, r, hr, ph, h};
}

GruNode::~GruNode(){}

void GruNode::fprop(){
    for (int i = 0; i < internal_nodes.size(); i++){
        internal_nodes[i]->fprop();
    }
    vector<Vec*> data{z->v(), pred->v(), h->v()};
    layer->fprop(data, state);
}

void GruNode::bprop(){
    vector<Vec*> data{z->v(), pred->v(), h->v()};
    vector<Vec*> data_grad{z->d(), pred->d(), h->d()};
    layer->bprop(data, state, dstate, data_grad);

    for (int i = internal_nodes.size()-1; i >= 0; i--){
        internal_nodes[i]->bprop();
    }
}

void GruNode::get_memory_node(shared_ptr<AbstractNeuralNode> &hnode){
    hnode = h;
}









RnnNode::RnnNode(int size,
                 shared_ptr<AbstractNeuralNode> &predecessor,
                 vector<shared_ptr<AbstractNeuralNode>> &input,
                 RecurrentLayerWrapper &layers)
            : ComplexNode(size,nullptr,input){
    pred = predecessor;

    vector<shared_ptr<AbstractNeuralNode>> h_in(input);
    h_in.push_back(pred);
    h = shared_ptr<ComplexNode>(new ComplexNode(size, layers[REC], h_in));
    layer = layers[ACTIVATION];
}
RnnNode::~RnnNode(){}

void RnnNode::fprop(){
    h->fprop();
    vector<Vec*> data{h->v()};
    layer->fprop(data, state);
}
void RnnNode::bprop(){
    vector<Vec*> data{h->v()};
    vector<Vec*> data_grad{h->d()};
    layer->bprop(data, state, dstate, data_grad);
    h->bprop();
}












LstmNode::LstmNode(int size,
                   shared_ptr<AbstractNeuralNode> &predecessor,
                   vector<shared_ptr<AbstractNeuralNode>> &input,
                   RecurrentLayerWrapper &layers)
                : ComplexNode(size,nullptr,input){

    this->pred = std::static_pointer_cast<LstmNode>(predecessor);

    layer = layers[H];

    vector<shared_ptr<AbstractNeuralNode>> in(input);
    in.push_back(predecessor);

    ia = shared_ptr<ComplexNode>(new ComplexNode(size, layers[I], in));
    ih = shared_ptr<SimpleNode>(new SimpleNode(size, layers[IS], ia));

    fa = shared_ptr<ComplexNode>(new ComplexNode(size, layers[F], in));
    fh = shared_ptr<SimpleNode>(new SimpleNode(size, layers[FS], fa));

    oa = shared_ptr<ComplexNode>(new ComplexNode(size, layers[O], in));
    oh = shared_ptr<SimpleNode>(new SimpleNode(size, layers[OS], oa));

    ga = shared_ptr<ComplexNode>(new ComplexNode(size, layers[G], in));
    gh = shared_ptr<SimpleNode>(new SimpleNode(size, layers[GT], ga));

    shared_ptr<AbstractNeuralNode> memory_node;
    pred->get_memory_node(memory_node);

    in = {memory_node, fh};
    cf_mult = shared_ptr<ComplexNode>(new ComplexNode(size, layers[CF], in));
    in = {gh, ih};
    gi_mult = shared_ptr<ComplexNode>(new ComplexNode(size, layers[GI], in));
    in = {cf_mult, gi_mult};
    c = shared_ptr<ComplexNode>(new ComplexNode(size, layers[C], in));
    in = {c};
    ch = shared_ptr<SimpleNode>(new SimpleNode(size, layers[CT], c));

    internal_nodes = {ia, ih, fa, fh, oa, oh, ga, gh, cf_mult, gi_mult, c, ch};

}

LstmNode::~LstmNode(){}

void LstmNode::fprop(){
    for (int i = 0; i < internal_nodes.size(); i++){
        internal_nodes[i]->fprop();
    }
    vector<Vec*> data{ch->v(), oh->v()};
    layer->fprop(data, state);
}

void LstmNode::bprop(){
    vector<Vec*> data{ch->v(), oh->v()};
    vector<Vec*> data_grad{ch->d(), oh->d()};
    layer->bprop(data, state, dstate, data_grad);

    for (int i = internal_nodes.size()-1; i >= 0; i--){
        internal_nodes[i]->bprop();
    }

}

void LstmNode::get_memory_node(shared_ptr<AbstractNeuralNode> &hnode){
    hnode = c;
}







