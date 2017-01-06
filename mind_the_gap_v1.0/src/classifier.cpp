#include "classifier.h"


/////////////////////////////////////////////////////////
///
///
///     Classifier
///
///
/// ////////////////////////////////////////////////////

Classifier::Classifier(int n_classes) : n_updates_(0), T_(0), N_(n_classes){
    scores_ = vector<double>(n_classes, 0.0);
}


Classifier::~Classifier(){}

void Classifier::print_stats(ostream &os){
    os << "Nothing to see here" << endl;
}

double Classifier::operator[](int i){
    return scores_[i];
}

int Classifier::get_n_updates(){
    return n_updates_;
}

void Classifier::reset_updates(){
    n_updates_ = 0;
}

int Classifier::argmax(){
    int imax  = -1;
    double max = MINUS_INFINITY;
    for (int i = 0; i < scores_.size(); i++){
        if (scores_[i] > max){
            imax = i;
            max = scores_[i];
        }
    }
    assert(imax != -1);
    return imax;
}

void Classifier::increment_T(){
    T_++;
}

void Classifier::increment_updates(){
    n_updates_++;
}


//void Classifier::export_model(const string &outdir){
//    assert(false &&"Not implemented error");
//}

Classifier* Classifier::import_model(const string &outdir){
    ifstream is(outdir + "/classifier_id_classes");
    int id_classifier;
    int n_classes;
    is >> id_classifier;
    is >> n_classes;
    switch (id_classifier){
    case FASTEST_PER:
        return new FastestPerceptron(n_classes, outdir);
    case FFNN:
        return new NeuralNet(n_classes, outdir);
    default:
        assert(false && "Not implemented yet");
    }
    return nullptr;
}





/////////////////////////////////////////////////////////
///
///
///     Perceptron
///
///
/// ////////////////////////////////////////////////////



Perceptron::Perceptron(int n_classes): Classifier(n_classes){
    cerr << "WARNING: this perceptron class is deprecated, use FastPercetron instead." << endl;
    weights = vector<unordered_map<int, double>>(n_classes);
    cached = vector<unordered_map<int, double>>(n_classes);
}

int Perceptron::get_id(){
    return Classifier::PERCEPTRON;
}

Classifier* Perceptron::copy(){
    Perceptron *p = new Perceptron(N_);
    p->weights = weights;
    p->cached = cached;
    p->scores_ = scores_;
    p->n_updates_ = n_updates_;
    p->T_ = T_;
    return p;
}

void Perceptron::average_weights(){
    for (int i = 0; i < N_; i++){
        unordered_map<int, double> *w = &(weights[i]);
        unordered_map<int, double> *cw = &(cached[i]);
        for (auto &it : *w){
            (*w)[it.first] = it.second - (*cw)[it.first] / T_;
        }
    }
}



void Perceptron::score(const vector<int> &features, const vector<bool> &allowed){
    for (int i = 0; i < N_; i++){
        scores_[i] = 0.0;
        if (allowed[i]){
            for (int f : features){
                auto it = weights[i].find(f);
                if (it != weights[i].end()){
                    scores_[i] += it->second;
                }
            }
        }
    }
}

void Perceptron::global_update_one(const vector<int> &feats,
                       int action_num,
                       int increment){
    for (int f : feats){ // REMINDER: f might not be the map (IDLE action -> no feature extraction) but this should work altogether (defaultdict style)
        weights[action_num][f] += increment;
        cached[action_num][f] += T_ * increment;
    }
}


void Perceptron::export_model(const string &outdir){
    assert(false &&"Not implemented error");
}


/////////////////////////////////////////////////////////
///
///
///     SparseVector
///
///
/// ////////////////////////////////////////////////////






double& SparseVector::operator[](int i){                  // REMINDER: this should not be used at test time
    for (int k = 0; k < loc.size(); k++){
        if (loc[k] == i)
            return val[k];
    }
    loc.push_back(i);
    val.push_back(0.0);
    return val.back();
}

void SparseVector::accumulate(vector<double> &s){
    for (int i = 0; i < loc.size(); i++){
        s[loc[i]] += val[i];
    }
}
SparseVector& SparseVector::operator-=(const SparseVector &v){
    for (int i = 0; i < v.loc.size(); i++){
        (*this)[v.loc[i]] -= v.val[i];
    }
    return *this;
}
SparseVector& SparseVector::operator+=(const SparseVector &v){
    for (int i = 0; i < v.loc.size(); i++){
        (*this)[v.loc[i]] += v.val[i];
    }
    return *this;
}
SparseVector& SparseVector::operator*=(const double d){
    for (int i = 0; i < loc.size(); i++){
        val[i] *= d;
    }
    return *this;
}
SparseVector& SparseVector::operator/=(const double d){
    for (int i = 0; i < loc.size(); i++){
        val[i] /= d;
    }
    return *this;
}

int SparseVector::size(){
    return loc.size();
}

ostream & operator<<(ostream &os, const SparseVector &v){
    if (v.loc.size() > 0)
        os << v.loc[0] << ":" << v.val[0];
    for (int i = 1; i < v.loc.size(); i++){
        os << " " << v.loc[i] << ":" << v.val[i];
    }
    return os;
}

void SparseVector::test(){
    SparseVector a;
    a[0] = 4;
    a[6] = 2;
    cerr << "a=" << a << endl;
    SparseVector b;
    b[1] = 6;
    b[6] = 1;
    cerr << "b=" << b << endl;
    b *= 4;
    cerr << "b=" << b << endl;
    a -= b;
    cerr << a << endl;
}





/////////////////////////////////////////////////////////
///
///
///     FastPerceptron
///
///
/// ////////////////////////////////////////////////////





FastPerceptron::FastPerceptron(int n_classes): Classifier(n_classes){}

int FastPerceptron::get_id(){
    return Classifier::FAST_PER;
}

Classifier* FastPerceptron::copy(){
    FastPerceptron *p = new FastPerceptron(N_);
    p->weights = weights;
    p->scores_ = scores_;
    p->n_updates_ = n_updates_;
    p->T_ = T_;
    return p;
}

void FastPerceptron::average_weights(){
    double d = T_;
    for (auto &it : weights){
        it.second.second /= d;
        it.second.first -= it.second.second;
    }
}

void FastPerceptron::score(const vector<int> &features, const vector<bool> &allowed){

    for (int i = 0; i < scores_.size(); i++){
        scores_[i] = 0.0;
    }
    for (int f :features){
        auto it = weights.find(f);
        if (it != weights.end()){
            it->second.first.accumulate(scores_);
        }
    }
}

void FastPerceptron::global_update_one(const vector<int> &feats,
                                       int action_num,
                                       int increment){
    for (int f : feats){
        auto it = weights.find(f);
        if (it == weights.end()){
            weights[f] = std::pair<SparseVector, SparseVector>();
            weights[f].first[action_num] += increment;
            weights[f].second[action_num] += T_ * increment;
        }else{
            it->second.first[action_num] += increment;
            it->second.second[action_num] += T_ * increment;
        }
    }
}

void FastPerceptron::print_stats(ostream &os){
    os << "Number of features : " << weights.size() << endl;

//    vector<int> v;
//    for (auto &it : weights){
//        v.push_back(it.first);
//    }
//    std::sort(v.begin(), v.end());
//    for (int i : v){
//       cout << i << " ";
//    }cout << endl;
}


void FastPerceptron::export_model(const string &outdir){
    assert(false &&"Not implemented error");
}



/////////////////////////////////////////////////////////
///
///
///     FasterPerceptron
///
///
/// ////////////////////////////////////////////////////


FasterPerceptron::FasterPerceptron(int n_classes): Classifier(n_classes){}

int FasterPerceptron::get_id(){
    return Classifier::FASTER_PER;
}


Classifier* FasterPerceptron::copy(){
    FasterPerceptron *p = new FasterPerceptron(N_);
    p->weights = weights;
    p->scores_ = scores_;
    p->n_updates_ = n_updates_;
    p->T_ = T_;
    return p;
}

void FasterPerceptron::average_weights(){
    double d = T_;
    for (int i = 0; i < weights.size(); i++){
        weights[i].second /= d;
        weights[i].first -= weights[i].second;
    }
}

void FasterPerceptron::score(const vector<int> &features, const vector<bool> &allowed){

    for (int i = 0; i < scores_.size(); i++){
        scores_[i] = 0.0;
    }
    for (int f : features){
        if (weights.size() <= f){
            weights.resize(f+1);
        }
    }
    for (int f :features){
        weights[f].first.accumulate(scores_);
    }
}

void FasterPerceptron::global_update_one(const vector<int> &feats,
                                       int action_num,
                                       int increment){
    for (int f : feats){
        if (weights.size() <= f){
            weights.resize(f+1);
        }
    }
    for (int f : feats){
        weights[f].first[action_num] += increment;
        weights[f].second[action_num] += T_ * increment;
    }
}

void FasterPerceptron::print_stats(ostream &os){
    os << "Number of features : " << weights.size() << endl;
}


void FasterPerceptron::export_model(const string &outdir){
    assert(false &&"Not implemented error");
}


/////////////////////////////////////////////////////////
///
///
///     FastestPerceptron
///
///
/// ////////////////////////////////////////////////////


FastestPerceptron::FastestPerceptron(int n_classes): Classifier(n_classes){
    weights.resize(KERNEL_SIZE);
}

int FastestPerceptron::get_id(){
    return Classifier::FASTEST_PER;
}


FastestPerceptron::FastestPerceptron(int n_classes, const string &outdir) : FastestPerceptron(n_classes){
    ifstream in(outdir + "/classifier_weights");
    int ksize;
    in >> ksize;
    weights.resize(ksize);
    string buffer;
    while (getline(in, buffer)){
        boost::char_separator<char> sep(" ");
        boost::tokenizer<boost::char_separator<char>> toks(buffer, sep);
        vector<string> tokens(toks.begin(), toks.end());
        if (tokens.size() > 0){
            int index = stoi(tokens[0]);
            for (int i = 1; i < tokens.size(); i++){
                int f = tokens[i].find(":");
                assert(f != string::npos);
                string s1 = tokens[i].substr(0, f);
                string s2 = tokens[i].substr(f+1, string::npos);

                weights[index].first[stoi(s1)] = stof(s2);
            }
        }
    }
}


Classifier* FastestPerceptron::copy(){
    FastestPerceptron *p = new FastestPerceptron(N_);
    p->weights = weights;
    p->scores_ = scores_;
    p->n_updates_ = n_updates_;
    p->T_ = T_;
    return p;
}

void FastestPerceptron::average_weights(){
    double d = T_;
    for (int i = 0; i < weights.size(); i++){
        weights[i].second /= d;
        weights[i].first -= weights[i].second;
    }
}

void FastestPerceptron::score(const vector<int> &features, const vector<bool> &allowed){

    for (int i = 0; i < scores_.size(); i++){
        scores_[i] = 0.0;
    }
    for (int f :features){
        weights[f].first.accumulate(scores_);
    }
}

void FastestPerceptron::global_update_one(const vector<int> &feats,
                                       int action_num,
                                       int increment){
    for (int f : feats){
        weights[f].first[action_num] += increment;
        weights[f].second[action_num] += T_ * increment;
    }
}

void FastestPerceptron::print_stats(ostream &os){
    os << "Kernel size: " << KERNEL_SIZE << endl;
}

void FastestPerceptron::export_model(const string &outdir){
    ofstream out1(outdir + "/classifier_id_classes");
    out1 << FASTEST_PER << endl;
    out1 << N_ << endl;
    out1.close();

    ofstream out2(outdir + "/classifier_weights");
    out2 << KERNEL_SIZE << endl; // make sure that kernel size global constant is really constant across experiments
    for (int i = 0; i < weights.size(); i++){
        if (weights[i].first.size() > 0)
            out2 << i << " " << weights[i].first << endl;
    }
}













/////////////////////////////////////////////////////////////////////
///
///
///     Neural nets
///
///
/// //////////////////////////////////////////////////////////////////



NetTopology::NetTopology():n_hidden_layers(2), size_hidden_layers(16), embedding_size_type{8,8,8,8}{}

NeuralNetParameters::NeuralNetParameters():learning_rate(0.02),decrease_constant(1e-6), clip_value(10.0), gaussian_noise(false), gradient_clipping(false){}

void NeuralNetParameters::print(ostream &os){
    os << "    - learning rate: "       << learning_rate << endl;
    os << "    - decrease constant: "   << decrease_constant << endl;
    os << "    - gradient clipping: "   << gradient_clipping << endl;
    os << "    - clip value: "          << clip_value << endl;
    os << "    - gaussian noise: "      << gaussian_noise << endl;
    os << "    - num hidden layers: "   << topology.n_hidden_layers << endl;
    os << "    - size hidden layers: "  << topology.size_hidden_layers << endl;
    os << "    - size embeddings:";
    for (int &i : topology.embedding_size_type){
        os << " " << i;
    } os << endl;
}

void NeuralNetParameters::read_option_file(const string &filename, NeuralNetParameters &p){
    enum {LEARNING_RATE, DECREASE_CONSTANT, GRADIENT_CLIPPING, CLIP_VALUE, GAUSSIAN_NOISE, HIDDEN_LAYERS, SIZE_HIDDEN, EMBEDDING_SIZE};
    unordered_map<string,int> dictionary{
        {"learning rate", LEARNING_RATE},
        {"decrease constant", DECREASE_CONSTANT},
        {"gradient clipping", GRADIENT_CLIPPING},
        {"clip value", CLIP_VALUE},
        {"gaussian noise", GAUSSIAN_NOISE},
        {"hidden layers", HIDDEN_LAYERS},
        {"size hidden layers", SIZE_HIDDEN},
        {"embedding sizes", EMBEDDING_SIZE},
    };
    ifstream is(filename);
    string buffer;
    vector<string> tokens;
    while (getline(is,buffer)){
        str::split(buffer, "\t", "", tokens);
        if (tokens.size() == 2){
            int id = dictionary[tokens[0]];
            switch (id){
            case LEARNING_RATE:     p.learning_rate = stod(tokens[1]);              break;
            case DECREASE_CONSTANT: p.decrease_constant = stod(tokens[1]);          break;
            case GRADIENT_CLIPPING: p.gradient_clipping = stoi(tokens[1]);          break;
            case CLIP_VALUE:        p.clip_value = stod(tokens[1]);                 break;
            case GAUSSIAN_NOISE:    p.gaussian_noise = stoi(tokens[1]);             break;
            case HIDDEN_LAYERS:     p.topology.n_hidden_layers = stoi(tokens[1]);   break;
            case SIZE_HIDDEN:       p.topology.size_hidden_layers = stoi(tokens[1]);break;
            case EMBEDDING_SIZE:{
                vector<string> sizes;
                str::split(tokens[1], " ", "", sizes);
                p.topology.embedding_size_type.clear();
                for (string &s : sizes){
                    p.topology.embedding_size_type.push_back(stoi(s));
                }
                break;
            }
            default: assert(false && "Unknown nn option");
            }
        }else{
            cerr << "Unknown nn option or problematic formatting : " << buffer << endl;
        }
    }
}















/////////////////////////////////////////////////////////////////////
///
///
///     Neural nets
///
///
/// //////////////////////////////////////////////////////////////////




NeuralNet::NeuralNet(int n_classes, const string &outdir) : Classifier(n_classes){
    assert(false);
}

NeuralNet::NeuralNet(int n_classes) : Classifier(n_classes){}

NeuralNet::NeuralNet(int n_classes, FeatureExtractor *fe, NeuralNetParameters &params) :
        Classifier(n_classes),
        params_(params){
    fe->get_template_type(feature_types);
    n_features = fe->n_templates();

    int max_type = 0;
    for (int i = 0; i < feature_types.size(); i++){
        if (feature_types[i] > max_type){
            max_type = feature_types[i];
        }
    }
    lu.resize(max_type+1);
    cerr << "Max type id = " << max_type << endl;
    for (int i = 0; i < lu.size(); i++){
        lu[i] = LookupTable(enc::hodor.size(i), params_.topology.embedding_size_type[i]);
        cerr << "Lookup table " << i << " has " << params_.topology.embedding_size_type[i] << " dimensions (vocsize = " << enc::hodor.size(i) << ")" << endl;
    }
    initialize_network();
}

int NeuralNet::get_id(){
    return Classifier::FFNN;
}

NeuralNet::~NeuralNet(){
    for (int i = 0; i < layers.size(); i++){
        delete layers[i];
    }
}

void NeuralNet::initialize_network(){
    vector<int> embedding_size_token;
    for (int i = 0; i < feature_types.size(); i++){
        embedding_size_token.push_back(params_.topology.embedding_size_type[feature_types[i]]);
    }
    layers.push_back(new MultipleLinearLayer(feature_types.size(), embedding_size_token, params_.topology.size_hidden_layers));
    layers.push_back(new ReLU());
    for (int i = 1; i < params_.topology.n_hidden_layers; i++){
        layers.push_back(new LinearLayer(params_.topology.size_hidden_layers, params_.topology.size_hidden_layers));
        layers.push_back(new ReLU());
    }
    layers.push_back(new LinearLayer(params_.topology.size_hidden_layers, N_));
    layers.push_back(new SoftmaxFilter());

    edata.resize(n_features);
    edata_grad.resize(n_features);
    states.resize(layers.size());
    dstates.resize(layers.size());
    for (int i = 0; i < layers.size() -2; i++){
        states[i] = dstates[i] = Vec::Zero(params_.topology.size_hidden_layers);
    }
    states[layers.size()-2] = Vec::Zero(N_);
    states[layers.size()-1] = Vec::Zero(N_);
    dstates[layers.size()-2] = Vec::Zero(N_);
    dstates[layers.size()-1] = Vec::Zero(N_);


    for (int i = 0; i < layers.size(); i++){
        layers[i]->get_params(parameters);
    }
    softmax_filter = Vec::Zero(N_);
}

void NeuralNet::average_weights(){
    for (int i = 0; i < lu.size(); i++){
        lu[i].average(T_);
    }
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->average(T_);
    }
}

double NeuralNet::fprop(const vector<int> &features, const vector<bool> &allowed, int target){

    assert(softmax_filter.size() == allowed.size());
    for (int i = 0; i < allowed.size(); i++){
        softmax_filter[i] = allowed[i];
    }

    assert(features.size() == n_features);
    for (int i = 0; i < n_features; i++){
        shared_ptr<VecParam> p;
        lu[feature_types[i]].get(features[i], p);
        edata[i] = p->b;
        edata_grad[i] = p->db;
    }
    layers[0]->fprop(edata, states[0]);
    for (int i = 1; i < layers.size()-1; i++){
        vector<Vec*> input{&states[i-1]};
        layers[i]->fprop(input, states[i]);
    }
    vector<Vec*> input{&states[layers.size()-2], &softmax_filter};
    layers.back()->fprop(input, states.back());

    return - log(states.back()[target]);
}

void NeuralNet::bprop(const vector<int> &features, const vector<bool> &allowed, int target){
    for (int i = 0; i < dstates.size(); i++){
        dstates[i].fill(0.0);
    }
    layers.back()->target = target;
    vector<Vec*> data{&states[layers.size()-2], &softmax_filter};
    vector<Vec*> data_grad{&dstates[layers.size()-2]};
    layers.back()->bprop(data, states.back(), dstates.back(), data_grad);
    for (int i = layers.size() -1; i > 0; i--){
        data = {&states[i-1]};
        data_grad = {&dstates[i-1]};
        layers[i]->bprop(data, states[i], dstates[i], data_grad);
    }
    layers[0]->bprop(edata, states[0], dstates[0], edata_grad);
}

void NeuralNet::update(){
    double lr = get_learning_rate();
    for (int i = 0; i < lu.size(); i++){
        lu[i].update(lr, T_, params_.clip_value, params_.gradient_clipping, params_.gaussian_noise);
    }
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->update(lr, T_, params_.clip_value, params_.gradient_clipping, params_.gaussian_noise);
    }
}

void NeuralNet::gradient_check(const vector<int> &features, const vector<bool> &allowed, int target, double epsilon){
    fprop(features, allowed, target);
    bprop(features, allowed, target);

    for (int i = 0; i < lu.size(); i++){
        lu[i].get_active_params(parameters);
    }

    for (int i = 0; i < parameters.size(); i++){
        for (int k = 0; k < parameters[i]->size(); k++){
            parameters[i]->add_epsilon(k, epsilon);
            double a = fprop(features, allowed, target);
            parameters[i]->add_epsilon(k, -epsilon);
            parameters[i]->add_epsilon(k, -epsilon);
            double c = fprop(features, allowed, target);
            parameters[i]->add_epsilon(k, epsilon);
            parameters[i]->set_empirical_gradient(k, (a-c) / (2 * epsilon));
        }
        parameters[i]->print_gradient_differences();
    }
}

double NeuralNet::get_learning_rate(){
    return params_.learning_rate / (1.0 + T_ * params_.decrease_constant);
}

Classifier* NeuralNet::copy(){
    NeuralNet *nn = new NeuralNet(N_);  // need a new constructor

    nn->scores_ = this->scores_;
    nn->T_ = this->T_;
    nn->n_updates_ = this->n_updates_;

    nn->n_features = this->n_features;
    nn->params_ = this->params_;
    nn->feature_types = this->feature_types;
    nn->lu = this->lu;

    nn->initialize_network();

    for (int i = 0; i < nn->parameters.size(); i++){
        nn->parameters[i]->assign(parameters[i]);
    }
    return nn;
}

void NeuralNet::score(const vector<int> &features, const vector<bool> &allowed){
    fprop(features, allowed,0);
    for (int i = 0; i < N_; i++){
        scores_[i] = log(states.back()[i]);
    }
}

void NeuralNet::global_update_one(const vector<int> &feats,
                       int action_num,
                       int increment){
    assert(false);
}

void NeuralNet::print_stats(ostream &os){
    assert(false);
}
void NeuralNet::export_model(const string &outdir){


    for (int i = 0; i < lu.size(); i++){
        lu[i].export_model(outdir + "/lu" + std::to_string(i));
    }

    assert(false);
}

void NeuralNet::print_parameters(ostream &os){
    params_.print(os);
}

void NeuralNet::check_gradient(){
    DenseFeatureExtractor fe("../data/neural_templates/simple.md");
    NeuralNetParameters p;
    NeuralNet nn(20, &fe, p);
    vector<int> features(fe.n_templates());
    for (int i = 0; i < features.size(); i++){
        features[i] = i;
    }
    vector<bool> allowed(20, true);
    nn.gradient_check(features, allowed, 0, 1e-6);
}






























