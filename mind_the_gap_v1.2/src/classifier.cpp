#include "classifier.h"
#include "random_utils.h"


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

void Classifier::export_classifier_id(const string &outdir){
    ofstream out(outdir + "/classifier_id_classes");
    out << get_id() << endl;
    out << N_ << endl;
    out.close();
}


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
    case RNN:
        return new Rnn(n_classes, outdir);
    case RNN_LABEL_STRUCTURE:
        return new RnnStructureLabel(n_classes, outdir, false);
    case RNN_LABEL_STRUCTURE_LEX:
        return new RnnStructureLabel(n_classes, outdir, true);
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
    export_classifier_id(outdir);

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

CharRnnParameters::CharRnnParameters():dim_char(16), dim_char_based_embeddings(32), crnn(0){}

RnnParameters::RnnParameters()
    : cell_type(RecurrentLayerWrapper::GRU),
      depth(2),
      hidden_size(64),
      features(2),
      //char_rnn_feature_extractor(false),
      auxiliary_task(false),
      auxiliary_task_max_target(0){};


NeuralNetParameters::NeuralNetParameters():
    learning_rate(0.02),
    decrease_constant(1e-6),
    clip_value(10.0),
    gaussian_noise_eta(0.1),
    gaussian_noise(false),
    gradient_clipping(false),
    soft_clipping(false),
    rnn_feature_extractor(false),
    header{"word", "tag"}{}

void NeuralNetParameters::print(ostream &os){
    os << "learning rate\t"       << learning_rate << endl;
    os << "decrease constant\t"   << decrease_constant << endl;
    os << "gradient clipping\t"   << gradient_clipping << endl;
    os << "clip value\t"          << clip_value << endl;
    os << "gaussian noise\t"      << gaussian_noise << endl;
    os << "gaussian noise eta\t"  << gaussian_noise_eta << endl;
    os << "hidden layers\t"       << topology.n_hidden_layers << endl;
    os << "size hidden layers\t"  << topology.size_hidden_layers << endl;
    os << "embedding sizes\t";
    for (int &i : topology.embedding_size_type){
        os << " " << i;
    } os << endl;
    os << "bi-rnn\t" << rnn_feature_extractor << endl;
    os << "cell type\t" << rnn.cell_type << endl;
    os << "rnn depth\t" << rnn.depth << endl;
    os << "rnn state size\t" << rnn.hidden_size << endl;
    os << "number of token feature (rnn)\t" << rnn.features <<endl;
    os << "char rnn\t" << rnn.crnn.crnn << endl;
    os << "char embedding size\t" << rnn.crnn.dim_char << endl;
    os << "char based embedding size\t" << rnn.crnn.dim_char_based_embeddings << endl;
    os << "auxiliary task\t" << rnn.auxiliary_task << endl;
    os << "auxiliary task max idx\t" << rnn.auxiliary_task_max_target << endl;
    os << "voc sizes\t";
    for (int &i : voc_sizes){
        os << " " << i;
    } os << endl;
}

void NeuralNetParameters::read_option_file(const string &filename, NeuralNetParameters &p){
    enum {CHECK_VALUE, LEARNING_RATE, DECREASE_CONSTANT,
          GRADIENT_CLIPPING, CLIP_VALUE, GAUSSIAN_NOISE,
          HIDDEN_LAYERS, SIZE_HIDDEN, EMBEDDING_SIZE,
          BI_RNN,
          RNN_CELL_TYPE, RNN_DEPTH, RNN_STATE_SIZE, RNN_FEATURE,
          CHAR_BIRNN, CHAR_EMBEDDING_SIZE, CHAR_BASED_EMBEDDING_SIZE,
          GAUSSIAN_NOISE_ETA,
         AUX_TASK, AUX_TASK_IDX,
         VOC_SIZES};
    unordered_map<string,int> dictionary{
        {"learning rate", LEARNING_RATE},
        {"decrease constant", DECREASE_CONSTANT},
        {"gradient clipping", GRADIENT_CLIPPING},
        {"clip value", CLIP_VALUE},
        {"gaussian noise", GAUSSIAN_NOISE},
        {"hidden layers", HIDDEN_LAYERS},
        {"size hidden layers", SIZE_HIDDEN},
        {"embedding sizes", EMBEDDING_SIZE},
        {"bi-rnn", BI_RNN},
        {"cell type", RNN_CELL_TYPE},
        {"rnn depth", RNN_DEPTH},
        {"rnn state size", RNN_STATE_SIZE},
        {"number of token feature (rnn)", RNN_FEATURE},
        {"char rnn", CHAR_BIRNN},
        {"char embedding size", CHAR_EMBEDDING_SIZE},
        {"char based embedding size", CHAR_BASED_EMBEDDING_SIZE},
        {"gaussian noise eta", GAUSSIAN_NOISE_ETA},
        {"auxiliary task", AUX_TASK},
        {"auxiliary task max idx", AUX_TASK_IDX},
        {"voc sizes", VOC_SIZES}
    };
    ifstream is(filename);
    string buffer;
    vector<string> tokens;
    while (getline(is,buffer)){
        str::split(buffer, "\t", "", tokens);
        if (tokens.size() == 2){
            int id = dictionary[tokens[0]];
            assert(id != CHECK_VALUE);
            switch (id){
            case LEARNING_RATE:     p.learning_rate = stod(tokens[1]);              break;
            case DECREASE_CONSTANT: p.decrease_constant = stod(tokens[1]);          break;
//            case GRADIENT_CLIPPING: p.soft_clipping = stoi(tokens[1]);              break;
            case GRADIENT_CLIPPING: p.gradient_clipping = stoi(tokens[1]);          break;
            case CLIP_VALUE:        p.clip_value = stod(tokens[1]);                 break;
            case GAUSSIAN_NOISE:    p.gaussian_noise = stoi(tokens[1]);             break;
            case GAUSSIAN_NOISE_ETA:p.gaussian_noise_eta = stod(tokens[1]);         break;
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
            case VOC_SIZES:{
                vector<string> sizes;
                str::split(tokens[1], " ", "", sizes);
                p.voc_sizes.clear();
                for (string &s : sizes){
                    p.voc_sizes.push_back(stoi(s));
                }
                break;
            }
            case BI_RNN: p.rnn_feature_extractor = stoi(tokens[1]);     break;
            case RNN_CELL_TYPE: p.rnn.cell_type = stoi(tokens[1]);      break;
            case RNN_DEPTH: p.rnn.depth = stoi(tokens[1]);              break;
            case RNN_STATE_SIZE: p.rnn.hidden_size = stoi(tokens[1]);   break;
            case RNN_FEATURE: p.rnn.features = stoi(tokens[1]);         break;
            case CHAR_BIRNN: p.rnn.crnn.crnn = stoi(tokens[1]);         break;
            case CHAR_EMBEDDING_SIZE: p.rnn.crnn.dim_char = stoi(tokens[1]);      break;
            case CHAR_BASED_EMBEDDING_SIZE: p.rnn.crnn.dim_char_based_embeddings = stoi(tokens[1]); break;
            case AUX_TASK: p.rnn.auxiliary_task = stoi(tokens[1]);                break;
            case AUX_TASK_IDX: p.rnn.auxiliary_task_max_target = stoi(tokens[1]); break;
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
    FeatureExtractor *fe = FeatureExtractor::import_model(outdir);

    params_.print(cerr);
    NeuralNetParameters::read_option_file(outdir + "/network_info", params_);
    params_.print(cerr);

    init_feature_types_and_lu(fe);

    for (int i = 0; i < lu.size(); i++){
        lu[i].load(outdir + "/lu" + std::to_string(i));
    }

    initialize_network();
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->load(outdir+"/parameters" + std::to_string(i));
    }
}

NeuralNet::NeuralNet(int n_classes) : Classifier(n_classes){}

NeuralNet::NeuralNet(int n_classes, FeatureExtractor *fe, NeuralNetParameters &params) :
        Classifier(n_classes),
        params_(params){

    init_feature_types_and_lu(fe);

    lu[enc::CAT] = LookupTable(enc::hodor.size(enc::CAT), params_.topology.embedding_size_type[enc::CAT]);
    cerr << "Lookup table " << enc::CAT << " has " << params_.topology.embedding_size_type[enc::CAT] << " dimensions (vocsize = " << enc::hodor.size(enc::CAT) << ")" << endl;
    for (int i = 1; i < lu.size(); i++){
        lu[i] = LookupTable(params_.voc_sizes[i], params_.topology.embedding_size_type[i]);
        cerr << "Lookup table " << i << " has " << params_.topology.embedding_size_type[i]
                << " dimensions (vocsize = " << params_.voc_sizes[i] << ")"
                << " hodor size = " << enc::hodor.size(i) << endl;
    }
    initialize_network();
}

void NeuralNet::init_feature_types_and_lu(FeatureExtractor *fe){

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
        lu[i].update(lr, T_, params_.clip_value, params_.gradient_clipping, params_.gaussian_noise, params_.gaussian_noise_eta);
    }
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->update(lr, T_, params_.clip_value, params_.gradient_clipping, params_.gaussian_noise, params_.gaussian_noise_eta);
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
    cerr << endl;
}
void NeuralNet::export_model(const string &outdir){

    export_classifier_id(outdir);

    ofstream os(outdir + "/network_info");
    params_.print(os);
    os.close();

    for (int i = 0; i < lu.size(); i++){
        lu[i].export_model(outdir + "/lu" + std::to_string(i));
    }

    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->export_model(outdir+"/parameters" + std::to_string(i));
    }
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





/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

CharBiRnnFeatureExtractor::CharBiRnnFeatureExtractor(){}
CharBiRnnFeatureExtractor::CharBiRnnFeatureExtractor(CharRnnParameters *nn_parameters)
    : params(nn_parameters){
    encoder = SequenceEncoder(nn_parameters->crnn);
    vector<int> input_sizes{params->dim_char};

    int cell_type = RecurrentLayerWrapper::LSTM;
    int hidden_size = params->dim_char_based_embeddings;

    // 2 layers only
    layers.push_back(shared_ptr<RecurrentLayerWrapper>(new RecurrentLayerWrapper(cell_type, input_sizes, hidden_size)));
    layers.push_back(shared_ptr<RecurrentLayerWrapper>(new RecurrentLayerWrapper(cell_type, input_sizes, hidden_size)));

    for (int i = 0; i < layers.size(); i++){
        for (int j = 0; j < layers[i]->size(); j++){
            (*layers[i])[j]->get_params(parameters);
        }
    }
}

CharBiRnnFeatureExtractor::~CharBiRnnFeatureExtractor(){}

void CharBiRnnFeatureExtractor::precompute_lstm_char(){
    cerr << "Precomputing char-lstm for known words" << endl;
    vector<shared_ptr<Node>> fake_buffer; // contain the list of tokens in vocabulary
    for (STRCODE i = 0; i < enc::hodor.size(enc::TOK); i++){
        const vector<STRCODE> morph{i};
        fake_buffer = {shared_ptr<Node>(new Leaf(i,i,morph))};

        build_computation_graph(fake_buffer);
        fprop();

        vector<Vec> pair{*(states[0][0].back()->v()), *(states[0][1].front()->v())};
        precomputed_embeddings.push_back(pair);
    }


//    for (STRCODE i = 0; i < enc::hodor.size(enc::TOK); i++){
//        vector<Vec> pair{*(states[i][0].back()->v()), *(states[i][1].front()->v())};
//        precomputed_embeddings.push_back(pair);
//    }
    cerr << "Precomputing char-lstm for known words: done" << endl;
//    input.clear();
//    states.clear();
//    init_nodes.clear();
}

bool CharBiRnnFeatureExtractor::has_precomputed(){
    return precomputed_embeddings.size() > 0;
}

void CharBiRnnFeatureExtractor::init_encoders(){
    encoder.init();
    lu = LookupTable(encoder.char_voc_size(), params->dim_char);
}


void CharBiRnnFeatureExtractor::build_computation_graph(vector<shared_ptr<Node>> &buffer){

    input = vector<NodeMatrix>(buffer.size());
    states.resize(input.size());

    init_nodes.clear();
    for (int depth = 0; depth < 2; depth++){
        add_init_node(depth);
    }

    for (int w = 0; w < input.size(); w++){
        STRCODE tokcode = buffer[w]->get_field(Leaf::FIELD_TOK);

        // If a precomputed vector is available
        if (tokcode < precomputed_embeddings.size()){
            shared_ptr<AbstractNeuralNode> forwardnode(new ConstantNode(&precomputed_embeddings[tokcode][0]));
            shared_ptr<AbstractNeuralNode> backwardnode(new ConstantNode(&precomputed_embeddings[tokcode][1]));
            vector<shared_ptr<AbstractNeuralNode>> forward{forwardnode};
            vector<shared_ptr<AbstractNeuralNode>> backward{backwardnode};
            states[w] = {forward, backward};
        }else{
            vector<int> *sequence = encoder(tokcode);
            for (int c = 0; c < sequence->size(); c++){
                shared_ptr<VecParam> e;
                lu.get(sequence->at(c), e);
                vector<shared_ptr<AbstractNeuralNode>> proxy{shared_ptr<AbstractNeuralNode>(new LookupNode(*e))};
                input[w].push_back(proxy);
            }

            states[w] = {vector<shared_ptr<AbstractNeuralNode>>(sequence->size()),
                         vector<shared_ptr<AbstractNeuralNode>>(sequence->size())};

            int depth = 0;
            states[w][depth][0] = shared_ptr<AbstractNeuralNode>(
                        new LstmNode(params->dim_char_based_embeddings,
                                     init_nodes[depth],
                                     input[w][0],*layers[depth]));

            for (int c = 1; c < sequence->size(); c++){
                states[w][depth][c] = shared_ptr<AbstractNeuralNode>(
                            new LstmNode(params->dim_char_based_embeddings,
                                         states[w][depth][c-1],
                            input[w][c], *layers[depth]));
            }
            depth = 1;
            states[w][depth].back() = shared_ptr<AbstractNeuralNode>(
                        new LstmNode(params->dim_char_based_embeddings,
                                     init_nodes[depth],
                                     input[w].back(), *layers[depth]));

            for (int c = sequence->size()-2; c >= 0; c--){
                states[w][depth][c] = shared_ptr<AbstractNeuralNode>(
                            new LstmNode(params->dim_char_based_embeddings,
                                         states[w][depth][c+1],
                            input[w][c], *layers[depth]));
            }
        }
    }
}


void CharBiRnnFeatureExtractor::add_init_node(int depth){
    shared_ptr<ParamNode> init11(new ParamNode(params->dim_char_based_embeddings, (*layers[depth])[GruNode::INIT2]));
    shared_ptr<AbstractNeuralNode> init1(new MemoryNodeInitial(
                                             params->dim_char_based_embeddings,
                                             (*layers[depth])[GruNode::INIT1],
                                         init11));
    init_nodes.push_back(init1);
}

void CharBiRnnFeatureExtractor::fprop(){
    for (int i = 0; i < init_nodes.size(); i++){
        init_nodes[i]->fprop();
    }
    for (int w = 0; w < states.size(); w++){
        for (int c = 0; c < states[w][0].size(); c++){
            states[w][0][c]->fprop();
        }
        for (int c = states[w][1].size() -1; c >= 0; c--){
            states[w][1][c]->fprop();
        }
    }
}

void CharBiRnnFeatureExtractor::bprop(){
    for (int w = 0; w < states.size(); w++){
        for (int c = states[w][0].size() -1; c >= 0; c--){
            states[w][0][c]->bprop();
        }
        for (int c = 0; c < states[w][1].size(); c++){
            states[w][1][c]->bprop();
        }
    }
    for (int i = 0; i < init_nodes.size(); i++){
        init_nodes[i]->bprop();
    }
}

void CharBiRnnFeatureExtractor::update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->update(lr, T, clip, clipping, gaussian, gaussian_eta);
    }
    lu.update(lr, T, clip, clipping, gaussian, gaussian_eta);
}

double CharBiRnnFeatureExtractor::gradient_squared_norm(){
    double gsn = 0;
    for (int i = 0; i < parameters.size(); i++){
        gsn += parameters[i]->gradient_squared_norm();
    }
    gsn += lu.gradient_squared_norm();
    return gsn;
}

void CharBiRnnFeatureExtractor::scale_gradient(double scale){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->scale_gradient(scale);
    }
    lu.scale_gradient(scale);
}


void CharBiRnnFeatureExtractor::operator()(int i, vector<shared_ptr<AbstractNeuralNode>> &output){
    assert( i >= 0 && i < size() );
    output = {states[i][0].back(), states[i][1].front()};
}

int CharBiRnnFeatureExtractor::size(){
    assert(input.size() == states.size());
    return input.size();
}

void CharBiRnnFeatureExtractor::copy_encoders(CharBiRnnFeatureExtractor &other){
    lu = other.lu;
    encoder = other.encoder;
}

void CharBiRnnFeatureExtractor::assign_parameters(CharBiRnnFeatureExtractor &other){
    assert(parameters.size() == other.parameters.size());
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->assign(other.parameters[i]);
    }
}

void CharBiRnnFeatureExtractor::average_weights(int T){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->average(T);
    }
    lu.average(T);
}
void CharBiRnnFeatureExtractor::get_parameters(vector<shared_ptr<Parameter>> &weights){ // REMINDER: this is used by gradient checker
    weights.insert(weights.end(), parameters.begin(), parameters.end());
    lu.get_active_params(weights);
}

void CharBiRnnFeatureExtractor::export_model(const string &outdir){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->export_model(outdir+"/char_rnn_parameters" + std::to_string(i));
    }
    lu.export_model(outdir+"/lu_char_rnn");
}

void CharBiRnnFeatureExtractor::load_parameters(const string &outdir){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->load(outdir+"/char_rnn_parameters" + std::to_string(i));
    }
    lu.clear();
    lu.load(outdir+"/lu_char_rnn");
}

void CharBiRnnFeatureExtractor::reset_gradient_history(){
    lu.reset_gradient_history();
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->reset_gradient_history();
    }
}



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////





BiRnnFeatureExtractor::BiRnnFeatureExtractor():train_time(false), parse_time(false){}
BiRnnFeatureExtractor::BiRnnFeatureExtractor(NeuralNetParameters *nn_parameters,
                      vector<LookupTable> *lookup)
    :lu(lookup), params(nn_parameters), aux_start(0), aux_end(0), train_time(false), parse_time(false){

    vector<int> input_sizes;

    if (params->rnn.crnn.crnn > 0){
        input_sizes.push_back(params->rnn.crnn.dim_char_based_embeddings);
        input_sizes.push_back(params->rnn.crnn.dim_char_based_embeddings);
    }
    for (int i = 0; i < params->rnn.features; i++){
        input_sizes.push_back(params->topology.embedding_size_type[i+1]);
    }

    layers.push_back(shared_ptr<RecurrentLayerWrapper>(new RecurrentLayerWrapper(params->rnn.cell_type, input_sizes, params->rnn.hidden_size)));
    layers.push_back(shared_ptr<RecurrentLayerWrapper>(new RecurrentLayerWrapper(params->rnn.cell_type, input_sizes, params->rnn.hidden_size)));
    for (int i = 2; i < params->rnn.depth; i++){
        vector<int> prec_layer_sizes{params->rnn.hidden_size, params->rnn.hidden_size};
        layers.push_back(shared_ptr<RecurrentLayerWrapper>(new RecurrentLayerWrapper(params->rnn.cell_type, prec_layer_sizes, params->rnn.hidden_size)));
    }

    for (int i = 0; i < layers.size(); i++){
        for (int j = 0; j < layers[i]->size(); j++){
            (*layers[i])[j]->get_params(parameters);
        }
    }

    out_of_bounds = Vec::Zero(params->rnn.hidden_size);
    out_of_bounds_d = Vec::Zero(params->rnn.hidden_size);

    if (params->rnn.crnn.crnn > 0){
        char_rnn = CharBiRnnFeatureExtractor(& params->rnn.crnn);
        char_rnn.init_encoders();
    }


    if (params->rnn.auxiliary_task){
        aux_start = params->rnn.features;
        aux_end = params->rnn.auxiliary_task_max_target;
        auxiliary_layers.resize(aux_end - aux_start);
        aux_output_sizes.resize(aux_end - aux_start);
        vector<int> input_sizes{params->rnn.hidden_size, params->rnn.hidden_size};
        for (int i = aux_start; i < aux_end; i++){
            int output_size = params->voc_sizes[i+1]; // +1 -> 0 is non terminals
            aux_output_sizes[i-aux_start] = output_size;
            auxiliary_layers[i-aux_start] ={
                //shared_ptr<Layer>(new MultipleLinearLayer(2, input_sizes, AUX_HIDDEN_LAYER_SIZE)),
                //shared_ptr<Layer>(new ReLU()),
                //shared_ptr<Layer>(new AffineLayer(AUX_HIDDEN_LAYER_SIZE, output_size)),
                shared_ptr<Layer>(new MultipleLinearLayer(2, input_sizes, output_size)),
                shared_ptr<Layer>(new Softmax())
            };
            for (int j = 0; j < auxiliary_layers[i-aux_start].size(); j++){
                auxiliary_layers[i-aux_start][j]->get_params(auxiliary_parameters);
                //auxiliary_layers[i-aux_start][1]->get_params(auxiliary_parameters);
            }
        }
    }
}


BiRnnFeatureExtractor::~BiRnnFeatureExtractor(){}

void BiRnnFeatureExtractor::precompute_char_lstm(){
    parse_time = true;
    char_rnn.precompute_lstm_char();
}

void BiRnnFeatureExtractor::build_computation_graph(vector<shared_ptr<Node>> &buffer, bool aux_task){

    if (params->rnn.crnn.crnn > 0){
        char_rnn.build_computation_graph(buffer);
    }

    int add_features = (params->rnn.crnn.crnn > 0) ? 2 : 0;

    input = NodeMatrix(
                buffer.size(),
                vector<shared_ptr<AbstractNeuralNode>>(
                    params->rnn.features + add_features,
                    nullptr));  // +2 if char rnn

    for (int i = 0; i < buffer.size(); i++){

        if (params->rnn.crnn.crnn > 0){
            vector<shared_ptr<AbstractNeuralNode>> char_based_embeddings;
            char_rnn(i, char_based_embeddings);
            assert(char_based_embeddings.size() == 2);
            input[i][0] = char_based_embeddings[0];
            input[i][1] = char_based_embeddings[1];
        }

        shared_ptr<VecParam> e;
        for (int f = 0; f < params->rnn.features; f++){
            STRCODE word_code = buffer[i]->get_field(f);
            if (train_time && f == 0 && word_code != enc::UNDEF){ // 2% unknown words   --> won't work unless prob depends on frequency
                assert(word_code != enc::UNKNOWN);
                double threshold = 0.8375 / (0.8375 + enc::hodor.get_freq(word_code));
                if (rd::random() < threshold){
                    word_code = enc::UNKNOWN;
                }
            }

            (*lu)[f+1].get(word_code, e);
            input[i][f+add_features] = shared_ptr<AbstractNeuralNode>(new LookupNode(*e));
        }
    }

    int aux_depth = aux_task ? 2 : params->rnn.depth;

    //states.resize(params->rnn.depth);
    states.resize(aux_depth);
    for (int d = 0; d < states.size(); d++){
        states[d].resize(buffer.size());
    }

    init_nodes.clear();
    for (int i = 0; i < aux_depth; i++){
        add_init_node(i);
    }

    int depth = 0;
    states[depth][0]=  shared_ptr<AbstractNeuralNode>(get_recurrent_node(init_nodes[depth], input[0], *layers[depth]));
    for (int i = 1; i < buffer.size(); i++){
        states[depth][i]= shared_ptr<AbstractNeuralNode>(get_recurrent_node(states[depth][i-1], input[i], *layers[depth]));
    }

    depth = 1;
    states[depth].back() = shared_ptr<AbstractNeuralNode>(get_recurrent_node(init_nodes[depth], input.back(), *layers[depth]));
    for (int i = buffer.size()-2; i >=0 ; i--){
        states[depth][i] = shared_ptr<AbstractNeuralNode>(get_recurrent_node(states[depth][i+1], input[i], *layers[depth]));
    }

    if (! aux_task){
        for (depth = 2; depth < params->rnn.depth; depth++){
            if (depth % 2 == 0){
                vector<shared_ptr<AbstractNeuralNode>> rnn_in{states[depth-1][0], states[depth-2][0]};
                states[depth][0] = shared_ptr<AbstractNeuralNode>(get_recurrent_node(init_nodes[depth], rnn_in, *layers[depth]));
                for (int i = 1; i < buffer.size(); i++){
                    rnn_in = {states[depth-1][i], states[depth-2][i]};
                    states[depth][i]= shared_ptr<AbstractNeuralNode>(get_recurrent_node(states[depth][i-1], rnn_in, *layers[depth]));
                }
            }else{
                vector<shared_ptr<AbstractNeuralNode>> rnn_in{states[depth-2].back(), states[depth-3].back()};
                states[depth].back() = shared_ptr<AbstractNeuralNode>(get_recurrent_node(init_nodes[depth], rnn_in, *layers[depth]));
                for (int i = buffer.size()-2; i >=0 ; i--){
                    rnn_in = {states[depth-2][i], states[depth-3][i]};
                    states[depth][i] = shared_ptr<AbstractNeuralNode>(get_recurrent_node(states[depth][i+1], rnn_in, *layers[depth]));
                }
            }
        }
    }
}



void BiRnnFeatureExtractor::add_init_node(int depth){
    switch(params->rnn.cell_type){
    case RecurrentLayerWrapper::GRU:
    case RecurrentLayerWrapper::LSTM:{
        shared_ptr<ParamNode> init11(new ParamNode(params->rnn.hidden_size, (*layers[depth])[GruNode::INIT2]));
        shared_ptr<AbstractNeuralNode> init1(new MemoryNodeInitial(
                                                 params->rnn.hidden_size,
                                                 (*layers[depth])[GruNode::INIT1],
                                                  init11));
        init_nodes.push_back(init1);
        break;
    }
    case RecurrentLayerWrapper::RNN:{
        shared_ptr<AbstractNeuralNode> init(new ParamNode(params->rnn.hidden_size, (*layers[depth])[RnnNode::INIT]));
        init_nodes.push_back(init);
        break;
    }
    default:
        assert(false && "Not Implemented error or unknown rnn cell type");
    }
}

AbstractNeuralNode* BiRnnFeatureExtractor::get_recurrent_node(
        shared_ptr<AbstractNeuralNode> &pred,
        vector<shared_ptr<AbstractNeuralNode> > &input_nodes,
        RecurrentLayerWrapper &l){

    switch(params->rnn.cell_type){
    case RecurrentLayerWrapper::GRU:
        return new GruNode(params->rnn.hidden_size, pred, input_nodes, l);
    case RecurrentLayerWrapper::RNN:
        return new RnnNode(params->rnn.hidden_size, pred, input_nodes, l);
    case RecurrentLayerWrapper::LSTM:
        return new LstmNode(params->rnn.hidden_size, pred, input_nodes, l);
    default:
        assert(false);
    }
    assert(false);
    return nullptr;
}


void BiRnnFeatureExtractor::fprop(){
    if (params->rnn.crnn.crnn > 0){
        char_rnn.fprop();
    }
    for (int i = 0; i < init_nodes.size(); i++){
        init_nodes[i]->fprop();
    }

    for (int d = 0; d < states.size(); d++){
        if (d % 2 == 0){
            for (int i = 0; i < states[d].size(); i++){
                states[d][i]->fprop();
            }
        }else{
            for (int i = states[d].size() -1; i >= 0; i--){
                states[d][i]->fprop();
            }
        }
    }
}

void BiRnnFeatureExtractor::bprop(){
    for (int d = states.size()-1; d >= 0; d--){
        if (d % 2 == 0){
            for (int i = states[d].size() -1; i >= 0; i--){
                states[d][i]->bprop();
            }
        }else{
            for (int i = 0; i < states[d].size(); i++){
                states[d][i]->bprop();
            }
        }
    }
    for (int i = 0; i < init_nodes.size(); i++){
        init_nodes[i]->bprop();
    }
    if (params->rnn.crnn.crnn > 0){
        char_rnn.bprop();
    }
}

void BiRnnFeatureExtractor::update(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->update(lr, T, clip, clipping, gaussian, gaussian_eta);
    }
    if (params->rnn.crnn.crnn > 0){
        char_rnn.update(lr, T, clip, clipping, gaussian, gaussian_eta);
    }
}

double BiRnnFeatureExtractor::gradient_squared_norm(){
    double gsn = 0;
    for (int i = 0; i < parameters.size(); i++){
        gsn += parameters[i]->gradient_squared_norm();
    }
    if (params->rnn.crnn.crnn > 0){
        gsn += char_rnn.gradient_squared_norm();
    }
    for (int i = 0; i < auxiliary_parameters.size(); i++){
        gsn += auxiliary_parameters[i]->gradient_squared_norm();
    }
    return gsn;
}

void BiRnnFeatureExtractor::scale_gradient(double scale){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->scale_gradient(scale);
    }
    for (int i = 0; i < auxiliary_parameters.size(); i++){
        auxiliary_parameters[i]->scale_gradient(scale);
    }
    char_rnn.scale_gradient(scale);
}


void BiRnnFeatureExtractor::operator()(int i, vector<Vec*> &data, vector<Vec*> &data_grad){
    if (i >= 0 && i < size()){
        int j = params->rnn.depth - 2;
        assert((j+2) == states.size());
        data.push_back(states[j][i]->v());
        data_grad.push_back(states[j][i]->d());
        data.push_back(states[j+1][i]->v());
        data_grad.push_back(states[j+1][i]->d());
    }else{
        // TODO: find cleverer way (use start / stop symbols ??)
        for (int d = 0; d < 2; d++){
            data.push_back(&out_of_bounds);
            data_grad.push_back(&out_of_bounds_d);
        }
    }
}
int BiRnnFeatureExtractor::size(){
    assert( states.size() > 0 );
    assert(input.size() == states[0].size());
    return input.size();
}

void BiRnnFeatureExtractor::assign_parameters(BiRnnFeatureExtractor &other){
    assert(parameters.size() == other.parameters.size());
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->assign(other.parameters[i]);
    }
    for (int i = 0; i < auxiliary_parameters.size(); i++){
        auxiliary_parameters[i]->assign(other.auxiliary_parameters[i]);
    }
}

void BiRnnFeatureExtractor::copy_char_birnn(BiRnnFeatureExtractor &other){
    if (params->rnn.crnn.crnn > 0){
        char_rnn.copy_encoders(other.char_rnn);
        char_rnn.assign_parameters(other.char_rnn);
    }
}

void BiRnnFeatureExtractor::average_weights(int T){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->average(T);
    }
    if (params->rnn.crnn.crnn > 0){
        char_rnn.average_weights(T);
    }
    if (params->rnn.auxiliary_task){
        for (int i = 0; i < auxiliary_parameters.size(); i++){
            auxiliary_parameters[i]->average(T);
        }
    }
}

void BiRnnFeatureExtractor::get_parameters(vector<shared_ptr<Parameter>> &weights){
    weights.insert(weights.end(), parameters.begin(), parameters.end());
    if (params->rnn.crnn.crnn){
        char_rnn.get_parameters(weights);
    }
}

void BiRnnFeatureExtractor::export_model(const string &outdir){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->export_model(outdir+"/rnn_parameters" + std::to_string(i));
    }
    if (params->rnn.crnn.crnn){
        char_rnn.export_model(outdir);
    }
    for (int i = 0; i < auxiliary_parameters.size(); i++){
        auxiliary_parameters[i]->export_model(outdir+"/rnn_aux_parameters" + std::to_string(i));
    }
}

void BiRnnFeatureExtractor::load_parameters(const string &outdir){
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->load(outdir+"/rnn_parameters" + std::to_string(i));
    }
    if (params->rnn.crnn.crnn){
        char_rnn.load_parameters(outdir);
    }
    for (int i = 0; i < auxiliary_parameters.size(); i++){
        auxiliary_parameters[i]->load(outdir+"/rnn_aux_parameters" + std::to_string(i));
    }
}

void BiRnnFeatureExtractor::auxiliary_task_summary(ostream &os){
    os << "Auxiliary tasks summary:" << endl;
    for (int i = 0; i < aux_output_sizes.size(); i++){
        os << "    task " << i << ": output size = " << aux_output_sizes[i] << endl;
    }
}


void BiRnnFeatureExtractor::add_aux_graph(vector<shared_ptr<Node> > &buffer, bool aux_only=true){
    build_computation_graph(buffer, aux_only);
    assert(input.size() == buffer.size());
    auxiliary_output_nodes.resize(input.size());

    for (int i = 0; i < buffer.size(); i++){
        int depth = 0;  /// params->rnn.depth - 2;  -> supervise lower tasks at lower layers
        vector<shared_ptr<AbstractNeuralNode>> input_nodes = {
                states[depth][i],
                states[depth+1][i]};
        auxiliary_output_nodes[i].resize(aux_end - aux_start);

        for (int j = 0; j < aux_end - aux_start; j++){

            auxiliary_output_nodes[i][j].clear();
            // Use an individual hidden layer for each aux task (--> probably better without this
//            auxiliary_output_nodes[i][j].push_back(
//                        shared_ptr<AbstractNeuralNode>(
//                            new ComplexNode(AUX_HIDDEN_LAYER_SIZE, //aux_output_sizes[j],
//                                           auxiliary_layers[j][0].get(), input_nodes)));  // size / layer / vector input
//            auxiliary_output_nodes[i][j].push_back(
//                        shared_ptr<AbstractNeuralNode>(
//                            new SimpleNode(AUX_HIDDEN_LAYER_SIZE, //aux_output_sizes[j],
//                                           auxiliary_layers[j][1].get(), auxiliary_output_nodes[i][j][0])));
//            auxiliary_output_nodes[i][j].push_back(
//                        shared_ptr<AbstractNeuralNode>(
//                            new SimpleNode(aux_output_sizes[j],
//                                           auxiliary_layers[j][2].get(), auxiliary_output_nodes[i][j][1])));
//            auxiliary_output_nodes[i][j].push_back(
//                        shared_ptr<AbstractNeuralNode>(
//                            new SimpleNode(aux_output_sizes[j],
//                                           auxiliary_layers[j][3].get(), auxiliary_output_nodes[i][j][2])));
            auxiliary_output_nodes[i][j].push_back(
                        shared_ptr<AbstractNeuralNode>(
                            new ComplexNode(aux_output_sizes[j],
                                            auxiliary_layers[j][0].get(), input_nodes)));  // size / layer / vector input
            auxiliary_output_nodes[i][j].push_back(
                        shared_ptr<AbstractNeuralNode>(
                            new SimpleNode(aux_output_sizes[j],
                                           auxiliary_layers[j][1].get(), auxiliary_output_nodes[i][j][0])));
        }
    }

    aux_targets = vector<vector<int>>(buffer.size());
    for (int i = 0; i < buffer.size(); i++){
        aux_targets[i].resize(aux_end - aux_start);
        for (int j = 0; j < aux_end - aux_start; j++){
            if (buffer[i]->n_fields() > (j+aux_start)){
                aux_targets[i][j] = buffer[i]->get_field(j + aux_start);
                if (aux_targets[i][j] >= aux_output_sizes[j]){
//                    cerr << (j+aux_start) << endl;
//                    cerr << "unknown:" << enc::hodor.decode(aux_targets[i][j], j + aux_start + 1) << endl;
//                    for (int k = 0; k < buffer[i]->n_fields(); k++){
//                        cerr << enc::hodor.decode(buffer[i]->get_field(k), k+1) << " ";
//                    }cerr << endl;
                    aux_targets[i][j] = enc::UNKNOWN;
                }
            }
        }
    }
}

void BiRnnFeatureExtractor::fprop_aux(){
    fprop();
    for (int i = 0; i < auxiliary_output_nodes.size(); i++){
        for (int j = 0; j < auxiliary_output_nodes[i].size(); j++){
            for (int k = 0; k < auxiliary_output_nodes[i][j].size(); k++){
                auxiliary_output_nodes[i][j][k]->fprop();
            }
        }
    }
}

void BiRnnFeatureExtractor::bprop_aux(){
    for (int i = 0; i < auxiliary_output_nodes.size(); i++){
        //int j = rand() % auxiliary_output_nodes[i].size(); // random auxiliary task
        //std::uniform_int_distribution<int> distribution(0,auxiliary_output_nodes[i].size()-1);
        //int j = distribution(Parameter::random);

        for (int j = 0; j < auxiliary_output_nodes[i].size(); j++){
            auxiliary_layers[j].back()->target = aux_targets[i][j];
            for (int k = auxiliary_output_nodes[i][j].size() - 1; k >= 0; k--){
                auxiliary_output_nodes[i][j][k]->bprop();
            }
        }
    }
    bprop();
}

void BiRnnFeatureExtractor::update_aux(double lr, double T, double clip, bool clipping, bool gaussian, double gaussian_eta){
    double learning_rate_aux = lr;// / aux_output_sizes.size();
    for (int i = 0; i < auxiliary_parameters.size(); i++){
        auxiliary_parameters[i]->update(learning_rate_aux, T, clip, clipping, gaussian, gaussian_eta);
    }
    for (int i = 0; i < lu->size(); i++){
        lu->at(i).update(learning_rate_aux, T, clip, clipping, gaussian, gaussian_eta);
    }
    update(learning_rate_aux, T, clip, clipping, gaussian, gaussian_eta);
}

void BiRnnFeatureExtractor::eval_aux(AuxiliaryTaskEvaluator &evaluator){
    evaluator.total += auxiliary_output_nodes.size();
    for (int i = 0; i < auxiliary_output_nodes.size(); i++){
        bool all_good = true;
        for (int k = 0; k < auxiliary_output_nodes[i].size(); k++){
            Vec* output = auxiliary_output_nodes[i][k].back()->v();
            int argmax;
            output->maxCoeff(&argmax);
            //cerr << enc::hodor.decode(argmax, enc::TAG) << " " << enc::hodor.decode(aux_targets[i][k], enc::TAG) << endl;
            assert(argmax < aux_output_sizes[k]);
            if (argmax == aux_targets[i][k]){
                evaluator.good[k] ++;
            }else{
                all_good = false;
            }
        }
        if (all_good){
            evaluator.complete_match ++;
        }
    }
    //cerr << endl;
}

void BiRnnFeatureExtractor::assign_deplabels(vector<shared_ptr<Node>> &buffer, int deplabel_id){
    int task_id = deplabel_id - aux_start;
    for (int i = 0; i < buffer.size(); i++){
        Vec* output = auxiliary_output_nodes[i][task_id].back()->v();
        int argmax;
        output->maxCoeff(&argmax);
        //cerr << "Assigning label " << argmax << "   " << enc::hodor.decode(argmax, deplabel_id+1) << endl;
        buffer[i]->set_dlabel(argmax);
    }
}

void BiRnnFeatureExtractor::assign_tags(vector<shared_ptr<Node>> &buffer){
    for (int i = 0; i < buffer.size(); i++){
        Vec* output = auxiliary_output_nodes[i][0].back()->v();  // task 0 is necessarily tag
        int argmax;
        output->maxCoeff(&argmax);
//        if (argmax == enc::UNDEF || argmax == enc::UNKNOWN){
//            cerr << "Assigning " << argmax << "  as tag" << endl;
//            cerr << *buffer[i] << endl;
//        }
        buffer[i]->set_label(enc::hodor.code(enc::hodor.decode(argmax, enc::TAG), enc::CAT));
    }
}

void BiRnnFeatureExtractor::assign_morphological_features(vector<shared_ptr<Node>> &buffer, int deplabel_id){
    for (int i = 0; i < buffer.size(); i++){
        for (int task = 1; task < auxiliary_output_nodes[i].size(); task++){
            if (task != deplabel_id){
                Vec* output = auxiliary_output_nodes[i][task].back()->v();
                int argmax;
                output->maxCoeff(&argmax);
                buffer[i]->set_pred_field(task, argmax);
            }
        }
    }
}

void BiRnnFeatureExtractor::auxiliary_gradient_check(vector<shared_ptr<Node>> &buffer, double epsilon){
    cerr << "Gradient Checking auxiliary task" << endl;

    add_aux_graph(buffer);
    fprop_aux();
    bprop_aux();

    for (int i = 0; i < lu->size(); i++){
        (*lu)[i].get_active_params(auxiliary_parameters);
    }
    get_parameters(auxiliary_parameters);
//    if (params->rnn.char_rnn_feature_extractor){
//        char_rnn.get_parameters(auxiliary_parameters);
//    }
//    auxiliary_parameters.insert(auxiliary_parameters.end(), parameters.begin(), parameters.end());

    for (int i = 0; i < auxiliary_parameters.size(); i++){
        for (int k = 0; k < auxiliary_parameters[i]->size(); k++){
            auxiliary_parameters[i]->add_epsilon(k, epsilon);
            double a = full_fprop_aux(buffer);
            auxiliary_parameters[i]->add_epsilon(k, -epsilon);
            auxiliary_parameters[i]->add_epsilon(k, -epsilon);
            double c = full_fprop_aux(buffer);
            auxiliary_parameters[i]->add_epsilon(k, epsilon);
            auxiliary_parameters[i]->set_empirical_gradient(k, (a-c) / (2 * epsilon));
        }
        cerr << "p[" << i << "] -> " << std::flush;
        auxiliary_parameters[i]->print_gradient_differences();
    }

}

double BiRnnFeatureExtractor::aux_loss(){
    double loss = 0.0;
    for (int i = 0; i < auxiliary_output_nodes.size(); i++){
        for (int j = 0; j < auxiliary_output_nodes[i].size(); j++){
            Vec* v = auxiliary_output_nodes[i][j].back()->v();
            loss += - log((*v)[aux_targets[i][j]]);
        }
    }
    return loss;
}

double BiRnnFeatureExtractor::full_fprop_aux(vector<shared_ptr<Node>> &buffer){
    add_aux_graph(buffer);
    fprop_aux();
    return aux_loss();
}

void BiRnnFeatureExtractor::aux_reset_gradient_history(){
    for (int i = 0; i < auxiliary_parameters.size(); i++){
        auxiliary_parameters[i]->reset_gradient_history();
    }
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->reset_gradient_history();
    }
    for (int i = 0; i < lu->size(); i++){
        (*lu)[i].reset_gradient_history();
    }
    char_rnn.reset_gradient_history();
}

void BiRnnFeatureExtractor::set_train_time(bool b){
    train_time = b;
}

//int BiRnnFeatureExtractor::n_aux_tasks(){
//    return aux_end - aux_start;
//}




/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////


Rnn::Rnn(int n_classes, const string &outdir, bool initialize) : Classifier(n_classes){
    FeatureExtractor *fe = FeatureExtractor::import_model(outdir);

    params_.print(cerr);
    NeuralNetParameters::read_option_file(outdir + "/network_info", params_);
    //params_.rnn.auxiliary_task = false; // WARNING/TODO: temporary workaround
    params_.print(cerr);

    init_feature_types_and_lu(fe);

    for (int i = 0; i < lu.size(); i++){
        lu[i].load(outdir + "/lu" + std::to_string(i));
    }

    if (initialize){
        initialize_network();
        for (int i = 0; i < parameters.size(); i++){
            parameters[i]->load(outdir+"/parameters" + std::to_string(i));
        }
        rnn.load_parameters(outdir);
    }
}



Rnn::Rnn(int n_classes):Classifier(n_classes){}

Rnn::Rnn(int n_classes, FeatureExtractor *fe, NeuralNetParameters &params, bool initialize) :
        Classifier(n_classes),
        params_(params){

    if (params_.rnn.features > params_.header.size()){
        params_.rnn.features = params_.header.size();
    }

    if (params_.rnn.auxiliary_task && params_.rnn.auxiliary_task_max_target > params_.header.size()){
        params_.rnn.auxiliary_task_max_target = params_.header.size();
    }

    fe->get_template_type(feature_types);

    init_feature_types_and_lu(fe);

    lu[enc::CAT] = LookupTable(enc::hodor.size(enc::CAT), params_.topology.embedding_size_type[enc::CAT]);
    cerr << "Lookup table " << enc::CAT << " has " << params_.topology.embedding_size_type[enc::CAT] << " dimensions (vocsize = " << enc::hodor.size(enc::CAT) << ")" << endl;

    for (int i = 1; i < lu.size(); i++){
        lu[i] = LookupTable(params_.voc_sizes[i], params_.topology.embedding_size_type[i]);
        cerr << "Lookup table " << i << " has " << params_.topology.embedding_size_type[i]
                << " dimensions (vocsize = " << params_.voc_sizes[i] << ")"
                << " hodor size = " << enc::hodor.size(i) << endl;
    }

    if (initialize){
    initialize_network();
    }
}

Rnn::~Rnn(){
    for (int i = 0; i < layers.size(); i++){
        delete layers[i];
    }
}

void Rnn::precompute_char_lstm(){
    rnn.precompute_char_lstm();
}

void Rnn::init_feature_types_and_lu(FeatureExtractor *fe){

    fe->get_template_type(feature_types);
    n_features = fe->n_templates();

    int max_type = 0;
    for (int i = 0; i < feature_types.size(); i++){
        if (feature_types[i] > max_type){
            max_type = feature_types[i];
        }
    }
    if (params_.rnn.features > max_type){
        max_type =  params_.rnn.features;
    }
    lu.resize(max_type+1);
    cerr << "Max type id = " << max_type << endl;
}

int Rnn::get_id(){
    return Classifier::RNN;
}


void Rnn::initialize_network(){
    vector<int> embedding_size_token;
    for (int i = 0; i < feature_types.size(); i++){
        if (feature_types[i] == enc::CAT){
            embedding_size_token.push_back(params_.topology.embedding_size_type[enc::CAT]);
        }else{
            for (int d = 0; d < 2; d++){ // magic number: bi-rnn -> concatenate forward and backward embedding
                embedding_size_token.push_back(params_.rnn.hidden_size);
            }
        }
    }

    layers.push_back(new MultipleLinearLayer(embedding_size_token.size(), embedding_size_token, params_.topology.size_hidden_layers));
    layers.push_back(new ReLU());
    for (int i = 1; i < params_.topology.n_hidden_layers; i++){
        layers.push_back(new AffineLayer(params_.topology.size_hidden_layers, params_.topology.size_hidden_layers));
        layers.push_back(new ReLU());
    }
    layers.push_back(new AffineLayer(params_.topology.size_hidden_layers, N_));
    layers.push_back(new SoftmaxFilter());

    for (int i = 0; i < layers.size(); i++){
        layers[i]->get_params(parameters);
    }

    rnn = BiRnnFeatureExtractor(&params_, &lu);
}

void Rnn::run_rnn(vector<shared_ptr<Node>> &buffer){
    rnn.build_computation_graph(buffer);
    rnn.fprop();

    if (params_.rnn.features == 1 && ! params_.rnn.auxiliary_task){
        // Here: tags are not input to the bi-rnn, and not predicted as aux tasks
        // The parser should not be able to use tag information as non-terminal features
        for (int i = 0; i < buffer.size(); i++){
            buffer[i]->set_label(enc::UNKNOWN);
        }
    }
}

void Rnn::flush(){
    t_edata.clear();
    t_edata_grad.clear();
    t_states.clear();
    t_dstates.clear();
    softmax_filters.clear();
}

void Rnn::get_feature(int f, int type, vector<Vec*> &input, vector<Vec*> &dinput){
    if (type == enc::CAT){
        shared_ptr<VecParam> p;
        lu[enc::CAT].get(f, p);
        input.push_back(p->b);
        dinput.push_back(p->db);
    }else{
        rnn(f, input, dinput);
    }
}

double Rnn::fprop(const vector<int> &features, const vector<bool> &allowed, int target){

    Vec s_filter = Vec::Zero(allowed.size());
    for (int i = 0; i < allowed.size(); i++){
        s_filter[i] = allowed[i];
    }

    vector<Vec*> input;
    vector<Vec*> dinput;

    for (int i = 0; i < features.size(); i++){
        get_feature(features[i], feature_types[i], input, dinput);
    }
    vector<Vec> states(layers.size(), Vec::Zero(params_.topology.size_hidden_layers));
    states[layers.size()-2] = Vec::Zero(N_);
    states[layers.size()-1] = Vec::Zero(N_);

    layers[0]->fprop(input, states[0]);

    for (int i = 1; i < layers.size()-1; i++){
        vector<Vec*> data{&states[i-1]};
        layers[i]->fprop(data, states[i]);
    }
    vector<Vec*> data{&states[layers.size()-2], &s_filter};
    layers.back()->fprop(data, states.back());

    t_edata.push_back(input);
    t_edata_grad.push_back(dinput);
    t_states.push_back(states);
    softmax_filters.push_back(s_filter);

    return - log(states.back()[target]);
}

void Rnn::bprop(const vector<int> &targets){

    t_dstates.resize(t_states.size());
    for (int i = 0; i < t_states.size(); i++){
        t_dstates[i].resize(t_states[i].size());
        for (int j = 0; j < t_states[i].size(); j++){
            t_dstates[i][j] = Vec::Zero(t_states[i][j].size());
        }
    }
    for (int t = t_dstates.size()-1; t >= 0; t--){

        layers.back()->target = targets[t];
        vector<Vec*> data{&t_states[t][layers.size()-2], &softmax_filters[t]};
        vector<Vec*> data_grad{&t_dstates[t][layers.size()-2]};
        layers.back()->bprop(data, t_states[t].back(), t_dstates[t].back(), data_grad);
        for (int i = layers.size() -1; i > 0; i--){
            data = {&t_states[t][i-1]};
            data_grad = {&t_dstates[t][i-1]};
            layers[i]->bprop(data, t_states[t][i], t_dstates[t][i], data_grad);
        }
        layers[0]->bprop(t_edata[t], t_states[t][0], t_dstates[t][0], t_edata_grad[t]);
    }

    rnn.bprop();
}


void Rnn::score(const vector<int> &features,
           const vector<bool> &allowed){
    fprop(features, allowed, 0);
    for (int i = 0; i < N_; i++){
        scores_[i] = log(t_states.back().back()[i]);
    }
}

double Rnn::get_learning_rate(){
    return params_.learning_rate / (1.0 + T_ * params_.decrease_constant);
}

void Rnn::update(){
    if (params_.soft_clipping){
        soft_gradient_clipping();
    }
    double lr = get_learning_rate();
    for (int i = 0; i < lu.size(); i++){
        lu[i].update(lr, T_, params_.clip_value, params_.gradient_clipping, params_.gaussian_noise, params_.gaussian_noise_eta);
    }
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->update(lr, T_, params_.clip_value, params_.gradient_clipping, params_.gaussian_noise, params_.gaussian_noise_eta);
    }
    rnn.update(lr, T_, params_.clip_value, params_.gradient_clipping, params_.gaussian_noise, params_.gaussian_noise_eta);

    flush();
}

void Rnn::soft_gradient_clipping(){
    double squared_norm = 0;
    for (int i = 0; i < parameters.size(); i++){
        squared_norm += parameters[i]->gradient_squared_norm();
    }
    for (int i = 0; i < lu.size(); i++){
        squared_norm += lu[i].gradient_squared_norm();
    }
    squared_norm += rnn.gradient_squared_norm();

    double norm = sqrt(squared_norm);
    double scale_factor = params_.clip_value / norm;

    //cerr << norm << endl;

    if (norm > params_.clip_value){
        for (int i = 0; i < parameters.size(); i++){
            parameters[i]->scale_gradient(scale_factor);
        }
        for (int i = 0; i < lu.size(); i++){
            lu[i].scale_gradient(scale_factor);
        }
        rnn.scale_gradient(scale_factor);
    }
}

void Rnn::train_auxiliary_task(vector<shared_ptr<Node>> &buffer){
    if (params_.rnn.auxiliary_task){
        rnn.add_aux_graph(buffer);
        rnn.fprop_aux();
        rnn.bprop_aux();
        rnn.update_aux(get_learning_rate(), T_, params_.clip_value, params_.gradient_clipping, params_.gaussian_noise, params_.gaussian_noise_eta);

        if (params_.rnn.features == 1){
            assert(params_.rnn.auxiliary_task_max_target > 1);
            rnn.assign_tags(buffer);
        }
    }
}

void Rnn::predict_auxiliary_task(vector<shared_ptr<Node>> &buffer, bool aux_only){
    if (params_.rnn.auxiliary_task){
        rnn.add_aux_graph(buffer, aux_only);
        rnn.fprop_aux();
    }
}

void Rnn::compare_auxiliary_task(AuxiliaryTaskEvaluator &evaluator){
    rnn.eval_aux(evaluator);
}

void Rnn::assign_deplabels(vector<shared_ptr<Node>> &buffer, int deplabel_id){
    rnn.assign_deplabels(buffer, deplabel_id);
}

void Rnn::assign_tags(vector<shared_ptr<Node>> &buffer){
    rnn.assign_tags(buffer);
}

void Rnn::assign_morphological_features(vector<shared_ptr<Node>> &buffer, int deplabel_id){
    rnn.assign_morphological_features(buffer, deplabel_id);
}

void Rnn::average_weights(){
    for (int i = 0; i < lu.size(); i++){
        lu[i].average(T_);
    }
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->average(T_);
    }
    rnn.average_weights(T_);
}

Classifier* Rnn::copy(){
    Rnn *newcf = new Rnn(N_);  // need a new constructor

    newcf->scores_ = this->scores_;
    newcf->T_ = this->T_;
    newcf->n_updates_ = this->n_updates_;

    newcf->n_features = this->n_features;
    newcf->params_ = this->params_;
    newcf->feature_types = this->feature_types;
    newcf->lu = this->lu;

    newcf->initialize_network();

    assert(newcf->parameters.size() == parameters.size());
    for (int i = 0; i < newcf->parameters.size(); i++){
        newcf->parameters[i]->assign(parameters[i]);
    }
    newcf->rnn.assign_parameters(rnn);
    newcf->rnn.copy_char_birnn(rnn);
    return newcf;
}



double Rnn::full_fprop(vector<shared_ptr<Node>> &buffer,
                const vector<vector<int>> &features,
                const vector<vector<bool>> &allowed,
                const vector<int> &targets){
    double res = 0;
    run_rnn(buffer);
    for (int i = 0; i < features.size(); i++){
        res += fprop(features[i], allowed[i], targets[i]);
    }
    return res;
}



void Rnn::gradient_check(vector<shared_ptr<Node>> &buffer,
                    const vector<vector<int>> &features,
                    const vector<vector<bool>> &allowed,
                    const vector<int> &targets,
                    double epsilon){

    run_rnn(buffer);
    for (int i = 0; i < features.size(); i++){
        fprop(features[i], allowed[i], targets[i]);
    }
    bprop(targets);

    vector<int> n_params{0,0,0};
    n_params[0] = parameters.size();
    cerr << parameters.size() << " before lu" << endl;
    for (int i = 0; i < lu.size(); i++){
        lu[i].get_active_params(parameters);
        cerr << parameters.size() << " column " << i << endl;
    }
    n_params[1] = parameters.size();
    rnn.get_parameters(parameters);
    n_params[2] = parameters.size();
    flush();

    cerr << "Params: output/embeddings/rnn : " << n_params[0] << " " << n_params[1] << " " << n_params[2] << endl;

    for (int i = 0; i < parameters.size(); i++){
        for (int k = 0; k < parameters[i]->size(); k++){
            parameters[i]->add_epsilon(k, epsilon);
            double a = full_fprop(buffer, features, allowed, targets);
            flush();
            parameters[i]->add_epsilon(k, -epsilon);
            parameters[i]->add_epsilon(k, -epsilon);
            double c = full_fprop(buffer, features, allowed, targets);
            flush();
            parameters[i]->add_epsilon(k, epsilon);
            parameters[i]->set_empirical_gradient(k, (a-c) / (2 * epsilon));
        }
        cerr << "p[" << i << "] -> " << std::flush;
        parameters[i]->print_gradient_differences();
    }

}


void Rnn::print_parameters(ostream &os){
    params_.print(os);
    rnn.auxiliary_task_summary(os);
    if (get_id() == Classifier::RNN){
        os << "Using RNN class" << endl;
    }else if (get_id() == Classifier::RNN_LABEL_STRUCTURE){
        os << "Using RNN label structure class" << endl;
    }else if (get_id() == Classifier::RNN_LABEL_STRUCTURE_LEX){
        os << "Using RNN lexicalized label structure class" << endl;
    }else{
        assert(false);
    }
}

void Rnn::export_model(const string &outdir){
    export_classifier_id(outdir);

    ofstream os(outdir + "/network_info");
    params_.print(os);
    os.close();

    for (int i = 0; i < lu.size(); i++){
        lu[i].export_model(outdir + "/lu" + std::to_string(i));
    }

    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->export_model(outdir+"/parameters" + std::to_string(i));
    }
    rnn.export_model(outdir);
}






void Rnn::print_stats(ostream &os){
    cerr << endl;
}


void Rnn::global_update_one(const vector<int> &feats,
                       int action_num,
                       int increment){
    assert(false);
}

int Rnn::n_aux_tasks(){
    return params_.rnn.auxiliary_task_max_target - params_.rnn.features;
}

bool Rnn::auxiliary_task(){
    return params_.rnn.auxiliary_task;
}
bool Rnn::auxiliary_tags(){
    return params_.rnn.features == 1 && auxiliary_task() && params_.rnn.auxiliary_task_max_target > 1;
}

void Rnn::auxiliary_gradient_check(vector<shared_ptr<Node>> &buffer, double epsilon){
    rnn.auxiliary_gradient_check(buffer, epsilon);
}

void Rnn::aux_reset_gradient_history(){
    rnn.aux_reset_gradient_history();
}

void Rnn::set_train_time(bool b){
    rnn.set_train_time(b);
}



bool RnnStructureLabel::is_struct_action(const vector<bool> &allowed){
    //sanity_check(allowed);
    for (int i = 0; i < I; i++){
        if (allowed[i]){
            return true;
        }
    }
    return false;

    /*return allowed[MergeLabelTS::SHIFT_I]
        || allowed[MergeLabelTS::MERGE_I]
        || allowed[MergeLabelTS::IDLE_I]
        || allowed[MergeLabelTS::GAP_I];*/
}

//void RnnStructureLabel::sanity_check(const vector<bool> &allowed){
//    assert(allowed.size() == N_);
//    if (allowed[MergeLabelTS::SHIFT_I]
//            || allowed[MergeLabelTS::MERGE_I]
//            || allowed[MergeLabelTS::IDLE_I]
//            || allowed[MergeLabelTS::GAP_I]){
//        for (int i = I; i < N_; i++){
//            assert(! allowed[i]);
//        }
//    }else{
//        bool tmp = false;
//        for (int i = I; i < N_; i++){
//            if (allowed[i]) tmp = true;
//        }
//        assert(tmp);
//    }
//}


// Note to self: calling virtual functions in constructor / destructor is a sin in c++
RnnStructureLabel::RnnStructureLabel(int n_classes, const string &outdir, bool lex) : Rnn(n_classes, outdir, false){
//    Classifier(n_classes){
//    FeatureExtractor *fe = FeatureExtractor::import_model(outdir);

//    params_.print(cerr);
//    NeuralNetParameters::read_option_file(outdir + "/network_info", params_);
//    //params_.rnn.auxiliary_task = false; // WARNING/TODO: temporary workaround
//    params_.print(cerr);

//    init_feature_types_and_lu(fe);

//    for (int i = 0; i < lu.size(); i++){
//        lu[i].load(outdir + "/lu" + std::to_string(i));
//    }
    this->lex = lex;
    if (lex){
        I = LexicalizedMergeLabelTS::GAP_I + 1;
    }else{
        assert(I == MergeLabelTS::GAP_I + 1);
    }

    initialize_network();
    for (int i = 0; i < parameters.size(); i++){
        parameters[i]->load(outdir+"/parameters" + std::to_string(i));
    }
    rnn.load_parameters(outdir);
}


RnnStructureLabel::RnnStructureLabel(int n_classes) : Rnn(n_classes){}

// : Rnn(n_classes, fe, params){}
RnnStructureLabel::RnnStructureLabel(int n_classes, FeatureExtractor *fe, NeuralNetParameters &params, bool lex)
    : Rnn(n_classes, fe, params, false){

    this->lex = lex;
    if (lex){
        I = LexicalizedMergeLabelTS::GAP_I + 1;
    }else{
        assert(I == MergeLabelTS::GAP_I + 1);
    }

    initialize_network();
}
//    : Classifier(n_classes),
//      params_(params){

//    if (params_.rnn.features > params_.header.size()){
//        params_.rnn.features = params_.header.size();
//    }

//    if (params_.rnn.auxiliary_task && params_.rnn.auxiliary_task_max_target > params_.header.size()){
//        params_.rnn.auxiliary_task_max_target = params_.header.size();
//    }

//    fe->get_template_type(feature_types);

//    init_feature_types_and_lu(fe);

//    lu[enc::CAT] = LookupTable(enc::hodor.size(enc::CAT), params_.topology.embedding_size_type[enc::CAT]);
//    cerr << "Lookup table " << enc::CAT << " has " << params_.topology.embedding_size_type[enc::CAT] << " dimensions (vocsize = " << enc::hodor.size(enc::CAT) << ")" << endl;

//    for (int i = 1; i < lu.size(); i++){
//        lu[i] = LookupTable(params_.voc_sizes[i], params_.topology.embedding_size_type[i]);
//        cerr << "Lookup table " << i << " has " << params_.topology.embedding_size_type[i]
//                << " dimensions (vocsize = " << params_.voc_sizes[i] << ")"
//                << " hodor size = " << enc::hodor.size(i) << endl;
//    }

//    initialize_network();
//}

RnnStructureLabel::~RnnStructureLabel(){
    for (int i = 0; i < layers_struct.size(); i++){
        delete layers_struct[i];
    }
}

int RnnStructureLabel::get_id(){
    if (lex)
        return Classifier::RNN_LABEL_STRUCTURE_LEX;
    else
        return Classifier::RNN_LABEL_STRUCTURE;
}

void RnnStructureLabel::initialize_network(){
    vector<int> embedding_size_token;
    for (int i = 0; i < feature_types.size(); i++){
        if (feature_types[i] == enc::CAT){
            embedding_size_token.push_back(params_.topology.embedding_size_type[enc::CAT]);
        }else{
            for (int d = 0; d < 2; d++){ // magic number: bi-rnn -> concatenate forward and backward embedding
                embedding_size_token.push_back(params_.rnn.hidden_size);
            }
        }
    }

    // Output layers for label actions
    layers.push_back(new MultipleLinearLayer(embedding_size_token.size(), embedding_size_token, params_.topology.size_hidden_layers));
    layers.push_back(new ReLU());
    for (int i = 1; i < params_.topology.n_hidden_layers; i++){
        layers.push_back(new AffineLayer(params_.topology.size_hidden_layers, params_.topology.size_hidden_layers));
        layers.push_back(new ReLU());
    }
    layers.push_back(new AffineLayer(params_.topology.size_hidden_layers, N_ - I));
    layers.push_back(new SoftmaxFilter());

    for (int i = 0; i < layers.size(); i++){
        layers[i]->get_params(parameters);
    }

    // Output layers for structure actions
    layers_struct.push_back(new MultipleLinearLayer(embedding_size_token.size(), embedding_size_token, params_.topology.size_hidden_layers));
    layers_struct.push_back(new ReLU());
    for (int i = 1; i < params_.topology.n_hidden_layers; i++){
        layers_struct.push_back(new AffineLayer(params_.topology.size_hidden_layers, params_.topology.size_hidden_layers));
        layers_struct.push_back(new ReLU());
    }
    layers_struct.push_back(new AffineLayer(params_.topology.size_hidden_layers, I));
    layers_struct.push_back(new SoftmaxFilter());


    for (int i = 0; i < layers_struct.size(); i++){
        layers_struct[i]->get_params(parameters);
    }


    rnn = BiRnnFeatureExtractor(&params_, &lu);

}

void RnnStructureLabel::flush(){
    Rnn::flush();
    struct_or_label.clear();
}

double RnnStructureLabel::fprop(const vector<int> &features, const vector<bool> &allowed, int target){

    bool struct_action = is_struct_action(allowed);
    int n_output = -1;
    int shift = -1;
    vector<Layer*> *l_ptrs = nullptr;
    if (struct_action){
        n_output = I;
        shift = 0;
        l_ptrs = &layers_struct;
    }else{
        n_output = N_ - I;
        shift = I;
        l_ptrs = &layers;
    }

    Vec s_filter = Vec::Zero(n_output);
    for (int i = 0; i < n_output; i++){
        s_filter[i] = allowed[i+shift];
    }

    vector<Vec*> input;
    vector<Vec*> dinput;

    for (int i = 0; i < features.size(); i++){
        get_feature(features[i], feature_types[i], input, dinput);
    }

    vector<Vec> states(l_ptrs->size(), Vec::Zero(params_.topology.size_hidden_layers));
    states[l_ptrs->size()-2] = Vec::Zero(n_output);
    states[l_ptrs->size()-1] = Vec::Zero(n_output);

    (*l_ptrs)[0]->fprop(input, states[0]);

    for (int i = 1; i < l_ptrs->size()-1; i++){
        vector<Vec*> data{&states[i-1]};
        (*l_ptrs)[i]->fprop(data, states[i]);
    }

    vector<Vec*> data{&states[l_ptrs->size()-2], &s_filter};
    l_ptrs->back()->fprop(data, states.back());

    t_edata.push_back(input);
    t_edata_grad.push_back(dinput);
    t_states.push_back(states);
    softmax_filters.push_back(s_filter);
    struct_or_label.push_back(struct_action);

    return - log(states.back()[target-shift < 0 ? 0 : target-shift]);
}


void RnnStructureLabel::bprop(const vector<int> &targets){

    t_dstates.resize(t_states.size());
    for (int i = 0; i < t_states.size(); i++){
        t_dstates[i].resize(t_states[i].size());
        for (int j = 0; j < t_states[i].size(); j++){
            t_dstates[i][j] = Vec::Zero(t_states[i][j].size());
        }
    }

    for (int t = t_dstates.size()-1; t >= 0; t--){

        if (struct_or_label[t]){
            assert(targets[t] < I);
            layers_struct.back()->target = targets[t];
            vector<Vec*> data{&t_states[t][layers_struct.size()-2], &softmax_filters[t]};
            vector<Vec*> data_grad{&t_dstates[t][layers_struct.size()-2]};
            layers_struct.back()->bprop(data, t_states[t].back(), t_dstates[t].back(), data_grad);
            for (int i = layers_struct.size() -1; i > 0; i--){
                data = {&t_states[t][i-1]};
                data_grad = {&t_dstates[t][i-1]};
                layers_struct[i]->bprop(data, t_states[t][i], t_dstates[t][i], data_grad);
            }
            layers_struct[0]->bprop(t_edata[t], t_states[t][0], t_dstates[t][0], t_edata_grad[t]);
        }else{
            assert(targets[t] >= I);
            layers.back()->target = targets[t] - I; // shifted by I
            vector<Vec*> data{&t_states[t][layers.size()-2], &softmax_filters[t]};
            vector<Vec*> data_grad{&t_dstates[t][layers.size()-2]};
            layers.back()->bprop(data, t_states[t].back(), t_dstates[t].back(), data_grad);
            for (int i = layers.size() -1; i > 0; i--){
                data = {&t_states[t][i-1]};
                data_grad = {&t_dstates[t][i-1]};
                layers[i]->bprop(data, t_states[t][i], t_dstates[t][i], data_grad);
            }
            layers[0]->bprop(t_edata[t], t_states[t][0], t_dstates[t][0], t_edata_grad[t]);
        }
    }

    rnn.bprop();
}

void RnnStructureLabel::score(const vector<int> &features, const vector<bool> &allowed){
    fprop(features, allowed, 0);

    if (struct_or_label.back()){
        for (int i = 0; i < I; i++){
            scores_[i] = log(t_states.back().back()[i]);
        }
        for (int i = I; i < N_; i++){
            scores_[i] = MINUS_INFINITY;
        }
    }else{
        for (int i = 0; i < I; i++){
            scores_[i] = MINUS_INFINITY;
        }
        for (int i = I; i < N_; i++){
            scores_[i] = log(t_states.back().back()[i-I]);
        }
    }
}


Classifier* RnnStructureLabel::copy(){
    RnnStructureLabel *newcf = new RnnStructureLabel(N_);  // need a new constructor
    newcf->I = this->I;
    newcf->lex = lex;
//    if (lex){
//        I = LexicalizedMergeLabelTS::GAP_I + 1;
//    }else{
//        assert(I == MergeLabelTS::GAP_I + 1);
//    }

    newcf->scores_ = this->scores_;
    newcf->T_ = this->T_;
    newcf->n_updates_ = this->n_updates_;
    newcf->struct_or_label = this->struct_or_label;

    newcf->n_features = this->n_features;
    newcf->params_ = this->params_;
    newcf->feature_types = this->feature_types;
    newcf->lu = this->lu;

    newcf->initialize_network();

    assert(newcf->parameters.size() == parameters.size());
    for (int i = 0; i < newcf->parameters.size(); i++){
        newcf->parameters[i]->assign(parameters[i]);
    }
    newcf->rnn.assign_parameters(rnn);
    newcf->rnn.copy_char_birnn(rnn);
    return newcf;
}



