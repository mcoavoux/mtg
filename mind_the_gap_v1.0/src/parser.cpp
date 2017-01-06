#include <sys/stat.h>
#include "parser.h"



////////////////////////////////////////////////////////////
///
///
///  Parser
///
///
/// ////////////////////////////////////////////////////////////


Parser::Parser(int tsi, int cfi, int fei,
       Treebank &train, Treebank &dev, const Grammar &grammar, const string &tpl_filename, NeuralNetParameters &params):
        transition_system_id(tsi), classifier_id(cfi), feature_extractor_id(fei){

    fe = feature_extractor_factory(fei, tpl_filename);
    ts = transition_system_factory(tsi, grammar);
    for (int i = 0; i < train.size(); i++){
        Derivation d;
        ts->compute_derivation(*train[i], d);
    }
    for (int i = 0; i < dev.size(); i++){
        Derivation d;
        ts->compute_derivation(*dev[i], d);
    }
    cf = classifier_factory(cfi, ts->num_actions(), fe, params);

    allowed = vector<bool>(ts->num_actions(), false);
    gold_actions = vector<bool>(ts->num_actions(), false);
    features = vector<int>(fe->n_templates());

    #ifdef DEBUG
        cerr << *ts->grammar_ptr() << endl;
        ts->print_transitions(cerr);
    #endif
}

Parser::Parser(const string &outdir):transition_system_id(0), classifier_id(0), feature_extractor_id(0){
    ts = TransitionSystem::import_model(outdir);
    cf = Classifier::import_model(outdir);
    fe = FeatureExtractor::import_model(outdir);
    allowed = vector<bool>(ts->num_actions(), false);
    gold_actions = vector<bool>(ts->num_actions(), false);
    features = vector<int>(fe->n_templates());
}

Parser::~Parser(){
    delete ts;
    delete cf;
    delete fe;
}


void Parser::summary(int beamsize, int epochs, const string &outdir, int train_size, int dev_size, int test_size){
    cerr << endl;
    cerr << "Summary" << endl;
    cerr << "=======" << endl << endl;
    switch(transition_system_id){
    case TransitionSystem::GAP_TS:
        cerr << "- Transition System: simple gap,  actions=[shift, gap, ru, rr, rl, idle]" << endl; break;
    case TransitionSystem::CGAP_TS:
        cerr << "- Transition System: compound gap,  actions=[shift, gap_i (i can be 0), ru, rr, rl, ghost reduce]" << endl; break;
    case TransitionSystem::MERGE_LABEL_TS:
        cerr << "- Transition System: merge-label, actions=[shift, gap, ru, merge, label, idle]" << endl; break;
    }
    cerr << "- feature extractor " << feature_extractor_id << endl;
    cerr << "- Classifier:";
    switch (classifier_id){
    case Classifier::FASTEST_PER:
        cerr << " Averaged structured perceptron" << endl;
        break;
    case Classifier::FFNN:{
        cerr << " Feedforward neural network" << endl;
        static_cast<NeuralNet*>(cf)->print_parameters(cerr);
        break;
    }
    default:
        assert(false && "Error : not a valid classifier");
    }

    cerr << "- Number of actions: " << ts->num_actions() << endl;
    cerr << "- Number of templates: " << fe->n_templates() << endl;
    cerr << "- Number of training epochs: " << epochs << endl;
    cerr << "- Size of beam: " << beamsize << endl;
    cerr << "- Results will be written in " << outdir << endl;
    cerr << "- Number of sentences in " << endl;
    cerr << "    - train: " << train_size << endl;
    cerr << "    - dev: " << dev_size << endl;
    cerr << "    - test: " << test_size << endl;
    cerr << "- Known vocabulary size: " << enc::hodor.size(enc::TOK) << endl;

}

void Parser::train_global(Treebank &train_tbk,
                          Treebank &dev_tbk,
                          Treebank &train_nary,
                          Treebank &dev_nary,
                          int epochs,
                          int beamsize,
                          const std::string &outdir,
                          vector<vector<shared_ptr<Node>>> &raw_test,
                          vector<vector<std::pair<string,string>>> &str_raw_test){


    summary(beamsize, epochs, outdir, train_tbk.size(), dev_tbk.size(), raw_test.size());
    cerr << "Training started" << endl;

    Treebank sample_train;
    Treebank sample_train_nary;
    for (int i = 0; i < dev_tbk.size(); i++){
        sample_train.add_tree(*train_tbk[i]);
        sample_train_nary.add_tree(*train_nary[i]);
    }

    for (int e = 0; e < epochs; e++){

        if (cf->get_id() >= Classifier::FFNN){
            nn = static_cast<NeuralNet*>(cf);
        }

        train_tbk.shuffle();
        cf->reset_updates();

        for (int i = 0; i < train_tbk.size(); i++){
            if (i% 100 == 0) cout << "\rTree " << i << "      " << std::flush;
            if (cf->get_id() < Classifier::FFNN){
                train_global_one(*train_tbk[i], beamsize);
            }else{
                train_neural_one(*train_tbk[i], beamsize);
            }
        }

        Classifier *tmp = cf->copy();
        cf->average_weights();

        eval::EvalTriple fscore_train;
        eval::EvalTriple fscore_dev;

        evaluate_global(sample_train, sample_train_nary, beamsize, fscore_train);
        evaluate_global(dev_tbk, dev_nary, beamsize, fscore_dev);

        cerr << "Epoch " << e
           << "  train={F="<<fscore_train.fscore() << " "
                    << "P="<<fscore_train.precision() << " "
                    << "R="<<fscore_train.recall() << "} "
                  "dev={F="<<fscore_dev.fscore() << " "
                    << "P="<<fscore_dev.precision() << " "
                    << "R="<<fscore_dev.recall() << "} "
                    << "up=" << cf->get_n_updates() <<endl;

        if (e % 4 == 3  && e >= 19){
            if (raw_test.size() > 0){
                Treebank pred_tbk;
                predict_treebank(raw_test, beamsize, pred_tbk);
                pred_tbk.detransform(*(ts->grammar_ptr()));
                pred_tbk.write(outdir+"/test_beam"+ std::to_string(beamsize) + "_it" + std::to_string(e) + ".mrg", str_raw_test);
            }
            string out_at_iteration = outdir + "/iteration" + std::to_string(e);
            mkdir(out_at_iteration.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
            export_model(out_at_iteration);
            enc::hodor.export_model(out_at_iteration);
        }

        delete cf;
        cf = tmp;
    }
    cf->average_weights();
}

void Parser::train_global_one(Tree &tree, int beamsize){

    vector<shared_ptr<Node>> buffer;
    tree.get_buffer(buffer);
    TssBeam beam(beamsize, buffer);

    Derivation gold;
    ts->compute_derivation(tree,gold);

    while ((! beam.finished(*ts->grammar_ptr())) && beam.gold_in_beam(gold)){
        for (int k = 0; k < beam.size(); k++){
            shared_ptr<ParseState> state;
            beam.get(k, state);
            assert( ! std::isnan(state->score()) && ! std::isinf(state->score()) && "state score is nan or inf");
            ts->allowed(*state, buffer.size(), allowed);

            assert(check(allowed)&& "Error: no action allowed (train_global_one function)");

            (*fe)(*state, buffer, features);
            cf->score(features, allowed);

            for (int i = 0; i < allowed.size(); i++){
                if (allowed[i]){
                    assert( ! std::isnan((*cf)[i]) && ! std::isinf((*cf)[i]) && "classifier result is nan or inf");
                    Candidate candidate(k, state->score() + (*cf)[i], *(*ts)[i]);
                    beam.add_candidate(candidate);
                }
            }
        }
        beam.next_step(ts);
    }

    Derivation best;
    beam.best_derivation(best);

    if (gold.size() < best.size()){
        assert(transition_system_id != TransitionSystem::CGAP_TS && "Padding with compound gap should not happen");
        gold.pad(best.size(), *(ts->get_idle()));
    }
    if (! (gold == best)){
        update_global(gold, best, tree, buffer);
        cf->increment_updates();
    }
    cf->increment_T();
}

void Parser::train_neural_one(Tree &tree, int beamsize){

    vector<shared_ptr<Node>> buffer;
    tree.get_buffer(buffer);
    TssBeam beam(beamsize, buffer);

    Derivation gold;
    ts->compute_derivation(tree,gold);

    shared_ptr<ParseState> state(new ParseState());
    while (! state->is_final(*ts->grammar_ptr(), buffer.size())){
        int time_step = state->time_step();
        int target = gold[time_step]->code();

        ts->allowed(*state, buffer.size(), allowed);
        (*fe)(*state, buffer, features);

        nn->fprop(features, allowed, target);
        nn->bprop(features, allowed, target);;
        nn->update();

        shared_ptr<ParseState> next_state;

        ts->next(state, *gold[time_step], buffer, next_state);
        state = next_state;
        cf->increment_T();
    }
}











bool Parser::check(const vector<bool> &v){
    for (bool b : v)
        if (b) return true;
    return false;
}

void Parser::update_global(const Derivation &gold, const Derivation &best, const Tree &tree, vector<shared_ptr<Node>> &buffer){
    shared_ptr<ParseState> gold_state = shared_ptr<ParseState>(new ParseState());
    shared_ptr<ParseState> pred_state = shared_ptr<ParseState>(new ParseState());
    shared_ptr<ParseState> tmp_state;
    int min_size = std::min(gold.size(), best.size());
    int i = 0;
    while (i < min_size && (*gold[i] == *best[i])){
        ts->next(gold_state, *gold[i], buffer, tmp_state);
        gold_state = tmp_state;
        ts->next(pred_state, *best[i], buffer, tmp_state);
        pred_state = tmp_state;
        i++;
    }

    int j = i;
    while (j < min_size){
        (*fe)(*gold_state, buffer, features);
        cf->global_update_one(features, gold[j]->code(), 1);
        assert(ts->allowed(*gold_state, buffer.size(), gold[j]->code()));
        ts->next(gold_state, *gold[j], buffer, tmp_state);
        gold_state = tmp_state;
        j++;
    }

    j = i;
    while (j < min_size){
        (*fe)(*pred_state, buffer, features);
        cf->global_update_one(features, best[j]->code(), -1);
        ts->next(pred_state, *best[j], buffer, tmp_state);
        pred_state = tmp_state;
        j++;
    }
}

void Parser::evaluate_global(Treebank &tbk, Treebank &nary, int beamsize, eval::EvalTriple &fscore){
    for (int i = 0; i < tbk.size(); i++){
        if (i % 100==0) cout << "\rEval, Tree " << i << "   " << std::flush;;
        Tree pred;
        vector<shared_ptr<Node>> buffer;
        tbk[i]->get_buffer(buffer);
        predict_tree(buffer, beamsize, pred);

        pred.unbinarize(*ts->grammar_ptr());

        eval::compare(*nary[i], pred, fscore);
    }
}

void Parser::predict_tree(vector<shared_ptr<Node>> &buffer, int beamsize, Tree &pred){
    TssBeam beam(beamsize, buffer);

    while (! beam.finished(*ts->grammar_ptr())){
        for (int k = 0; k < beam.size(); k++){
            shared_ptr<ParseState> state;

            beam.get(k, state);

            ts->allowed(*state, buffer.size(), allowed);
            assert(check(allowed)&& "Error: no action allowed (train_global_one function)");

            assert( ! std::isnan(state->score()) && ! std::isinf(state->score()) && "state score is nan or inf");


            (*fe)(*state, buffer, features);
            cf->score(features, allowed);

            for (int i = 0; i < allowed.size(); i++){
                if (allowed[i]){
                    assert( ! std::isnan((*cf)[i]) && ! std::isinf((*cf)[i]) && "classifier result is nan or inf");
                    Candidate candidate(k, state->score() + (*cf)[i], *(*ts)[i]);
                    beam.add_candidate(candidate);
                }
            }
        }
        beam.next_step(ts);
    }
    beam.best_tree(pred);
}

void Parser::predict_treebank(vector<vector<shared_ptr<Node>>> &raw_test, int beamsize, Treebank &tbk){
    tbk.clear();
    Tree tree;
    for (int i = 0; i < raw_test.size(); i++){
        predict_tree(raw_test[i], beamsize, tree);
        tbk.add_tree(tree);
    }
}

void Parser::export_model(const string &outdir){
    cerr << "Exporting classifier weights" << endl;
    cf->export_model(outdir);
    cerr << "Exporting transition system and grammar" << endl;
    ts->export_model(outdir);
    cerr << "Exporting feature templates" << endl;
    fe->export_model(outdir);
}


Classifier* Parser::classifier_factory(int classifier_id, int num_classes, FeatureExtractor *fe, NeuralNetParameters &params){
    switch (classifier_id){
        case Classifier::PERCEPTRON: return new Perceptron(num_classes);
        case Classifier::FAST_PER: return new FastPerceptron(num_classes);
        case Classifier::FASTER_PER: return new FasterPerceptron(num_classes);
        case Classifier::FASTEST_PER: return new FastestPerceptron(num_classes);
        case Classifier::FFNN : return new NeuralNet(num_classes, fe, params);
        default:
            assert(false && "Unknown classifier or not implemented");
    }
    assert(false);
    return nullptr;
}

TransitionSystem* Parser::transition_system_factory(int ts_id, const Grammar &grammar){
    switch(ts_id){
        case TransitionSystem::GAP_TS: return new GapTS(grammar);
        case TransitionSystem::CGAP_TS: return new CompoundGapTS(grammar);
        case TransitionSystem::MERGE_LABEL_TS: return new MergeLabelTS(grammar);
        default:
            assert(false && "Unknown transition system or not implemented");
    }
    assert(false);
    return nullptr;
}

FeatureExtractor* Parser::feature_extractor_factory(int feat_extractor_id, const string &filename){
    switch(feat_extractor_id){
        case feature::STD_FEATURE_EXTRACTOR : return new StdFeatureExtractor(filename);
        case feature::FAST_FEATURE_EXTRACTOR : return new FastFeatureExtractor(filename);
        case feature::DENSE_FEATURE_EXTRACTOR : return new DenseFeatureExtractor(filename);
        default:
            assert(false && "Unknown feature extractor or not implemented");
    }
    assert(false);
    return nullptr;
}





