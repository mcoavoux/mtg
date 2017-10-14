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

void Parser::precompute_char_lstm(){
    cf->precompute_char_lstm();
}


void Parser::summary(int beamsize, int epochs, const string &outdir, int train_size, int dev_size){
    cerr << endl;
    cerr << "Summary" << endl;
    cerr << "=======" << endl << endl;
    switch(transition_system_id){
    case TransitionSystem::GAP_TS:
        cerr << "- Transition System: simple gap,  actions=[shift, gap, ru, rr, rl, idle]" << endl; break;
    case TransitionSystem::CGAP_TS:
        cerr << "- Transition System: compound gap,  actions=[shift, gap_i (i can be 0), ru, rr, rl, ghost reduce]" << endl; break;
    case TransitionSystem::MERGE_LABEL_TS:
        cerr << "- Transition System: merge-label, actions=[shift, gap, merge, label / no label, idle]" << endl; break;
    case TransitionSystem::SHIFT_REDUCE:
        cerr << "- Transition System: standard shift-reduce,  actions=[shift, ru, rr, rl, idle]" << endl; break;
    case TransitionSystem::LEXICALIZED_MERGE_LABEL_TS:
        cerr << "- Transition System: lexicalized merge-label, actions=[shift, gap, left, right, label / no label, idle]" << endl; break;
    case TransitionSystem::MERGE_LABEL_PROJ_TS:
        cerr << "- Transition System: merge-label (projective version), actions=[shift, merge, label / no label, idle]" << endl; break;
    case TransitionSystem::LEXICALIZED_MERGE_LABEL_PROJ_TS:
        cerr << "- Transition System: lexicalized merge-label (projective version), actions=[shift, left, right, label / no label, idle]" << endl; break;
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
    case Classifier::RNN:{
        cerr << "bi-rnn feature extractor + feedforward on top" << endl;
        static_cast<Rnn*>(cf)->print_parameters(cerr);
        break;
    }
    case Classifier::RNN_LABEL_STRUCTURE:
    case Classifier::RNN_LABEL_STRUCTURE_LEX:{
        cerr << "bi-rnn feature extractor + 2 feedforward networks for structure and label actions" << endl;
        static_cast<Rnn*>(cf)->print_parameters(cerr);
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
    cerr << "- Known vocabulary size: " << enc::hodor.size(enc::TOK) << endl;

}

void Parser::train_global(Treebank &train_tbk,
                          Treebank &dev_tbk,
                          Treebank &train_nary,
                          Treebank &dev_nary,
                          int epochs,
                          int beamsize,
                          const std::string &outdir){

//    train_tbk.write("footrain");
//    dev_tbk.write("foodev");
//    train_nary.write("footrain_nary");
//    dev_nary.write("foodev_nary");

    summary(beamsize, epochs, outdir, train_tbk.size(), dev_tbk.size());
    cerr << "Training started" << endl;

    Treebank sample_train;
    Treebank sample_train_nary;
    for (int i = 0; i < dev_tbk.size(); i++){
        sample_train.add_tree(*train_tbk[i]);
        sample_train_nary.add_tree(*train_nary[i]);
    }

    //train_aux_task(train_tbk, sample_train, dev_tbk, 20);  // no pretraining for now

    for (int e = 0; e < epochs; e++){

        train_tbk.shuffle();
        cf->reset_updates();

        for (int i = 0; i < train_tbk.size(); i++){
            if (i% 20 == 0) cout << "\rTree " << i << "(" << ((100.0*i) / train_tbk.size()) << "%)" <<"      " << std::flush;
            switch(cf->get_id()){
            case Classifier::FFNN:
                train_neural_one(*train_tbk[i], beamsize); break;
            case Classifier::RNN:
            case Classifier::RNN_LABEL_STRUCTURE:
            case Classifier::RNN_LABEL_STRUCTURE_LEX:
                //gradient_check_one(*train_tbk[i], beamsize, 1e-6); exit(0);
                train_rnn_one(*train_tbk[i], beamsize); break;
            default:
                train_global_one(*train_tbk[i], beamsize); break;
            }
        }

        Classifier *tmp = cf->copy();
        cf->average_weights();

        eval::EvalTriple fscore_train;
        eval::EvalTriple fscore_dev;


        evaluate_global(sample_train, sample_train_nary, beamsize, fscore_train);
        evaluate_global(dev_tbk, dev_nary, beamsize, fscore_dev);

        AuxiliaryTaskEvaluator aux_train;
        AuxiliaryTaskEvaluator aux_dev;
        bool aux_task = false;
        if (cf->get_id() >= Classifier::RNN){
            Rnn *rnn = static_cast<Rnn*>(cf);
            aux_task = rnn->auxiliary_task();
            if (aux_task){
                evaluate_auxiliary_tasks(sample_train, aux_train);
                evaluate_auxiliary_tasks(dev_tbk, aux_dev);
            }
        }

        cerr << "Epoch " << e
           << "  train={F="<< std::setprecision(4) << fscore_train.fscore() << " "
                    << "P="<< std::setprecision(4) << fscore_train.precision() << " "
                    << "R="<< std::setprecision(4) << fscore_train.recall() << "} "
                  "dev={F="<< std::setprecision(4) << fscore_dev.fscore() << " "
                    << "P="<< std::setprecision(4) << fscore_dev.precision() << " "
                    << "R="<< std::setprecision(4) << fscore_dev.recall() << "} "
                    << "up=" << cf->get_n_updates();
        if (aux_task){
            cerr << "    train aux = " << aux_train << "  dev aux = " << aux_dev;
        }
        cerr << endl;

        if (e % 4 == 3  &&
                ((e >= 19 && cf->get_id() < Classifier::FFNN) || // structured perceptron is slower to converge (in number of iterations ;)
                 (e >= 3 && cf->get_id() >= Classifier::FFNN))){ // for some datasets, 4 / 8 iterations can be enouch for bi-lstm
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


/// TODO: could you merge those two ?
void Parser::train_neural_one(Tree &tree, int beamsize){

    NeuralNet *nn = static_cast<NeuralNet*>(cf);

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

void Parser::train_rnn_one(Tree &tree, int beamsize){

    Rnn *nn = static_cast<Rnn*>(cf);
    nn->set_train_time(true);

    vector<shared_ptr<Node>> buffer;
    tree.get_buffer(buffer);
    TssBeam beam(beamsize, buffer);

    Derivation gold;
    ts->compute_derivation(tree,gold);

    shared_ptr<ParseState> state(new ParseState());
    vector<int> targets;

    nn->train_auxiliary_task(buffer);

    nn->run_rnn(buffer);

    while (! state->is_final(*ts->grammar_ptr(), buffer.size())){
        int time_step = state->time_step();
        int target = gold[time_step]->code();
        targets.push_back(target);

        ts->allowed(*state, buffer.size(), allowed);
        if (! allowed[target]){
            cerr << *state << endl;
            cerr << "buffer size " << buffer.size() << endl;
            //cerr << target << endl;
            cerr << gold << endl;
            cerr << time_step << endl;
            cerr << *gold[time_step] << endl;
        }
        assert(allowed[target]);
        (*fe)(*state, buffer, features);

        vector<Vec*> input;
        vector<Vec*> dinput;


        nn->fprop(features, allowed, target);
        shared_ptr<ParseState> next_state;
        ts->next(state, *gold[time_step], buffer, next_state);
        state = next_state;

    }

    nn->bprop(targets);;
    nn->update();
    nn->increment_T();
    nn->set_train_time(false);



    // Bug fix (?) 10/03/2017 : confusion about label and field[1] of Leafs
    // field[1] contains gold tag (when provided), label_ contains predicted tag (after first iteration)
    // Note: this might have been beneficial (add noise) so far, but it is the cause of a bug when using MergeLabel TS
//    for (int i = 0; i < buffer.size(); i++){
//        buffer[i]->set_label(enc::hodor.code(enc::hodor.decode(buffer[i]->get_field(Leaf::FIELD_TAG), enc::TAG), enc::CAT));
//    }
}


void Parser::gradient_check_one(Tree &tree, int beamsize, double epsilon){
    Rnn *nn = static_cast<Rnn*>(cf);

    vector<shared_ptr<Node>> buffer;
    tree.get_buffer(buffer);
    TssBeam beam(beamsize, buffer);

    Derivation gold;
    ts->compute_derivation(tree,gold);

    shared_ptr<ParseState> state(new ParseState());
    vector<int> targets;

    vector<vector<int>> tfeats;
    vector<vector<bool>> tallow;

    while (! state->is_final(*ts->grammar_ptr(), buffer.size())){
        int time_step = state->time_step();
        int target = gold[time_step]->code();
        targets.push_back(target);

        ts->allowed(*state, buffer.size(), allowed);
        tallow.push_back(allowed);
        (*fe)(*state, buffer, features);
        tfeats.push_back(features);

        shared_ptr<ParseState> next_state;
        ts->next(state, *gold[time_step], buffer, next_state);
        state = next_state;
    }

    nn->gradient_check(buffer, tfeats, tallow, targets, epsilon);
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

//        cerr << "pred" << endl << pred << endl;
//        cerr << "gold" << endl << *nary[i] << endl;
        eval::compare(*nary[i], pred, fscore);
    }
}

void Parser::evaluate_auxiliary_tasks(Treebank &tbk, AuxiliaryTaskEvaluator &evaluator){
    assert(cf->get_id() >= Classifier::RNN);
    Rnn *rnn = static_cast<Rnn*>(cf);
    evaluator.total = 0;
    evaluator.complete_match = 0;
    evaluator.good = vector<int>(rnn->n_aux_tasks(), 0);

    //cerr << "Evaluating auxiliary task " << endl << endl;
    for (int i = 0; i < tbk.size(); i++){
        if (i % 100==0) cout << "\rEval aux, Tree " << i << "   " << std::flush;;
        vector<shared_ptr<Node>> buffer;
        tbk[i]->get_buffer(buffer);
        //eval_aux_one(buffer, evaluator);
        //rnn->run_rnn(buffer);
        rnn->predict_auxiliary_task(buffer, true);
        rnn->compare_auxiliary_task(evaluator);
    }
}

void Parser::train_aux_task(Treebank &train_tbk, Treebank &sample_train, Treebank &dev_tbk, int epochs){

    if (epochs == 0) return;
    if (cf->get_id() < Classifier::RNN) return;
    Rnn *rnn = static_cast<Rnn*>(cf);
    bool aux_task = rnn->auxiliary_task();
    if (! aux_task) return;

    cerr << "Pre training on auxiliary tasks" << endl;
    cf->reset_updates();
    for (int i = 0; i < epochs; i++){
        train_tbk.shuffle();
        for (int k = 0; k < train_tbk.size(); k++){
            if (k% 100 == 0) cout << "\rTree " << k << "      " << std::flush;
            train_aux_task_one(rnn, *train_tbk[k]);
            rnn->increment_T();
        }

        Classifier *tmp = cf->copy();
        cf->average_weights();


        AuxiliaryTaskEvaluator aux_train;
        AuxiliaryTaskEvaluator aux_dev;

        evaluate_auxiliary_tasks(sample_train, aux_train);
        evaluate_auxiliary_tasks(dev_tbk, aux_dev);

        cerr << "Aux iteration " << i << " train = " << aux_train << " dev = " << aux_dev << endl;

        delete cf;
        cf = tmp;
        rnn = static_cast<Rnn*>(cf);
    }
    rnn->aux_reset_gradient_history();
}


void Parser::train_aux_task_one(Rnn *rnn, Tree &tree){
    vector<shared_ptr<Node>> buffer;
    tree.get_buffer(buffer);
//    rnn->auxiliary_gradient_check(buffer, 1e-6);
//    exit(0);
    rnn->train_auxiliary_task(buffer);
}

//void Parser::eval_aux_one(vector<shared_ptr<Node>> &buffer, AuxiliaryTaskEvaluator &evaluator){
//    assert(cf->get_id() == Classifier::RNN);
//    Rnn *rnn = static_cast<Rnn*>(cf);
//    rnn->run_rnn(buffer);
//    rnn->predict_auxiliary_task(buffer);
//    rnn->compare_auxiliary_task(evaluator);
//}

void Parser::predict_tree(vector<shared_ptr<Node>> &buffer, int beamsize, Tree &pred){
    if (cf->get_id() >= Classifier::RNN){
        Rnn *rnn = static_cast<Rnn*>(cf);
        if (rnn->auxiliary_task()){
            rnn->predict_auxiliary_task(buffer, false);
            int field_dep = enc::hodor.get_dep_idx();
            if (field_dep != -1){
                rnn->assign_deplabels(buffer, field_dep);
            }
            rnn->assign_morphological_features(buffer, field_dep);
            if (rnn->auxiliary_tags()){
                rnn->assign_tags(buffer);
            }
        }else{
            rnn->run_rnn(buffer);
        }
    }

    TssBeam beam(beamsize, buffer);

    while (! beam.finished(*ts->grammar_ptr())){
        for (int k = 0; k < beam.size(); k++){
            shared_ptr<ParseState> state;

            beam.get(k, state);

            ts->allowed(*state, buffer.size(), allowed);
            if (! check(allowed)){
                cerr << beam << endl;
            }
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
    if (cf->get_id() >= Classifier::RNN){
        static_cast<Rnn*>(cf)->flush();
    }
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

int Parser::get_classifier_id(){
    return cf->get_id();
}

int Parser::get_transition_system_id(){
    return ts->get_id();
}


Classifier* Parser::classifier_factory(int classifier_id, int num_classes, FeatureExtractor *fe, NeuralNetParameters &params){
    switch (classifier_id){
    case Classifier::PERCEPTRON: return new Perceptron(num_classes);
    case Classifier::FAST_PER: return new FastPerceptron(num_classes);
    case Classifier::FASTER_PER: return new FasterPerceptron(num_classes);
    case Classifier::FASTEST_PER: return new FastestPerceptron(num_classes);
    case Classifier::FFNN : return new NeuralNet(num_classes, fe, params);
    case Classifier::RNN : return new Rnn(num_classes, fe, params);
    case Classifier::RNN_LABEL_STRUCTURE : return new RnnStructureLabel(num_classes, fe, params, false);
    case Classifier::RNN_LABEL_STRUCTURE_LEX : return new RnnStructureLabel(num_classes, fe, params, true);
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
    case TransitionSystem::SHIFT_REDUCE: return new ShiftReduce(grammar);
    case TransitionSystem::LEXICALIZED_MERGE_LABEL_TS: return new LexicalizedMergeLabelTS(grammar);
    case TransitionSystem::MERGE_LABEL_PROJ_TS: return new MergeLabelProjTS(grammar);
    case TransitionSystem::LEXICALIZED_MERGE_LABEL_PROJ_TS: return new LexicalizedMergeLabelProjTS(grammar);
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
    case feature::RNN_FEATURE_EXTRACTOR : return new RnnFeatureExtractor(filename);
    default:
        assert(false && "Unknown feature extractor or not implemented");
    }
    assert(false);
    return nullptr;
}





