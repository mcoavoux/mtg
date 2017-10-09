#include "transition_system.h"

///////////////////////////////////////////////////////
///
/// Derivation
///
/// ///////////////////////////////////////////////////////

Derivation::Derivation(){}
Derivation::Derivation(const vector<Action> &derivation)
    : derivation_(derivation){}

void Derivation::add(const Action &a){
    derivation_.push_back(a);
}
int Derivation::size()const{
    return derivation_.size();
}

void Derivation::pad(int i, const Action &a){
    while (size() < i)
        add(a);
}

void Derivation::clear(){
    derivation_.clear();
}

const Action* Derivation::operator[](int i)const{
    return &derivation_[i];
}

bool Derivation::operator==(const Derivation &d){
    if (size() != d.size()) return false;
    for (int i = 0; i < size(); i++){
        if (derivation_[i] != *(d[i]))
            return false;
    }
    return true;
}
bool Derivation::operator!=(const Derivation &d){
    return ! ((*this) == d);
}

bool Derivation::prefix_equals(const Derivation &d){
    int min = size();
    if (d.size() < min){
        min = d.size();
    }
    for (int i = 0; i < min; i++){
        if (derivation_[i] != *(d[i]))
            return false;
    }
    return true;
}

ostream& operator<<(ostream &os, const Derivation &d){
    os << "[";
    for (int i =0; i < d.size(); i++){
        os << *(d[i]) << ", ";
    }return os << "]";
}

/////////////////////////////////////////////////////////////
///
///
///         Transition System
///
/// ///////////////////////////////////////////////////////

TransitionSystem::TransitionSystem(int s_id):system_id(s_id){}
TransitionSystem::~TransitionSystem(){}



void TransitionSystem::allowed(ParseState &state, int buffer_size, vector<bool> &allowed_actions){

    assert(allowed_actions.size() == num_actions());
    for (int i = 0; i < num_actions(); i++){
        allowed_actions[i] = allowed(state, buffer_size, i);
    }


}

void TransitionSystem::add_action(const Action &a){
    assert(encoder.find(a) == encoder.end() && "TransitionSystem::add_action : Trying to add a known action");
    Action action_with_code(a, actions.size());
    encoder[action_with_code] = action_with_code.code();
    actions.push_back(action_with_code);
}

Action* TransitionSystem::operator[](int i){
    assert(i < actions.size() && i >= 0 && "TransitionSystem::get_action -> index out of range");
    return &actions[i];
}

int TransitionSystem::num_actions(){  return actions.size(); }

void TransitionSystem::next(const shared_ptr<ParseState> &state,
          const Action &a,
          const vector<shared_ptr<Node>> &buffer,
          shared_ptr<ParseState> &newstate){

//    cerr << "state = " << *state << endl;
//    cerr << "action = " << a << endl;
    assert(allowed(*state, buffer.size(), a.code()));
    switch (a.type()){
        case Action::SHIFT: {
            newstate =
                shared_ptr<ParseState>(
                    new ParseState(
                        state,
                        shared_ptr<StackItem>(
                                new StackItem(
                                    state->top_,
                                    buffer[state->buffer_j_])),
                        state->top_,
                        state->buffer_j_+1,
                        a
                    )
                 );
            return;
        }
        case Action::REDUCE_L:
        case Action::REDUCE_R:
        case Action::LEFT:
        case Action::RIGHT:
        {
            shared_ptr<StackItem> newbranch;
            state->grow_new_branch(newbranch);

            shared_ptr<Node> node(new Node(a.label(), {state->mid_->n, state->top_->n}));
            if (a.type() == Action::REDUCE_L || a.type() == Action::LEFT){
                // REMINDER: head assignment has been modified wrt java implementation
                // bc reordering can take place during node construction
                if (state->mid_->n->index() < state->top_->n->index())
                    node->set_h(0);
                else
                    node->set_h(1);
            }else{
                assert( (a.type() == Action::REDUCE_R || a.type() == Action::RIGHT) && "This should not happen, TransitionSystem::next");
                if (state->mid_->n->index() < state->top_->n->index())
                    node->set_h(1);
                else
                    node->set_h(0);
            }
            newstate =
                shared_ptr<ParseState>(
                    new ParseState(
                        state,
                        shared_ptr<StackItem>(
                            new StackItem(
                                newbranch,
                                node)),
                        newbranch,
                        state->buffer_j_,
                        a
                    )
                );
            return;
        }
        case Action::REDUCE_U: {
            vector<shared_ptr<Node>> children;
            shared_ptr<Node> child;
            state->top_->get(child);
            children.push_back(child);
            assert(children.size() > 0);
            shared_ptr<Node> node(new Node(a.label(), children));
            node->set_h(0);
            newstate =
                shared_ptr<ParseState>(
                    new ParseState(
                        state,
                        shared_ptr<StackItem>(
                                new StackItem(
                                    state->top_->predecessor,
                                    node)),
                        state->mid_,
                        state->buffer_j_,
                        a
                    )
                );
            return;
        }
        case Action::GAP:{
            newstate = shared_ptr<ParseState>(
                        new ParseState(
                            state,
                            state->top_,
                            state->mid_->predecessor,
                            state->buffer_j_,
                            a
                        ));
            return;
        }
        case Action::IDLE:
        case Action::GHOST_REDUCE:
        case Action::NULL_ACTION:
        {
            newstate = shared_ptr<ParseState>(
                        new ParseState(
                            state,
                            state->top_,
                            state->mid_,
                            state->buffer_j_,
                            a
                        ));
            return;
        }
        case Action::COMPOUND_GAP:{
            int i = a.order();
            shared_ptr<StackItem> item = state->mid_;
            while (i > 0){
                item = item->predecessor;
                i--;
                assert(item != nullptr);
            }
            newstate = shared_ptr<ParseState>(
                        new ParseState(
                            state,
                            state->top_,
                            item,
                            state->buffer_j_,
                            a
                        ));
            return;
        }
        case Action::MERGE:
        {
            shared_ptr<StackItem> newbranch;
            state->grow_new_branch(newbranch);

            vector<shared_ptr<Node>> newchildren;
//            cerr << state->mid_->n->has_label() << endl;
//            cerr << state->mid_->n->is_preterminal() << endl;
//            cerr << *state->mid_->n << endl;
//            cerr << state->top_->n->has_label() << endl;
//            cerr << state->top_->n->is_preterminal() << endl;
//            cerr << *state->top_->n << endl;

            if (state->mid_->n->has_label() || state->mid_->n->is_preterminal()){
                newchildren.push_back(state->mid_->n);
            }else{
                state->mid_->n->get_children(newchildren);
            }
            if (state->top_->n->has_label() || state->top_->n->is_preterminal()){
                newchildren.push_back(state->top_->n);
            }else{
                state->top_->n->get_children(newchildren);
            }
            assert(newchildren.size() > 1);
            shared_ptr<Node> node(new Node(newchildren));
            newstate =
                shared_ptr<ParseState>(
                    new ParseState(
                        state,
                        shared_ptr<StackItem>(
                            new StackItem(
                                newbranch,
                                node)),
                        newbranch,
                        state->buffer_j_,
                        a
                    )
                );
            return;
        }
        case Action::LABEL:
        {
            assert ((! state->top_->n->has_label() || state->top_->n->is_preterminal()) && "Node already has a label");
            if (state->top_->n->is_preterminal()){
                vector<shared_ptr<Node>> newchildren{state->top_->n};
                assert(newchildren.size() > 0);
                shared_ptr<Node> node(new Node(a.label(), newchildren));
                node->set_h(0);
                newstate =
                        shared_ptr<ParseState>(
                            new ParseState(
                                state,
                                shared_ptr<StackItem>(
                                    new StackItem(
                                        state->top_->predecessor,
                                        node)),
                                state->mid_,
                                state->buffer_j_,
                                a
                                )
                            );
            }else{
                assert(! state->top_->n->has_label());
                vector<shared_ptr<Node>> newchildren;
                state->top_->n->get_children(newchildren);
//                if (newchildren.size() <= 1){
//                    cerr << *state << endl;
//                }
                assert(newchildren.size() > 1);
                shared_ptr<Node> node(new Node(a.label(), newchildren));
                int head_idx = state->top_->n->h();
                if (head_idx > -1){
                    node->set_h(head_idx);
                }
                newstate =
                        shared_ptr<ParseState>(
                            new ParseState(
                                state,
                                shared_ptr<StackItem>(
                                    new StackItem(
                                        state->top_->predecessor,
                                        node)),
                                state->mid_,
                                state->buffer_j_,
                                a
                                )
                            );
            }
            return;
        }
        default:
            assert(false && "Unknown or illegal action");
    }
}

Grammar* TransitionSystem::grammar_ptr(){
    return &grammar;
}


void TransitionSystem::print_transitions(ostream &os) const{
    for (int i = 0; i < actions.size(); i++){
        os << actions.at(i) << endl;
    }
}




void TransitionSystem::export_model(const string &outdir){
    ofstream out1(outdir + "/system_id");
    out1 << system_id << endl;
    out1.close();
    grammar.export_model(outdir);

    ofstream out2(outdir + "/system_actions");
    for (int i = 0; i < actions.size(); i++){
        out2 << actions[i].type() << " " << actions[i].label() << " " << actions[i].code() << endl;
    }
    out2.close();
}

TransitionSystem* TransitionSystem::import_model(const string &outdir){
    Grammar grammar(outdir);
    ifstream in1(outdir + "/system_id");
    int s_id;
    in1 >> s_id;
    in1.close();

    TransitionSystem *ts = nullptr;
    switch (s_id){
    case GAP_TS:
        ts = new GapTS(grammar);
        break;
    case CGAP_TS:
        ts = new CompoundGapTS(grammar);
        break;
    case MERGE_LABEL_TS:
        ts = new MergeLabelTS(grammar);
        break;
    case SHIFT_REDUCE:
        ts = new ShiftReduce(grammar);
        break;
    case LEXICALIZED_MERGE_LABEL_TS :
        ts = new LexicalizedMergeLabelTS(grammar);
        break;
    case MERGE_LABEL_PROJ_TS:
        ts = new MergeLabelProjTS(grammar);
        break;
    case LEXICALIZED_MERGE_LABEL_PROJ_TS:
        ts = new LexicalizedMergeLabelProjTS(grammar);
        break;
    default:
        assert(false && "Unknown transition system");
    }

    ts->actions.clear();
    ifstream in2(outdir + "/system_actions");
    string buffer;
    while (getline(in2, buffer)){
//        boost::char_separator<char> sep(" ");
//        boost::tokenizer<boost::char_separator<char>> toks(buffer, sep);
//        vector<string> tokens(toks.begin(), toks.end());
        vector<string> tokens;
        str::split(buffer, " ", "", tokens);
        assert(tokens.size() == 3);
        ts->actions.push_back(Action(stoi(tokens[0]), stoi(tokens[1]), stoi(tokens[2])));
    }
    in2.close();
    for (int i = 0; i < ts->actions.size(); i++){
        ts->encoder[ts->actions[i]] = i;
    }
    return ts;
}




void TransitionSystem::shift(vector<shared_ptr<Node>> &stack, std::deque<shared_ptr<Node>> &deque, vector<shared_ptr<Node>> &buffer, int &j){
    while (deque.size() > 0){
        stack.push_back(deque.front());
        deque.pop_front();
    }
    deque.push_back(buffer[j++]);
}

void TransitionSystem::test_derivation_extraction(const string &tbk_filename, const string &hr_filename, int transition_system){
    cerr << "Reading treebank" << endl;
    enc::hodor.reset();
    Treebank tbk(tbk_filename, Treebank::ENCODE_EVERYTHING, Treebank::DISCBRACKET, true);
    Grammar grammar;

    if (transition_system == MERGE_LABEL_TS){
        grammar.do_not_binarise();
    }

    HeadRules hr(hr_filename, &grammar);
    tbk.annotate_heads(hr);
    tbk.transform(grammar);

    TransitionSystem *ts = nullptr;
    switch (transition_system) {
        case GAP_TS:    ts = new GapTS(grammar); break;
        case CGAP_TS:   ts = new CompoundGapTS(grammar); break;
        case MERGE_LABEL_TS:   ts = new MergeLabelTS(grammar); break;
        default: assert(false && "Unknown transition system");
    }

    vector<Derivation> derivations(tbk.size());

    for (int i = 0; i < tbk.size(); i++){
        tbk[i]->annotate_parent_ptr();
        ts->compute_derivation(*tbk[i],derivations[i]);
    }

    for (int i = 0; i < tbk.size(); i++){
        cout << *tbk[i] << endl;
        cout << derivations[i] << endl;
    }

    for (int i = 0; i < ts->actions.size(); i++){
        cerr << ts->actions[i].code() << "  " << ts->actions[i] << endl;
    }
    cerr << ts->grammar << endl;
    delete ts;
}



bool TransitionSystem::has_head(shared_ptr<Node> &parent, shared_ptr<Node> child){
    shared_ptr<Node> head;
    parent->head(head);
    if (child->is_preterminal()){
        return head == child;
    }
    shared_ptr<Node> headc;
    child->head(headc);
    return head == headc;
}


void TransitionSystem::print_stats(const string &train_file, const string &outfile, const string &headrules_filename, int tbk_format, int transition_system){
    // TODO: update with new transition system
    cerr << "Reading treebank" << endl;
    Treebank tbk(train_file, Treebank::ENCODE_EVERYTHING, tbk_format, true);
    Grammar grammar;

    Treebank tbk_bin(tbk);
    if (tbk_format == Treebank::DISCBRACKET){
        HeadRules hr(headrules_filename, &grammar);
        tbk_bin.annotate_heads(hr);
    }
    tbk_bin.transform(grammar);

    TreebankStats tbk_stats(tbk);
    TreebankStats tbk_bin_stats(tbk_bin);

    ofstream out1(outfile + "/stats.md");
    out1 << "Stats n-ary corpus" << endl;
    out1 << "==================" << endl << endl;
    out1 << tbk_stats << endl << endl;

    out1 << "Stats binarized corpus" << endl;
    out1 << "======================" << endl << endl;
    out1 << tbk_bin_stats << endl << endl;

    print_swap_stats(tbk_bin, out1, grammar);

    //enum {GAP_TS, CGAP_TS, MERGE_LABEL_TS, SHIFT_REDUCE, LEXICALIZED_MERGE_LABEL_TS, MERGE_LABEL_PROJ_TS, LEXICALIZED_MERGE_LABEL_PROJ_TS};// transition system identifiers
    string ts_str[7] = {"Gap transition system", "Compound gap transition system", "Merge Label transition system", "Shift Reduce", "Lexicalized Merge Label transition system",
                       "Merge Label Projective transition system", "Lexicalized Merge Label projective transition system"};
    string ts_str_short[7] = {"gap", "cgap", "mergelabel", "shiftred", "lex_mergelabel", "proj_merge_label", "lex_proj_merge_label"};
    vector<int> tss;
    if (is_projective(transition_system)){
        tss = {SHIFT_REDUCE, LEXICALIZED_MERGE_LABEL_PROJ_TS, MERGE_LABEL_PROJ_TS};
    }else{
        tss = {GAP_TS, CGAP_TS, LEXICALIZED_MERGE_LABEL_TS, MERGE_LABEL_TS};
    }

    for (int id : tss){
        //cerr << "ID " << id << endl;
//        if (transition_system == SHIFT_REDUCE && id != SHIFT_REDUCE){ continue;}
//        if (transition_system != SHIFT_REDUCE && id == SHIFT_REDUCE){ continue;}
        TransitionSystem *ts = nullptr;

        switch (id){
        case SHIFT_REDUCE: ts = new ShiftReduce(grammar);
        case GAP_TS: ts = new GapTS(grammar); break;
        case CGAP_TS: ts = new CompoundGapTS(grammar); break;
        case LEXICALIZED_MERGE_LABEL_TS: ts = new LexicalizedMergeLabelTS(grammar);  break;
        case LEXICALIZED_MERGE_LABEL_PROJ_TS: ts = new LexicalizedMergeLabelProjTS(grammar);  break;
        case MERGE_LABEL_PROJ_TS:
        case MERGE_LABEL_TS: {
            tbk_bin.clear();
            for (int i = 0; i < tbk.size(); i++){
                Tree t;
                tbk[i]->copy(t);
                tbk_bin.add_tree(t);
            }
            Grammar gram;
            gram.do_not_binarise();
            tbk_bin.transform(gram);
            if (id == MERGE_LABEL_TS){
                ts = new MergeLabelTS(gram);
            }else{
                ts = new MergeLabelProjTS(gram);
            }
            break;
        }

        default: assert(false && "unknown transition system");
        }


        vector<Derivation> derivations(tbk_bin.size());

        for (int i = 0; i < tbk_bin.size(); i++){
            tbk_bin[i]->annotate_parent_ptr();
            ts->compute_derivation(*tbk_bin[i],derivations[i]);
        }

        vector<int> action_freq(ts->num_actions(), 0);
        int longest_derivation = 0;
        int gap_longest_sequence = 0;
        double total_number_of_actions = 0;

        int num_gap_seqs = 0;
        int sum_gap_seqs = 0;

        for (int i = 0; i < derivations.size(); i++){
            total_number_of_actions += derivations[i].size();
            longest_derivation = std::max(longest_derivation, derivations[i].size());
            int gap_seq = 0;
            for (int k = 0; k < derivations[i].size(); k++){
                const Action *a = derivations[i][k];
                action_freq[a->code()] ++;
                if (a->type() == Action::GAP){
                    gap_seq ++;
                }else{
                    gap_longest_sequence = std::max(gap_longest_sequence, gap_seq);
                    if (gap_seq > 0){
                        sum_gap_seqs += gap_seq;
                        num_gap_seqs += 1;
                    }
                    gap_seq = 0;
                }
            }
        }

        vector<int> connected_components_stats(100,0);
        incrementality_stats(derivations, connected_components_stats);

        out1 << endl << "# " << ts_str[id] << endl << endl;
        out1 << "- longest derivation: " << longest_derivation << endl;
        out1 << "- longest consecutive gap sequence: " << gap_longest_sequence << endl;
        out1 << "- average derivation length (over all sentences): " << (total_number_of_actions / tbk_bin.size()) << endl;
        out1 << "- average derivation length (over all tokens, i.e. for a sentence of length n): " << (total_number_of_actions / tbk_bin_stats.n_tokens) << " n" << endl<< endl;
        out1 << "- number of action types: " << ts->num_actions() << endl;
        out1 << "- average length of gap seq: " << (1.0 * sum_gap_seqs) / num_gap_seqs << endl;

//        out1 << "Incrementality stats" << endl;
//        out1 << "--------------------" << endl << endl;


        double sum_connex = 0;
        double tot_eff    = 0;
        for (int i = 0; i < connected_components_stats.size(); i++){
            sum_connex += i * connected_components_stats[i];
            tot_eff += connected_components_stats[i];
        }
        out1 <<"- average incrementality = " << (sum_connex / tot_eff) << endl;

        ofstream out_ts_stats(outfile + "/" + ts_str_short[id] + "_stats.txt");
        for (int i = 0; i < connected_components_stats.size(); i++){
            if (connected_components_stats[i] > 0){
                out_ts_stats << i << "\t" << connected_components_stats[i] << "\t" << (connected_components_stats[i]/tot_eff) << endl;
            }
        }


//        out1 << endl;
//        out1 << "Action frequencies" << endl;
//        out1 << "------------------" << endl << endl;

        ofstream out_ts(outfile + "/" + ts_str_short[id] + ".txt");
        for (int i = 0; i < action_freq.size(); i++){
            if (action_freq[i] > 0){
                //out1 << ts->actions[i] << "\t" << action_freq[i] << endl;
                //out1 << action_freq[i] << " (" << (action_freq[i] * 100.0)/ total_number_of_actions << "%)\t" << ts->actions[i] << endl;
                out_ts << action_freq[i] << "\t" << ts->actions[i] << "\t" << (action_freq[i] * 100.0)/ total_number_of_actions << endl;
            }
//            else{
//                cerr << "0 freq action " << action_freq[i]  << " \t " << ts->actions[i] << endl;
//            }
        }
        out_ts.close();

        delete ts;
    }

    out1.close();
}

void TransitionSystem::incrementality_stats(vector<Derivation> &derivations, vector<int> &counts){
    for (int i = 0; i < counts.size(); i++){
        counts[i] = 0;
    }
    //{SHIFT, REDUCE_L, REDUCE_R, REDUCE_U, GAP, IDLE, GHOST_REDUCE, COMPOUND_GAP, NULL_ACTION, MERGE, LABEL, RIGHT, LEFT};
    for (int d = 0; d < derivations.size(); d++){
        int connected_components = 0;
        counts[connected_components] += 1;
        for (int i = 0; i < derivations[d].size(); i++){
            assert(connected_components >= 0 && connected_components < counts.size());
            switch(derivations[d][i]->type()){
            case Action::SHIFT:
                counts[++connected_components] += 1;
                break;
            case Action::REDUCE_L:
            case Action::REDUCE_R:
            case Action::MERGE:
            case Action::RIGHT:
            case Action::LEFT:
                counts[--connected_components] += 1;
                break;
            case Action::REDUCE_U:
            case Action::GAP:
            case Action::NULL_ACTION:
            case Action::LABEL:
            case Action::GHOST_REDUCE:
            case Action::COMPOUND_GAP:
                counts[connected_components] += 1;
                break;
            case Action::IDLE:
                assert(false && "Should not happen in this setting...");
            }
        }
    }
}


void TransitionSystem::print_swap_stats(Treebank &tbk_bin, ofstream & outfile, Grammar &grammar){
    int n_swaps = 0;
    int n_cswaps = 0;
    int n_shifts = 0;
    // these ones for sanity check / comparison with gap
    int n_binary_reduce = 0;
    int n_unary_reduce = 0;

    int max_swaps = 0;
    int longest_derivation = 0;
    int longest_derivation_cswap = 0;
    int n_tokens = 0;

    for (int tree_id = 0; tree_id < tbk_bin.size(); tree_id ++){
        int derivation_length = 0;
        int derivation_length_cswap = 0;

        tbk_bin[tree_id]->annotate_parent_ptr();
        vector<shared_ptr<Node>> buffer;
        tbk_bin[tree_id]->get_buffer(buffer);

        n_tokens += buffer.size();

        std::deque<shared_ptr<Node>> deque;

        for (int i = 0; i < buffer.size(); i++){
            deque.push_back(buffer[i]);
        }

        vector<shared_ptr<Node>> stack;


        stack.push_back(deque.front());
        deque.pop_front();
        n_shifts += 1;
        derivation_length += 1;
        derivation_length_cswap += 1;

        while (! grammar.is_axiom(stack.front()->label())){
            shared_ptr<Node> s0 = stack.back();
            shared_ptr<Node> s0p;
            s0->get_parent(s0p);

            if (s0p->arity() == 1){
                n_unary_reduce += 1;
                derivation_length += 1;
                derivation_length_cswap += 1;
                stack.pop_back();
                stack.push_back(s0p);
                continue;
            }
            if (stack.size() > 1){
                shared_ptr<Node> s1 = stack.at(stack.size() - 2);
                shared_ptr<Node> s1p;
                s1->get_parent(s1p);
                if (s0p == s1p){
                    stack.pop_back();
                    stack.pop_back();
                    stack.push_back(s0p);
                    n_binary_reduce += 1;
                    derivation_length += 1;
                    derivation_length_cswap += 1;
                    continue;
                }
            }
            shared_ptr<Node> s0p_right_child;
            s0p->get(1, s0p_right_child);
            int s = 0;
            while (s0p_right_child->index() != deque.front()->index()){
                s += 1;
                stack.push_back(deque.front());
                deque.pop_front();
            }
            if (s > max_swaps){
                max_swaps = s;
            }
            n_shifts += s + 1;
            n_swaps += s;
            if (s > 0){
                n_cswaps += 1;
                derivation_length_cswap += s + 2;
            }else{
                derivation_length_cswap += 1;
            }
            derivation_length += s + s + 1;
            shared_ptr<Node> last_to_shift = deque.front();
            deque.pop_front();
            for (int i = 0; i < s; i++){
                deque.push_front(stack.back());
                stack.pop_back();
            }
            stack.push_back(last_to_shift);
        }
        if (derivation_length > longest_derivation){
            longest_derivation = derivation_length;
        }
        if (derivation_length_cswap > longest_derivation_cswap){
            longest_derivation_cswap = derivation_length_cswap;
        }
    }

    double total_num_actions = n_shifts + n_swaps + n_binary_reduce + n_unary_reduce;
    double total_num_actions_cswaps = n_shifts + n_cswaps + n_binary_reduce + n_unary_reduce;

    outfile << endl;
    outfile << "Swap statistics" << endl;
    outfile << "===============" << endl << endl;

    outfile << "longest derivation       \t" << longest_derivation << endl;
    outfile << "longest swap sequence    \t" << max_swaps << endl;
    outfile << "avg derivation length sw \t" << total_num_actions / tbk_bin.size() << endl;
    outfile << "avg length (n tokens) sw \t" << total_num_actions / n_tokens << endl;

    outfile << "longest derivation       \t" << longest_derivation_cswap << endl;
    outfile << "avg derivation length csw\t" << total_num_actions_cswaps / tbk_bin.size() << endl;
    outfile << "avg length (n tokens) csw\t" << total_num_actions_cswaps/ n_tokens << endl << endl;


    outfile << "shifts    \t" << n_shifts << endl;
    outfile << "binary red\t" << n_binary_reduce << endl;
    outfile << "unary red \t" << n_unary_reduce << endl;
    outfile << "swaps     \t" << n_swaps << endl;
    outfile << "cswaps     \t" << n_cswaps << endl;
}


bool TransitionSystem::is_projective(int transition_system_id){
    switch(transition_system_id){
    case GAP_TS: return false;
    case CGAP_TS: return false;
    case MERGE_LABEL_TS: return false;
    case SHIFT_REDUCE: return true;
    case LEXICALIZED_MERGE_LABEL_TS: return false;
    case MERGE_LABEL_PROJ_TS: return true;
    case LEXICALIZED_MERGE_LABEL_PROJ_TS: return true;
    default:
        cerr << "Error, unknown transition system, aborting" << endl;
        exit(1);
    }
}

/////////////////////////////////////////////////////////////
///
///
///         GapTS
///
/// //////////////////////////////////////////////////////////


GapTS::GapTS(const Grammar &g):TransitionSystem(GAP_TS){
    for (int type : {Action::SHIFT, Action::GAP, Action::IDLE}){
        add_action(Action(type, -1));
    }
    grammar = g;
}

int GapTS::get_id(){
    return TransitionSystem::GAP_TS;
}

Action* GapTS::get_idle(){
    return &actions[IDLE_I];
}

void GapTS::compute_derivation(Tree &tree, Derivation &derivation){

    vector<Action> deriv;
    vector<shared_ptr<Node>> buffer;
    tree.get_buffer(buffer);

    vector<shared_ptr<Node>> stack;
    std::deque<shared_ptr<Node>> deque;
    int j = 0;

    while (j < buffer.size() ||
           deque.size() != 1 ||
           stack.size() > 0  ||
           ! grammar.is_axiom(deque.back()->label())){


        if (stack.size() == 0 && deque.size() == 0){
            deriv.push_back(actions[SHIFT_I]);
            shift(stack, deque, buffer, j);
            continue;
        }

        shared_ptr<Node> s0 = deque.back();
        shared_ptr<Node> s0p;
        s0->get_parent(s0p);

        if (s0p->arity() == 1){
            assert(deque.size() == 1 && "Error in derivation extraction");
            deriv.push_back(Action(Action::REDUCE_U, s0p->label(), -1));
            deque.pop_back();
            deque.push_back(s0p);
            continue;
        }

        if (stack.size() > 0){
            shared_ptr<Node> s1 = stack.back();
            shared_ptr<Node> s1p;
            s1->get_parent(s1p);

            if (s0p == s1p){
                // REMINDER: there might be something incorrect here, see java code -> no correct because use of h_ field to get head
                int hpos = s0p->h();                // in constituent, head is hpos when children are sorted by left corner index
                if (s0->index() < s1->index()){     // if reduced nodes are not sorted, head position is reversed
                    hpos = 1 - hpos;                // REMIDNER: for some reason, commenting out these 3 lines has little to no effect on accuracy,
                }                                   // strange .... you might wanna check head assignement again
                if (hpos == 0){
                    assert( has_head( s0p, s1 ));
                    deriv.push_back(Action(Action::REDUCE_L, s0p->label(), -1));
                }else{
                    assert( has_head( s0p, s0 ));
                    assert ( hpos == 1 );
                    deriv.push_back(Action(Action::REDUCE_R, s0p->label(), -1));
                }

                stack.pop_back();
                deque.pop_back();
                while(deque.size() > 0){
                    stack.push_back(deque.front());
                    deque.pop_front();
                }
                deque.push_back(s0p);
                continue;
            }
        }

        if (stack.size() > 1){
            int gaps = -1;
            for (int i = 0; i < stack.size(); i++){         // TODO: who on earth coded this ? use parent ptr instead
                shared_ptr<Node> candidate = stack[stack.size() - 1 - i];
                shared_ptr<Node> f1,f2;
                s0p->get(0,f1);
                s0p->get(1,f2);
                if (candidate == f1 || candidate == f2){
                    gaps = i;
                    break;
                }
            }
            if (gaps != -1){
                for (int i = 0; i < gaps; i++){
                    deriv.push_back(actions[GAP_I]);
                    deque.push_front(stack.back());
                    stack.pop_back();
                }
                continue;
            }
        }
        deriv.push_back(actions[SHIFT_I]);
        shift(stack, deque, buffer, j);
    }

    for (int i = 0; i < deriv.size(); i++){
        if (encoder.find(deriv[i]) == encoder.end()){
            add_action(deriv[i]);
        }
        int j = encoder[deriv[i]];
        deriv[i] = actions[j];
    }
    derivation = Derivation(deriv);

}


bool GapTS::allowed(ParseState &state, int buffer_size, int i){
    const Action a = actions[i];
    switch(a.type()){
    case Action::SHIFT:
        return state.is_init() ||
                (state.buffer_j_ < buffer_size &&
                 ! (state.last_action_.code() == GAP_I));
    case Action::REDUCE_L:
    case Action::REDUCE_R: return state.top_ != nullptr &&
                state.mid_ != nullptr &&
                allowed_reduce(state, a, buffer_size);
    case Action::REDUCE_U: return state.top_ != nullptr &&
                allowed_reduce_u(state, a, buffer_size);
    case Action::GAP:      return state.top_ != nullptr &&
                state.mid_ != nullptr &&
                allowed_gap(state);
    case Action::IDLE:     return state.is_final(grammar, buffer_size);
    }
    assert(false && "GapTS: unknown action");
    return false;
}

bool GapTS::allowed_reduce(ParseState &state, const Action &a, int buffer_size){
    if (grammar.is_tmp(state.top_->n->label()) && grammar.is_tmp(state.mid_->n->label())) return false; // cannot reduce 2 tmps
    if (buffer_size == state.buffer_j_ && state.mid_->is_bottom() && state.top_->predecessor == state.mid_){ // if last reduce -> to axiom
        return grammar.is_axiom(a.label());
    }
    // REMINDER: why is this line decommented ??
    if (grammar.is_axiom(a.label())){               // if axiom -> it must be the last reduce
        return buffer_size == state.buffer_j_ &&
               state.mid_->is_bottom() &&
               state.top_->predecessor == state.mid_;
    }
    if (grammar.is_tmp(a.label())){ // there must be a non tmp symbol in the stack
        if (buffer_size == state.buffer_j_){
            bool has_ntmp = false;
            shared_ptr<StackItem> si(state.top_->predecessor);
            while (si != nullptr){
                if (si != state.mid_ && ! (grammar.is_tmp(si->n->label()))){
                    has_ntmp = true;
                    break;
                }
                si = si->predecessor;
            }
            if (! has_ntmp) return false;
        }
        if (a.type() == Action::REDUCE_L) return ! grammar.is_tmp(state.top_->n->label());
        if (a.type() == Action::REDUCE_R) return ! grammar.is_tmp(state.mid_->n->label());
    }
    return true;
}

bool GapTS::allowed_reduce_u(ParseState &state, const Action &a, int buffer_size){// TODO: something imprecise here: in every case, last action is shift
    return state.last_action_.code() == SHIFT_I
        || (state.top_->is_bottom() &&
            state.buffer_j_ == buffer_size &&               // TODO: you need to correct this, if axiom -> one word sentence, if one word sentence -> axiom
            ! grammar.is_axiom(state.top_->n->label()) &&
            grammar.is_axiom(a.label()));
}

bool GapTS::allowed_gap(ParseState &state){
    if (state.mid_->is_bottom()) return false;
    if (! grammar.is_tmp(state.top_->n->label())){ return true; }

    shared_ptr<StackItem> si(state.mid_->predecessor);
    while ( si != nullptr ){
        if (! grammar.is_tmp(si->n->label()))
            return true;
        si = si->predecessor;
    }
    return false;
}



//////////////////////////////////////////////////////////
///
///
///
/// Compound Gap transition system
///
///
//////////////////////////////////////////////////////////



CompoundGapTS::CompoundGapTS(const Grammar &g):TransitionSystem(CGAP_TS){
    for (int type : {Action::SHIFT, Action::GHOST_REDUCE, Action::IDLE}){
        add_action(Action(type, -1));
    }
    grammar = g;
}

int CompoundGapTS::get_id(){
    return TransitionSystem::CGAP_TS;
}

Action* CompoundGapTS::get_idle(){
    return &actions[IDLE_I];
}

void CompoundGapTS::compute_derivation(Tree &tree, Derivation &derivation){

    vector<Action> deriv;
    vector<shared_ptr<Node>> buffer;
    tree.get_buffer(buffer);

    vector<shared_ptr<Node>> stack;
    std::deque<shared_ptr<Node>> deque;
    int j = 0;

    deriv.push_back(actions[SHIFT_I]);
    shift(stack, deque, buffer, j);


    while (j < buffer.size() ||
           deque.size() != 1 ||
           stack.size() > 0  ||
           ! grammar.is_axiom(deque.back()->label())){

        shared_ptr<Node> s0 = deque.back();
        shared_ptr<Node> s0p;
        s0->get_parent(s0p);

        if (deriv.back().type() == Action::SHIFT){
            if (s0p->arity() == 1){
                assert(deque.size() == 1 && "Error in derivation extraction");
                deriv.push_back(Action(Action::REDUCE_U, s0p->label(), -1));
                deque.pop_back();
                deque.push_back(s0p);
                continue;
            }else{
                deriv.push_back(Action(Action::GHOST_REDUCE, -1));
                continue;
            }
        }

        if (s0p->arity() == 1){
            assert(deque.size() == 1 && stack.size() == 0 && grammar.is_axiom(s0p->label()) && "Error in derivation extraction");
            deriv.push_back(Action(Action::REDUCE_U, s0p->label(), -1));
            deque.pop_back();
            deque.push_back(s0p);
            continue;
        }

        // From this point: chose shift or (gap + reduce)
        if (stack.size() > 0){
            bool keep_going = true;
            for (int gaporder = 0; gaporder < stack.size() && keep_going; gaporder ++){
                shared_ptr<Node> sn = stack[stack.size()-1-gaporder];
                shared_ptr<Node> snp;
                sn->get_parent(snp);
                if (s0p == snp){

                    deriv.push_back(Action(Action::COMPOUND_GAP, gaporder, -1));
                    for (int i = 0; i < gaporder; i++){  // gaporder consecutive gaps
                        deque.push_front(stack.back());
                        stack.pop_back();
                    }

                    int hpos = s0p->h();        // REMINDER: here too, probably incorrect -> no see above
                    if (s0->index() < sn->index()){
                        hpos = 1 - hpos;
                    }
                    if (hpos == 0){
                        deriv.push_back(Action(Action::REDUCE_L, s0p->label(), -1));
                    }else{
                        assert ( hpos == 1 );
                        deriv.push_back(Action(Action::REDUCE_R, s0p->label(), -1));
                    }

                    stack.pop_back();
                    deque.pop_back();
                    while(deque.size() > 0){
                        stack.push_back(deque.front());
                        deque.pop_front();
                    }
                    deque.push_back(s0p);
                    keep_going = false;
                }
            }
            if (! keep_going){
                continue;
            }
        }

        // No reduce has been found, time to shift
        deriv.push_back(actions[SHIFT_I]);
        shift(stack, deque, buffer, j);
    }

    for (int i = 0; i < deriv.size(); i++){
        if (encoder.find(deriv[i]) == encoder.end()){
            add_action(deriv[i]);
        }
        int j = encoder[deriv[i]];
        deriv[i] = actions[j];
    }
    derivation = Derivation(deriv);
}


bool CompoundGapTS::allowed(ParseState &state, int buffer_size, int i){
    const Action a = actions[i];
    switch(a.type()){
        case Action::SHIFT:
            return state.is_init() ||
                  (state.buffer_j_ < buffer_size &&
                   state.last_action_.type() != Action::COMPOUND_GAP &&
                    state.last_action_.type() != Action::SHIFT);
        case Action::REDUCE_L:
        case Action::REDUCE_R:
           return state.last_action_.type() == Action::COMPOUND_GAP &&
                  state.top_ != nullptr &&  // TODO: these two should probably be commented out
                  state.mid_ != nullptr &&  //
                  allowed_reduce(state, a, buffer_size);
        case Action::REDUCE_U:
            return state.last_action_.type() == Action::SHIFT &&
                   //state.top_ != nullptr && // obvious
                   allowed_reduce_u(state, a, buffer_size);
        case Action::COMPOUND_GAP:
            return state.top_ != nullptr &&
                   state.mid_ != nullptr &&
                   state.last_action_.type() != Action::COMPOUND_GAP &&
                   state.last_action_.type() != Action::SHIFT &&
                   allowed_cgap(state, a, buffer_size);
        case Action::IDLE:
            return state.is_final(grammar, buffer_size);
        case Action::GHOST_REDUCE :
            return state.last_action_.type() == Action::SHIFT && // impossible to GR if need RU to root
                  (state.buffer_j_ != buffer_size ||
                   ! state.top_->is_bottom());      // REMINDER: add not axiom -> no need
    }
    assert(false && "CompoundGapTS: unknown action");
    return false;
}

bool CompoundGapTS::allowed_reduce(ParseState &state, const Action &a, int buffer_size){
    // Now's the delicate part
    // From last step, we know that there is at least one allowed reduce

    assert(!(grammar.is_tmp(state.top_->n->label()) && grammar.is_tmp(state.mid_->n->label()))); // this should have been tested previously

    // this part doesn't change ...
    if (buffer_size == state.buffer_j_ &&
            state.mid_->is_bottom() &&
            state.top_->predecessor == state.mid_){ // if last reduce -> to axiom
        return grammar.is_axiom(a.label());
    }
    // REMINDER: why is this line decommented ??
    if (grammar.is_axiom(a.label())){               // if axiom -> it must be the last reduce
        return buffer_size == state.buffer_j_ &&
               state.mid_->is_bottom() &&
               state.top_->predecessor == state.mid_;
    } // ... till here

    // so, label is not axiom
    if (grammar.is_tmp(a.label())){ // there must be a non tmp symbol in the stack
        if (buffer_size == state.buffer_j_){
            bool has_ntmp = false;
            shared_ptr<StackItem> si(state.top_->predecessor);
            while (si != nullptr){
                if (si != state.mid_ && ! (grammar.is_tmp(si->n->label()))){
                    has_ntmp = true;
                    break;
                }
                si = si->predecessor;
            }
            if (! has_ntmp) return false;
        }
        if (a.type() == Action::REDUCE_L) return ! grammar.is_tmp(state.top_->n->label());
        if (a.type() == Action::REDUCE_R) return ! grammar.is_tmp(state.mid_->n->label());
    }
    // if label is not a tmp, we're safe
    return true;
}

bool CompoundGapTS::allowed_reduce_u(ParseState &state, const Action &a, int buffer_size){
    if (grammar.is_axiom(a.label())){
        return state.top_->is_bottom() &&
               state.buffer_j_ == buffer_size &&
               ! grammar.is_axiom(state.top_->n->label());
    }
    return state.buffer_j_ != buffer_size || ! state.top_->is_bottom();
}

bool CompoundGapTS::allowed_cgap(ParseState &state, const Action &action, int buffer_size){

    int i = action.order();
    shared_ptr<StackItem> si(state.mid_);
    while (i > 0 && si != nullptr){
        si = si->predecessor;
        i--;
    }
    if (si == nullptr){
        return false;
    }
    return !(grammar.is_tmp(state.top_->n->label()) && grammar.is_tmp(si->n->label()));
}





//////////////////////////////////////////////////////////
///
///
///
/// Merge Label transition system
///
///
//////////////////////////////////////////////////////////



MergeLabelTS::MergeLabelTS(const Grammar &g):TransitionSystem(MERGE_LABEL_TS){
    for (int type : {Action::SHIFT, Action::MERGE, Action::IDLE, Action::GAP, Action::NULL_ACTION}){
        add_action(Action(type, -1));
    }
    grammar = g;
}

int MergeLabelTS::get_id(){
    return TransitionSystem::MERGE_LABEL_TS;
}

Action* MergeLabelTS::get_idle(){
    return &actions[IDLE_I];
}

void MergeLabelTS::compute_derivation(Tree &tree, Derivation &derivation){

    vector<Action> deriv;
    vector<shared_ptr<Node>> buffer;
    tree.get_buffer(buffer);

    vector<shared_ptr<Node>> stack;
    std::deque<shared_ptr<Node>> deque;
    int j = 0;

    deriv.push_back(actions[SHIFT_I]);
    shift(stack, deque, buffer, j);



    bool structure = false;

    while (j < buffer.size() ||
           deque.size() != 1 ||
           stack.size() > 0  ||
           ! grammar.is_axiom(deque.back()->label())){

        if (structure){
            //shift
            // gap
            // merge
            if (stack.size() == 0 && deque.size() == 0){
                deriv.push_back(actions[SHIFT_I]);
                shift(stack, deque, buffer, j);
                structure = false;
                continue;
            }

            shared_ptr<Node> s0 = deque.back();
            shared_ptr<Node> s0p;
            s0->get_parent(s0p);

            if (stack.size() > 0){
                shared_ptr<Node> s1 = stack.back();
                shared_ptr<Node> s1p;
                s1->get_parent(s1p);

                if (s0p == s1p){
                    deriv.push_back(Action(Action::MERGE, -1));

                    stack.pop_back();
                    deque.pop_back();

                    while(deque.size() > 0){
                        stack.push_back(deque.front());
                        deque.pop_front();
                    }

                    vector<shared_ptr<Node>> newchildren;

                    if (s0->has_label() || s0->is_preterminal()){
                        newchildren.push_back(s0);
                    }else{
                        s0->get_children(newchildren);
                    }
                    if (s1->has_label() || s1->is_preterminal()){
                        newchildren.push_back(s1);
                    }else{
                        s1->get_children(newchildren);
                    }
//                    if (newchildren.size() == 0){
//                        cerr << *s0 << endl;
//                        cerr << *s1 << endl;
//                    }
                    assert(newchildren.size() > 0);
                    shared_ptr<Node> node = shared_ptr<Node>(new Node(newchildren));
                    node->set_parent(s0p);       // intermediary node
                    deque.push_back(node);
                    structure = false;
                    continue;
                }
            }

            if (stack.size() > 1){
                int gaps = -1;
                for (int i = 0; i < stack.size(); i++){         // TODO: who on earth coded this ? use parent ptr instead
                    shared_ptr<Node> candidate = stack[stack.size() - 1 - i];
                    shared_ptr<Node> candparent;
                    candidate->get_parent(candparent);
                    if (candparent == s0p){
                        gaps = i;
                        break;
                    }
                }
                if (gaps != -1){
                    for (int i = 0; i < gaps; i++){
                        deriv.push_back(actions[GAP_I]);
                        deque.push_front(stack.back());
                        stack.pop_back();
                    }
                    continue;
                }
            }
//            if (j >= buffer.size()){
//                cerr << "Tree" << endl;
//                cerr << tree << endl;
//                cerr << "stack size " << stack.size() << endl;
//                for (int i = 0; i < stack.size(); i++){
//                    cerr << i << "  " << *stack[i] << endl;
//                    cerr << i << " " << *shared_ptr<Node>(stack[i]->parent_) << endl;
//                }
//                cerr << "buffer size " << buffer.size() << endl;
//                cerr << "deque size " << deque.size() << endl;
//                for (int i = 0; i < deque.size(); i++){
//                    cerr << i << "  " << *deque[i] << endl;
//                    cerr << i << " " << *shared_ptr<Node>(deque[i]->parent_) << endl;
//                }
//                cerr << "j " << j << endl;
//                for (int i = 0; i < deriv.size(); i++){
//                    cerr << deriv[i] << " ";
//                }cerr << endl;
//            }
            assert(j < buffer.size());
            deriv.push_back(actions[SHIFT_I]);
            shift(stack, deque, buffer, j);
            structure = false;
        }else{
            structure = true;
            // label action
            shared_ptr<Node> s0 = deque.back();
            shared_ptr<Node> s0p;
            s0->get_parent(s0p);

//            if (deriv.size() == 1){
//                cerr << "HHHHH " << *s0 << endl;
//                cerr << "HHHHH " << *s0p << endl;
//            }

            // label / no label
            if (s0->is_preterminal()){
                //if (deriv.size() == 1){ cerr << " has label" << endl; }
                //assert(s0->is_preterminal());
                if (s0p->arity() == 1){
                    //if (deriv.size() == 1){ cerr << " arity == 1" << endl; }
                    assert(deque.size() == 1 && "Error in derivation extraction");
                    deriv.push_back(Action(Action::LABEL, s0p->label(), -1));
                    deque.pop_back();
                    deque.push_back(s0p);
                    continue;
                }else{
                    deriv.push_back(actions[NULL_ACTION_I]);
                }
            }else{
//                if (deriv.size() == 1){
//                    cerr << " has no label" << endl;
//                    cerr << *s0 << endl;
//                }
                if (s0p->arity() == s0->arity()){
                    //if (deriv.size() == 1){ cerr << " arity == s0 arity" << endl; }
                    deriv.push_back(Action(Action::LABEL, s0p->label(), -1));
                    deque.pop_back();
                    deque.push_back(s0p);
                    continue;
                }else{
                    deriv.push_back(actions[NULL_ACTION_I]);
                }
            }
        }

    }

    for (int i = 0; i < deriv.size(); i++){
        if (encoder.find(deriv[i]) == encoder.end()){
            add_action(deriv[i]);
        }
        int j = encoder[deriv[i]];
        deriv[i] = actions[j];
    }
    derivation = Derivation(deriv);

    //cerr << tree << endl;
    //cerr << derivation << endl;
}


bool MergeLabelTS::allowed(ParseState &state, int buffer_size, int i){
    const Action a = actions[i];
    switch(a.type()){
    case Action::SHIFT:
        return state.is_init() ||
                (state.buffer_j_ < buffer_size &&
                 //! (state.last_action_.code() == GAP_I));
                 (state.last_action_.type() == Action::LABEL || state.last_action_.type() == Action::NULL_ACTION));
    case Action::MERGE:
        return state.structure_action() &&
                state.top_ != nullptr &&
                state.mid_ != nullptr;
    case Action::LABEL:
        return allowed_label(state, a, buffer_size);
    case Action::IDLE:
        return state.is_final(grammar, buffer_size);
    case Action::GAP:
        return state.structure_action() &&
                state.top_ != nullptr &&
                state.mid_ != nullptr &&
                !state.mid_->is_bottom();
    case Action::NULL_ACTION:
        //        cerr << "structure" << state.label_action() << endl;
        //        cerr << "axiom" << (state.buffer_j_ != buffer_size || ! state.top_->is_bottom()) << endl;
        return state.label_action() &&
                (state.buffer_j_ != buffer_size || ! state.top_->is_bottom());
        //        case Action::REDUCE_U:
        //            return state.last_action_.type() == Action::SHIFT       // also requires s0 already labeled (triially true after shift)
        //                 && allowed_reduce_u(state, a, buffer_size);
    default:
        assert(false && "Merge and label ts: unknown action");
    }
    return false;
}

bool MergeLabelTS::allowed_label(ParseState &state, const Action &a, int buffer_size){

    if (! state.label_action()) return false;
    if (state.top_ == nullptr) return false;
    if (state.top_->n->has_label() && ! state.top_->n->is_preterminal()) return false;

    if (state.buffer_j_ == buffer_size && state.top_->is_bottom()){
        return grammar.is_axiom(a.label());
    }else{
        return ! grammar.is_axiom(a.label());
    }
//    return buffer_j_ == buffer_size
//            && top_->is_bottom()
//            && grammar.is_axiom(top_->n->label());

//    state.label_action() &&
//    state.top_ != nullptr &&
//   (! state.top_->n->has_label() || );

}

//bool MergeLabelTS::allowed_reduce_u(ParseState &state, const Action &a, int buffer_size){
//    if (grammar.is_axiom(a.label())){
//        if (state.top_->is_bottom() &&
//               state.buffer_j_ == buffer_size &&
//               ! grammar.is_axiom(state.top_->n->label())){
//            assert(buffer_size == 1);
//            return true;
//        }
//        return false;
//    }
//    return state.buffer_j_ != buffer_size || ! state.top_->is_bottom();
//}




//////////////////////////////////////////////////////////
///
///
///
///    Lexicalized Merge Label transition system
///
///
///
//////////////////////////////////////////////////////////


LexicalizedMergeLabelTS::LexicalizedMergeLabelTS(const Grammar &g):TransitionSystem(LEXICALIZED_MERGE_LABEL_TS){
    for (int type : {Action::SHIFT, Action::LEFT, Action::RIGHT, Action::IDLE, Action::GAP, Action::NULL_ACTION}){
        add_action(Action(type, -1));
    }
    grammar = g;
}

// HERE:

int LexicalizedMergeLabelTS::get_id(){
    return TransitionSystem::LEXICALIZED_MERGE_LABEL_TS;
}

Action* LexicalizedMergeLabelTS::get_idle(){
    return &actions[IDLE_I];
}

void LexicalizedMergeLabelTS::compute_derivation(Tree &tree, Derivation &derivation){

    vector<Action> deriv;
    vector<shared_ptr<Node>> buffer;
    tree.get_buffer(buffer);

    vector<shared_ptr<Node>> stack;
    std::deque<shared_ptr<Node>> deque;
    int j = 0;

    deriv.push_back(actions[SHIFT_I]);
    shift(stack, deque, buffer, j);



    bool structure = false;

    while (j < buffer.size() ||
           deque.size() != 1 ||
           stack.size() > 0  ||
           ! grammar.is_axiom(deque.back()->label())){

        if (structure){
            //shift
            // left
            // right
            // gap
            if (stack.size() == 0 && deque.size() == 0){
                deriv.push_back(actions[SHIFT_I]);
                shift(stack, deque, buffer, j);
                structure = false;
                continue;
            }

            shared_ptr<Node> s0 = deque.back();
            shared_ptr<Node> s0p;
            s0->get_parent(s0p);

            if (stack.size() > 0){
                shared_ptr<Node> s1 = stack.back();
                shared_ptr<Node> s1p;
                s1->get_parent(s1p);

                if (s0p == s1p){
                    int hpos = s0p->h();
                    if (s0->index() < s1->index()){
                        hpos = 1 - hpos;
                    }
                    if (hpos == 0){
                        assert( has_head( s0p, s1 ));
                        deriv.push_back(Action(Action::LEFT, -1));
                    }else{
                        assert( has_head( s0p, s0 ));
                        assert ( hpos == 1 );
                        deriv.push_back(Action(Action::RIGHT, -1));
                    }

                    //deriv.push_back(Action(Action::MERGE, -1));

                    stack.pop_back();
                    deque.pop_back();

                    while(deque.size() > 0){
                        stack.push_back(deque.front());
                        deque.pop_front();
                    }


                    vector<shared_ptr<Node>> newchildren{s0, s1};

//                    if (s0->has_label()){
//                        newchildren.push_back(s0);
//                    }else{
//                        s0->get_children(newchildren);
//                    }
//                    if (s1->has_label()){
//                        newchildren.push_back(s1);
//                    }else{
//                        s1->get_children(newchildren);
//                    }
                    assert(newchildren.size() > 0);
                    shared_ptr<Node> node = shared_ptr<Node>(new Node(newchildren));
                    node->set_parent(s0p);       // intermediary node
                    node->set_h(s0p->h());
                    deque.push_back(node);
                    structure = false;
                    continue;
                }
            }

            if (stack.size() > 1){
                int gaps = -1;
                for (int i = 0; i < stack.size(); i++){         // TODO: who on earth coded this ? use parent ptr instead
                    shared_ptr<Node> candidate = stack[stack.size() - 1 - i];
                    shared_ptr<Node> candparent;
                    candidate->get_parent(candparent);
                    if (candparent == s0p){
                        gaps = i;
                        break;
                    }
                }
                if (gaps != -1){
                    for (int i = 0; i < gaps; i++){
                        deriv.push_back(actions[GAP_I]);
                        deque.push_front(stack.back());
                        stack.pop_back();
                    }
                    continue;
                }
            }
//            if (j >= buffer.size()){
//                cerr << "Tree" << endl;
//                cerr << tree << endl;
//                cerr << "stack size " << stack.size() << endl;
//                for (int i = 0; i < stack.size(); i++){
//                    cerr << i << "  " << *stack[i] << endl;
//                    cerr << i << " " << *shared_ptr<Node>(stack[i]->parent_) << endl;
//                }
//                cerr << "buffer size " << buffer.size() << endl;
//                cerr << "deque size " << deque.size() << endl;
//                for (int i = 0; i < deque.size(); i++){
//                    cerr << i << "  " << *deque[i] << endl;
//                    cerr << i << " " << *shared_ptr<Node>(deque[i]->parent_) << endl;
//                }
//                cerr << "j " << j << endl;
//                for (int i = 0; i < deriv.size(); i++){
//                    cerr << deriv[i] << " ";
//                }cerr << endl;
//            }
            assert(j < buffer.size());
            deriv.push_back(actions[SHIFT_I]);
            shift(stack, deque, buffer, j);
            structure = false;
        }else{
            structure = true;
            // label action
            shared_ptr<Node> s0 = deque.back();
            shared_ptr<Node> s0p;
            s0->get_parent(s0p);

//            if (deriv.size() == 1){
//                cerr << "HHHHH " << *s0 << endl;
//                cerr << "HHHHH " << *s0p << endl;
//            }

            // label / no label
            if (s0->is_preterminal()){
                //if (deriv.size() == 1){ cerr << " has label" << endl; }
                //assert(s0->is_preterminal());
                if (s0p->arity() == 1){
                    //if (deriv.size() == 1){ cerr << " arity == 1" << endl; }
                    assert(deque.size() == 1 && "Error in derivation extraction");
                    deriv.push_back(Action(Action::LABEL, s0p->label(), -1));
                    deque.pop_back();
                    deque.push_back(s0p);
                    continue;
                }else{
                    deriv.push_back(actions[NULL_ACTION_I]);
                }
            }else{
//                if (deriv.size() == 1){
//                    cerr << " has no label" << endl;
//                    cerr << *s0 << endl;
//                }
                STRCODE label = s0p->label();
                if (! grammar.is_tmp(label)){
                    deriv.push_back(Action(Action::LABEL, label, -1));
                }else{
                    deriv.push_back(actions[NULL_ACTION_I]);
                }
                deque.pop_back();
                deque.push_back(s0p);
                continue;
            }
        }
    }

    for (int i = 0; i < deriv.size(); i++){
        if (encoder.find(deriv[i]) == encoder.end()){
            add_action(deriv[i]);
        }
        int j = encoder[deriv[i]];
        deriv[i] = actions[j];
    }
    derivation = Derivation(deriv);

//    cerr << tree << endl;
//    cerr << derivation << endl;
}


bool LexicalizedMergeLabelTS::allowed(ParseState &state, int buffer_size, int i){
    const Action a = actions[i];
    switch(a.type()){
        case Action::SHIFT:
            return state.is_init() ||
                              (state.buffer_j_ < buffer_size &&
                            //! (state.last_action_.code() == GAP_I));
                              (state.last_action_.type() == Action::LABEL || state.last_action_.type() == Action::NULL_ACTION));
//        case Action::MERGE:
//            return state.structure_action() &&
//                    state.top_ != nullptr &&
//                   state.mid_ != nullptr;
        case Action::LEFT:
        case Action::RIGHT:
            return state.structure_action() &&
                   state.top_ != nullptr &&
                   state.mid_ != nullptr &&
                    allowed_reduce(state, a, buffer_size);
        case Action::LABEL:
            return allowed_label(state, a, buffer_size);
        case Action::IDLE:
            return state.is_final(grammar, buffer_size);
        case Action::GAP: // TODO: additional constraints due to lexicalisation
            return state.structure_action() &&
                    state.top_ != nullptr &&
                   state.mid_ != nullptr &&
                    allowed_gap(state);
                  //!state.mid_->is_bottom();
        case Action::NULL_ACTION:
//        cerr << "structure" << state.label_action() << endl;
//        cerr << "axiom" << (state.buffer_j_ != buffer_size || ! state.top_->is_bottom()) << endl;
//             return state.label_action() &&
//                     (state.buffer_j_ != buffer_size || ! state.top_->is_bottom());
            return allowed_no_label(state, a, buffer_size);
//        case Action::REDUCE_U:
//            return state.last_action_.type() == Action::SHIFT       // also requires s0 already labeled (triially true after shift)
//                 && allowed_reduce_u(state, a, buffer_size);
        default:
            assert(false && "Merge and label ts: unknown action");
    }
    return false;
}

bool LexicalizedMergeLabelTS::allowed_label(ParseState &state, const Action &a, int buffer_size){

    if (! state.label_action()) return false;
    if (state.top_ == nullptr) return false;
    if (state.top_->n->has_label() && ! state.top_->n->is_preterminal()) return false;

    if (state.buffer_j_ == buffer_size && state.top_->is_bottom()){
        return grammar.is_axiom(a.label());
    }else{
        return ! grammar.is_axiom(a.label());
    }
}

bool LexicalizedMergeLabelTS::allowed_no_label(ParseState &state, const Action &a, int buffer_size){
    if (! state.label_action()) return false;
    if (state.buffer_j_ == buffer_size && state.top_->is_bottom()) return false;
    if (state.top_->n->is_preterminal()) return true;
    if (buffer_size == state.buffer_j_){
        bool has_ntmp = false;
        shared_ptr<StackItem> si(state.top_->predecessor);
        while (si != nullptr){
            if ((! grammar.is_tmp(si->n->label())) || si->n->is_preterminal()){
                has_ntmp = true;
                break;
            }
            si = si->predecessor;
        }
        if (! has_ntmp) return false;
    }
    return true;
}

bool LexicalizedMergeLabelTS::allowed_reduce(ParseState &state, const Action &a, int buffer_size){
//    if (grammar.is_tmp(state.top_->n->label()) && grammar.is_tmp(state.mid_->n->label()))
//        return false;
    assert(a.type() == Action::LEFT || a.type() == Action::RIGHT);
    bool left = a.type() == Action::LEFT;
    if (left){
//        cerr << "left" << endl;
//        cerr << *state.top_->n << endl;
//        cerr << enc::hodor.decode_to_str(state.top_->n->label(), enc::CAT) << endl;
//        cerr << ! grammar.is_tmp(state.top_->n->label()) << endl;
        return (! grammar.is_tmp(state.top_->n->label())) || state.top_->n->is_preterminal();
    }else{
//        cerr << "right" << endl;
//        cerr << *state.mid_->n << endl;
//        cerr << enc::hodor.decode_to_str(state.mid_->n->label(), enc::CAT) << endl;
//        cerr << ! grammar.is_tmp(state.mid_->n->label()) << endl;
        return (! grammar.is_tmp(state.mid_->n->label())) || state.mid_->n->is_preterminal();
    }
}

bool LexicalizedMergeLabelTS::allowed_gap(ParseState &state){
    //cerr << "AG 0" << endl;
    if (state.mid_->is_bottom()) return false;
    //cerr << "AG 1" << endl;
    if (! grammar.is_tmp(state.top_->n->label())){ return true; }
    //cerr << "AG 2" << endl;

    shared_ptr<StackItem> si(state.mid_->predecessor);
    //cerr << "AG 3" << endl;
    while ( si != nullptr ){
        if (! grammar.is_tmp(si->n->label()))
            return true;
        si = si->predecessor;
    }
    //cerr << "AG 4" << endl;
    return false;
}

////////////////////////////////////////////////////////////////////////////////
///
///
///     Shift Reduce
///
///
////////////////////////////////////////////////////////////////////////////////



ShiftReduce::ShiftReduce(const Grammar &g) : GapTS(g){
    system_id = SHIFT_REDUCE;
}

int ShiftReduce::get_id(){
    return TransitionSystem::SHIFT_REDUCE;
}

bool ShiftReduce::allowed_reduce(ParseState &state, const Action &a, int buffer_size){
    if (grammar.is_tmp(state.top_->n->label()) && grammar.is_tmp(state.mid_->n->label())) return false; // cannot reduce 2 tmps
    if (buffer_size == state.buffer_j_ && state.mid_->is_bottom() && state.top_->predecessor == state.mid_){ // if last reduce -> to axiom
        return grammar.is_axiom(a.label());
    }
    // REMINDER: why is this line decommented ??
    if (grammar.is_axiom(a.label())){               // if axiom -> it must be the last reduce
        return buffer_size == state.buffer_j_ &&
               state.mid_->is_bottom() &&
               state.top_->predecessor == state.mid_;
    }
    if (grammar.is_tmp(a.label())){ // there must be a non tmp symbol in the stack
        if (buffer_size == state.buffer_j_){
            if (grammar.is_tmp(state.mid_->predecessor->n->label()))
                return false;
        }
        if (a.type() == Action::REDUCE_L) return ! grammar.is_tmp(state.top_->n->label());
        if (a.type() == Action::REDUCE_R) return ! grammar.is_tmp(state.mid_->n->label());
    }
    return true;

}

bool ShiftReduce::allowed_gap(ParseState &state){
    return false;
}


////////////////////////////////////////////////////////////////////////////////
///
///
///     MergeLabelProjTs
///
///
////////////////////////////////////////////////////////////////////////////////

MergeLabelProjTS::MergeLabelProjTS(const Grammar &g):MergeLabelTS(g){
    system_id = TransitionSystem::MERGE_LABEL_PROJ_TS;
}

int MergeLabelProjTS::get_id(){
    return TransitionSystem::MERGE_LABEL_PROJ_TS;
}

bool MergeLabelProjTS::allowed(ParseState &state, int buffer_size, int i){
    const Action a = actions[i];
    switch(a.type()){
    case Action::SHIFT:
        return state.is_init() ||
                (state.buffer_j_ < buffer_size &&
                 //! (state.last_action_.code() == GAP_I));
                 (state.last_action_.type() == Action::LABEL || state.last_action_.type() == Action::NULL_ACTION));
    case Action::MERGE:
        return state.structure_action() &&
                state.top_ != nullptr &&
                state.mid_ != nullptr;
    case Action::LABEL:
        return allowed_label(state, a, buffer_size);
    case Action::IDLE:
        return state.is_final(grammar, buffer_size);
    case Action::GAP:
        return false;
    case Action::NULL_ACTION:
        //        cerr << "structure" << state.label_action() << endl;
        //        cerr << "axiom" << (state.buffer_j_ != buffer_size || ! state.top_->is_bottom()) << endl;
        return state.label_action() &&
                (state.buffer_j_ != buffer_size || ! state.top_->is_bottom());
        //        case Action::REDUCE_U:
        //            return state.last_action_.type() == Action::SHIFT       // also requires s0 already labeled (triially true after shift)
        //                 && allowed_reduce_u(state, a, buffer_size);
    default:
        assert(false && "Merge and label ts: unknown action");
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////
///
///
///     LexicalizedMergeLabelProjTs
///
///
////////////////////////////////////////////////////////////////////////////////

LexicalizedMergeLabelProjTS::LexicalizedMergeLabelProjTS(const Grammar &g):LexicalizedMergeLabelTS(g){
    system_id = TransitionSystem::LEXICALIZED_MERGE_LABEL_PROJ_TS;
}

int LexicalizedMergeLabelProjTS::get_id(){
    return TransitionSystem::LEXICALIZED_MERGE_LABEL_PROJ_TS;
}

bool LexicalizedMergeLabelProjTS::allowed(ParseState &state, int buffer_size, int i){
    const Action a = actions[i];
    switch(a.type()){
    case Action::SHIFT:
        return state.is_init() ||
                (state.buffer_j_ < buffer_size &&
                 //! (state.last_action_.code() == GAP_I));
                 (state.last_action_.type() == Action::LABEL || state.last_action_.type() == Action::NULL_ACTION));
    case Action::LEFT:
    case Action::RIGHT:
        return state.structure_action() &&
                state.top_ != nullptr &&
                state.mid_ != nullptr &&
                allowed_reduce(state, a, buffer_size);
    case Action::LABEL:
        return allowed_label(state, a, buffer_size);
    case Action::IDLE:
        return state.is_final(grammar, buffer_size);
    case Action::GAP:
        return false;
    case Action::NULL_ACTION:
        return allowed_no_label(state, a, buffer_size);
    default:
        assert(false && "Merge and label ts: unknown action");
    }
    return false;
}

bool LexicalizedMergeLabelProjTS::allowed_no_label(ParseState &state, const Action &a, int buffer_size){
    if (! state.label_action()) return false;
    if (state.buffer_j_ == buffer_size && state.top_->is_bottom()) return false;
    if (state.top_->n->is_preterminal()) return true;
    if (buffer_size == state.buffer_j_){
        if (state.mid_ != nullptr){
            if (grammar.is_tmp(state.mid_->n->label())
                    && ! state.mid_->n->is_preterminal()){
                return false;
            }
        }
    }
    return true;
}

