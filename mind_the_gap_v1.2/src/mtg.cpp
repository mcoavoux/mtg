#include <getopt.h>
#include <sys/stat.h>


#include "tree.h"
#include "treebank.h"
#include "utils.h"
#include "str_utils.h"
#include "tss_beam.h"
#include "grammar.h"
#include "parser.h"
#include "transition_system.h"
#include "features.h"



using namespace std;


void test_various_modules(){
    cerr << "Testing treebank reading binarization..." << endl;
    Treebank::test_binarization("../data/debug_data/tiger_train5.mrg", "../data/debug_data/negra.headrules");

    cerr << "Testing derivation extraction for Gap ts..." << endl;
    TransitionSystem::test_derivation_extraction("../data/debug_data/foo.mrg", "../data/debug_data/negra.headrules", TransitionSystem::GAP_TS);

    cerr << "Testing feature extraction..." << endl;
    StdFeatureExtractor::test("../data/debug_data/gap_minimal_templates.md");

    cerr << "Testing raw (tagged) sentence reading..." << endl;
    Treebank::test_raw_sentence_reading("../data/debug_data/test.raw");

    cerr << "Testing derivation extraction for Compound Gap ts..." << endl;
    TransitionSystem::test_derivation_extraction("../data/debug_data/foo.mrg", "../data/debug_data/negra.headrules", TransitionSystem::CGAP_TS);

    cerr << "Testing derivation extraction for Merge Label ts..." << endl;
    TransitionSystem::test_derivation_extraction("../data/debug_data/foo.mrg", "../data/debug_data/negra.headrules", TransitionSystem::MERGE_LABEL_TS);

    cerr << "Tests finished" << endl;
}


struct Options{
    string train_file;
    string dev_file;
    string output_dir = "./foo";
    string tpl_file;
    string head_rules_file = "../data/negra.headrules";
    int transition_system = TransitionSystem::GAP_TS;
    int classifier = Classifier::FASTEST_PER;
    int feature_extractor = feature::FAST_FEATURE_EXTRACTOR;

    int epochs = 20;
    int beamsize = 4;
    int unknown = Treebank::ENCODE_EVERYTHING;

    bool stats = false;

    int format = Treebank::DISCBRACKET;
    NeuralNetParameters nn_params;
} options;

void print_help(){
    cout << endl << endl <<"This is Mind The Gap Parser, a transition based constituent parser that handles discontinuous constituents." << endl << endl <<

        "Usage:" << endl <<
        "      ./mtg2_trainer -t <trainfile> - d <devfile> -x <testfile> -o <outputdirectory> -f <feature templates filename> [options]" << endl << endl <<

        "Options:" << endl <<
        "  -h     --help                  displays this message and quits" << endl <<
        "  -t     --train     [STRING]    training corpus (discbracket format)   " << endl <<
        "  -d     --dev       [STRING]    developpement corpus (discbracket format)   " << endl <<
        "  -i     --epochs    [INT]       number of iterations [default=20]" << endl <<
        "  -b     --beam      [INT]       size of beam [default=4]" << endl <<
        "  -f     --features  [STRING]    templates filename (see mind_the_gap/data for examples)" << endl <<
        "  -o     --out       [STRING]    output directory (where parse results will be written) [default=../experiments/foo]" << endl <<
        "  -g     --cgap                  uses compound gap transition system (no idle action, every derivation has same length)" << endl <<
        "  -m     --mergelabel[INT]       uses merge-label transition system (no binarisation, basially: Cross and Huang 2016 + gap action) [0: use 2 classifiers for structure and label actions 1: use only 1]" << endl <<
        "  -r     --shift-red             uses standard shift-reduce transition system" << endl <<
        "  -N     --neural    [STRING]    uses a neural network instead of a structured perceptron. Argument: file containing neural net options" << endl <<
        "  -u     --unknown               replaces hapaxes by UNKNOWN pseudo-word in train corpus (learn parameters for unknown words)" << endl <<
        "  -F     --inputfmt  [INT]       0) discbracket 1) tbk" << endl <<
        "  -s     --stats                 writes statistics about the training treebank and the transition systems and exits. You only need -t and -o (filename) to use that." << endl <<
        "  -H     --headrules [STRING]    use customized head-rules file (default: negra.headrules)" << endl <<
        "  -X     --debug                 tests various modules / classes and exits. You probably want to compile with 'make debug' if you need that" << endl << endl;
}

int main(int argc, char *argv[]){
    srand(rd::Random::SEED);

    while(true){
        static struct option long_options[] ={
            {"help",no_argument,0,'h'},
            {"train", required_argument, 0, 't'},
            {"dev", required_argument, 0, 'd'},
            {"epochs", required_argument, 0, 'i'},
            {"beam", required_argument, 0, 'b'},
            {"out", required_argument, 0, 'o'},
            {"features", required_argument, 0, 'f'},
            {"debug", no_argument, 0, 'X'},
            {"cgap", no_argument, 0, 'g'},
            {"unknown", no_argument, 0, 'u'},
            {"stats", no_argument, 0, 's'},
            {"mergelabel", required_argument, 0, 'm'},
            {"neural", no_argument, 0, 'N'},
            {"inputfmt", required_argument, 0, 'F'},
            {"shift-red", no_argument, 0, 'r'},
            {"headrules", required_argument, 0, 'H'},
        };
        int option_index = 0;
        char c = getopt_long (argc, argv, "hm:Xgust:d:i:b:o:f:N:F:rH:",long_options, &option_index);
        if(c==-1){break;}
        switch(c){
        case 'h': print_help(); exit(0);
        case 't': options.train_file = optarg;              break;
        case 'd': options.dev_file = optarg;                break;
        case 'i': options.epochs = atoi(optarg);            break;
        case 'b': options.beamsize = atoi(optarg);          break;
        case 'o': options.output_dir = optarg;              break;
        case 'f': options.tpl_file = optarg;                break;
        case 'H': options.head_rules_file = optarg;         break;
        case 'u': options.unknown
                = Treebank::CUTOFF;                         break;
        case 'X':
            test_various_modules();
            exit(0);
        case 's': options.stats = true;                     break;
        case 'F': options.format = atoi(optarg);
            if (options.format < 0 || options.format > 2){
                cerr << "-F option illegal argument (possible choices: 0,1,2)" << endl;
                cerr << "Aborting" << endl;
                return 1;
            }
            break;
        case 'g': options.transition_system
                = TransitionSystem::CGAP_TS;                break;
        case 'r': options.transition_system
                = TransitionSystem::SHIFT_REDUCE;           break;
        case 'm': {
            options.transition_system = TransitionSystem::MERGE_LABEL_TS;
            int v = atoi(optarg);
            switch(v){
            case 0:{
                options.classifier = Classifier::RNN_LABEL_STRUCTURE;
                break;
            }
            case 1:{
                options.classifier = Classifier::RNN;
                break;
            }
            case 2:{
                options.transition_system = TransitionSystem::LEXICALIZED_MERGE_LABEL_TS;
                options.classifier = Classifier::RNN_LABEL_STRUCTURE_LEX;
                break;
            }
            case 3:{
                options.transition_system = TransitionSystem::LEXICALIZED_MERGE_LABEL_TS;
                options.classifier = Classifier::RNN;
                break;
            }
            case 10:{
                options.transition_system = TransitionSystem::MERGE_LABEL_PROJ_TS;
                options.classifier = Classifier::RNN_LABEL_STRUCTURE;
                break;
            }
            case 11:{
                options.transition_system = TransitionSystem::MERGE_LABEL_PROJ_TS;
                options.classifier = Classifier::RNN;
                break;
            }
            case 12:{
                options.transition_system = TransitionSystem::LEXICALIZED_MERGE_LABEL_PROJ_TS;
                options.classifier = Classifier::RNN_LABEL_STRUCTURE_LEX;
                break;
            }
            case 13:{
                options.transition_system = TransitionSystem::LEXICALIZED_MERGE_LABEL_PROJ_TS;
                options.classifier = Classifier::RNN;
                break;
            }
            default:
                cerr << "Unrecognised argument for -m option, aborting..." << endl;
                exit(1);
            }
            break;// huge bug fix.
        }
        case 'N':{
            //cerr << "N options" << endl;
            NeuralNetParameters::read_option_file(optarg, options.nn_params);
            if (options.nn_params.rnn_feature_extractor){
                if (options.classifier != Classifier::RNN_LABEL_STRUCTURE
                    && options.classifier != Classifier::RNN_LABEL_STRUCTURE_LEX){// prevent option interactions
                    options.classifier = Classifier::RNN;
                }
                options.feature_extractor = feature::RNN_FEATURE_EXTRACTOR;
            }else{
                options.classifier = Classifier::FFNN;
                options.feature_extractor = feature::DENSE_FEATURE_EXTRACTOR;
            }
            break;
        }
        default:
            cerr << "unknown option" << endl;
            print_help();
            exit(0);
        }
    }

    mkdir(options.output_dir.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

    if (options.stats){
        assert(! options.train_file.empty());
        cerr << "Writing stats about " << options.train_file << " corpus in " << options.output_dir << endl;
        TransitionSystem::print_stats(options.train_file, options.output_dir, options.head_rules_file, options.format, options.transition_system);
        cerr << "Done." << endl;
        exit(0);
    }

    if (options.train_file.empty() ||
            options.dev_file.empty() ||
            options.tpl_file.empty()){
        cerr << "At least one mandatory option is missing." << endl;
        cerr << "Please make sure you provided a training set, a developement set, and a feature template file." << endl;
        print_help();
        exit(0);
    }


    int unknown_policy_train = options.unknown;
    if (options.classifier > Classifier::FFNN){
        unknown_policy_train = Treebank::ENCODE_EVERYTHING;
    }
    Treebank train(options.train_file, unknown_policy_train, options.format, options.nn_params.header, true);
    if (options.classifier > Classifier::FFNN && options.unknown == Treebank::ENCODE_EVERYTHING){
        train.encode_all_from_freqs();
    }
    train.get_vocsizes(options.nn_params.voc_sizes);

    vector<string> tbk_header;
    train.get_header(tbk_header);
    enc::hodor.set_header(tbk_header);

    cerr << "Known voc sizes" << endl;
    for (int i = 0; i < options.nn_params.voc_sizes.size(); i++){
        if (options.nn_params.voc_sizes[i] > 2)
            cerr << "Type " << i << " " << options.nn_params.voc_sizes[i] << endl;
    }


    int unknown_policy_dev =  // REMINDER: neural nets can test if word is unknown with their lookup tables (strcode > lu.size()). they also need to access the full form of unknown words (char lstm)
            options.classifier < Classifier::FFNN ? Treebank::UNKNOWN_CODING
                                                  : Treebank::ENCODE_EVERYTHING;

    cerr << "Unknown policy dev " << unknown_policy_dev << endl;
    Treebank dev(options.dev_file, unknown_policy_dev, options.format, options.nn_params.header, false);

    if (train.size() == 0 || dev.size() == 0){
        cerr << "Treebank reading failed for some reason." << endl;
        cerr << "Double check the path to the train and dev treebanks." << endl;
        cerr << "Aborting..." << endl;
        exit(1);
    }

    Grammar grammar;

    grammar.mark_punct();
    if (options.transition_system == TransitionSystem::MERGE_LABEL_TS
            ||options.transition_system == TransitionSystem::MERGE_LABEL_PROJ_TS){
        grammar.do_not_binarise();
    }

    Treebank train_bin = train;
    Treebank dev_bin = dev;

    if (options.format == Treebank::DISCBRACKET){// Treebank::TBK format corpora are natively head-annotated
        HeadRules headrules(options.head_rules_file, &grammar);
        train_bin.annotate_heads(headrules);
        dev_bin.annotate_heads(headrules);
    }

    train_bin.transform(grammar);
    train_bin.annotate_parent_ptr();

    dev_bin.transform(grammar);
    dev_bin.annotate_parent_ptr();



    // REMINDER: tbk format -> parser use header to make sure number of rnn feature is consistent
    Parser mtg(options.transition_system, options.classifier, options.feature_extractor,
               train_bin, dev_bin, grammar, options.tpl_file,
               options.nn_params);

    mtg.train_global(train_bin,
                     dev_bin,
                     train,
                     dev,
                     options.epochs,
                     options.beamsize,
                     options.output_dir);


    mtg.cf->print_stats(cout);

    mtg.export_model(options.output_dir);
    enc::hodor.export_model(options.output_dir);

//    Parser newmtg(options.output_dir);
//    newmtg.export_model(options.output_dir+"_2");
}
