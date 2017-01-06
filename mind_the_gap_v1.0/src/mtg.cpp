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
    string output_dir = "../experiments/foo";
    string tpl_file;
    int transition_system = TransitionSystem::GAP_TS;
    int classifier = Classifier::FASTEST_PER;
    int feature_extractor = feature::FAST_FEATURE_EXTRACTOR;

    int epochs = 20;
    int beamsize = 4;
    int unknown = Treebank::ENCODE_EVERYTHING;

    bool stats = false;

    NeuralNetParameters nn_params;
} options;

void print_help(){
    cout << endl << endl <<"This is Mind The Gap Parser, a transition based constituent parser that handles discontinuous constituents." << endl << endl <<

        "Usage:" << endl <<
        "      ./mrg_gcc -t <trainfile> - d <devfile> -x <testfile> -o <outputdirectory> -f <feature templates filename> [options]" << endl << endl <<

        "Options:" << endl <<
        "  -h     --help                  displays this message and quits" << endl <<
        "  -t     --train     [STRING]    training corpus (discbracket format)   " << endl <<
        "  -d     --dev       [STRING]    developpement corpus (discbracket format)   " << endl <<
        "  -i     --epochs    [INT]       number of iterations [default=20]" << endl <<
        "  -b     --beam      [INT]       size of beam [default=4]" << endl <<
        "  -f     --features  [STRING]    templates filename (see mind_the_gap/data for examples)" << endl <<
        "  -o     --out       [STRING]    output directory (where parse results will be written) [default=../experiments/foo]" << endl <<
        "  -g     --cgap                  uses compound gap transition system (no idle action, every derivation has same length)" << endl <<
        //"  -m     --mergelabel            uses merge-label transition system (no binarisation)" << endl <<
        //"  -N     --neural    [STRING]    uses a neural network instead of a structured perceptron. Argument: file containing neural net options" << endl <<
        "  -u     --unknown               replaces hapaxes by UNKNOWN pseudo-word in train corpus (learn parameters for unknown words)" << endl <<
        //"  -K     --no-hash-trick         do not use hash trick" << endl <<
        "  -s     --stats                 writes statistics about the training treebank and the transition systems and exits. You only need -t and -o (filename) to use that." << endl <<
        "  -X     --debug                 tests various modules / classes and exits. You probably want to compile with 'make debug' if you need that" << endl << endl;
}

int main(int argc, char *argv[]){
    srand(SEED);

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
            //{"no-hash-trick", no_argument, 0, 'K'},
            {"mergelabel", no_argument, 0, 'm'},
            {"neural", no_argument, 0, 'N'},
        };
        int option_index = 0;
        char c = getopt_long (argc, argv, "hmXgust:d:i:b:o:f:KN:",long_options, &option_index);
        if(c==-1){break;}
        switch(c){
        case 'h': print_help(); exit(0);
        case 't': options.train_file = optarg;              break;
        case 'd': options.dev_file = optarg;                break;
        case 'i': options.epochs = atoi(optarg);            break;
        case 'b': options.beamsize = atoi(optarg);          break;
        case 'o': options.output_dir = optarg;              break;
        case 'f': options.tpl_file = optarg;                break;
        case 'g': options.transition_system
                = TransitionSystem::CGAP_TS;                break;
        case 'm': options.transition_system
                = TransitionSystem::MERGE_LABEL_TS;         break;
        case 'u': options.unknown
                = Treebank::CUTOFF;                         break;
        case 'X':
            test_various_modules();
            exit(0);
        case 's': options.stats = true;                     break;
        case 'N':{
            options.classifier = Classifier::FFNN;
            options.feature_extractor = feature::DENSE_FEATURE_EXTRACTOR;
            NeuralNetParameters::read_option_file(optarg, options.nn_params);
            break;
        }
        case 'K':{
            cerr << "Deprecated option, aborting" << endl;
            exit(1);
            options.classifier = Classifier::FASTER_PER;
            options.feature_extractor = feature::STD_FEATURE_EXTRACTOR;
            break;
        }
        default:
            cerr << "unknown option" << endl;
            print_help(); exit(0);
        }
    }



    if (options.stats){
        cerr << "Writing stats about " << options.train_file << " corpus in " << options.output_dir << endl;
        TransitionSystem::print_stats(options.train_file, options.output_dir);
        cerr << "Done." << endl;
        exit(0);
    }

    if (options.train_file.empty() ||
            options.dev_file.empty() ||
            options.tpl_file.empty()){
        cerr << "At least one mandatory option is missing." << endl;
        print_help();
        exit(0);
    }

    mkdir(options.output_dir.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

    Treebank train(options.train_file, options.unknown);
    Treebank dev(options.dev_file, Treebank::UNKNOWN_CODING);


    Grammar grammar;
    HeadRules headrules("../data/negra.headrules", &grammar);

    grammar.mark_punct();
    if (options.transition_system == TransitionSystem::MERGE_LABEL_TS){
        grammar.do_not_binarise();
    }

    Treebank train_bin = train;
    Treebank dev_bin = dev;

    train_bin.annotate_heads(headrules);
    train_bin.transform(grammar);
    train_bin.annotate_parent_ptr();

    dev_bin.annotate_heads(headrules);
    dev_bin.transform(grammar);
    dev_bin.annotate_parent_ptr();

    Parser mtg(options.transition_system, options.classifier, options.feature_extractor,
               train_bin, dev_bin, grammar, options.tpl_file,
               options.nn_params);

    vector<vector<shared_ptr<Node>>> raw_test;
    vector<vector<std::pair<string,string>>> str_raw_test;

    mtg.train_global(train_bin,
                     dev_bin,
                     train,
                     dev,
                     options.epochs,
                     options.beamsize,
                     options.output_dir,
                     raw_test,
                     str_raw_test);


    mtg.cf->print_stats(cout);

    mtg.export_model(options.output_dir);
    enc::hodor.export_model(options.output_dir);
}






