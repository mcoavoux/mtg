#include <getopt.h>
#include <sys/stat.h>

#include "logger.h"
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


struct Options{
    string test_file;
    string output_file = "../experiments/foo_file";
    string model_dir;
    int beamsize = 4;
    int unknown = Treebank::UNKNOWN_CODING;
    int format = 0;
} options;

void print_help(){
    cout << endl << endl <<"This is Mind The Gap Parser, a transition based constituent parser that handles discontinuous constituents. Command line for loading a model and parsing" << endl << endl <<

        "Usage:" << endl <<
        "      ./mrg_parse_gcc -x <testfile> -o <outputfile> -m <model dir> -b <beam size> -F <format id>" << endl << endl <<

        "Options:" << endl <<
        "  -h     --help                  displays this message and quits" << endl <<
        "  -x     --test      [STRING]    test corpus (discbracket format)   " << endl <<
        "  -b     --beam      [INT]       size of beam [default=4]" << endl <<
        "  -o     --out       [STRING]    output directory (where parse results will be written) [default=../experiments/foo]" << endl <<
        "  -m     --model     [STRING]    folder  containing a model" << endl <<
        "  -F     --inputfmt  [INT]       format 0) tok/tag (1 sentence per line)" << endl <<
        "                                        1) tok    tag    (<attribute>    )* (1 token per line)" << endl << endl;
}

int main(int argc, char *argv[]){
    srand(0);

    while(true){
        static struct option long_options[] ={
            {"help",no_argument,0,'h'},
            {"test", required_argument, 0, 'x'},
            {"beam", required_argument, 0, 'b'},
            {"out", required_argument, 0, 'o'},
            {"model", required_argument, 0, 'm'},
            {"inputfmt", required_argument, 0, 'F'},
        };
        int option_index = 0;
        char c = getopt_long (argc, argv, "hx:b:o:m:F:",long_options, &option_index);
        if(c==-1){break;}
        switch(c){
        case 'h': print_help(); exit(0);
        case 'x': options.test_file = optarg;       break;
        case 'b': options.beamsize = atoi(optarg);  break;
        case 'o': options.output_file = optarg;     break;
        case 'm': options.model_dir = optarg;       break;
        case 'F': options.format = atoi(optarg);    break;
        default:
            cerr << "unknown option" << endl;
            print_help(); exit(0);
        }
    }


    if (options.model_dir.empty()
        || options.output_file.empty()
        || options.test_file.empty()){
            cerr << "At least one mandatory option is missing." << endl;
            print_help();
            exit(0);
    }

    enc::hodor.import_model(options.model_dir);

    Parser mtg(options.model_dir);

    if (mtg.get_classifier_id() >= Classifier::FFNN){
        options.unknown = Treebank::ENCODE_EVERYTHING;
    }

    vector<vector<shared_ptr<Node>>> raw_test;
    vector<vector<std::pair<String,String>>> str_raw_test;

    // TODO: add options for unknown coding
    Treebank::read_raw_input_sentences(options.test_file, raw_test, str_raw_test, options.format, options.unknown);

    Treebank final_pred;

    Logger logger;
    logger.start();
    mtg.predict_treebank(raw_test, options.beamsize, final_pred);
    logger.stop();

    int n_tokens = 0;
    for (int i = 0; i < raw_test.size(); i++){
        n_tokens += raw_test[i].size();
    }
    cerr << "Parsing time :" << endl;
    cerr << "   n sentences     : " << raw_test.size() << endl;
    cerr << "   n tokens        : " << n_tokens << endl;
    cerr << "   time (seconds)  : " << logger.get_total_time() << endl;
    cerr << n_tokens / logger.get_total_time() << " tokens per second" << endl;
    cerr << raw_test.size() / logger.get_total_time() << " sentences per second" << endl;

    final_pred.write_conll(options.output_file + ".conll", str_raw_test);
    final_pred.detransform(*(mtg.ts->grammar_ptr()));
    final_pred.write(options.output_file, str_raw_test);

}







