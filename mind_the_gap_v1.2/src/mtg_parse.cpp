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
    int beamsize = 1;
    int unknown = Treebank::UNKNOWN_CODING;
    int format = 0;
    bool precompute_char_lstm = false;
    bool log_prob = false;
} options;

void print_help(){
    cout << endl << endl <<"This is Mind The Gap Parser, a transition based constituent parser that handles discontinuous constituents. Command line for loading a model and parsing" << endl << endl <<

        "Usage:" << endl <<
        "  (1) " << endl <<
        "      ./mtg2_parser -x <testfile> -o <outputfile> -m <model dir> -b <beam size> -F <format id>" << endl <<
        "  (2)   " << endl <<
        "      ./mtg2_parser -m <model dir> -b <beam size> < input > ouput" << endl <<
        "  (3)   " << endl <<
        "      ./mtg2_parser -m <model dir> -b <beam size> file1 [file2 [file3 ... ]]" << endl << endl <<

        "Options:" << endl <<
        "  -h     --help                  displays this message and quits" << endl <<
        "  -x     --test      [STRING]    test corpus (discbracket format)   " << endl <<
        "  -b     --beam      [INT]       size of beam [default=4]" << endl <<
        "  -o     --out       [STRING]    output directory (where parse results will be written) [default=../experiments/foo]" << endl <<
        "  -m     --model     [STRING]    folder  containing a model" << endl <<
        "  -F     --inputfmt  [INT]       format 0) tok/tag (1 sentence per line)" << endl <<
        "                                        1) tok    tag    (<attribute>    )* (1 token per line)" << endl <<
        "  -p     --precompute            precomputes char-based embeddings for known words (experimental, higher loading time, faster parsing)" << endl <<
        "  -L     --log-prob              outputs each tree with its log-probability (in discbracket file)" << endl;

    cout << "In all cases, make sure you replace '(' and ')' in your input by '-LRB-' and '-RRB-'." << endl;

    cout << "(2) format: 1 sentence per line, tokens separated by spaces" << endl <<
            "    outputs only const-trees, only works for models that require only" << endl <<
            "    tokens as input (no tag or morphological attribute)" << endl << endl <<
            "        echo \"Le chat mange une pomme .\" | ./mtg2_parser -m ../pretrained_models/FRENCH -b 1 > output" << endl << endl <<
            "(3) format: 1 sentence per line, tokens separated by spaces" << endl <<
            "    for each fileN, the parser will output:" << endl <<
            "      fileN.discbracket : constituency trees" << endl <<
            "      fileN.conll : morphological analysis, dependency labels," << endl <<
            "        and dependency trees if the model is lexicalized" << endl << endl;
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
            {"precompute", no_argument, 0, 'p'},
            {"log-prob", no_argument, 0, 'L'},
        };
        int option_index = 0;
        char c = getopt_long (argc, argv, "hx:b:o:m:F:pL",long_options, &option_index);
        if(c==-1){break;}
        switch(c){
        case 'h': print_help(); exit(0);
        case 'x': options.test_file = optarg;          break;
        case 'b': options.beamsize = atoi(optarg);     break;
        case 'o': options.output_file = optarg;        break;
        case 'm': options.model_dir = optarg;          break;
        case 'F': options.format = atoi(optarg);       break;
        case 'p': options.precompute_char_lstm = true; break;
        case 'L': options.log_prob = true;             break;
        default:
            cerr << "unknown option" << endl;
            print_help(); exit(0);
        }
    }


    if (options.model_dir.empty()){
//        || options.output_file.empty()
//        || options.test_file.empty())
            cerr << "At one mandatory option is missing." << endl;
            print_help();
            exit(0);
    }

    enc::hodor.import_model(options.model_dir);

    Parser mtg(options.model_dir);

    if (options.precompute_char_lstm){
        mtg.precompute_char_lstm();
    }

    if (mtg.get_classifier_id() >= Classifier::FFNN){
        options.unknown = Treebank::ENCODE_EVERYTHING;
    }

    bool unlexicalised = (mtg.get_transition_system_id() == TransitionSystem::MERGE_LABEL_TS
                          || mtg.get_transition_system_id() == TransitionSystem::MERGE_LABEL_PROJ_TS);

    if (! options.output_file.empty() && ! options.test_file.empty()){

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


        final_pred.write_conll(options.output_file + ".conll", str_raw_test, unlexicalised);
        final_pred.detransform(*(mtg.ts->grammar_ptr()));
        final_pred.write(options.output_file, str_raw_test, options.log_prob);

        return 0;
    }

    vector<std::pair<String,String>> tmp;
    string bline;
    String line;
    String empty;

    if (optind < argc){


        for (int file_i = optind; file_i < argc; file_i ++){

            Logger logger;
            logger.start();


            vector<vector<std::pair<String,String>>> str_raw_test;

            string filename(argv[file_i]);
            ifstream input_file(filename);

            cerr << "Parsing filename: " << filename << endl;

            Treebank pred;
            int n_tokens = 0;

            while(std::getline(input_file, bline)){
#ifdef WSTRING
                line = str::decode(bline);
#else
                line = bline;
#endif
                vector<std::pair<String,String>> raw_str;
                vector<String> tokens;

                vector<shared_ptr<Node>> sentence;
                str::split(line, " ", "", tokens);

                n_tokens += tokens.size();

                for (int i = 0; i < tokens.size(); i++){
                    vector<STRCODE> fields{enc::hodor.code(tokens[i], enc::TOK), enc::UNKNOWN};
                    sentence.push_back(shared_ptr<Node>(new Leaf(enc::UNKNOWN, i, fields)));
                    raw_str.push_back(std::make_pair(tokens[i],empty));
                }
                str_raw_test.push_back(raw_str);

                Tree tree;
                mtg.predict_tree(sentence, options.beamsize, tree);
                pred.add_tree(tree);
            }

            pred.write_conll(filename+".conll", str_raw_test, unlexicalised);
            pred.detransform(*(mtg.ts->grammar_ptr()));
            pred.write(filename+".discbracket", str_raw_test, options.log_prob);


            logger.stop();
            cerr << "Parsing time for " << filename << endl;
            cerr << "   n sentences     : " << str_raw_test.size() << endl;
            cerr << "   n tokens        : " << n_tokens << endl;
            cerr << "   time (seconds)  : " << logger.get_total_time() << endl;
            cerr << n_tokens / logger.get_total_time() << " tokens per second" << endl;
            cerr << str_raw_test.size() / logger.get_total_time() << " sentences per second" << endl;
        }

        return 0;
    }else{

        while(std::getline(cin, bline)){
#ifdef WSTRING
            line = str::decode(bline);
#else
            line = bline;
#endif

            vector<String> tokens;
            vector<shared_ptr<Node>> sentence;
            str::split(line, " ", "", tokens);
            for (int i = 0; i < tokens.size(); i++){
                vector<STRCODE> fields{enc::hodor.code(tokens[i], enc::TOK), enc::UNKNOWN};
                sentence.push_back(shared_ptr<Node>(new Leaf(enc::UNKNOWN, i, fields)));
            }
            Tree tree;
            mtg.predict_tree(sentence, options.beamsize, tree);
            tree.unbinarize(*(mtg.ts->grammar_ptr()));
            tree.write(cout,tmp, options.log_prob);
            cout << endl;
        }
    }

}
