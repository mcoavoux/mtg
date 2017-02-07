#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include <fstream>
#include <memory>
#include <boost/functional/hash.hpp>
#include <boost/tokenizer.hpp>

#include "utils.h"
#include "tree.h"
#include "tss_beam.h"
#include "hash_utils.h"


using std::vector;
using std::shared_ptr;

//typedef unsigned long long Int;
typedef unsigned int Int;

namespace feature{
    enum {BUFFER, QUEUE, STACK};
    enum {TOP, LEFT, RIGHT, LEFT_CORNER, RIGHT_CORNER, LEFT_CORNER_OUT, RIGHT_CORNER_OUT, LEFT_CORNER_2, RIGHT_CORNER_2, LEFT_CORNER_OUT_2, RIGHT_CORNER_OUT_2};
    enum {UNDEFINED_POSITION = -1};

    // maps template file tokens to internal corresponding ints
    extern const unordered_map<std::string, int> reader;
    extern const vector<std::string> writer_datastruct;
    extern const vector<std::string> writer_type;
    extern const vector<std::string> writer_depth;

    enum {STD_FEATURE_EXTRACTOR, FAST_FEATURE_EXTRACTOR, DENSE_FEATURE_EXTRACTOR, RNN_FEATURE_EXTRACTOR};
}

struct FeatureTemplate{
    static int N;
    int data_struct;    // buffer / queue / stack
    int index;          // idx in structure
    int depth;          // top / left / right / left span / right span
    int type;           // cat / token / tag / morph1 ..... morph n

    int unique_id;

    FeatureTemplate(int data_struct, int index, int depth, int type);

    // code the feature on a 64 bit integer unsigned long long
    // 3 bits for data_struct, 3 bits for index
    // 3 bits for index
    // 3 bits for depth    TODO: probably need more when you use span features
    // 6 bits for type
    // rest for the STRCODE which instantiates the address
    Int feature_id(STRCODE code);

    STRCODE decode(Int f_id, int &ds, int &idx, int &dep, int &typ);

    // instantiate feature with parse configuration
    STRCODE get_val(ParseState &state, vector<shared_ptr<Node>> &buffer);

    STRCODE get_val_buffer(int j, vector<shared_ptr<Node>> &buffer);

    STRCODE get_val_queue(ParseState &state, vector<shared_ptr<Node>> &buffer);

    STRCODE get_val_stack(ParseState &state, vector<shared_ptr<Node>> &buffer);

    STRCODE get_val_(shared_ptr<Node> &node, vector<shared_ptr<Node>> &buffer);

    int get_index_in_sent(shared_ptr<Node> &node, vector<shared_ptr<Node>> &buffer);

    Int operator()(ParseState &state, vector<shared_ptr<Node>> &buffer);

    Int get_raw_feature(ParseState &state, vector<shared_ptr<Node>> &buffer);

    Int get_raw_rnn(ParseState &state, vector<shared_ptr<Node>> &buffer);

    friend ostream & operator<<(ostream & os, const FeatureTemplate &ft);

    static void test();
};


// Abstract class for feature extraction
//   state -> buffer -> int list
class FeatureExtractor{

public:
    //virtual void extract_features(ParseState &state, vector<Node> &buffer, vector<int> &features) = 0;
    virtual void operator()(ParseState &state, vector<shared_ptr<Node>> &buffer, vector<int> &features) = 0;
    virtual int n_templates() = 0;

    virtual ~FeatureExtractor();

    virtual void get_template_type(vector<int> &types){
        assert(false && "Error: this cannot be used with sparse features");
    }

    // This class should know the type (enc::type) of every feature
    // also : ensure contiguity for same-type elements, in the case of dense features

    // Sparse feature = sequence of unsigned long long (unigram : 1, bigram : 2, etc)
    //      hash immediately to a single 64int ?

    // Dense feature : (STRCODE, type)
    // the type should be implicit (known by the classifier, as instantiation of feature_i has always the same type
    //  the STRCODE should be the index of the embedding in the lookup table

    static void read_templates(const std::string &filename, vector<vector<FeatureTemplate>> &fts);

    static void read_template(const std::string &line, vector<FeatureTemplate> &ft);

    virtual void export_model(const string &outdir);

    static FeatureExtractor* import_model(const string &outdir);

};




struct Feature{
    vector<Int> vals;  // each value corresponds to a condition of a template (3-gram feature -> vals.size() ==3)

    Feature();
    Feature(int arity);

    // TODO: bool operator ==     -> collision wise -> tester avec return true ? probably won't work less you set the hashtable size on creation
    bool operator==(const Feature &f) const;
};

struct FeatureHasher{
    std::size_t operator()(const Feature & f) const;
};



class StdFeatureExtractor : public FeatureExtractor{
    vector<vector<FeatureTemplate>> templates;
    vector<Feature> features;

    unordered_map<Feature, int, FeatureHasher> dict;
    int size;
public:
    StdFeatureExtractor(const std::string &filename);

    int n_templates();

    void operator()(ParseState &state, vector<shared_ptr<Node>> &buffer, vector<int> &feats);

    friend ostream & operator<<(ostream &os, const StdFeatureExtractor &fe);

    static void test(const std::string &tpls_filename);
};


/////
/// Use Jenkins Hash function
///  To be used with FastestPerceptron: kernel trick
///
class FastFeatureExtractor : public FeatureExtractor{
    vector<vector<FeatureTemplate>> templates;
    vector<Feature> features;
    KernelIndexer indexer;

public:
    FastFeatureExtractor(const std::string &filename);

    int n_templates();

    void operator()(ParseState &state, vector<shared_ptr<Node>> &buffer, vector<int> &feats);

    friend ostream & operator<<(ostream &os, const FastFeatureExtractor &fe);

    //static void test(const std::string &tpls_filename);

    void export_model(const string &outdir);
};


// Computes only unigram (atomic) feature
// Designed to return the index of a typed symbol
//  (index for string_encoder and for lookup table)
class DenseFeatureExtractor :public FeatureExtractor{
protected:
    vector<FeatureTemplate> templates;
public:
    DenseFeatureExtractor(const std::string &filename);

    int n_templates();

    virtual void operator()(ParseState &state, vector<shared_ptr<Node>> &buffer, vector<int> &feats);

    friend ostream & operator<<(ostream &os, const DenseFeatureExtractor &fe);

    virtual void export_model(const string &outdir);

    void get_template_type(vector<int> &types);
};


// When using a Bi-Rnn feature extractor:
// need to access indices of terminals in sentence
// (and not in lookup table)
class RnnFeatureExtractor : public DenseFeatureExtractor{
public:
    RnnFeatureExtractor(const std::string &filename);
    void operator()(ParseState &state, vector<shared_ptr<Node>> &buffer, vector<int> &feats);
    void export_model(const string &outdir);
};





#endif // FEATURES_H
