
#!/bin/usr/python3

import argparse
import subprocess
from joblib import Parallel, delayed
import random
import sys
import os

from launch_experiments import *

def do_parse_experiment(parameters):
    
    lang = parameters["language"]
    
    parser ="../../bin/mtg_parse_gcc"
    
    outdir = parameters["expe"]
    
    beam = parameters["beamsize"]
    while beam > 0 :
        
        for corpus in ["dev", "test"] :
            
            
            pred_file = "{}/pred_{}b{}.mrg".format(outdir, corpus, beam)
            pred_file2= "{}/pred_{}b{}.brackets".format(outdir, corpus, beam)
            
            command="{parser} -F 1 -x {data} -b {b} -o {out} -m {model} 2> {log}".format(
                    parser = parser,
                    data = parameters["datadir"] + "/{}/{}.raw".format(lang, corpus),
                    b = beam,
                    out = pred_file,
                    model = outdir,
                    log = "{}/parse_log_{}_b{}".format(outdir, corpus, beam))
            print(command)
            unix(command)
            
            
            unix("discodop treetransforms {} {} --inputfmt=discbracket --outputfmt=bracket".format(pred_file, pred_file2))
            unix("evalb_spmrl -X -p ~/bin/evalb_spmrl2013/spmrl.prm {test} {pred} > {res}".format(
                    test = "{}/{}_SPMRL/gold/{}.ptb".format(parameters["golddir"], lang, corpus),
                    pred = pred_file2,
                    res = "{}/eval_{}_b{}_it{}".format(outdir, corpus, beam, parameters["iterations"])))
            
            unix("perl eval07.pl -g {test} -s  {pred}.conll -o {out}".format(
                    test="{}/{}_SPMRL/gold/{}.conll".format(parameters["golddir"], lang, corpus),
                    pred=pred_file,
                    out="{}/depeval_{}_b{}_it{}".format(outdir, corpus, beam, parameters["iterations"])))
        
        
        for i in range(3, parameters["iterations"], 4) :
            for corpus in ["dev", "test"] :
        
                pred_file = "{}/iteration{}/pred_{}b{}.mrg".format(outdir, i, corpus, beam)
                pred_file2= "{}/iteration{}/pred_{}b{}.brackets".format(outdir, i, corpus, beam)
                
                command="{parser} -F 1 -x {data} -b {b} -o {out} -m {model} 2> {log}".format(
                        parser = parser,
                        data = parameters["datadir"] + "/{}/{}.raw".format(lang, corpus),
                        b = beam,
                        out = pred_file,
                        model = "{}/iteration{}".format(outdir, i),
                        log = "{}/iteration{}/parse_log_{}_b{}".format(outdir, i, corpus, beam))
                print(command)
                unix(command)
                
                
                unix("discodop treetransforms {} {} --inputfmt=discbracket --outputfmt=bracket".format(pred_file, pred_file2))
                unix("evalb_spmrl -X -p ~/bin/evalb_spmrl2013/spmrl.prm {test} {pred} > {res}".format(
                        test = "{}/{}_SPMRL/gold/{}.ptb".format(parameters["golddir"], lang, corpus),
                        pred = pred_file2,
                        res = "{}/eval_{}_b{}_it{}".format(outdir, corpus, beam, i)))
                
                unix("perl eval07.pl -g {test} -s  {pred}.conll -o {out}".format(
                        test="{}/{}_SPMRL/gold/{}.conll".format(parameters["golddir"], lang, corpus),
                        pred=pred_file,
                        out="{}/depeval_{}_b{}_it{}".format(outdir, corpus, beam, i)))
        
        
        beam //= 2


def main() :
    
    
    usage = """
        This is a front end script to launch a bunch of experiments in parallel.
        To run it, you need Python >= 3.2, discodop, evalb_spmrl.
        Warning: some filenames (such as evalb_spmrl parameter file) are hardcoded.
        
        Note that hyperparameters options take lists as argument.
        The parser will do experiments with the cartesian product of
        every lists (so use either short lists or lots of cpus).
        
        One experiment:
            - train the parser for ITERATIONS epochs and outputs model every other 4 epochs.
            - test the parser on dev and test for each model.
    """
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    #parser.add_argument("positionalargument", type = str, default = "coarse", choices = ["e","j","k"], help="[coarse(=default)|fine|ignore]")
    #parser.add_argument("--optionalargument","-o", type=, default=, action=, choices=, help=)
    
    parser.add_argument("datadir", type=str, help="datadir/LANGUAGE contains {train|dev}.tbk, {dev|test}.raw")
    parser.add_argument("golddir", type=str, help="golddir/LANGUAGE_SPMRL/gold/ contains {dev|test}.{ptb|conll} for evaluation")
    #parser.add_argument("tpl", type=str, help="template files")
    parser.add_argument("outputdir", type=str, help="create this folder and stores all experiments in it")
    
    parser.add_argument("--languages", type=str, nargs="+", default = ["POLISH"], help="List of languages")
    parser.add_argument("--iterations", "-i", type=int, default=30, help="Number of iterations per experiment")
    parser.add_argument("--beamsize", "-b", type=int, default=1, help="Beam size for parsing")
    parser.add_argument("--threads", "-N", type=int, default=1, help="Max number of experiments in parallel")
    
    #parser.add_argument("--learningrate", "-l", type=float, nargs='+', default=[0.01,0.02], help="learning rate")
    #parser.add_argument("--decayrate", "-d",    type=float, nargs='+', default=[1e-6], help="learning rate decay constant")
    #parser.add_argument("--clip", "-c", type=float, nargs='+', default=[5.0], help="(hard) gradient clipping")
    #parser.add_argument("--gaussian", "-g", type=float, nargs="+", default=[0.01], help="gaussian noise hyperparameter")
    #parser.add_argument("--hiddenlayers", type=int, nargs="+", default=[2], help="number of hidden layers (feed forward component)")
    #parser.add_argument("--sizelayers", "-L", type=int, nargs="+", default=[128], help="size of hidden layers")
    #parser.add_argument("--depth", type=int, nargs="+", choices=[2,4,6], default=[4], help="depth of sentence level bi-lstm")
    #parser.add_argument("--rnnsize", "-S", type=int, nargs='+', default=[64], help="number of units per sentence level lstm")
    #parser.add_argument("--ninput", "-n", type=int, default=1, help="number of word attributes for bi-lstm input. [ex: 1 -> use token, 2 -> use tok+tag, etc")
    #parser.add_argument("--aux", action="store_true", help="auxiliary tasks")
    #parser.add_argument("--auxids", type=int, default=100, help="auxiliary task identifier (use NINPUT to AUXIDS attributes as aux tasks")
    #parser.add_argument("--charrnn", "-C", type=int, default=0, help="use char level bi-lstm. 0) no char bilstm, 1) split on chars, 2) split on __. other options are lazy splitting to speed training")
    #parser.add_argument("--charsize", "-p", type=int, nargs="+", default=[16], help="char embedding size")
    #parser.add_argument("--charrnnsize", "-P", type=int, nargs="+", default=[32], help="char based embedding size")
    #parser.add_argument("--embeddingsizedir", type=str, default=None, help="dir for files containing embedding size for each field (cat,tok,tag,morph1,...morphn")
    #parser.add_argument("--suffix", type=str, default="", help="add suffix to training and dev set filenames")
    
    args = parser.parse_args()
    
    parameters_list = []
    
    def get_params() :
        return {"datadir" : args.datadir,
                "golddir" : args.golddir,
                "iterations": args.iterations,
                "outdir" : args.outputdir,
                "beamsize" : args.beamsize,
                }

    for lang in args.languages :
        for exp in os.listdir("{}/{}".format(args.outputdir, lang)) :
            hyper = std_hyperparameters()
            params = get_params()
            params["hyper"] = None
            params["language"] = lang
            params["expe"] = "{}/{}/{}".format(args.outputdir, lang, exp)
            
            parameters_list.append(params)
    
    unix("echo {}  > {}/{}parse_command.txt".format(" ".join(sys.argv), args.outputdir, random.randint(100,1000)))
    
    Parallel(n_jobs=args.threads)(delayed(do_parse_experiment)(i) for i in parameters_list)

if __name__ == "__main__" :
    
    main()
