

def get_tags(filename):
    res = []
    buff = []
    with open(filename, "r") as instream:
        for line in instream:
            line = line.strip().split()
            if line == [] :
                res.append(buff)
                buff = []
            else:
                buff.append(line[0])
    return res

def assign_raw(filein, tags, fileout):
    tot = 0.0
    acc = 0.0
    with open(filein, "r") as instream:
        with open(fileout, "w") as outstream:
            offset = 0
            for i,line in enumerate(instream) :
                sentence = [ word.rsplit("/",1) for word in line.split() ]
                if len(sentence) != len(tags[i+offset]) :
                    print("Sent n° {}. There is a problem, sent: {}, tags[i]: {}".format(i, len(sentence), len(tags[i])))
                    offset += 1
                    #continue
                for j,tag in enumerate(tags[i+offset]):
                    if sentence[j][1] == tag :
                        acc += 1
                    tot += 1
                    sentence[j][1] = tag
                outstream.write( " ".join([ "/".join(word) for word in sentence ]) + "\n" )
    return acc / tot

def linearize(tokens):
    res = []
    for i,tok in enumerate(tokens) :
        res.append(tok)
        if i < len(tokens) - 1 and tokens[i+1] != ")" and tok != "(" :
            res.append(" ")
    return "".join(res)


def assign_mrg(filein, tags, fileout):
    tot = 0.0
    acc = 0.0
    with open(filein, "r") as instream:
        with open(fileout, "w") as outstream:
            offset = 0
            for i,line in enumerate(instream) :
                sentence = line.strip().replace("(", " ( ").replace(")", " ) ").split()
                length = sum([1 if "=" in word else 0 for word in sentence])
                if length != len(tags[i+offset]) :
                    print("Sent n° {}. There is a problem, sent: {}, tags[i]: {}, {}".format(i, len(sentence), len(tags[i]), line))
                    offset += 1
                    #continue
                for j in range(len(sentence)) :
                    if "=" in sentence[j]:
                        id,*word = sentence[j].split("=")
                        if sentence[j-1] == tags[i+offset][int(id)]:
                            acc += 1
                        tot += 1
                        sentence[j-1] = tags[i+offset][int(id)]
                
                
                outstream.write(linearize(sentence) + "\n" )
    return acc / tot


def main(train_tagsf, dev_tagsf, test_tagsf, corpusin, corpusout):

    dev = corpusin + "/dev.mrg"
    devraw = corpusin + "/dev.raw"
    
    test = corpusin + "/test.mrg"
    testraw = corpusin + "/test.raw"
    
    train = corpusin + "/train.mrg"
    
    train_tags = get_tags(train_tagsf)
    dev_tags = get_tags(dev_tagsf)
    test_tags = get_tags(test_tagsf)
    
    print("raw dev")
    acc = assign_raw(devraw, dev_tags, corpusout + "/dev.raw")
    print("tagging accuracy:",acc)
    
    print("raw test")
    acc = assign_raw(testraw, test_tags, corpusout + "/test.raw")
    print("tagging accuracy:",acc)
    
    print("mrg dev")
    acc = assign_mrg(dev, dev_tags, corpusout + "/dev.mrg")
    print("tagging accuracy:",acc)
    
    print("mrg test")
    acc = assign_mrg(test, test_tags, corpusout + "/test.mrg")
    print("tagging accuracy:",acc)

    print("mrg train")
    acc = assign_mrg(train, train_tags, corpusout + "/train.mrg")
    print("tagging accuracy:",acc)


train_tags="marmot_tags/marmot_spmrl/train.German.marmot.txt"
dev_tags="marmot_tags/marmot_spmrl/dev.German.marmot.txt"
test_tags="marmot_tags/marmot_spmrl/test.German.marmot.txt"

main(train_tags, dev_tags, test_tags, "tigerM15", "tigerM15pred_marmot")

train_tags = "german_train_spmrl_pred_tags.txt"
dev_tags   = "german_dev_spmrl_pred_tags.txt"
test_tags  = "german_test_spmrl_pred_tags.txt"

main(train_tags, dev_tags, test_tags, "tigerM15", "tigerM15pred_spmrl")

