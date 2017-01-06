
import sys

assert len(sys.argv) == 2

f = sys.argv[1]

corpus=[]
instream = open(f, "r", encoding="utf8")

d = { "(" : "#LRB#", ")" : "#RRB#" }
e = { "$(" : "$[", "$)" : "$]" }

for line in instream:
    line = line.strip().split(" ")
    tokens = [tok.rsplit("/", 1) for tok in line]
    for i in range(len(tokens)) :
        assert(len(tokens[i]) == 2)
        if tokens[i][0] in d :
            tokens[i][0] = d[tokens[i][0]]
        if tokens[i][1] in e :
            tokens[i][1] = e[tokens[i][1]]
    
    newline= " ".join(["/".join(tok) for tok in tokens])
    if "(" in newline or ")" in newline:
        print(newline)
        newline = newline.replace("(", d["("])
        newline = newline.replace(")", d[")"])
        print(newline)
    corpus.append(newline)

instream.close()

outstream = open(f, "w", encoding="utf8")

for line in corpus:
    outstream.write(line)
    outstream.write("\n")

outstream.close()


