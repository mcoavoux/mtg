


mkdir negra
tar xvzf negra-corpus.tar.gz -C negra


mkdir negraall
mkdir negra30

treetools transform negra/negra-corpus.export negra-corpus_root_attach.export --trans root_attach --src-enc iso-8859-1 --dest-enc utf-8

#--inputenc=iso-8859-1 --outputenc=utf-8
discodop treetransforms negra-corpus_root_attach.export negra30/treebank.mrg  --inputfmt=export --outputfmt=discbracket  --punct=move --maxlen=30
discodop treetransforms negra-corpus_root_attach.export negraall/treebank.mrg --inputfmt=export --outputfmt=discbracket  --punct=move

# Negra30 décrit dans Maier 2015 (80,10,10) pas de précision sur le nombre de phrases (1833 ou 1834)
#                     Kallmeyer et Maier 2013 : reportent 1833 dernières phrases dans le test, tout le reste dans le train
tail -1833 negra30/treebank.mrg  > negra30/test.mrg
tail -3666 negra30/treebank.mrg | head -1833 > negra30/dev.mrg
head -14669 negra30/treebank.mrg > negra30/train.mrg

# 20602 sentences, 18602 train, 1000 test, 1000 dev
# Dubey & Keller 2003
tail -1000 negraall/treebank.mrg > negraall/dev.mrg
tail -2000 negraall/treebank.mrg | head -1000 > negraall/test.mrg
head -18602 negraall/treebank.mrg > negraall/train.mrg

bash generate_raw_test.sh negraall negra30

rm -r negra
rm negra-corpus_root_attach.export
