
wget http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/tigercorpus-2.2.xml.tar.gz
wget http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/tigercorpus2.1.zip

# Maier 2015 split

mkdir tiger22
mkdir tigerM15

tar xvzf tigercorpus-2.2.xml.tar.gz -C tiger22

## 46234 50224 skipped (annotation error)
treetools transform tiger22/tiger_release_aug07.corrected.16012013.xml tigerM15_root_attach.export --trans root_attach --src-format tigerxml --dest-format export

discodop treetransforms tigerM15_root_attach.export tigerM15/treebank.mrg --inputfmt=export --outputfmt=discbracket --punct=move

head -40468 tigerM15/treebank.mrg > tigerM15/train.mrg
tail -9998  tigerM15/treebank.mrg | head -5000 > tigerM15/dev.mrg
tail -4998  tigerM15/treebank.mrg > tigerM15/test.mrg

# Hall and Nivre 2008 split, script de van cranenburgh

mkdir tigerHN8
mkdir tigerHN8dev

mkdir tiger21
unzip tigercorpus2.1.zip -d tiger21

python3 tigersplit.py

for corpus in  tigerdev tigertest tigertraindev tigertraintest
do
    treetools transform tiger21/${corpus}.export tiger21/${corpus}_root_attach.export --trans root_attach
done

discodop treetransforms tiger21/tigertraindev_root_attach.export  tiger21/train.mrg     --inputfmt=export --outputfmt=discbracket --punct=move
discodop treetransforms tiger21/tigertraintest_root_attach.export tiger21/train_dev.mrg --inputfmt=export --outputfmt=discbracket --punct=move
discodop treetransforms tiger21/tigerdev_root_attach.export       tiger21/dev.mrg       --inputfmt=export --outputfmt=discbracket --punct=move
discodop treetransforms tiger21/tigertest_root_attach.export      tiger21/test.mrg      --inputfmt=export --outputfmt=discbracket --punct=move

cat tiger21/train.mrg   > tigerHN8dev/train.mrg
cat tiger21/dev.mrg     > tigerHN8dev/dev.mrg
cat tiger21/test.mrg    > tigerHN8dev/test.mrg


cat tiger21/train_dev.mrg > tigerHN8/train.mrg
cat tiger21/dev.mrg       > tigerHN8/dev.mrg
cat tiger21/test.mrg      > tigerHN8/test.mrg



bash generate_raw_test.sh tigerHN8 tigerHN8dev tigerM15

bash generate_pred_tags_data.sh

python3 to_predicted_tags.py


rm -r tiger21 tiger22 marmot_tags
rm tigercorpus-2.2.xml.tar.gz tigercorpus2.1.zip tigerM15_root_attach.export marmot_spmrl.tar.bz2










