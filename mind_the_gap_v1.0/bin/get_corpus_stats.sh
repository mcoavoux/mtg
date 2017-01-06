
currentdir=`pwd`

train=${currentdir}/../data/tigerM15/train.mrg
dev=${currentdir}/../data/tigerM15/dev.mrg
test=${currentdir}/../data/tigerM15/test.mrg

echo $train
echo $dev
echo $test

./mtg_gcc -s -t ${train} -o stats_tiger_trainM15.md

cd ../java_prototype/bin
echo `pwd`
java Lcfrs ${train} ${dev} ${test} foobar ../std_templates.md 0 local 1 > ${currentdir}/stats_tigerM15_gap_java.md
java Lcfrs ${train} ${dev} ${test} foobar ../std_templates.md 0 local 1 swap > ${currentdir}/stats_tigerM15_swap_java.md
java Lcfrs ${train} ${dev} ${test} foobar ../std_templates.md 0 local 1 cswap > ${currentdir}/stats_tigerM15_cswap_java.md
