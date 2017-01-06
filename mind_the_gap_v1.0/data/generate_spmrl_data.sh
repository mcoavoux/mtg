

## generate spmrl evaluation data

folder=SPMRL2014/german_spmrl_gold
mkdir $folder

for corpus in test
do
    treetools transform SPMRL2014/GERMAN_SPMRL/gold/xml/${corpus}/${corpus}.German.gold.xml  ${folder}/${corpus}.export --src-format tigerxml --dest-format export
    discodop treetransforms ${folder}/${corpus}.export ${folder}/${corpus}.mrg --inputfmt=export --outputfmt=discbracket
done


#for type in gold pred
#do
    #folder=SPMRL2014/polish_spmrl_${type}
    #mkdir $folder

    #for corpus in train dev test
    #do
        #treetools transform SPMRL2014/POLISH_SPMRL/${type}/xml/${corpus}/${corpus}.Polish.${type}.xml  ${folder}/${corpus}.export --src-format tigerxml --dest-format export
        #discodop treetransforms ${folder}/${corpus}.export ${folder}/${corpus}.mrg --inputfmt=export --outputfmt=discbracket
    #done
#done


#for type in gold pred
#do
    #folder=SPMRL2014/swedish_spmrl_${type}
    #mkdir $folder
    
    #for corpus in train dev test
    #do
        #treetools transform SPMRL2014/SWEDISH_SPMRL/${type}/xml/${corpus}/${corpus}.swedish.${type}.xml  ${folder}/${corpus}.export --src-format tigerxml --dest-format export
        #discodop treetransforms ${folder}/${corpus}.export ${folder}/${corpus}.mrg --inputfmt=export --outputfmt=discbracket
    #done
#done
