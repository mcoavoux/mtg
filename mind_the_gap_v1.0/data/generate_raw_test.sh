





#for corpus in negra30 negraall tigerHN8 tigerHN8dev tigerM15
for corpus in $@
do
    for type in test dev
    do
        discodop treetransforms ${corpus}/${type}.mrg ${corpus}/${type}.export --inputfmt=discbracket --outputfmt=export
        treetools transform ${corpus}/${type}.export ${corpus}/${type}.raw --dest-format terminals --dest-opts terminals_pos
        python3 replace_brackets.py ${corpus}/${type}.raw
    done
done
