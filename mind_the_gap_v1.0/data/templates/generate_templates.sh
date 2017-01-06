
# tpls1.md
# tpls2.md
# tpls3.md
# tpls5.md
# tpls6.md



cat gap_minimal.md tpls1.md                     >  gap_tpls+1.md
cat gap_minimal.md tpls2.md                     >  gap_tpls+2.md
cat gap_minimal.md tpls3.md                     >  gap_tpls+3.md

cat gap_minimal.md tpls1.md tpls2.md            >  gap_tpls+1+2.md
cat gap_minimal.md tpls1.md tpls3.md            >  gap_tpls+1+3.md
cat gap_minimal.md tpls2.md tpls3.md            >  gap_tpls+2+3.md
cat gap_minimal.md tpls1.md tpls2.md tpls3.md   >  gap_tpls+1+2+3.md


cat gap_minimal.md tpls1.md tpls2.md tpls5.md   > gap_tpls+1+2+5.md
cat gap_minimal.md tpls1.md tpls2.md tpls6.md   > gap_tpls+1+2+6.md
cat gap_minimal.md tpls1.md tpls5.md            > gap_tpls+1+5.md
cat gap_minimal.md tpls1.md tpls6.md            > gap_tpls+1+6.md

cat gap_minimal.md tpls1.md tpls2.md tpls5.md tpls6.md > gap_tpls+1+2+5+6.md



