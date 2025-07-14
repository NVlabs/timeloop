printf "\n=== Buffer ===\n"
printf "\n --- A --- \n"
grep "=== Buffer" *.stats.txt -A 70 | grep -e 'shape' -e ' data storage c' -e 'A[a-z]* scalar reads'
printf "\n --- B --- \n"
grep "=== Buffer" *.stats.txt -A 110 | grep "B:" -A 50 | grep -e 'shape' -e 'data storage c' -e 'A[a-z]* scalar reads' \
-e 'Gated scalar reads'
printf "\n --- Z --- \n"
grep "=== Buffer" *.stats.txt -A 170 | grep "Z:" -A 50 | grep -e 'shape' -e 'data storage c' -e 'A[a-z]* scalar [reads|updates]' \
 -e 'Gated scalar [reads|updates]'

echo ""

printf "=== MAC ===\n\n"
grep "=== MAC ===" timeloop-mapper.stats.txt -A 20 | grep "A[a-z]* Computes"
grep "=== MAC ===" timeloop-mapper.stats.txt -A 20 | grep Cycles

echo ""


printf "=== Summary Stats ===\n\n"
grep "Summary Stats" timeloop-mapper.stats.txt -B 8 | grep topology
echo " "
grep "Summary Stats" timeloop-mapper.stats.txt -A 50 | grep "Algorithmic Computes" -A 40 | grep -v "<==>"
