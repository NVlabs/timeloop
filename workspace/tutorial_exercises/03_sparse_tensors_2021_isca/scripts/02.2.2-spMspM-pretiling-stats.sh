
printf "\n=== BackingStorage ===\n"
printf "\n --- B --- \n"
grep "=== BackingStorage" *.stats.txt -A 120 | grep "B:" -A 70 | grep -e 'shape' -e ' storage c' -e 'format' \
-e reads -e Rank


printf "\n=== Buffer ===\n"
printf "\n --- A --- \n"
grep "=== Buffer" *.stats.txt -A 43 | grep -e 'shape' -e ' storage c' -e 'format' -e Rank
printf "\n --- B --- \n"
grep "=== Buffer" *.stats.txt -A 150 | grep "B:" -A 50 | grep -e 'shape' -e ' storage c' -e 'format' \
-e reads -e fills -e Rank
printf "\n --- Z --- \n"
grep "=== Buffer" *.stats.txt -A 170 | grep "Z:" -A 50 | grep -e 'shape' -e ' storage c'

echo ""

printf "=== MAC ===\n\n"
grep "=== MAC ===" timeloop-mapper.stats.txt -A 20 | grep "A[a-z]* Computes"
grep "=== MAC ===" timeloop-mapper.stats.txt -A 20 | grep Cycles

echo ""

printf "=== Summary Stats ===\n\n"
grep "Summary Stats" timeloop-mapper.stats.txt -B 8 | grep topology
echo " "
grep "Summary Stats" timeloop-mapper.stats.txt -A 50 | grep "Algorithmic Computes" -A 40 | grep -v "<==>"
