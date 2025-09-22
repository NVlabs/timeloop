

printf "\n=== BackingStorage ===\n"
printf "\n --- Weights --- \n"
grep "=== BackingStorage" *.stats.txt -A 50 | grep "Weights:" -A 26 | grep -e 'shape' -e ' storage c' -e 'format' \
-e reads -e Rank


printf "\n=== Buffer ===\n"
printf "\n --- Weights --- \n"
grep "=== Buffer" *.stats.txt -A 63 | grep -e 'shape' -e ' storage c' -e 'format' -e reads -e fills -e Rank

printf "\n --- Inputs  --- \n"
grep "=== Buffer" *.stats.txt -A 120 | grep "Inputs:" -A 40 | grep -e 'shape' -e 'storage c' -e 'A[a-z]* scalar [reads|fills]' -e 'Skipped scalar [reads|fills]' -e Rank


printf "\n --- Outputs --- \n"
grep "=== Buffer" *.stats.txt -A 170 | grep "Outputs:" -A 40 | grep -e 'shape' -e 'data storage c' -e 'A[a-z]* scalar [reads|updates]' -e 'Skipped scalar [reads|updates]' -e Rank

echo ""

printf "=== MAC ===\n\n"
grep "=== MACC ===" timeloop-mapper.stats.txt -A 20 | grep "A[a-z]* Computes"
grep "=== MACC ===" timeloop-mapper.stats.txt -A 20 | grep Cycles

echo ""

printf "=== Summary Stats ===\n\n"
grep "Summary Stats" timeloop-mapper.stats.txt -B 8 | grep topology
echo " "
grep "Summary Stats" timeloop-mapper.stats.txt -A 50 | grep "Algorithmic Computes" -A 40 | grep -v "<==>"
