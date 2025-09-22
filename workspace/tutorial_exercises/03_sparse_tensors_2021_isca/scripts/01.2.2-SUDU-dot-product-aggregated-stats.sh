printf "\n=== Buffer ===\n"
printf "\n --- A --- \n"
grep "=== Buffer" *.stats.txt -A 50  | grep  -e 'capacity' -e 'Scalar reads'
printf "\n --- B --- \n"
grep "=== Buffer" *.stats.txt -A 80  | grep "B:" -A 20 | grep  -e 'capacity' -e 'Scalar reads'
printf "\n --- Z --- \n"
grep "=== Buffer" *.stats.txt -A 100 | grep "Z:" -A 20 | grep  -e 'capacity' -e 'Scalar [reads|updates]'

echo ""

printf "=== MAC ===\n\n"
grep "=== MAC ===" timeloop-mapper.stats.txt -A 20 | grep Computes
grep "=== MAC ===" timeloop-mapper.stats.txt -A 20 | grep Cycles

echo ""


printf "=== Summary Stats ===\n\n"
grep "Summary Stats" timeloop-mapper.stats.txt -B 8 | grep topology
echo " "
grep "Summary Stats" timeloop-mapper.stats.txt -A 50 | grep "Computes" -A 40 | grep -v "<==>"
