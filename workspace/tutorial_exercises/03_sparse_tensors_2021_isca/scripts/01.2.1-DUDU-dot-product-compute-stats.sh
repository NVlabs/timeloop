printf "=== MAC ===\n\n"
grep "=== MAC ===" timeloop-mapper.stats.txt -A 20 | grep Computes
grep "=== MAC ===" timeloop-mapper.stats.txt -A 20 | grep Cycles