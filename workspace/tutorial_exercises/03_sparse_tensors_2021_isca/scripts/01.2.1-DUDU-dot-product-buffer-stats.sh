printf "\n=== Buffer ===\n"
printf "\n --- A --- \n"
grep "=== Buffer" *.stats.txt -A 50  | grep -e 'capacity' -e 'Scalar reads'
printf "\n --- B --- \n"
grep "=== Buffer" *.stats.txt -A 80  | grep "B:" -A 20 | grep -e 'capacity' -e 'Scalar reads'
printf "\n --- Z --- \n"
grep "=== Buffer" *.stats.txt -A 100 | grep "Z:" -A 20 | grep -e 'capacity' -e 'Scalar [reads|updates]'
