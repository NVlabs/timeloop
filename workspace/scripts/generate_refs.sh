#!/bin/bash
python3 generate_timeloop_spec.py


cd ..

cd example_designs
python3 run_example_designs.py --clear-outputs
python3 run_example_designs.py --architecture all --generate-ref-output

cd ../tutorial_exercises/01_accelergy_timeloop_2020_ispass/timeloop
python3 run_example.py --clear-outputs
python3 run_example.py all --generate-ref-output

cd ../timeloop+accelergy
python3 run_example.py --clear-outputs
python3 run_example.py all --generate-ref-output