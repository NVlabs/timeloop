One-time setup for Simba regression tests:
1. Download the conv_problems.h file from the DeepBench repository.
2. Build the c++_to_python translator by typing "scons".
3. Run "./c++_to_python > conv_problems.py".

To run the Simba regression tests:
1. Run `pip3 install libconf numpy pyyaml`.
2. Run "./test_simba_chip.py".
