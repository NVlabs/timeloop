# Timeloop user guide

## Getting started

* Install the following dependencies.
```
scons
libconfig++-dev
libboost-dev
libboost-iostreams-dev
libboost-serialization-dev
```

* Clone the timeloop repository.
```
mkdir timeloop-dev
cd timeloop-dev
git clone ssh://path/to/timeloop.git
```

* In addition to the main source code, you need the source code for a
  power-area-timing (pat) model to build timeloop. A placeholder pat model
  is included in the main repository. Before building timeloop, place a
  symbolic link to the pat model like so:
```
cd timeloop/src
ln -s ../pat-public/src/pat .
cd ..
```

* You may build against any other custom pat model as well, as long as
  it exports the same interface as the pat/pat.hpp in the included model.
  The implementation must be in a pat/pat.cpp file. As before, create a
  symbolic link to the source code for the power-area-timing model and
  place it in `src/pat`, for example:
```
git clone ssh://path/to/timeloop-pat[XXX].git
cd timeloop/src
ln -s ../../timeloop-pat[XXX]/src/pat .
cd ..
```

* Once the pat link is set up, you can build timeloop using scons.
```
scons -j4
```

* Run timeloop with a sample configuration.
```
./build/opt/timeloop configs/sample.cfg > sample.out
```
This will put timeloop's log in `sample.out` and the mapping and statistics
in `timeLoopOutput.xml`.

## Further reading
[ISPASS 2019 paper](http://parashar.org/ispass19.pdf)
