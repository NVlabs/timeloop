# Timeloop user guide

## Getting started

* Install the following dependencies.
```
scons
libconfig++-dev
libboost-dev
libboost-iostreams-dev
libboost-serialization-dev
libyaml-cpp-dev
libncurses-dev
libtinfo-dev
libgpm-dev
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

* Instead of the included placeholder pat model, you may build against any
  other custom pat model, as long as it exports the same interface as the
  pat/pat.hpp in the included model. The implementation must be in a
  pat/pat.cpp file. As before, create a symbolic link to the source code for
  the power-area-timing model and place it in `src/pat`, for example:
```
git clone ssh://path/to/timeloop-pat[XXX].git
cd timeloop/src
ln -s ../../timeloop-pat[XXX]/src/pat .
cd ..
```

* Another way to provide a power/energy model to Timeloop is to integrate
[Accelergy](http://accelergy.mit.edu) with Timeloop. To do so, you need to
either install Accelergy so that the shell can find it (i.e., `which accelergy`
works), or provide the path to Accelergy binary as an environmental variable,
`ACCELERGYPATH`, when running Timeloop.

When building timeloop in the next step, you also need to provide an extra
`--accelergy` flag to `scons` so that it builds Timeloop and makes it use
Accelergy for the energy model.

```
scons --accelergy
```

* Once the pat link is set up, you can build timeloop using scons.
```
scons -j4
```
This builds 3 different tools:
* `timeloop-mapper` is the complete application that instantiates an architecture,
  constructs its mapspace, searches for an optimal mapping within the mapspace
  and reports statistics for the optimal mapping.
* `timeloop-model` instantiates an architecture, evalutes a specific given
  mapping of a workload and reports the statistics.
* `timeloop-metrics` simply instantiates an architecture and reports its
  workload-independent characteristics such as area and energy-per-access
  for various architectural structures.

* Run timeloop with a sample configuration.
```
cd configs/timeloop
../../build/timeloop-mapper ./sample.yaml > sample.out
```

This will place timeloop's log in `sample.out` and generate the following outputs:
* `timeloop-mapper.stats.txt` Simulation stats (performance, energy, etc.)
* `timeloop-mapper.map.txt/cfg` The optimal mapping in different formats
  (the latter can be used in conjunction with the
  input architecture and problem spec to re-run the model on the optimal
  mapping.)
* `timeloop-mapper.map+stats.xml` An XML-formatted copy of the stats and optimal mapping
  which is used by various Python scripts to extract results from batch runs.

## Further reading

Serially walking through the exercises in our [Timeloop tutorial series](https://github.com/jsemer/timeloop-accelergy-exercises/tree/master/exercises/timeloop) serves as an excellent hands-on introduction to the tool:

For a deeper technical overview please read our [ISPASS 2019 paper](http://parashar.org/ispass19.pdf)
