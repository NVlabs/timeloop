## Dependencies
- gmp-6.2.0
- barvinok-0.41.3

## Codebase overview
- The main Tenssella code is in `main.cpp`.
- User-specified architecture (e.g., `arch-2D-3L.hpp`), problem (e.g., `problem-GEMM.hpp`) and mapping (e.g., `mapping-GEMM-3L.hpp`) specs are currently described as ISL/C++ sets and relations and included into `main.cpp`. They will be parsed in from YAML files in future.

## Building and running Tenssella
To build Tenssella type:

```scons```

To run Tenssella type:

```./tenssella```

There are no command-line arguments because as mentioned before, the architecture, problem and mapping are baked in as C++ files and included into the build.

Tenssella has trace levels 0,1,2 for debugging. Default is 0, which does not emit any debugging messages. Other levels will emit intermediate sets and relations that are being constructed. To run with a higher trace level set the environment variable `TENSSELLA_TRACE_LEVEL=x`, e.g.,

```TENSSELLA_TRACE_LEVEL=2 ./tenssella >& tenssella.log```

Running Tenssella will produce the following files:

- `out/out.cpp` for (a) human inspection and (b) building into an executable emulator for functional validation.
- `out/out.ast`for further transformation into machine-level configuration.

## Running the output code on a host emulator.
- Go into the `out/` directory.
- Type `scons` to process and build the `out.cpp` file against the emulator.
- Run the emulator by typing `./emulator`.
