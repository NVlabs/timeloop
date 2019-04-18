* `timeloop.py` - This has a function called `run_timeloop(dirname, configfile, logfile='timeloop.log', workload_bounds=None)` which
    * copies `configfile` to `dirname`
    * modifies the copied configuration file with `workload_bounds`, if given
    * runs timeloop
    * puts all results in `dirname`. `dirname` will contain:
        * `logfile` which is the text log from timeloop
        * the (possibly modified) configuration file copy
        * the xml output file from timeloop

* `parse_timeloop_output.py` - This has a function called `parse_timeloop_stats(path)` which looks for `timeLoopOutput.xml` at `path` (can be a full file path or just a path to the directory) and parses it and returns a python dictionary with the statistics we care about.
This file is also a command-line tool that uses this functionality to produce pickle files of these dictionaries, which can be used to store and compare parsed outputs over time.
