Example Designs for Timeloop-Accelergy Evaluation System
------------------------------------------------------------
This folder contains some baseline implementations of an Eyeriss-like architecture,a simba-like (NVDLA style) architecture,
a simple weight stationary architecture, a processing-in-memory architecture, and an output stationary architecture.
Please find them in the `designs` folder. 

### System requirement
In order to run the example designs, you need to be either inside a docker with installed tools 
(e.g., [infrastructure docker](https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure) 
or manually install the Accelergy-Timeloop evaluation system. 

### File Structure
- example_designs: 
   - architecture descriptions, compound component descriptions.
   - top.yaml.jinja2: Top-level file gathering
   - _components directory: Compound components
   - _include directory: Default problem file and mapper description
- layer_shapes: 
    - Example workloads: AlexNet, VGG01, VGG02
- scripts
    - A set of scripts for generating your own workloads in Timeloop format
    - Instructions:
        - `cd scripts`
        - modify the `cnn_layers.py` file to describe your own workload
        - `python3 construct_workloads.py <my_dnn_model_name>`

### Run simulations

To run a simulation using timeloop-accelergy system you should run the `run_example_designs.py` file. 

Here is an example for running AlexNet Layer1 on the `simple_weight_stationary` architecture: 
```
python3 run_example_designs.py --architecture simple_weight_stationary
```

Outputs will be generated in the `example_designs/simple_output_stationary/outputs` directory.

Run `python3 run_example_designs.py -h` to see available arguments. 

```
usage: run_example_designs.py [-h] [--clear_outputs] [--architecture ARCHITECTURE] [--generate_ref_outputs] [--problem PROBLEM] [--n_jobs N_JOBS]

options:
  -h, --help            show this help message and exit
  --clear_outputs       Clear all generated outputs
  --architecture ARCHITECTURE
                        Architecture to run in the example_designs directory. If not specified, all architectures will be run.
  --generate_ref_outputs
                        Generate reference outputs instead of outputs
  --problem PROBLEM     Problem to run in the layer_shapes directory. If a directory is specified, all problems in the directory will be run. If not specified, the default problem will be run.
  --n_jobs N_JOBS       Number of jobs to run in parallel
```

Full DNNs can be run by specifying a directory containing multiple layer files:
```
python3 run_example_designs.py --clear_outputs --architecture eyeriss_like --problem CONV/VGG02
```

Outputs for each layer will be generated in subdirectories of the `example_designs/eyeriss_like/outputs` directory.


** Note that for the provided designs and workloads, your simulation should generally converge within 30 mins. Once you see
the simulations converging, you can press `ctrl + C` to manually stop them. They sometimes will take much longer to 
automaticaly stop as we set the converging criteria to be pretty high to avoid early-stop with subooptimal mappings. 
Please use you own judgement. **


###  Related reading
 - [eyeriss-like design](https://people.csail.mit.edu/emer/papers/2017.01.jssc.eyeriss_design.pdf)
 - [simba-like architecture](https://people.eecs.berkeley.edu/~ysshao/assets/papers/shao2019-micro.pdf)
 - simple weight stationary architecture: you can refer to the related lecture notes
 - simple output stationary architecture: you can refer to the related lecture notes
