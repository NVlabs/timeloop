Simba-Like Architecture
----------------------------
This folder contains an architecture based on the Simba design description proposed
[here](https://people.csail.mit.edu/emer/papers/2017.01.jssc.eyeriss_design.pdf). This architecture is similar to the
NVDLA architecture as well.

Q&As:
----------------------------
1. Why is the technology different?

   Since the open-sourced energy estimation plug-ins have the most flexible support
    for the 45nm components, we changed the technology to 45nm.
    
2. What are major the differences between this design and the original design?
   - Simba is a multi-chip design, but here we only show the architecture description of a single chip.
   - We have reduced the number of MACs inside each PE from 64 to 16.
   
3. How long do the Timeloop simulations take?
  
   Depending on your workload, the simulation takes various amount of time to finish. Generally, they should 
   converge within 30 mins. You can manually stop the exploration when you see things are converging by 
   pressing `ctrl + C`. They sometimes will take much longer to automaticaly stop as we set the converging cretiria to be pretty high to avoid early-stop with subooptimal mappings. Use you own
   judgement.