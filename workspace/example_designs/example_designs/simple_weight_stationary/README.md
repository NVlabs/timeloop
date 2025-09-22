Simple Weight Stationary Architecture
----------------------------
This folder contains a simple weight stationary architecture. 

Q&As:
----------------------------
1. How long do the Timeloop simulations take?
  
   Depending on your workload, the simulation takes various amount of time to finish. Generally, they should 
   converge within 30 mins. You can manually stop the exploration when you see things are converging by 
   pressing `ctrl + C`. They sometimes will take much longer to 
   automaticaly stop as we set the converging cretiria to be pretty high to avoid early-stop with subooptimal mappings. Use you own
   judgement.
   
2. How to get started on using the architecture skeleton to model architectures with advanced technologies?
   
   You generally need to modify the definitions of the compound components. If needed, you are also like required to 
   make updated the architecture description to include the additional setup for your architecture. 
   
   An example design for compute-in-memory architecture using ReRAM can be found 
   [here](https://github.com/Accelergy-Project/processing-in-memory-design)
