Eyeriss-Like Architecture
----------------------------
This folder contains an architecture based on the Eyeriss design description proposed
[here](https://people.csail.mit.edu/emer/papers/2017.01.jssc.eyeriss_design.pdf).

Q&As:
----------------------------
1. What is the usage of the "DummyBuffer"?

   DummyBuffer is placeholder memory block that helps Timeloop to correctly model 
   the row stationary mappings. With the insertion of the Dummybuffer, we can map 
   M dimension to both the rows of the PE and cols of PE -- otherwise, Timeloop can 
   only map the model dimensions to either the rows or columns of the PE array.
   
2. Does this design perform exactly as the Eyeriss design in the paper?
    
   For many cases, it does. However, since Timeloop's mapper is not as flexible as a manually 
   generated mapping, sometimes it does not produce the exact results as shown in the paper.
   The main limitation is that Timeloop considers the factors of a dimension, e.g., 4 is a factor 
   of 256, when it performs mapping space explorations. Therefore, when using a non-divisible factor 
   result in better mappings, Timeloop is not able to find it. However, padding the workload dimensions 
   will solve these types of problems.
   
3. How long do the Timeloop simulations take?
  
   Depending on your workload, the simulation takes various amounts of time to finish. Generally, they should 
   converge within 30 mins. You can manually stop the exploration when you see things are converging by 
   pressing `ctrl + C`. They sometimes will take much longer to automatically stop as we set the converging criteria to be pretty high to avoid early-stop with suboptimal mappings. Use you own
   judgement.
   



