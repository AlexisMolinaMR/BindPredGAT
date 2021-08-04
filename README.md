# BindPredGNN
Implementation of a Graph Neural Networks for binding affinity/energy prediction.

### Generate graphs 

**Parameters**

```
#control file for PeleAI3D - Graph statistics

path: 
output: 
run_name: 
ligand_name: 
selection_radius: 
center: 
decay_function:
nodes: 
``` 

### Fitting graph 

**Parameters**

```
#control file for PeleAI3D - Fitting a model

path_graph: 
output: 
test_size:
seed: 
task: 
cpus: 
scaler: 
algorithm: 
``` 

### Pipelining (_pipeline_)

**Parameters**

```
#control file for PeleAI3D - Pipeline
path: 
output: 
run_name: 
ligand_name: 
selection_radius: 
center: 
decay_function:
nodes: 
# ------------------------------------
#pipe: True
#target: 
# ------------------------------------
test_size: 
seed: 
task: 
cpus: 
scaler: 
algorithm: 
``` 

### Execution

You may pass the _input.yaml_ file to the ```peleAI3d.py``` as follows:

```
python3 peleAI3d.py input.yaml
```
