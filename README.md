# BindPredGNN
Implementation of a Graph Neural Networks for binding affinity/energy prediction.

### Generate graphs 

**Parameters**

```
#control file for BindPred - Graph statistics

path: '/input/path/folder/'
output: '/output/path/folder/'
ligand_name: 'LIG'
selection_radius: 20
center: 'geometric'
nodes: 'atoms'
decay_function: 'expo'
fitting: 'GNN'
target: '/path/to/target.csv'
batch_size: 10
epochs: 45
learning_rate: 0.01

``` 

### Execution

You may pass the _input.yaml_ file to the ```bindPred.py``` as follows:

```
python3 bindPred.py input.yaml
```
