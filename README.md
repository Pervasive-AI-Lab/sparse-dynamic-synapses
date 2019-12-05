The Unreasonable Effectiveness of Sparse Dynamic Synapses for Continual Learning
======

### Introduction


At [Numenta](https://www.numenta.com) *continual learning* is mostly believed 
to happen in the brain thanks to sparsity and dynamically growing synaptic 
connections. Sparsity of activations and connections allows to condense in a 
reasonable low dimension (e.g. 10k bits) an enormously large quantity of 
non-overlapping distributed patterns.

This means that once you want to learn a new pattern you just need to grow new 
synapses to encode that knowledge and thanks to sparsity, they will rarely 
interfere with one another. This idea of learning by simply encoding knowledge 
in different sparse weights is quite powerful for continual learning since it
removes the problems of interference among weights. In standard deep nets, the 
contributions of the weights is much more distributed and difficult to 
disentangle.

This is due to full connectivity, and the very nature of gradient descent
optimization.The idea of this project would be to work on highly sparse 
deep nets (2-10% connectivity) and slowly grow connections maintaining 
sparsity in the activations and eventually preserving old weights as much 
as possible (i.e. fixed or slow learning rate?) but still using backprop. 

### Papers

Possibly related, interesting papers:

- [“Piggyback: Adapting a Single Network to Multiple Tasks by Learning to 
Mask Weights”](https://arxiv.org/abs/1801.06519)
- [“Compete to Compute”](https://papers.nips.cc/paper/5059-compete-to-compute)
- [“Superposition of many models into one”](https://arxiv.org/abs/1902.05522)
- [“The power of Sparsity in CNNs”](https://arxiv.org/abs/1702.06257)
- ["PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning"](https://arxiv.org/abs/1711.05769)
- [“Selfless Sequential Learning”](https://arxiv.org/abs/1806.05421)

### Exploratory Experiments

In this codebase you will find a just a few exploratory experiments, trying to 
apply sparsity in continual learning. In particular, sparsity of both 
the units and the weightsis enforced through the use of the `Kwinners` and 
`SparseWeights` implementations offered in `nupic.torch`.

At the moment, this codebase supports:

- 3 benchmarks: `Permuted MNIST`, `SPlit MNIST` and `ICifar10`.
- 2 architectures: Plain `MLPs` and `CNNs` with parametrized structure.

The main idea is to apply sparsity in these settings and see if we can have 
a better average accuracy across tasks at the end of the continual learning
process. Results up to now are promising, especially with MLPs where the 
difference in accuracy can exceed 10% in some cases. However, more work seems
to be done to scale these results to ConvNets. 

### Project Structure

Here we list the directory structure of the project:

- `benchmarks`: It contains all the data loaders and utility scripts for
                handling the 3 benchmarks provided.
- `exps`: It contains all the experiments config files.
- `models`: It contains the neural networks architectures considered.
- `results`: It's a void directory that will contain the results of the exps
             in the pkl format.
- `utils`: It contains all the utility scripts for the experiments,
           mostly building on top of numpy and pytorch.


### Getting Started

When using anaconda virtual environment all you need to do is run the following 
command and conda will install everything for you. 
See [environment.yml](./environment.yml):

    conda env create --file environment.yml
    conda activate sparse_syn
    pip install -r requirements.txt
    
and than run the default experiment:

    python run_exps.py
    
Or a specific experiment with its name configuration (all the exps names are 
listed in the `exps/exps_params.cfg` file.):

    python run_exps.py --name <exp_name>
    
    
### Experiments Parameters

For each experiment the following parameters has been considered:

- `benchmark`: (str) Continual learning benchmark used for the experiment (`"cifar"` 
               or `"mnist"`).
- `mnist_mode`: (str) In case the `"mnist"` benchmark is used it can be either `"perm"` 
                or `"split"`.
- `num_batch`: (int) Number of training batches/tasks to generate (for cifar or split 
               mnist this number should be fixed to 10 and 5 respectively).
- `cumul`: (bool) `True` if we want to run the cumulative baseline (training on
            the union of all the batches training sets.)  
- `sparsify`: (bool) `True` if we want to introduce the `Kwinners` and 
               `SparseWeights` layers after every fully connected layer or conv.
- `percent_on_fc`: (float) Percentage of active units after a fully connected layer.
- `percent_on_conv`: (float) Percentage of active units after a conv layer.
- `k_inference_factor`: (float) Boosting parameter for `Kwinners`.
- `boost_strength`: (float) Boosting parameter for `Kwinners` 
                    (`0` to shut it off completely).
- `boost_strength_factor`: (float) Boosting parameter for `Kwinners`.
- `duty_cycle_period`: (int) Boosting parameter for `Kwinners`.
- `weight_sparsity_fc`: (float) Weights sparsity percentage for a fully 
                        connected layer.
- `weight_sparsity_conv`: (float) Weights sparsity percentage for conv layer.
- `cnn`: (bool) `True` if the architecture is a CNN, otherwise MLP.  
- `hidden_units`: (int) Number of units in each hidden layer.
- `hidden_layers`: (int) Number of hidden layers.
- `dropout`: (int) Dropout percentage.
- `lr`: (float) Learning rate.
- `nesterov`: (bool) Nesterov optimizer.
- `momentum`: (float) Momentum.
- `weight_decay`: (float) Weight Decay
- `mb_size`: (int) Mini-Batch size.
- `train_ep`: (int) Training epochs for the first task.
- `train_ep_inc`: (int) Training epoch for the following tasks.
- `record_stats`: (bool) `True` to record stats about sparsity.
