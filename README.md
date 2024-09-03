# Transformer Tree Generation

Some Transformer Experimentation generating a tree structure.

## Basic Transformer Results

### Different tasks

We train model on permutations of a single tree and its ability to afterwards predict a correct permutation of the tree.
First we analyse how the model performs on different tasks using a tree with branch factor 10 (b=10) and depth 2 (d=2):

* fix: no permutations in the values [0, 1, ...]
* lwr: leaves values are samples (among all siblings) with replacement
* lwor: leaves values are samples (among all siblings) without replacement (tree permutation, only shuffling leaves)
* nwor: all node values are samples (among all siblings) without replacement (tree permutation, shuffling leaves and branches)

The fix task can be solved by just memorizing the single tree. 
The lwr task is only position-dependent and must learn to predict a uniform distribution over each leaf children.
The lwor task is harder, as previous siblings must be remembered to avoid duplicates, however the group of siblings is always the same.
The nwor task is the hardest, as previous siblings must be remembered and the group of siblings depends on the (shuffled) parent node.

![Different-task-(bf=5).png](imgs/Different-task-(bf=5).png)

### Global Positional Encoding

Given a fixed tree structure the model has to learn to attend specific positions (parent nodes).
For that task learnable positional embeddings seems to help on easier tasks (b=2, d=7):

![Positional Encoding x4_b2_d7.png](imgs/Positional-Encoding-x4_b2_d7.png)

And is crucial for complex tasks (b=2, d=8):

![Positional Encoding x4_b2_d8.png](imgs/Positional-Encoding-x4_b2_d8.png)

### Weight Initialization

Instead of using the default PyTorch weight initialization, the model uses a common custom initialization for Transformers (see [init_weight](dec_model.py)):
* Embedding (encode, decode) with xavier_uniform 
* LayerNorm: weight=1, bias=0 
* Linear (hidden): weights=xavier_uniform, bias=0

This initialization seems to help on easier tasks (b=2, d=7):

![Weight Initialization x4_b2_d7.png](imgs/Weight-Initialization-x4_b2_d7.png)

And is crucial for complex tasks (b=2, d=8):

![Weight Initialization x4_b2_d8.png](imgs/Weight-Initialization-x4_b2_d8.png)

### Tie embeddings

Seems to make no big difference, [this](https://arxiv.org/pdf/1611.01462) suggests to do this, this suggestion do not do this:

![Tie-embeddings.png](imgs%2FTie-embeddings.png)
