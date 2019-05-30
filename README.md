# Simple Genetic Programming 
### For Symbolic Regression
This Python 3 code is a simple implementation of genetic programming for symbolic regression, and has been developed for educational purposes.

## Dependencies
Numpy. The file `test.py` shows an example of usage, and requires Scikit-learn.

## Installation
You can install it with pip using `python3 -m pip install --user simplegp`, or locally by downloading the code and running `python3 setup.py install --user`.

## Random 
- [ ] Create RelU node

## Experiments
Measure
- Number of generations
- Time taken
- Maximum Fitness
- Some measure of diversity

#### Optimise general params first for Symbolic Regression
- [ ] Different initial population sizes
- [ ] Different mutation probabilities
- [ ] Tournament selection size


#### Optimise the gradient descent local search
- [ ] Uniformly randomly select individuals to do gradient descent
- [ ] Top k% of nodes
- [ ] When to apply graident descent (every x generations)
- [ ] Different graident descent learning rates
- [ ] Combinations of above
- [ ] Iterations of gradient descent, 2 levels, lower for everyone, and then higher for top k 


#### Reoptimise solution with optimal graident descent params



## Questions For Marco
- [ ] What sort of parameters should we test for in the general GP (not including gradient descent)? Do we even need to optimise these?

