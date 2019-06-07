# Experiments

## Things to investigate

- [ ] Tune Learning Rate, Every Generation, Number of Individuals, Number of Iterations
- [ ] Explore decaying learning rate or ADAM after the above
- [ ] Examine overfitting
- [ ] Examine trade-off between model complexity and fit (bias-variance)
- [ ] Plots for training vs test over a number of iterations and generations, etc

## Comparison Gotchas
- [ ] Ensure that our comparisons are fair to Baseline SimpleGP (number of generations vs number of evaluations) (time is fixed) Careful here, maybe should be number of evals so that the experiments are not machine dependent
- [ ] Cross-Validation needs to be done
- [ ] Compare to actual functions found
- [ ] Compare to linear regression model
- [ ] Examine the types of nodes found by both approaches
- [ ] Examine models returned by both in terms of bias-variance


## Hypotheses
- [ ] Smaller/simplier trees generalise better (higher bias, lower variance)
- [ ] Applying backpropagation at the beginning makes more impact because otherwise good structures are lost. But why does this not work all of the time. Unnecessary local search wastes time. Figure this out by looking at plots over number of generations for different runs
- [ ] Learning rate decay will help when we use high number of iterations, especially for finding good structures earlier as defined above.
- [ ] Applying backprop every generation is optimal vs every number of generation, if we have no time constraints and depending on termination criteria
- [ ] Better to apply to all individuals but is a trade-off between time and optimality
- [ ] Low learning Rate vs High Learing Rate
- [ ] Expect more iterations to help but is a trade-off between time and optimality