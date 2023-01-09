# SSL_OOD

1. I first trained the SSL models and ImageNet pretrained model on CIFAR-10 with `utis/full_train_cifar10.py` and `train_cifar10.ipynb`.
2. Then I have fine tuned all models with the energy loss function in `utils/train_energy.py` with `train_energy.ipynb`.
3. Later I have logged the results with `utils/test.py` by runing `test.ipynb`.

The code includes necessary comments internally and hyperparameters are shared in `ft_hyperparameters.yaml`.



# Personal Notes Regarding The Project
OOD with SSL
* Training SSL methods are infeasable (Barlow Twins 7 days on 16 v100s). Hence, I must use ResNet for OOD. I need CIFAR-10 pre-trained ResNet.


### Image classification
1. Steps without OOD fine tuning:
- First see the OOD results of regular (not pretrained) ResNet without fine tuning.
- First see the OOD results of pretrained ResNet without fine tuning.
- See OOD results of SSL (for BYOL, Barlow, DINO) Resnet.

2. Steps with OOD fine tuning.
- See the results of all the same methods as in the first steps


### Object Detection
*Skipped due to insufficient time*

### Energy based model

Softmax seems to be useless for OD since even if logits are small, the largest logit can be much larger than other -> which means a high class probability. Hence, entropy overall, can be misleading with regular train settings. The logits being low though means that the input caused less activation of the network neurons, hence, the denominator of softmax being low, shows high energy, hence high probability of being OOD.


### OPTIMIZATION TAKE-AWAYS
* Cosine schedule can be used to increase or decrease other parameters troughout training as well (weight decay and momentum parameters etc.).
* Calling scheduler.step() once every epoch: This increases current iteration (step) once every epoch (The same as in the paper.). The paper updates lr every epoch: https://github.com/Lasagne/Recipes/blob/7b4516e722aab6055841d02168eba049a11fe6da/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py#L280 
* Calling scheduler.step() once every batch iteration: This increases current iteration (step) once every iteration, this is different than the paper I guess.
* The cosine annealing in this paper is wrongly implemented. The cosine annealing is appled multiplicative factor lambda rather than the learning rate itself.

### MISSING PIECES IN THE PAPER
* There should have been a validation OOD set to select the best model.
* How exactly to pick m_in and m_out from the OOD+cifar10(val) sets.

### Codebase related questions

* Why was shuffle=False by default for the OOD train dataset?


* Nesterov momentum may converge more quickly than regular momentum, especially in the presence of strong convexity. It differs in that it calculates the gradient at the "lookahead" position, rather than at the current position. This can help the optimization process to converge more quickly, especially in the presence of strong convexity.

### ABBREVATIONS

- MSP: maximum softmax prob.

