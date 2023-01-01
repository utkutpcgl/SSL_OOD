# SSL_OOD
OOD with SSL
* Training SSL methods are infeasable (Barlow Twins 7 days on 16 v100s). Hence, I must use ResNet for OOD. I need CIFAR-10 pre-trained ResNet.


# Image classification
1. Steps without OOD fine tuning:
- First see the OOD results of regular (not pretrained) ResNet without fine tuning.
- First see the OOD results of pretrained ResNet without fine tuning.
- See OOD results of SSL (for BYOL, Barlow, DINO) Resnet.

2. Steps with OOD fine tuning.
- See the results of all the same methods as in the first steps



# Object Detection
*No sufficient time*

# Energy based model

Softmax seems to be useless for OD since even if logits are small, the largest logit can be much larger than other -> which means a high class probability. Hence, entropy overall, can be misleading with regular train settings. The logits being low though means that the input caused less activation of the network neurons, hence, the denominator of softmax being low, shows high energy, hence high probability of being OOD.

# ABBREVATIONS

- MSP: maximum softmax prob.
