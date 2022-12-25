# SSL_OOD
OOD with SSL


 
# Image classification
- First see the OOD results of regular WideResnet without fine tuning.
- Then see OOD resutls of WideResnet with fine tuning.

- Then see the OOD results trained with SSL:
    - Apply the train wide resnet with the proposed SLL methods.
        - Then evaluate the results both with and without fine tuning.
        - Finally compare SSL vs reguler performance.

# Object Detection


# Energy based model

Softmax seems to be useless for OD since even if logits are small, the largest logit can be much larger than other -> which means a high class probability. Hence, entropy overall, can be misleading with regular train settings. The logits being low though means that the input caused less activation of the network neurons, hence, the denominator of softmax being low, shows high energy, hence high probability of being OOD.

# ABBREVATIONS

- MSP: maximum softmax prob.
