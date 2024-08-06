# Don't Think It Twice: Exploit Shift Invariance for Efficient Online Streaming Inference of CNNs
Reproduction repository for the "Don't Think It Twice: Exploit Shift Invariance for Efficient Online Streaming Inference of CNNs"

<img src="./figures/repo_banner.svg" width="1920">

# Abstract 

Deep learning time-series processing often relies on convolutional neural networks with overlapping windows. This overlap allows the network to produce an output faster than the window length. However, it introduces additional computations. This work explores the potential to optimize computational efficiency during inference by exploiting convolution's shift-invariance properties to skip the calculation of layer activations between successive overlapping windows. Although convolutions are shift-invariant, zero-padding and pooling operations, widely used in such networks, are not efficient and complicate efficient streaming inference. We introduce StreamiNNC, a strategy to deploy Convolutional Neural Networks for online streaming inference. We explore the adverse effects of zero padding and pooling on the accuracy of streaming inference, deriving theoretical error upper bounds for pooling during streaming. We address these limitations by proposing signal padding and pooling alignment and provide guidelines for designing and deploying models for StreamiNNC. We validate our method in simulated data and on three real-world biomedical signal processing applications. StreamiNNC achieves a low deviation between streaming output and normal inference for all three networks (2.03 - 3.55\% NRMSE). This work demonstrates that it is possible to linearly speed up the inference of streaming CNNs processing overlapping windows, negating the additional computation typically incurred by overlapping windows.

# Run Experiments

### Pooling Error Bounds

[```pooling_error_bounds.py```](./padding_and_pooling/pooling_error_bounds.py)

### Zero Padding Effect

[```zero_padding_effects.py```](./padding_and_pooling/zero_padding_effects.py)

### Streaming Inference

Experiments for Photoplethysmography-based Heart Rate extraction are in [```hr_ppg```](./hr_ppg/README.md).

Experiments for electroencephalography-based seizure detection are in [```seizure_eeg```](./seizure_eeg/README.md).

Experiments for acceleration-based seizure detection are in [```seizure_acc```](./seizure_acc/README.md).

# Reference
```
TODO
```