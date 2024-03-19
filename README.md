# WACEfNet
WACEfNet: Widened Attention-Enhanced Atrous Convolutional Network for Efficient Embedded Vision Applications
WACEfNet is an innovative deep learning architecture designed specifically for efficient aerial image analysis on resource-constrained embedded platforms like UAVs. This repository contains the source code, pre-trained models, and documentation for WACEfNet.
# Key Features
Widened MBConv Blocks: WACEfNet introduces an expansion factor to scale the width of mobile inverted bottleneck (MBConv) blocks, enhancing representational power without excessive overhead.
Atrous Convolutions: Standard depthwise convolutions are replaced with atrous convolutions to enlarge the receptive field and capture richer contextual details without increasing computational complexity.
Attention Modules: Channel and spatial attention mechanisms are integrated to refine intermediate features and improve the network's ability to discern subtle patterns.
State-of-the-Art Performance: WACEfNet achieves superior accuracy on aerial image datasets like AIDER, CDD, and LDD while requiring significantly fewer parameters and FLOPs compared to existing lightweight architectures.
