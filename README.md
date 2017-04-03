# pt-styletransfer
Neural style transfer as a class in PyTorch

Based on:
https://github.com/alexis-jacq/Pytorch-Tutorials

Adds:
StyleTransferNet as a class that can be imported by other scripts
VGG support (this was before pretrained VGG models were available in PyTorch)
Convenience functions for saving intermediate style & content targets for display
Convenience function for examining gram matricies as images
Auto style, content, and product image saving
matplotlib plots of losses over time & record of hyperparameters to keep track of favorable results
