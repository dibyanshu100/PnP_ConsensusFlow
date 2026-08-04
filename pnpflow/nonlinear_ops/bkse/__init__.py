# Vendored, unmodified subset of VinAIResearch/blur-kernel-space-exploring
# (Tran et al., "Explore Image Deblurring via Encoded Blur Kernel Space",
# CVPR 2021), BSD-3-Clause licensed, https://github.com/VinAIResearch/blur-kernel-space-exploring
#
# Only the modules needed for KernelWizard.adaptKernel() inference (the
# non-blind forward "reblurring" path used as DPS's nonlinear-deblurring
# forward operator) are included -- training, blind-deconvolution, and
# StyleGAN-prior code from the original repo are not vendored.
