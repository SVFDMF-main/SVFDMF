SVFDMF(SVMF)[A Cross-modal Fusion Method for Short Video Fake News Detection via MambaFormer]

### Environment
Please refer to the file requirements.txt.

### Dataset
FakeSV:[FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms, AAAI 2023.]

FakeTT:[FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process, ACM MM 2024.]

### Data Processing
To facilitate reproduction, we provide papers on preprocessing methods that you can use to process the features. Please place these features in the specified location, which can be customized in dataloader.py.
Bert:Y. Cui, W. Che, T. Liu, B. Qin, and Z. Yang, “Pre-training with wholeword masking for chinese bert,” IEEE/ACM Transactions on Audio,Speech, and Language Processing, vol. 29, pp. 3504–3514, 2021.
MAE：K. He, X. Chen, S. Xie, Y. Li, P. Doll´ar, and R. Girshick, “Masked autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,2022, pp. 16000–16022.
HuBert：W. N. Hsu, B. Bolte, Y. H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451–3460, 2021.
