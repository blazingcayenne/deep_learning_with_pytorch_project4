# deep_learning_with_pytorch_project4
Kaggle Competition: Semantic Segmentation on the Aeroscapes Dataset

This project uses 

- pretrained segmentation models and decoders from the [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch) library,

- optional geometric and pixel-level data augmentation transforms from the [Albumentations](https://albumentations.ai/) library,

- optional copy-paste data augmentation,

- optional overlapping tiling of multi-scaled training images,

- weighted soft-dice loss function adapted from the [Pytorch Toolbelt](https://github.com/BloodAxe/pytorch-toolbelt) library,

- ranger optimizer from lessw2020's [GitHub](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) repository,

- a constant learning rate until the trainer loses patience followed by a cosine annealing, and

- ensembles that combine pixel-level probabilities.

**Notes:** The copy-paste data augmentation loosely follows the technique presented in the [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/pdf/2012.07177v1.pdf)  paper. PyTorch Toolbelt's soft dice loss function was modified to allow for one-hot encoded masks and per-class weighting was added.The Ranger optimizer is a combination of RAdam and LookAhead as explained in [New Deep Learning Optimizer, Ranger: Synergistic combination of RAdam + LookAhead for the best of both](https://lessw.medium.com/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d). Despite trying to eliminate training variability to make experiments deterministic and repeatable, small variations occur between runs. Since I could not locate the source of this randomness and decided to live with it, I enabled CUDNN benchmarking.

The choice of optimizer and LR scheduler was informed by [How we beat the FastAI leaderboard score by +19.77%: a synergy of new deep learning techniques for your consideration](https://lessw.medium.com/how-we-beat-the-fastai-leaderboard-score-by-19-77-a-cbb2338fab5c).

> **Optimizer:** Adam, introduced in 2015, has been the default optimizer for deep learning for many years. Ranger is a combination of two recent innovations that build on top of Adam — Rectified Adam, and Lookahead.
> Rectified Adam (RAdam) was the result of Microsoft research looking into why all adaptive optimizers need a warm-up or else they often shoot into bad optima at the start of training. Most everyone was aware a warm-up was needed to get better results with Adam, but researchers Liu, Jian, He et al made the effort to understand why. The answer was because Adam (and all adaptive optimizers) are making premature jumps when the variance of the adaptive learning rate is too high. Adam needs to wait to see more data to really start making larger decisions. RAdam achieves this automatically by adding in a rectifier that dynamically tamps down the adaptive learning rate until the variance stabilizes. Thus, your training gets off to a solid start intrinsically, with no warmup needed!
> LookAhead was developed in part by Geoffrey Hinton and keeps a separate set of weights from the optimizer. Then every k steps (5 or 6), it interpolates between it’s weights and the optimizer’s faster weights, and then splits the difference and updates the faster optimizer. The result is like having a buddy system to explore the loss terrain. The faster optimizer, RAdam for Ranger, explores the terrain as usual. LookAhead stays a bit behind and lets it scout things out, while making sure it can be pulled back if it’s a bad optima. Thus, safer / better exploration and faster convergence.

> **LR Scheduler:** FastAI has traditionally used a ramp up, ramp down curve for the learning rates during training. The idea is likely based on the need for a warmup (see above), but also the ramp up and ramp down are designed to help the optimizer ‘jump’ over smaller local minima. It’s been quite successful with Adam. However, it became apparent that the newer optimizers just did not do well that approach. GRankin thus developed a ‘flat + cosine anneal’ training curve, and that immediately jumped the results we saw with Ranger. In theory, the improvements from this new training curve are that by stabilizing the learning rate, we are letting RAdam properly get training off to a solid start, and then letting LookAhead explore in a steadier manner. Constantly changing the learning rates seems to negatively impact the exploration process.

The first experiment, `ExpAAA`, used a one cycle LR scheduler. Subsequent experiments switched to a ‘flat + cosine anneal’ training curve where training occurs over primary and annealing stages. The primary stage uses a constant learning rate of `SOLVER.MAX_LR`. After each epoch, the trainer creates a _last_ checkpoint and a _best_ checkpoint if the validation loss reached a new minimum. Primary training stops after `SOLVER.NUM_EPOCHS` epochs or `SOLVER.PATIENCE` epochs where the validation loss does not decrease. Prior to the annealing stage, the trainer reverts to the _best_ checkpoint. It then uses a cosine annealing scheduler to drop the LR from `SOLVER.MAX_LR` to `SOLVER.MIN_LR` over `SOLVER.ANN_EPOCHS`. The choice of `SOLVER.MAX_LR` is informed by a _LR sweep test_, which plots the validation loss as a function of LR. The optimal LR is where the model learns the fastest, i.e., the reduction in validation loss is the greatest.

This project ran experiments to test the impact of the following techniques on the validation dataset's mean Dice score. Unless noted, these experiments used a FPN model with EfficientNet-B3 encoder.

- data augmentation (positive impact)
- overlapping tiles on multi-scale training images (positive impact for ½-size tiles, negative impact for ⅓-size tiles)
- a DeepLab3+ model w/ EfficientNet and ResNet encoders (negligible impact at expense of significant increase in training time)
- a Linknet model w/ SK-ResNe(X)t encoder (negative impact)
- a TIMM-ResNest50d encoder (negligible impact at expense of significant increase in training time)
- ensembles over systematic slices of public dataset (positive impact)
- further train previous experiment w/ lower LR and full-size images or ½-size tiles (minor impact)
- weighted soft-dice loss function (minor positive to negative impact depending upon weights)
- copy-paste data augmentation (neglibly to negative impact)
- adaptive loss function weights and copy-paste class weights (negibly to negative impact)

## Lessons Learned

I made the following observations.

1. Surprisingly, the FPN model with an EfficientB3 encoder performed better than DeepLabV3+ models with EfficientB3, ResNet50, and ResNet152 encoders.

2. Another surprise was the FPN model with an EfficientB3 encoder, unlike other models. did not fully converge with larger batch sizes.

3. It was interesting to observe at what point during training the model began to properly classify specific classes.

4. Weighting the soft dice loss function was finicky. It may be possible to improve the mean dice score by tweaking the per-class weights. However, there is a greater chance that weighting will decrease the mean dice score.
   
5. I had high expectations for the copy-paste data augmentation. Unfortunately, the random and weighted variants did improve the mean dice score even with more complicated models.

## Remaining Question

I was not able to find papers that addressed the handling of padding.

* Should it be its own class? 
* Should it count as background? 
* Should it be ignored from the loss function?