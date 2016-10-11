---
layout: page
title: Project summary - Matt Graham
---

My project is focused on developing *Markov Chain Monte Carlo* (MCMC) methods. These are a group of techniques for generating random samples from complex probabilistic models. They work by simulating a stochastic (noisy) dynamical system where the probability distribution over the state of the system converges to the distribution of interest.

The samples of the system state can then be used to estimate expectations of the model. For instance given a set of data and a proposed parametric model, we might want to sample parameters of the model which explain the data and any prior beliefs well. Alternatively given a model of how some physical system works, we might want to sample predictions of how the system will behave in novel situations. In both cases as well as wanting to generate plausible explanations from a model it is often also useful to have some idea our degree of uncertainty in those explanations and to be able to weigh up the evidence for competing explanations. Bayesian inference using MCMC provides a principled computational framework for doing this.

I am particularly interested in applying MCMC methods to models in which we cannot directly evaluate the probability distribution over the variables we want to sample. This is often due to the presence of an intractable integral, i.e. one which we cannot come up with a closed form solution for or evaluate with numerical integration methods in a reasonable amount of time, within the probability distribution of interest. This might be because the model includes unobserved (latent) variables that we need to integrate out or because the probability distribution involves an intractable normalising term which depends on the variables we are interested in.

In some such problems although we cannot evaluate the probability distribution of interest, we can compute a noisy estimate of it. A particular group of MCMC methods called *pseudo-marginal* MCMC have been proposed for performing inference in these settings. One common problem with these methods is that the MCMC dynamic can often tend to 'get stuck' at points in the state space over a long series of updates as illustrated in the image below. One of the methods I have worked on (in collaboration with [Iain Murray](http://homepages.inf.ed.ac.uk/imurray2/)) was designed to help try to overcome this problem [[1]](#references).

<div class='figure'>
<img class='inline' src='pm-trace-examples.svg' width='90%' />
  <p class='caption'>
    Example traces of posterior samples of length-scale hyperparameter in a Gaussian process classification model. Position on vertical axis indicates the sample parameter value and horizontal axis shows evolution over successive MCMC updates. The left trace was produced using a standard pseudo-marginal MCMC approach and shows characteristic 'sticking' where the parameter values remains constant over a long series of updates due to the the dynamic reaching a state from which most of the proposed move lead to a rejection. The right trace shows a run using pseudo-marginal slice sampling which adapts the size of updates so as to avoid the rejections causing the sticking artifacts.
  </p>
</div>

Another common case where we do not have direct access to the probability distribution we are interested in is in *simulator* models. Here we can usually easily generate samples from the model but often cannot calculate the probability of producing a particular set of outputs. The model could be a scientific simulation, for example a population model in computational biology, a model of how 3D scenes are rendered to 2D projections or a generative model trained using machine learning methods such as variational autoencoder or generative-adversial network. Simulated samples from toy examples of three such models are shown in the image below.

<div class='figure'>
<img class='inline' src='lotka-volterra-samples.svg' width='30%' />
<img class='inline' src='pose-samples.svg' width='30%' />
<img class='inline' src='mnist-samples.svg' width='30%' />
  <p class='caption'>
    Example sampled outputs from simulator models. Left sampled predator-prey population traces over time for variant of the <a href='https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations'>Lotka-Volterra model</a>. Centre simulated 2D projections of human-poses from model learnt from <a href='http://poseprior.is.tue.mpg.de/'>motion capture data</a>. Right generated handwritten digit images from model trained as a variational autoencoder on <a href='http://yann.lecun.com/exdb/mnist/'>MNIST dataset</a>.
  </p>
</div>

Simulator models can be thought of as a fixed function which takes in random inputs (from a random number generator) and produce simulated outputs. In many cases such as those shown in the image above, the fixed function will also be *differentiable* with respect to its inputs - if we change the inputs by a small amount, the outputs will also change by a small amount. We have proposed a method for performing asymptotically exact inference in this setting [[2]](#references), exploiting gradient information to both coherently explore the sample state space and to allow producing samples exactly consistent with observations.

In this work the use of computational graph frameworks such as [Theano](http://deeplearning.net/software/theano/) and [TensorFlow](https://www.tensorflow.org) has been key in both allowing the necessary derivatives of the simulator models to be automatically computed, enabling the methods to be easily applied to a range of models, and also making the methods developed easily scalable to high-dimensional problems due to the optimised code generated for both the forward simulation of outputs of the model and back-propagation of derivative information.

Further the device-agnostic nature of models specified in these frameworks, and natural parallelism arising from expressing models as operations on multidimensional arrays, allows computation in larger models to easily deployed to parallel architectures such as general purpose GPU devices or across multiple CPUs. Although these frameworks have often been developed with the training of deep network models in mind, they also offer natural way for bringing the benefits of automatic differentiation and use of large scale parallelism on GPU devices to statistical inference tasks. This allows the use of Bayesian methodology with increasingly complex and high-dimensional models and data sets.

### References

  1. Pseudo-Marginal Slice Sampling, I. Murray & M.M. Graham,
     *Proceedings of the 19th International Conference on Artificial
     Intelligence and Statistics (AISTATS)*, 2016 [[PDF]](http://www.jmlr.org/proceedings/papers/v51/murray16.pdf)
  2. Asymptotically exact likelihood-free inference,
     M.M. Graham & A. Storkey, *in submission*, 2016,
     [[pre-print available on arXiv]](https://arxiv.org/abs/1605.07826)
