# Max entropy project


This repo is the result of a research idea I've tried in my master degree:

I was inspired by Francis Crick theory of [Sleep](https://www.nature.com/articles/304111a0) that sleep and REM sleep's function is for doing [Reverse learning](https://en.wikipedia.org/wiki/Reverse_learning) that is to remove parasitic connection that happened during wakeful learning.

My main insight for this project was to connect this idea with Jaynes's maximum entropy principle which states that the probability distribution which best represents the current state of knowledge about a system is the one with largest entropy

 The idea is to do entropy regularization via a wake-sleep algorithm where during the wake phase the neural network is learning in the normal setting backpropagating the gradient of the loss of interest with real data, and during the sleep phase give noise in the input and backpropagating the weights with respect to the softmax entropy.

 Since according to Jaynnes maximum entropy principle and how we would like the neural network to behave none of the categories should be highly active or inactive when the model get noise in input, doing so will naturally kill parasitic connections and regularize the network to a better set of weights without any additional data points.




 Unfortunately I wasn't able to make this idea work but I might come back to it.


**Idea:**
                     
![animation1](figures/mnist_digits.gif)  ![animation2](figures/noise2.gif) 
*Left: Wake phase Right: Sleep phase*



 
