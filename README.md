# Max entropy project


This repo is the result of a research idea I've tried in my master degree:

I was inspired by Francis Crick theory of [Sleep](https://www.nature.com/articles/304111a0) that sleep and REM sleep's function is for doing "Reverse learning" that is to remove parasitic connection that happened during wakeful learning.

My main insight for this project was to connect this idea with Jaynes's maximum entropy principle.

 The idea is to have a wake-sleep algorithm where during the wake phase the neural netowork is learning in the normal setting backpropagating the gradient of the loss of interest with real data, and during the sleep phase give noise in the input and backpropagating the weights with respect to the softmax entropy.

 The insight was none of the softmax category should be highly active or inactive when the model see noise in input so bu it would naturally kill parasitic connection and regularize the network.


 Unfortunately I wasn't able to make this idea work but I might come back to it.
