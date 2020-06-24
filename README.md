# MAML: Model-Agnostic-Meta-learning
This repository was made for paper review and replacing existing code with PyTorch code.

[Paper](https://arxiv.org/pdf/1703.03400.pdf)\
[Tensorflow code](https://github.com/cbfinn/maml)

### What is meta learning?
Deep learning has a great success in mastering one task using a large dataset. But, what we really want to achieve is few shot-meta learning which is an algorithm that trains a neural network to learn many different tasks using only a small data per task.
Meta-learning, also known as “learning to learn”, intends to design models that can learn new skills or adapt to new environments rapidly with a few training examples.
There are three common approaches: 
1) learn an efficient distance metric (metric-based); 
2) use (recurrent) network with external or internal memory (model-based); 
3) optimize the model parameters explicitly for fast learning (optimization-based).

### Model-Agnostic Meta-Learning (MAML)
In meta-learning, there is a meta-learner and a learner. The meta-learner (or the agent) trains the learner (or the model) on a training set that contains a large number of different tasks. In this stage of meta-learning, the model will acquire a prior experience from training and will learn the common features representations of all the tasks. Then, whenever, there is a new task to learn, the model with its prior experience will be fine-tuned using the small amount of the new training data brought by that task. But we don’t want to start from a random initialization of its parameters because if we do so, it will not converge to a good performance after only a few updates on each task.

Model-Agnostic Meta-Learning (MAML) provides a good initialization of a model’s parameters to achieve an optimal fast learning on a new task with only a small number of gradient steps while avoiding overfitting that may happen when using a small dataset.\
\
<img src='http://bair.berkeley.edu/static/blog/maml/maml.png' width='400'>
\
In the diagram above, θ is the model’s parameters and the bold black line is the meta-learning phase. When we have, for example, 3 different new tasks 1, 2 and 3, a gradient step is taken for each task (the gray lines). We can see that the parameters θ are close to all the 3 optimal parameters of task 1, 2, and 3 which makes θ the best parameters initialization that can quickly adapt to different new tasks. As a result, only a small change in the parameters θ will lead to an optimal minimization of the loss function of any task.

### MAML‘s Algorithm
There is no better way to understand MAMl than its algorithm:\
<img src='https://miro.medium.com/max/1400/1*_pgbRGIlmCRsYNBHl71mUA.png' width='700'>\


In meta-training, all the tasks are treated as training examples p(T). So, we start with randomly choosing the parameters θ, and we enter the first loop (while) that takes a batch of tasks from p(T). And for each task from that batch, we train the model f using K examples of that task (k-shot learning). Then, we get the feedback of its loss function, and test it on new example test set to improve the model’s parameters. If we use one gradient descent update, then the adapted parameters for that task are:\
<img src='https://miro.medium.com/max/1400/1*ZD8Al-jdAVPt409oOt6aew.png' width='800'>\
\
The step size or learning rate α is a hyperparameter.\
\
The test error on the batch of tasks is the training error of the meta-learning process. And here is the meta-objective:\
<img src='https://miro.medium.com/max/1400/1*zmJnsGs8jD7xsF-IA_56-g.png' width='800'>\
Before, we move to the next batch of tasks, we update the parameters of the model θ using Stochastic Gradient Descent SGD because we have batches here. The parameters of the model θ are updated as follows:\
<img src='https://miro.medium.com/max/1400/1*F6zUewG7rQ23ZgaS5F2kNQ.png' width='800'>\
As we can see, the meta-gradient update contains a gradient through a gradient which can be computed using the Hessian-vector product.\
\
Then, we repeat the same process until we train all the batches.
