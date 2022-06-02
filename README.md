# Machine Learning Design patterns

## Pipeline

A pipeline is about processing some data sequentially using an arbitrary number of functions. It's useful for data preprocessing or within the context of an inference framework.

For example you may want to do `preprocess -> inference -> postprocess`

```python

from typing import Union

def preprocess(input : Union[str, Image, Video, Audio]) -> Tensor:
    # implementation

def inference(input : Tensor) -> Tensor:
    # implementation

def postprocess(input : Tensor) -> Union[str, Image, Video, Audio]:
    # implementation
```

And then you'd run your pipeline by saying

```python
pipeline = [preprocess, inference, postprocess]

input = ...
for step in pipeline:
    input = step(input)

return input
```

An import detail is that the input and output types of function in a pipeline need to match.

This pattern isn't only limited to an inferencing framework but a framework like Keras explictly has a concept of a layer so if you were to implement it from scratch a grossly simplified version would be something like.

```python
class KerasModel():
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
```

An exercise to the reader is to make the above work with a batch of examples.

## Workflow
A workflow is a more complex version of a pipeline that allows for sequential behaviors. But the more general pattern is a Directed Acyclic Graph (DAG). This is what DAG providers like Airflow, metaflow and the ensemble support in torchserve do.


```python
# Dag example
graph = {
    'input': ['a'],
    'a': ['b', 'e'],
    'b': ['c', 'd'],
    'd': ['e']}
```

In the example we above we use a Python dictionary where the keys on the right hand side are the nodes where arrows are pointing out of and the values on the left hand side have arrows pointing into them. If you don't like python dictionaries you can also create a DAG using YAML or python decorators.

 Now imagine if every node was some Python function or even a Pytorch model how would you go about executing this DAG?

```python
class WorkflowEngine():
    def __init__(self, dag):
        self.dag = dag
    
    def execute():
        for key, value in dag.items():
            Step(key, value, False)

@dataclass
class Step():
    def __init(self, inputs, outputs, dependencies_met):
        self.inputs = inputs
        self.outputs = outputs
        self.dependencies_met = False
        self.resources = {"cpu" : 2, "gpu" : 1}
    
    def execute():
        if self.dependencies_met:
            # Execute steps
```

A real world orchestrator would need to take care of dependency management, scheduling and resource allocation.

## Function as data
Function as data is something LISP programmers talk a lot about. The main idea is you could have a function like 

```lisp
;; Add 1 and 2
(+ 1 2)
```

But if you add a quote at the beginning of it then it becomes a string

```lisp
;; The string (+ 1 2)
'(+ 1 2)
```

This is powerful because now you could have a seperate program analyze the string `(+ 1 2)` realize that the inputs never change, the function is pure so the outputs never change so this function can be replaced by `3`

PyTorch also has a similar idea but first let's define a very simple toy model.


```python
class myModel(torch.nn.Model):
    def __init__(self):
        self.linear = torch.nn.Linear(100)
    
    def forward(self, input):
        output = self.linear(input)
        return output
```

Run an inference with `myModel(torch.randn(100))` so it's a function! But also if you were to run `myModel.data` you would get the weights of the model so it's also data. So a `function = data`.

This is also made clearer if you've ever pickled model which is essentially a method to serialize some python objects as strings on disk so again `function = data`

```python
model = myModel()
pickle.dumps(model)
```

## Iterator design pattern

```python
for i in range(10):
    print(i)

```

But a more useful operation would be something like

```python
for batch in dataset:
    model(batch)
```

So how do you make something like `for _ in _` available for your classes. We do this by implementing the `__iter__()` and `__next()__` functions


```python
from typing import List

class Dataset:
    def __init__(self, data : List[str]):
        self.data = data
        self.elements = 0
        
    def __iter__(self):
        return data[0]
    
    def __next__(self, batch : int = 0):
        
        # Return a batch of examples
        if batch > 0:
            # TODO: Fix typo here, this will only return a single or 0 elements
            self.elements = self.elements + batch
            return self.data[self.elements : self.elements + batch]
        
        # Return a single example
        else:
            self.elements = self.elements + 1
            return self.data[self.elements]
```

## Job queues

Let's say we have a service that needs to pick one of `n` PyTorch models to run on some input

```python
from dataclass import dataclass

@dataclass
class Job:
    model : str
    input : Union[str, Image, Audio, Video]
    endpoint : Tuple[str, int] # url : port

class JobProcessor():
    def __init__(self):
        self.jobs : List[Job] = []
    
    def process_job(self):
        job = jobs.pop()
        execute(job)
    
    def execute(self, job):
        output = job.model(job.input)
        expose(output, endpoint)
    
    def expose(self, output, endpoint)L
        # Use FastAPI or something else
```

With only a couple of lines of code we've designed a multi model inferencing framework. Let's say you're not using Python to design this job manager you can also still just spawn a Python process, run the inference and then write it either to disk or stdout and pick it back up from the other language.


## Callbacks
Many trainer loops will implement callbacks where you can trigger some behavior if some condition is fulfilled for example

```python
on_training_ends -> do_something
on_epoch_end -> do_something

def do_something():
    save_logs_to_tensorboard()
    change_learning_rate()
```

A callback is a particular case of something called the Observer pattern so let's implement that. Code paraphrased from https://refactoring.guru/design-patterns/observer/python/example#lang-features

So an observer needs to subscribe to some subject that changes its behavior

```python
class ModelSubject():
    def __init__(self):
        state : Trainer = None # A trainer includes a model, which epoch its on, loss, model weights...
        observers : List[ModelObserver] = None

    def attach(self, observer : ModelObserver):
        observers.append(observer)


    def detach(self, observer : ModelObserver):
        observers.remove(observer)

    def notify(self):
        for observer in observers:
            observer.update(state)
```

The observer is notified of all state changes of the subject and then needs to do something when that happens

At a high level an Observer is an abstract class that implements a function called update

```python
from abc import abstractmethod, ABC

class Observer(ABC):

    @abstractmethod
    def update(self):
        """
        Implement your own observer here
        """
        pass
```

We can then build specific kinds of observers by by implementing the `update()` function. In the example below we build an observer to adjust the learning rate of a model when the loss increases

```python
class ChangeLearningRateObserver(Observer):
    def __init__(self):
        self.state : [TrainerState] = None
    
    def update(self, new_state):
        if self.state = None:
            pass
        
        else:
            # Do not use this in production code this is educational only
            if new_state.loss > state.loss:
                state.lr = state.lr * 0.1
        self.state = new_state

```

But this is a powerful framework and we can also implement something like logging without changing the library code.

```python
class LogObserver(Observer):
    def __init__(self, log_dir='/logs/'):
        self.state : [TrainerState] = None
        self.log_dir : str = log_dir

    def update(self, new_state : Dict): # Asssume new state is a dictionary
        with open(filename, "w") as f:
            for key, value in new_state.items():
                f.write(f"{key}:{value}")
        self.state = new_state
```


So the benefit of this approach you can extend functionality of a library without changing the core code which may require you to get a PR merged in by the core team that may make the core code unmaintable by adding all sorts of usecases that people care about. So the observer pattern is primarily a way to extend code which is why it's very popular in training frameworks like fast.ai or PyTorch LIghtning.

## Learner pattern

Learner pattern was popularized by frameworks like Sci-kit learn that started approach to modeling that was as simple as 

`model.fit(data)`

But implementing code for this at least within the context of neural networks is something you already do if you've used vanillay PyTorch without a training framework.

```python

# data[0][0] means the first input example
# data[1][5] means the label for the 5th input example
data = [[inputs], [labels]]

class Model:
    def __init__(self):
        self.model = nn_model()
        self.loss_function = substract/square_loss/l1 etc..
    
    def fit(self, data):
        # 1. Compute forward function
        output = self.model(data) 

        # 2. Get loss
        loss = loss_function(data)

        # 3. Update model
        model.update(loss)
    
    def update(self, loss):
        # 1. Compute gradients with autograd

        self.model.weights = ...     
```

## Batch processing

So suppose you'd like to run `model.forward()` on two different inputs. The naive way of doing this is running

```python
model.forward(input_1)
model.forward(input_2)
```

But this becomes painfully slow if you start dealing with a large number of examples

```python

# model.forward is called O(inputs)
for input in inputs:
    model.forward(input)
```

Generally in numerical code you should fear `for loops` like the plague and as much as possible try to replace them with batch operations.

So instead rewrite your code as 

```python

tensor = torch.Tensor
for input in inputs:
    tensor.stack(input)

# model.forward is called once
model.forward(tensor)
```

Remember GPUs aren't that great at doing many small operations because there's an overhead to sending data to it so as much as possible it's better to batch jobs into large ones to take advantage of speedups. (Technically this can be worked around with CUDA graphs but that's still a relatively new feature)

As another exercise vectorization on CPU is also another technique to eliminate for loops but by operating over chunks of data concurrently. So for example some new newer Intel CPUs will turn matrices into long vectors and do matrix math on them by using a large instruction width AVX512.

## Decorator
Decorators are a technique to add functionality to a function or class without modifying its code. You may have already heard of or used decorators like `@memoize, @lru_cache, @profile, @step`

As an example let's take a look at how to implement a `@profile` decorator borrowing code from https://medium.com/uncountable-engineering/pythons-line-profiler-32df2b07b290

```python
from line_profiler import LineProfiler

profiler = LineProfiler()

# A decorator is just a python function that takes in a function
def profile(func)
    # Inner function takes in unnamed and named arguments
    def inner(*args, **kwargs)
        # New code decorator adds
        profiler.add_function(func)
        profiler.enable_by_count()

        # Running the decorated function
        return func(*args, **kwargs)
    return inner
```

So now you can just run

```python
@profile
def my_slow_func():
    # some terrible code here
```

In the above decorator we ran some commands before returning `func` but we could also change `func`, its arguments or do whatever we please this is another one of those patterns like callbacks that let you extend some code without modifying it.

One of the most interesting decorators is the FastAPI one https://github.com/tiangolo/fastapi

```python
@app.get("/")
def read_root():
    return {"Hello": "World"}
```

The above application redirects calls to `/` to the `read_root()` function so digging into the code a bit you'll find a function called `get()` in `fastapi/application.py` https://github.com/tiangolo/fastapi/blob/master/fastapi/applications.py#L425

It's a complicated function but what we care about is

```python
def get(...) -> Callable[DecoratedCallable]:
    return self.router.get(...)
```

Digging through the code a bit more we find that `add_api_route()` whenever a new `@app.get()` is called where see `func` being returned in much the same way as it is in the plain profiling decorator https://github.com/tiangolo/fastapi/blob/87e29ec2c54ce3651939cc4d10e05d07a2f8b9ce/fastapi/applications.py#L378

The flipside of decorators is that they can lead you to a monolithic architecture where your infrastructure and deployment is tightly coupled to your implementation, this is generally fine if you're a startup but not so fine if multiple people are contributing code to the same place.

## Strategy Pattern

The strategy pattern is classic Object Oriented programming and is generally useful when you to set some particular strategy for an object without constraining it too much as a library designer.

For example suppose you're creating a new Trainer class and don't have time to implement all optimizers that people care about. So you start with adding support for an SGDOptimizer
```python
class Trainer:
    def __init__(self):
        optimizer : Optimizer = SGDOptimizer
        ...

# Create an abstract optimizer class
class Optimizer(ABC):
    @abstractmethod
    # We don't want to constrain the input types for such a function
    # Return type is a tensor because value in a tensor needs to be changed by a bit
    def step(*args, **kwargs) -> Tensor:
        pass 

class SGDOptimizer(Optimizer):
    def step(self, learn_rate : float, n_iter : int, tolerance : float):
        # Your SGD implementation here
```

So now someone else that doesn't understand how your whole trainer codebase works could create a new optimizer by just making sure to inherit from `Optimizer`

```python
class AdamOptimizer(Optimizer):
    def step(self, beta_1 : float, beta_2 : float, epsilon : float):
        # Out of core Adam implementation here
```


## TODO
* Autograd - https://marksaroufim.medium.com/automatic-differentiation-step-by-step-24240f97a6e6 (Maybe I need to update this tutorial with some python code)
* Matrix Multiplication
    * http://supertech.csail.mit.edu/papers/Prokop99.pdf
    * https://github.com/mitmath/18335/blob/spring21/notes/oblivious-matmul.pdf
* Distributed patterns: good tutorial here https://huggingface.co/docs/transformers/parallelism
* Strategy pattern

