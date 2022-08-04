# Which-type-of-data-fusion-is-better
### The impact of concatinating data in CNNs
Three ways of fusioning are implemented here in order to compare which one is better and helps to reach a higher accuracy.
By considering that 3 inputs are available to feed the CNNs model, it is important to find the best way of using them. The methods are:

        1-Training each model seperately and fusion their results
        2-Each input has its own CNNs, but they are concatenated in the output layer 
        3-First concating the data, then using a CNNs for training
        
### 1-Training each model seperately and fusion their results
We used UTKinect dataset for this test and the inputs are extracted from 3 plane(x-y, y-z, x-z). Three CNNs by same architecure traind on them and finally their outputs mixed together. The output of the softmax, which shows the probability of each class, is used for soft voting.

The code of this model is available in model_1.py.

![soft voting](https://user-images.githubusercontent.com/106428795/182706147-75dae7e0-4257-4d87-ad4e-e027cb72f66b.jpg)


##### *The blue block, which is defined by softmax, actually is a fully connected layer with softmax activation function. For more details see the code. 



#### Accuracy: 91.5%


### 2-Each input has its own CNNs, then they are concatenated to classify by one fully connected.
The last layer receive data from 3 parallel convolutional networks. This layer is a fully connected to define the class of data.

The code of this model is available in model_2.py.

![softmax fusion](https://user-images.githubusercontent.com/106428795/182701879-fd97a617-7a7d-4e66-8248-ff00205cef73.jpg)

##### *The blue block, which is defined by softmax, actually is a fully connected layer with softmax activation function. For more details see the code. 



#### Accuracy: 93.0%

### 3-First concatenating data, then using a CNNs for training
This fusion had the best result. 3 input data first concatenate together. After that, a convolutional neural networks trained by these data. Merging data in the first step had a significant impact on accuracy.
        
The code of this model is available in model_3.py.

![input fusion](https://user-images.githubusercontent.com/106428795/182699105-b58271ca-a77b-45ae-888e-7e7f11986fa8.jpg)

##### *The blue block, which is defined by softmax, actually is a fully connected layer with softmax activation function. For more details see the code.

#### Accuracy: 94.5%

### Conclusion:
we tried 3 differnet ways of data fusion to obtain higher accuracy. Results show that if we concatenate data in earlier step, the accuracy increases more. The highest accuracy obtained when we combined all data toghether in the input level. This approve that our convolutional neural networks had more apportunity to analyze data and extract relation between them.


### Citation:
If this is useful for you, please cite our paper:

    @Article{
    title={Time-Invariant 3D Human Action Recognition with Positive and Negative Movement Memory using Convolutional Neural Networks},
    author={Farhad Khezeli, Hoda Mohammadzade},
    year={2019},
    conference={International Conference on Pattern Recognition and Image Analysis (IPRIA)},
    volume={},
    number={},
    pages={},
    doi={10.1109/PRIA.2019.8785987}
}
