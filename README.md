# Which-type-of-data-fusion-is-better
### The impact of concatinating data in CNNs
Three ways of fusioning are implemented here in order to compare which one is better and helps to reach a higher accuracy.
By considering that 3 inputs are available to feed the CNNs model, it is important to find the best way of using them. The methods are:

        1-Training each model seperately and fusion their results
        2-Each input has its own CNNs, but they are concatenated in the output layer 
        3-First concating the data, then using a CNNs for training
        
#### 1-Training each model seperately and fusion their results
We used UTKinect dataset for this test and the inputs are extracted from 3 plane(x-y, y-z, x-z)



#### 2-Each input has its own CNNs, but they are concatenated in the output layer 


#### 3-First concating the data, then using a CNNs for training
        
