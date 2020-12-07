# mushrooms-analysis
Predicting mushrooms edibility using Machine Learning

# Objective
Use classification tree to train mushroom features and predict if a given mushroom is edible. 

# Method
After assigning the x and y varibles for the dataset, it is identified that x varibles need to be transformed into dummy varibles because the model could not train string varibles. Then the data is split into trainning and testing set by 0.7 and 0.3 with y stratified. A classification tree model is used for trainning, with a max depth of 6. 

<img src="https://user-images.githubusercontent.com/65926359/101330158-2294f700-3827-11eb-9377-6f24ba5a3a28.png" width="90%"></img> 

(It can be seen from the decision tree that the top three most important features in this model are: odor_n, stalk-root_c, spare_print_color_r)

Confusion Matrix: 

<img src="https://user-images.githubusercontent.com/65926359/101330250-3b051180-3827-11eb-9f09-d3ecb2213e39.png" width="45%"></img> 

The model produced an accuracy score of 0.9987694831829368, which means this model is highly reliable. 

# Outcome
This model could be used to predict on the edibility of a given mushroom. As an example, the following sample is given, and a prediction is produced using the model. 

<img src="https://user-images.githubusercontent.com/65926359/101330739-d5fdeb80-3827-11eb-9a33-f9fa616c1e9b.png" width="90%"></img> 
