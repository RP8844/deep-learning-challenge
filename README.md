# deep-learning-challenge
## Module 21: Neural Networks and Deep Learning

# Background: 
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

# Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.
Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
- What variable(s) are the target(s) for your model?
- What variable(s) are the feature(s) for your model?
2. Drop the EIN and NAME columns.
3. Determine the number of unique values for each column.
4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6. Use pd.get_dummies() to encode categorical variables.
7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

# Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every five epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

# Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  - Dropping more or fewer columns.
  - Creating more bins for rare occurrences in columns.
  - Increasing or decreasing the number of values for each bin.
  - Add more neurons to a hidden layer.
  - Add more hidden layers.
  - Use different activation functions for the hidden layers.
  - Add or reduce the number of epochs to the training regimen.
  **Note:** If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

  1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
  2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
  3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
  4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
  5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

# Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:
  1. **Overview** of the analysis: Explain the purpose of this analysis.
  2. **Results:** Using bulleted lists and images to support your answers, address the following questions:

- Data Preprocessing
    - What variable(s) are the target(s) for your model?
    - What variable(s) are the features for your model?
    - What variable(s) should be removed from the input data because they are neither targets nor features?

- Compiling, Training, and Evaluating the Model
  - How many neurons, layers, and activation functions did you select for your neural network model, and why?
  - Were you able to achieve the target model performance?
  - What steps did you take in your attempts to increase model performance?
3. **Summary:** Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem,                   and then explain your recommendation.

# Step 5: Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.

1. Download your Colab notebooks to your computer.
2. Move them into your Deep Learning Challenge directory in your local repository.
3. Push the added files to GitHub.

# Step 4: Writing a Report on the Neural Network Model - Alphabet Soup Charity Analysis Report
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Q1- Overview of the analysis: Explain the purpose of this analysis.

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Using machine learning and neural networks, we use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
Q2- Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?

Our target is the "y" which is the "IS_SUCCESSFUL" column

What variable(s) are the features for your model?

Our features is the "X" which is everything but the "IS_SUCCESSFUL" column as: "APPLICATION_TYPE" , "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", "ASK_AMT"

What variable(s) should be removed from the input data because they are neither targets nor features?

"EIN & "NAME" were removed from variables, since they are neither targets nor features.

Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?

I had 3 tries trying to get to best & higest accuracy

1st try: 2 hidden layers & an outer layer, layer 1: 10 neurons, tanh activation, layer 2: 20 neurons, sigmoid activations, and Outer layer: 1 unit, sigmoid activation & adam optimizers for complier.

2nd try: 3 hidden layers & an outer layer, layer 1: 15 neurons, tanh activation, layer 2: 25 neurons, sigmoid activations, layer 3: 25 neurons, relu activations, and Outer layer: 1 unit, sigmoid & adam optimizers for complier.

3rd try: 3 hidden layers & an outer layer, layer 1: 30 neurons, tanh activation, layer 2: 25 neurons, sigmoid activations, layer 3: 20 neurons, sigmoid activations, and Outer layer: 1 unit, sigmoid & adam optimizers for complier.

Were you able to achieve the target model performance? The best I got accuracy 0.7286

What steps did you take in your attempts to increase model performance? after cleanining & removing none features neither targets, & splitting used the deep neural net & trying diffrent number of layers, activations, optimizers & epochs

Q3- Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

unfortunatly diffrences between all 3 tries was not major unless maybe adding 5 layers & 100's of neurels but that will kill my processor that's why didn't go that far, just did the 2 to 3 layers & between 10-30 neurels on each layers but oticed sigmoid activation is the best for this module.
1st_try1 1st_try2

2nd_try1 2nd_try2

3rd_try1 3rd_try2


# Resources: 
1. Modules 19: In Class Activities
2. Class Instructor: Arooj A Qureshi
3. TA Instructor: Abdelrahman "Abdo" Elewah