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

# Step 4: Write a Report on the Neural Network Model: Alphabet Soup Charity Analysis Report

1. **Overview** of the analysis:
  The nonprofit foundation Alphabet Soup wants an effective tool which can help it select the applicants for funding with the best chance of success. Using machine       learning and neural networks, we use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if       funded by Alphabet Soup.
2. **Results:** 
- Data Preprocessing
- What variable(s) are the target(s) for your model?
    - The variable for the Target was the column "IS_SUCCESSFUL".
- What variable(s) are the features for your model?
    - The columns that were considered as features for the model were: "NAME", "APPLICATION_TYPE" , "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION",        "STATUS", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", "ASK_AMT"
- What variable(s) should be removed from the input data because they are neither targets nor features?
    - "EIN & "NAME" were removed from the variables, since they are neither targets nor features.

- Compiling, Training, and Evaluating the Model
- How many neurons, layers, and activation functions did you select for your neural network model, and why?
      In the Optimized version of the model, I used 3 hidden layers each with multiple neurons which increased the accuracy from 72% to 78%. The Initial model only had two layers. Although the number of epochs did not change between the Initial and the Optimized Model, adding a third layer increased the accuracy of the model.

- Were you able to achieve the target model performance?
  - Yes by optimizing the model, I was able to increase the accuracy from 72% to 78%.

- What steps did you take in your attempts to increase model performance? 
  - I dropped both the EIN and NAME columns.
  - I added a third activation layer from relu as the first and second layer, to tanh as the second layer and sigmoid as the third layer and in this way was able to boost the accuracy to over 79%:
      - 1st Layer - relu
      - 2nd Layer - tanh
      - 3rd Layer - sigmoid

3. **Summary**
By optimizing the model, I was able to increase the accuracy from 72% to 78%. This means that I was able to correctly classify each of the points in the test data 78% of the time, which translates to an applicant having a 78% chance of being successful if they have the following:
    - The NAME of the applicant appears more than 5 times (they have applied more than 5 times)
    - The type of APPLICATION is one of the following: T3, T4, T5, T6 and T19
    - The application has the following values for CLASSIFICATION: C1000, C1200, C2000,C2100 and C3000.

**Alternative Method**

Although this model worked very well and provided a great deal of accuracy, an alternative approach that I tried out (to see if it would provide more accurate results),  was the Random Forest model. I chose this model because it's suited for classification problems. Using the Random Forest model, it was interesting to discover and to note that the accuracy decreased by 1%, from 78% accuracy to 76.5% (or 77%), as can be seen in the code. This means that the model which was chosen was the best one suited for this optimization.

# Resources: 
1. Modules 21: In Class Activities
2. Class Instructor: Arooj A Qureshi
3. TA Instructor: Abdelrahman "Abdo" Elewah
