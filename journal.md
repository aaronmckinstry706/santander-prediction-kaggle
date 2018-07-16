# Development Journal for [Santander Value Prediction Competition on Kaggle](https://www.kaggle.com/c/santander-value-prediction-challenge#description)

## July 10, 2018

First thing's first: change the README file appropriately. Done! Now, we look into creating a downloading script via the Kaggle API. We can only download via the Kaggle CLI, so it's not worth creating a download script. Successfully downloaded the data. This weekend, we need to examine the data: how many numeric fields are there? What kind of data (all money values, or some other kinds of values) and whether any missing fields exist. Specifically:

1. How many numeric fields are there?

2. Is there any missing data (missing fields for any labeled samples)?

Then we can proceed from there. 

## July 15, 2018

Plan's changed. No data exploration today. We just need a full pipeline, from loading the data through outputting the results. This involves a few things. (For reference, I'm following steps from Andrew Ng's unpublished book, which he is emailing out to people who subscribed to his Twitter profile *and* subscribed to a special email list, a link to which I'm not sure I'm allowed to give.)

First, select an error measure. For a Kaggle competition, this is done for us, so we will just list it here. The error measure is Root Mean Square Logarithmic Error, which is calculated as $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\log(p_i + 1) - \log(a_i + 1))^2}$. In the given formula, $p_i$ is the predicted value and $a_i$ is the actual value. $a_i$ and $p_i$ are always positive, because they are the values of financial transactions (i.e., you spent \$500, or there was a \$20 deposit, etc.). Adding 1 to both predicted and actual values means that the logarithms are always nonnegative. In addition, because this error calculation uses logarithms, we get penalized linearly in the exponent. I.e., if I predict \$2.70 (about $e^1$), but the actual value is \$2980.95 (about $e^8$), then the calculated difference is only -7. This means that, as long as the predictions are on the same order of magnitude as the actual result, the difference in logs will be no more than 1. As for the root-mean-square component of the error measure, the important statistical property is that it penalizes outliers heavily (which is due to the mean being susceptible to outliers, and the squared term inflates large errors much more than smaller ones). So, if our predicted value is 10 orders of magnitude off from the actual value, that will be penalized much more heavily than if we are only 1 or 2 orders of magnitude off. 

Now that we understand the error measure, as well as what we are predicting, we need to randomly separate out an Eyeball Dev Set (one which you analyze to determine causes of high error, and debug the inference algorithm) and a Black Box Dev Set (one which is used to distinguish between the performance of different algorithms). From Andrew Ng's book, we want these datasets to have sizes that allow us to distinguish between the performance of different algorithms, and allows for good error analysis. We only have a training set of 4460 values! This is *alarmingly* small! Given this small training set, we will simply use the competition's Public Test Set (i.e., the one used to determine public leaderboard scores during the competition) as our Black Box Dev Set. For our Eyeball Dev Set, we will use the 70/30 rule: 3122 samples will be used for training, and 1338 samples will be used as our development set. These will be randomly selected *once*, and then used forever (unless, of course, we find that our training/development set performances are *dramatically* different; in that case, we can use techniques like PCA, or comparing mean/stdev vectors, to test for significant differences between training set distribution and development set distribution). 

The next step is to actually build the pipeline: load the data, train a baseline model, get predictions on dev set, calculate error, get predictions on test set, print those predictions, and submit a file to the leaderboard. Some details:

* we will be predicting `log(a_i + 1)`, rather than the raw values `a_i`, to help deal with scaling the outputs appropriately;

* due to the first point, when outputting predictions for the public leaderboard, we will have to print `exp(p_i) - 1`;

* also due to the first point, we will have to preprocess the targets in the loaded training set to be `log(a_i + 1)`, and then use Root Mean Square error--which is a common error measure in machine learning libraries (and included in `scikit-learn`, which I will be using for prediction!);

* we'll use pandas dataframes and pipelines to load and process the data, and then we'll use `scikit-learn` to train our model and get predictions. 

First thing's first: load the data into a Pandas dataframe, and check for missing data values. Done. Next: take the `target` column and create a `log_target` column. Done. Next: use `scikit-learn` to train a linear model with RMSE loss. Whoops! Can't do that, because `scikit-learn` can't use a custom loss function. I have three options: use PyTorch, figure out how to implement custom loss for `scikit-learn`, or find some other library which makes it easy. I don't feel like doing the third option, and I couldn't find a way to do the second option, so I'll choose the first option. 

PyTorch takes Numpy arrays, so I'll have to convert my dataframes to Numpy arrays before being able to convert them to tensors for PyTorch. In the future, any preprocessing/feature selection I do will have to be done on the Pandas dataframe before it gets converted to a PyTorch tensor. Once we have our PyTorch tensors, we can do the normal steps: create a class with a simple linear layer, use the L-BFGS optimizer with the `step(...)` function to optimize, and then get predictions from the model. After we have the predictions, we can convert back into a Numpy array and then add as another column to our original dataframe. Then we can do all post-prediction steps: plotting errors, etc. 

Bah! That's too much work for now, though. Let's just use `scikit-learn` with simple linear regression (i.e., using the wrong error function) to get a dumb model for now. Later, we'll get a good error function. 

Finished! As a slight modification, we had to bound the predictions `p_i` by `0` and `log(4600000000000 + 1)` in order to prevent NaN's and inf's. (The 4.6 trillion number comes from a quick Google search of the largest transaction ever made.)

Now we need to flesh out this Notebook into a proper script, so we can start copying it and developing new scripts based off of this short template. Done! Now, we are finally finished. Next week, we will add preprocessing: normalization, PCA, maybe taking logs of some fields which have a large maximum like we did for predictions, and more that I don't know about. And with that, we're done!
