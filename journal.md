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

## July 20, 2018

Some thoughts written in my phone from this week:

> Need to add preprocessing. Options include: normalization, PCA/whitening, binarizing values, bucketing, and taking the logs of values.

> Suppose the features are monetary, and there is a linear relationship between the raw features and the predicted monetary value. In this case, taking the log of each feature, and of the predicted value, would make that relationship nonlinear. E.g., y = m*x + b, so log(y) = log(m*x + b) =/= m*log(x) + b. Of course, there might be a linear relationship between log features and log predictions, but not from raw features to raw prediction. The only way to know is to test both.

> Another interesting question regards normalization for monetary values. How do you normalize monetary values? It's awkward, because the data doesn't follow a normal distribution: disproportionate mass near 0, and no negative values. So...what distribution do you use to shift and scale the data properly?

> One workaround to consider is using a mirrored normal: take negatives of existing values, add them to matrix while doing column-wise normalization, and then remove them afterwards. This scales the data, assuming it follows a half-normal curve on the positive side.

> Another workaround is to ignore 0-values entirely when calculating stdev/mean for normalization. However, in this case, any columns with just a few high values and mostly-0 values will normalize to be *extreme* outliers (on the lower end of the of the distribution).

> Another workaround is to consider some other one-sided distribution. It might be useful to examine a random sample of the columns to determine what kinds of distributions exist in the set of columns.

> Additionally, for all these workarounds, we must consider them with raw vs log values.

> Related to normalization, we might consider PCA. We would need to try it both with both raw vals and (separately) log vals, as we did with normalization.

> On a separate note, we need to implement a train/test set split. On the one hand, we don't want the matrix to be under-constrained (too few rows ) during regression; on the other hand, we'd like to have a validation set of decent size. When is a matrix too under-constrained? This point determines the size of my validation set.

> As it stands, we have ~4600 rows and ~4900 cols. This means that we are already under-constrained. So, rather than do a single train/validation split, we will use k-fold cross-validation. We'll use the standard k=10 (large k gets low bias but high variance; small k is the opposite; 10 strikes a balance).

> Some things we should look at, if time allows: how do other people do normalization for monetary values? How many more (or less) rows than columns do I have?

> Another thing to consider: we need to actually optimize on the true loss. For this, we require PyTorch. Bright side: we can totally use the GPU! Down side: extra work figuring out optimization, no CVGridSearch, etc.

> From these thoughts, I have three goals this weekend: 
  * implement, or find an existing implementation, of k-fold cross validation with arbitrary parameters;
  * use PyTorch to directly optimize the loss function (choose an optimizer).

> Well that's two. Whatever.

> I'll do the invert-then-normalize hack. Question is, if the data is inverted, then...what's the new stdev? Normally it's just 1.0. But if we're only taking the top half, then...nothing. We just use stdev 1. All that matters is that the data is properly scaled.

> Another issue: Many of these values are zero in each column. I wonder if it's not well-conditioned, and thus PCA won't work.

> If I did the exclude-zeroes normalization, then one option to avoid the zero-outlier problem is to replace the zero-values with the mean. This might eliminate their contribution to instability in the regression.

> UGH. I just need a proper Internet connection and a place to work.

Found the last one: a proper Internet connection and a place to work. Adding to the (official) list of things to do from my phone notes, I also want to do preprocessing. 

After some sample work in a Notebook, I now have code that shuffles the rows of a dataframe and loops through sequential chunks the shuffled dataframe; on each iteration, it separates the chunk into a validation subset, and puts the remaining rows into a training subset. Putting this into a function would give us an easy way to perform cross-validation. This brings into question, though, the project structure: how do we organize things from here? One option is to have a generator for yielding the training/validation subsets for cross-validation. This is nice, because it doesn't force us to decide on a structure yet for prediction code. It makes it easy to put what we've already done into an easy loop, change a few variable names, and have a working output. 

I could focus on creating a general interface for training, validating, predicting, etc. However, I don't want to get bogged down in creating my own mini-framework. That's what this scripting interface was for: to enable me to avoid creating my own mini-frameworks, by allowing me to track each experiment separately. This simple method allows me to do cross-validation, without forcing me to impose general structure on my experiments. Let's go with this, and then implement k-fold cross-validation in the baseline script. We won't change anything about the baseline results; we will just add our code to perform cross-validation before generating our final predictions on the full dataset. At some point later, we do want to develop a simple, general interface for cross-validation so that we don't have to duplicate training code within the same script. 

Hmmm. Code duplication is bad. Okay, what's the simplest interface that I can possibly come up with? This cross-validation has to be all done in 35 minutes. We'd like this generic cross-validation function to be independent of the model/optimization method/etc. The only thing that is guaranteed is that the cross-validation function will have to operate with pandas dataframes, and the training/scoring code will have to operate on pandas dataframes. The cross-validation function should take as parameters:

* a dataframe, comprised of the whole training set; and

* a function which takes two dataframes (training/validation), trains a model on the training set, and returns a score on the validation set.

How does this fit in with the generic train/prediction/evaluation code? Ah! Those will simply be composed in a larger function. All will operate with dataframes! That's that, then. 

Two components to this: the code for generating the train/validation datasets (this is the part I already have), and the function for generating a cross-validation score from a train-validate-score function and a whole labeled dataset. 

Let's switch back to dev-baseline and make changes. Actually, we'll stay in `scratch` to make changes, and then copy them manually to new modules in `scripts.baseline`. 

Writing `cross_validate`: the core is finished. I'm handling the only edge-case now: when the number of folds does not evenly divide the number of rows in the dataframe. What we need here is a way to do float computations in order to calculate the indices at each step. So, we don't do a chunk size; instead, we do a set of dividing indices. E.g., with 4 folds and a dataset of size 4, we want 5 dividing points `div_pts = [0, 1, 2, 3, 4]`, where `fold[i] = training_data.iloc[div_pts[i]:div_pts[i + 1]]`. Another example, 4 folds of dataset with size 6 yields dividing points `[0*6//4, 6//4, 2*6//4, 3*6//4, 4*6//4] = [0, 1, 3, 4, 6]`. There we go!

## July 21, 2018

Last night, I was still writing the test function for iteration over the folds. Let's finish that. Wrote some unit tests for each function, debugged them. Done!

