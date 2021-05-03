# matrix-factorization-numpy
Program that produces two recommendation systems based on neighboorhood model and matrix factorization model, and ultimately produce predicted ratings for a movie given by some user. Matrix factorization is done with `numpy.linalg.svd`
## Requirements
Requirements can be found in `requirements.txt`. Only `numpy` and `pandas` are needed.
## Datasets
Datasets are from the Movielens dataset
- ratings_train.csv: Ratings given to movies by users (Train set)
- ratings_valid.csv: Ratings given to movies by users (Test set)
## Model Explanation
1. Neighborhood-based Model: Create representations for all movies and calculate cosine similarity to find nearest neightbors. Predict rating for user based on similarities
2. Matrix Factorization: Decompose user-item matrix into user feature vector and item feature vector. Use 400 features only to produce predictions
3. Optimized Model: Optimized matrix factorization model by using 250 features. Number of features is selected by cross validation
## Recommendation Generation
In practice, adjust input.txt to change user ids and movie ids to generate recommendations for
1. Adjust input.txt. The input format is `user_id, movie_id` for each line
2. Run `python main.py`
3. Check `output.txt` for results. The output format is 3 lines of `user_id, movie_id, prediction_score` for each input line. Each line corresponds to score produced from the three models in model explanation (ex. 1st line is from neighboorhood-based model)
## Get RMSE
Run `python test.py` to check RMSE for some user in validation set, or for all the users in the validations set. Change variable `test_user_id` in `matrix_factor.py` to change test user id
