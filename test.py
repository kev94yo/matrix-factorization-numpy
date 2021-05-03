from matrix_factor import *
import pandas as pd
import numpy as np

"""
Task 1: CF Test Code Start (For RMSE Calculation)
"""
val_df = pd.read_csv('data/ratings_val.csv', index_col = 0)

# Set test_user_id
test_user_id = 73

test_user_df = val_df.loc[test_user_id]

user_df = rating_df.loc[test_user_id]
user_sim = []
user_rating = np.array(user_df['rating'])
for movie in user_df['movieId']:
    user_sim.append(sim_df.loc[movie])
user_sim = np.array(user_sim)
sim_sum = np.sum(user_sim, axis= 0)
score = np.matmul(user_sim.T, user_rating) / (1 + sim_sum)
print(f"CF Score list for user {test_user_id}: {score}")

sum = 0
count = 0
for id in test_user_df['movieId']:
    if id not in movies:
        continue
    index = np.where(movies == id)[0].item()
    error = test_user_df[test_user_df['movieId'] == id]['rating'] - score[index]
    sum += np.square(error)
    count += 1

print(f"CF RMSE for user {test_user_id}: {np.sqrt(sum / count).item()}")
"""
Task 1: CF Test Code End (For RMSE Calculation)
"""

"""
Task 2: MF Test Code Start (For RMSE Calculation)
"""
sum = 0
count = 0
user_index = np.where(users == test_user_id)[0].item()
user_rating = predicted_rating[:, user_index]
print(f"MF Score list for user{test_user_id}: {user_rating}")

for id in test_user_df['movieId']:
    if id not in movies:
        continue
    index = np.where(movies == id)[0].item()
    error = test_user_df[test_user_df['movieId'] == id]['rating'] - user_rating[index]
    sum += np.square(error)
    count += 1

print(f"MF RMSE for user {test_user_id}: {np.sqrt(sum / count).item()}")
"""
Task 2: MFTest Code End (For RMSE Calculation)
"""

"""
Task 3: RMSE for all movies in validation set. Test Code Start (For RMSE Calculation)
"""
def rmse_calculation(user_id):
    user_df = val_df.loc[user_id]
    user_index = np.where(users == user_id)[0].item()
    user_rating = opt_predicted_rating[:, user_index]

    sum = 0
    count = 0
    for id in user_df['movieId']:
        if id not in movies: # ignore movie if not in training set
            continue
        index = np.where(movies == id)[0].item()
        error = user_df[user_df['movieId'] == id]['rating'] - user_rating[index]
        if np.isnan(error.item()):
            continue
        sum += np.square(error).item()
        count += 1

    return sum, count

test_list = users
total_RMSE = 0
total_count = 0
for user_id in test_list:
    if user_id not in val_df.index:
        continue
    sum, count = rmse_calculation(user_id)
    total_RMSE += sum
    total_count += count

print(f"Optimized MF RMSE for all movies: {np.sqrt(total_RMSE / total_count).item()}")

"""
Task 3: Test Code End (For RMSE Calculation)
"""