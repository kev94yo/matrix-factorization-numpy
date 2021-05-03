import pandas as pd
import numpy as np

""" 
1. Item-based Collaborative Filtering
"""

# 1. Create base iu matrix
train_df = pd.read_csv("./data/ratings_train.csv")
users = train_df['userId'].unique()
movies = train_df['movieId'].unique()
movies = np.sort(movies)

base_iu_matrix = pd.DataFrame(0, index = movies, columns = users)

for index in train_df.index:
    data = train_df.loc[index]
    base_iu_matrix.loc[data['movieId'], data['userId']] = data['rating']

# 2. Get cosine similarity
movie_arr = base_iu_matrix.values

dot = movie_arr @ movie_arr.T
magnitude = np.sqrt(np.outer(np.sum(np.square(movie_arr), axis=1), np.sum(np.square(movie_arr), axis=1)))
sim_df = pd.DataFrame(dot / magnitude, index = base_iu_matrix.index, columns = base_iu_matrix.index)

# 3. Get Recommendation Scores
rating_df = pd.read_csv("./data/ratings_train.csv", index_col = 0)

def get_scores_cf(user_id, movie_id):
    user_df = rating_df.loc[user_id]
    user_sim = []
    user_rating = np.array(user_df['rating'])
    for movie in user_df['movieId']:
        user_sim.append(sim_df.loc[movie])
    user_sim = np.array(user_sim)
    sim_sum = np.sum(user_sim, axis= 0)

    score = np.matmul(user_sim.T, user_rating) / (1 + sim_sum)
    index = np.where(movies == movie_id)[0].item()
    return [user_id, movie_id, score[index]]

"""
2. Matrix Factorization
"""
movie_mean_series = base_iu_matrix.replace(0, np.NaN).mean(axis = 1)
filled_iu_matrix = base_iu_matrix.copy()

for index in filled_iu_matrix.index:
    mean = movie_mean_series[index]
    filled_iu_matrix.loc[index].replace(0, mean, inplace=True)

# Decompose
actual_rating = filled_iu_matrix.values
u, s, vh = np.linalg.svd(actual_rating, full_matrices=False)
u, s, vh = u[:, :400], np.diag(s[:400]), vh[:400, :]

# Predict
predicted_rating = ((u @ s) @ vh)

def get_scores_mf(user_id, movie_id):
    user_index = np.where(users == user_id)[0].item()
    movie_index = np.where(movies == movie_id)[0].item()

    return [user_id, movie_id, predicted_rating[movie_index, user_index]]

"""
3. Optimize MF
"""
k = 250 # 여기서 K 변화해보면서 체크
opt_u, opt_s, opt_vh = np.linalg.svd(actual_rating, full_matrices=False)
opt_u, opt_s, opt_vh = opt_u[:, :k], np.diag(opt_s[:k]), opt_vh[:k, :]
opt_predicted_rating = ((opt_u @ opt_s) @ opt_vh)

def get_scores_opt(user_id, movie_id):
    user_index = np.where(users == user_id)[0].item()
    movie_index = np.where(movies == movie_id)[0].item()

    return [user_id, movie_id, opt_predicted_rating[movie_index, user_index]]


# """
# 4. Output Functions
# """
# def read_user_movie_id():
#     with open('input.txt', 'r') as f:
#         return [l.strip().split(',') for l in f.readlines()]

# def predictions(id_list):
#     final_prediction = []
#     for ids in id_list:
#         user_id, movie_id = int(ids[0]), int(ids[1])
#         final_prediction.append(get_scores_cf(user_id, movie_id))
#         final_prediction.append(get_scores_mf(user_id, movie_id))
#         final_prediction.append(get_scores_opt(user_id, movie_id))
    
#     final_prediction = [['{},{},{}'.format(line[0], line[1], round(line[2], 4))] for line in final_prediction]
        
#     return final_prediction

# def write_output(prediction):
#     with open('output.txt', 'w') as f:
#         for p in prediction:
#             for r in p:
#                 f.write(r + "\n")
    

# if __name__ == "__main__":
#     user_movie_ids = read_user_movie_id()
#     result = predictions(user_movie_ids)
#     write_output(result)