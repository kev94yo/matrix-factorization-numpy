from matrix_factor import *

"""
Output Functions
"""
def read_user_movie_id():
    with open('input.txt', 'r') as f:
        return [l.strip().split(',') for l in f.readlines()]

def predictions(id_list):
    final_prediction = []
    for ids in id_list:
        user_id, movie_id = int(ids[0]), int(ids[1])
        final_prediction.append(get_scores_cf(user_id, movie_id))
        final_prediction.append(get_scores_mf(user_id, movie_id))
        final_prediction.append(get_scores_opt(user_id, movie_id))
    
    final_prediction = [['{},{},{}'.format(line[0], line[1], round(line[2], 4))] for line in final_prediction]
        
    return final_prediction

def write_output(prediction):
    with open('output.txt', 'w') as f:
        for p in prediction:
            for r in p:
                f.write(r + "\n")
    

if __name__ == "__main__":
    user_movie_ids = read_user_movie_id()
    result = predictions(user_movie_ids)
    write_output(result)