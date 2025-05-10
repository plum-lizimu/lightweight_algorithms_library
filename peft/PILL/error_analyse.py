import pandas as pd
import json

result_file = "/data/zfy/pill/preds.json"
result_lavin_file = "/data/zfy/LaVIN/preds.json"
data_file = "/data/zfy/dataset/scienceQA/problems.json"
# answer_file = "/data/zfy/answer.json"

# read result file
results = json.load(open(result_file))
results_lavin = json.load(open(result_lavin_file))
# answer = json.load(open(answer_file))
num = len(results)
assert num == 4241

sqa_data = json.load(open(data_file))

# construct pandas data
sqa_pd = pd.DataFrame(sqa_data).T
res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set

# update data
for index, row in res_pd.iterrows():

    label = row['answer']
    pred = int(results[index])
    pred_lavin = int(results_lavin[index])

    if pred_lavin != label and pred == label:
        print(index)
        # print(answer_file[index])

