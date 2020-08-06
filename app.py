import os, ast
import pandas as pd

dataset = os.environ["INPUT_DATASET"]
target = os.environ["INPUT_TARGET"]
usecase = os.environ["INPUT_USECASE"]

dataset_path = "https://raw.githubusercontent.com/" + os.environ["GITHUB_REPOSITORY"] + "/master/" + os.environ["INPUT_DATASET"] + '.csv'
print("Dataset path {}".format(dataset_path))
print("Listing current files in the same directory")
os.listdir()

print("Listing recursively")
files = []
path = os.getcwd()
print("Current working directory : {}".format(path))

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

for f in files:
    print(f)
    
data = pd.read_csv(dataset_path)
data.head()

if usecase == 'regression':
    from pycaret.regression import *
elif usecase == 'classification':
    from pycaret.classification import *

exp1 = setup(data, target = target, session_id=123, silent=True, html=False, log_experiment=True, experiment_name='exp_github')

best = compare_models()

best_model = finalize_model(best)

save_model(best_model, 'model')

logs_exp_github = get_logs(save=True)
