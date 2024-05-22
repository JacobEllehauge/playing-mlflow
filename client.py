from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor

import argparse


# first python3.11 client.py --create True
# then python3.11 client.py


# Optional positional argument
parser = argparse.ArgumentParser(description='Optional create bool')
parser.add_argument('--create', type=bool, nargs='?',
                    help='create')

args = parser.parse_args()
create_non_idempotent_stuff = args.create


client = MlflowClient(tracking_uri="http://127.0.0.1:50666")

all_experiments = client.search_experiments()

print(all_experiments)


default_experiment = [
    {"name": experiment.name, "lifecycle_stage": experiment.lifecycle_stage}
    for experiment in all_experiments
    if experiment.name == "Default"
][0]

pprint(default_experiment)


# tags etc

# Provide an Experiment description that will appear in the UI
experiment_description = (
    "This is the grocery forecasting project. "
    "This experiment contains the produce models for apples."
)

# Provide searchable tags that define characteristics of the Runs that
# will be in this Experiment
experiment_tags = {
    "project_name": "grocery-forecasting",
    "store_dept": "produce",
    "team": "stores-ml",
    "project_quarter": "Q3-2023",
    "mlflow.note.content": experiment_description,
}

if create_non_idempotent_stuff:
    # Create the Experiment, providing a unique name
    produce_apples_experiment = client.create_experiment(
        name="Apple_Models", tags=experiment_tags
    )


# Use search_experiments() to search on the project_name tag key
apples_experiment = client.search_experiments(
    filter_string="tags.`project_name` = 'grocery-forecasting'"
)

print(vars(apples_experiment[0]))
