from train_and_evaluate import train_and_evaluate
from pathlib import Path
from utils import *
import os
from metrics_aggregation import metrics_to_table, aggregate_metrics

def launch_training_job(model_dir, parameter_name, parameter_value, params, general_settings):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
       Args:
        parameter_naem: select which parameter to be varied
        params: (dict) containing basic setting of hyperparameters
       """
    # Create a new folder in parent_dir with unique_name "job_name"
    parameter_dir = os.path.join(model_dir, parameter_name+"_pred_step_"+str(general_settings.prediction_steps))
    if not os.path.exists(parameter_dir):
        os.makedirs(parameter_dir)
    job_dir = os.path.join(parameter_dir, parameter_name+"_"+str(parameter_value))
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    # Write parameters in json file
    json_path = os.path.join(job_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    train_and_evaluate(params, general_settings, job_dir)

if __name__ == "__main__":
    ##### Initializing #####

    # define paths
    # use the Python3 Pathlib modul to create platform independent path
    general_settings = Params.update(
        Path("C:/Users/zhouyuxuan/PycharmProjects/Masterarbeit/experiments/general_settings.json"))

    model_dir = os.path.join(general_settings.experiments_path, general_settings.model_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # load the parameters for the experiment params.json file in model path
    json_path = Path(model_dir) / 'params.json'
    params = Params.update(json_path)

    ### hyperparameter search ####
    # # learning rate search
    # for lr in [0.1, 0.05]:
    #     params.learning_rate = lr
    #     # Launch a training in this directory -- it will call `train_and_evaluate.py`
    #     launch_training_job(model_dir, "learning_rate", lr, params, general_settings)

    # epochs search
    for epochs in [5 ,10]:
        params.num_epochs = epochs
        launch_training_job(model_dir, "num_epochs", epochs, params, general_settings)


    # Aggregate metrics from args.parent_dir directory
    # parent_dir = os.path.join(model_dir, "num_epochs_pred_step_1")
    parent_dir = model_dir
    for channel_name in general_settings.channels:
        metrics = dict()
        aggregate_metrics(parent_dir, metrics, channel_name)
        table = metrics_to_table(metrics)

        # Display the table to terminal
        print(table)

        # Save results in parent_dir/results.md
        save_file = os.path.join(parent_dir, channel_name+"_results.md")
        with open(save_file, 'w') as f:
            f.write(table)
