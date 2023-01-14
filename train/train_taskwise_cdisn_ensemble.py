# train_taskwise_cdisn_ensemble.py: a file defining functions for training a CDISN Ensemble task-wise
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from numpy.core.fromnumeric import nonzero
import torch
import os
import numpy as np
import pickle as pkl
import copy
from data_preprocessing.data_preprocessing_utils import CachedDatasetForTraining

from models.cdisn_models import *

from results_analytics.training_analytics_utils import *


def train_cdisn_ensemble_taskwise(
    save_dir,
    root_data_dirs,  # list of datasets for each task (ordered according to requested_tasks)
    train_set_foldernames,
    val_set_foldernames,
    test_set_foldernames,
    requested_tasks=["RP", "BehavioralTUAB"],
    cv_split_nums=[0, 0],
    max_num_retrain_iterations=10,
    batch_size=250,
    shuffle=True,
    max_epochs=150,
    max_evals_after_saving=10,
    save_freq=20,
    models_root_file_name="RP_with_BehavioralTUAB_Training",
    dropout_rate=0.5,
    embed_dim=100,
    num_channels=22,
    window_len=600, 
    num_classes=[
        2,
        2,
    ],  # list of classes for each task (ordered according to requested_tasks)
    beta_vals=(0.9, 0.999),
    learning_rate=5e-4,
    weight_decay=0.001,
    former_state_dict_file=None,
    pretrained_upstream_cdisn_file_path=None,
    max_data_samp_ratios_by_task_and_split = {
        "RP":{
            "train":0.01, 
            "validation":0.01, 
            "test":0.01
        }, 
        "BehavioralTUAB":{
            "train":1.0, # may be 0.1 or 0.01, depending on how much data is in original dataset
            "validation":1.0, 
            "test":1.0
        }, 
    }
):
    """
    train_taskwise_cdisn_ensemble.train_cdisn_ensemble_taskwise: code for training a cdisn ensemble in a taskwise fashion
    - Inputs:
        * save_dir (str): the directory in which the trained model and statistics are to be stored,
        * root_data_dirs (list): list of root directories to (PreprocessedDataset) datasets for each task (ordered according to requested_tasks),
        * train_set_foldernames (list): list of train subset foldernames to use for each task (ordered according to requested_tasks),
        * val_set_foldernames (list): list of validation subset foldernames to use for each task (ordered according to requested_tasks),
        * test_set_foldernames (list): list of test subset foldernames to use for each task (ordered according to requested_tasks),
        * requested_tasks (list): list of task_ids which the cdisn ensemble will be required to model during training; default=["RP", "BehavioralFluoxetine"],
        * cv_split_nums (list): list of the particular CV-split fold numbers for each train_set; default=[0, 0],
        * max_num_retrain_iterations (int): the maximum number of times to retrain the cdisn ensemble on each task; default=10,
        * batch_size (int): the number of data samples to feed into each model in the ensemble at once; default=50,
        * shuffle (boolean): whether to shuffle the data once loaded; default=True,
        * max_epochs (int): the maximum number of epochs to train the models on a given task; default=100,
        * max_evals_after_saving (int): the number of times a cdisn ensemble will be evaluated on a task before training is stopped; default=6,
        * save_freq (int): the maximum number of epochs a model will be trained for before a checkpoint is saved; default=20,
        * models_root_file_name (str): a string that will be included in the name of every checkpoint that is saved to identify a cached ensemble; default="RP_with_BehavioralFluoxetine_Training",
        * dropout_rate (float): the probability with which to dropout a given parameter during training; default=0.5,
        * embed_dim (int): the size of the latent embeddings produced by StagerNet-based models; default=100,
        * num_channels (int): the number of channels in a time-series recording; default=11,
        * num_classes (list): the number of classes to be predicted for each task (ordered according to requested_tasks); default=[2,4,],
        * beta_vals (tuple): the beta vals to be used for the optimizer; default=(0.9, 0.999),
        * learning_rate (float): the learning rate to be applied to each gradient update; default=2e-4,
        * weight_decay (float): the weight decay rate to be applied during training; default=0.001,
        * former_state_dict_file (str): the path to an entire cached cdisn ensemble that is to be retrained; default=None,
        * pretrained_upstream_cdisn_file_path (str): the path to a single pretrained model that is to be incorporated into a new cdisn ensemble; default=None,
        * max_data_samp_ratios_by_task_and_split (dict(dict(float))): the fraction of data samples in a given dataset to use during training
    - Outputs:
        * cached cdisn ensemble (.pkl files): either fully (marked with 'vFinal') or partially (marked with 'temp_cdisn_model_ensemble_epoch') trained cdisn ensembles
        * plots (.png files): plots depicting various aspects of training (such as trends in validation accuracy)
        * recorded statistics (.pkl files): various files recording statistics recorded during training
        * meta data files (.pkl files): files containing the arguments used to train a specific checkpoint, along with statistics recorded for that checkpoint
    - Usage:
        * N/A
    """
    START_TASK_INDEX = 0
    supported_tasks = [
        "RP",
        "TS",
        "BehavioralTST",
        "BehavioralFluoxetine",
        "BehavioralTUAB",
    ]
    for requested_task in requested_tasks:
        assert requested_task in supported_tasks

    # cuda setup if allowed
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Pytorch v0.4.0
    # # due to issues with cuda (suspected memory problems), just run on cpu
    # device = torch.device("cpu") # Pytorch v0.4.0

    save_dirs_for_tasks = {
        task_name: save_dir + os.sep + str(task_name) for task_name in requested_tasks
    }
    for key in save_dirs_for_tasks.keys():
        os.mkdir(
            save_dirs_for_tasks[key]
        )  # see geeksforgeeks.org/create-a-directory-in-python/

    # initialize models
    cdisn_model_ensemble = []
    if former_state_dict_file is not None:
        with open(former_state_dict_file, "rb") as infile:
            cdisn_model_ensemble = pkl.load(infile)
    else:
        for i, requested_task in enumerate(requested_tasks):
            cdisn_model_ensemble.append(
                FullCDISNTaskModel( # FullCDISNStagerNetAndDecoder(
                    requested_task,
                    len(requested_tasks),
                    num_channels,
                    num_classes=num_classes[i],
                    dropout_rate=dropout_rate,
                    embed_dim=embed_dim,
                    device=device,
                )
            )
            if (
                pretrained_upstream_cdisn_file_path is not None
                and "Behavioral" in requested_task
            ):
                cdisn_model_ensemble[i].load_pretrained_upstream_params(
                    pretrained_upstream_cdisn_file_path
                )
    cdisn_model_ensemble = [x.to(device) for x in cdisn_model_ensemble]

    # begin training
    print(
        "train.train_taskwise_cdisn_ensemble.train_cdisn_ensemble_taskwise: beginning training at START_TASK_INDEX == ",
        START_TASK_INDEX,
    )
    historical_avg_test_accs = {task_name: [] for task_name in requested_tasks}
    historical_avg_test_accs_by_class = {
        task_name: {label_id: [] for label_id in range(num_classes[i])}
        for i, task_name in enumerate(requested_tasks)
    }
    historical_test_roc_aucs = {task_name: [] for task_name in requested_tasks}
    final_test_roc_metrics = {
        task_name: {"fpr": None, "tpr": None, "roc_auc": None}
        for task_name in requested_tasks
    }

    for retrain_iter_num in range(max_num_retrain_iterations):
        curr_save_dirs_for_tasks = {
            task_name: save_dirs_for_tasks[task_name]
            + os.sep
            + "retrainIter"
            + str(retrain_iter_num)
            for task_name in requested_tasks
        }
        for save_dir_key in curr_save_dirs_for_tasks.keys():
            os.mkdir(curr_save_dirs_for_tasks[save_dir_key])

        avg_train_losses = {task_name: [] for task_name in requested_tasks}
        avg_train_accs = {task_name: [] for task_name in requested_tasks}
        avg_train_roc_aucs = {task_name: [] for task_name in requested_tasks}
        avg_val_accs = {task_name: [] for task_name in requested_tasks}
        avg_val_roc_aucs = {task_name: [] for task_name in requested_tasks}
        avg_train_accs_by_class = {
            task_name: {label_id: [] for label_id in range(num_classes[i])}
            for i, task_name in enumerate(requested_tasks)
        }
        avg_val_accs_by_class = {
            task_name: {label_id: [] for label_id in range(num_classes[i])}
            for i, task_name in enumerate(requested_tasks)
        }
        avg_test_accs = {task_name: None for task_name in requested_tasks}
        test_roc_metrics = {
            task_name: {"fpr": None, "tpr": None, "roc_auc": None}
            for task_name in requested_tasks
        }

        for curr_task_ind, curr_task_id in enumerate(requested_tasks):
            # prep cdisn ensemble for current retraining run
            for model_ind, model in enumerate(cdisn_model_ensemble):
                model.train()
                if model_ind != curr_task_ind:
                    for p in model.parameters():
                        p.requires_grad = False
                else:
                    for p in model.parameters():
                        p.requires_grad = True

            # initialize training state
            min_val_inaccuracy = float("inf")
            num_evaluations_since_model_saved = 0
            saved_ensemble = None
            stopped_early = False

            # see https://neptune.ai/blog/pytorch-loss-functions
            loss_fn = nn.CrossEntropyLoss()

            # define optimizer: see https://discuss.pytorch.org/t/train-only-part-of-variable-in-the-network/16425/3
            optimizer = torch.optim.Adam(
                cdisn_model_ensemble[curr_task_ind].parameters(),
                betas=beta_vals,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            # ACCESS TRAIN/VAL DATASETS
            # curr_train_set = CachedDatasetForTraining(
            #     root_data_dirs[curr_task_ind], 
            #     train_set_foldernames[curr_task_ind], 
            #     max_data_samp_ratios_by_task_and_split[curr_task_id]["train"]
            # )
            # curr_train_loader = torch.utils.data.DataLoader(
            #     curr_train_set, batch_size=batch_size, shuffle=shuffle
            # )

            # curr_val_set = CachedDatasetForTraining(
            #     root_data_dirs[curr_task_ind], 
            #     val_set_foldernames[curr_task_ind], 
            #     max_data_samp_ratios_by_task_and_split[curr_task_id]["validation"]
            # )
            # curr_val_loader = torch.utils.data.DataLoader(
            #     curr_val_set, batch_size=batch_size, shuffle=shuffle
            # )

            # LOOP THROUGH EPOCHS
            for epoch in range(max_epochs):
                running_train_loss = 0
                num_correct_train_preds = 0
                total_num_train_preds = 0
                num_correct_train_preds_by_class = {
                    i: 0 for i in range(num_classes[curr_task_ind])
                }
                total_num_train_preds_by_class = {
                    i: 0 for i in range(num_classes[curr_task_ind])
                }
                running_train_roc_auc = 0
                total_num_train_batches = 0

                num_correct_val_preds = 0
                total_num_val_preds = 0
                num_correct_val_preds_by_class = {
                    i: 0 for i in range(num_classes[curr_task_ind])
                }
                total_num_val_preds_by_class = {
                    i: 0 for i in range(num_classes[curr_task_ind])
                }
                running_val_roc_auc = 0
                total_num_val_batches = 0

                ##LOOP THROUGH FILE 
                #load_subset returns just one file
                train_subset_foldernames = train_set_foldernames[curr_task_ind]

                for subset_foldername in train_subset_foldernames:
                    #curr_train_set contains numpy arrays (a few rows)
                    curr_train_set = CachedDatasetForTraining(
                    root_data_dirs[curr_task_ind], 
                    subset_foldername,
                    max_data_samp_ratios_by_task_and_split[curr_task_id]["train"]
                    )
                    curr_train_loader = torch.utils.data.DataLoader(
                    curr_train_set, batch_size=batch_size, shuffle=shuffle
                    )


                    # LOOP THROUGH BATCHES IN TRAIN LOADER
                    for batch_ind, curr_sample in enumerate(curr_train_loader):
                        # TRAIN
                        # transfer to GPU
                        curr_sample = list(curr_sample)
                        for item_index, item in enumerate(curr_sample):
                            # curr_sample[item_index] = item.to(device)
                            if item_index < len(curr_sample)-1:
                                curr_sample[item_index] = item.float().view(-1, window_len, num_channels).to(device)
                            else:
                                curr_sample[item_index] = item.float().view(-1).to(device)

                        # zero out any existing gradients
                        optimizer.zero_grad()

                        # make predictions
                        preds, embeds = cdisn_model_ensemble[curr_task_ind](
                            curr_task_id,
                            curr_sample[:-1]
                            + [
                                cdisn_model_ensemble[:curr_task_ind]
                                + cdisn_model_ensemble[curr_task_ind + 1 :]
                            ],
                        )
                        # print("preds after model forward pass == ", preds)
                        # print("curr_sample[-1].long().view(-1) == ", curr_sample[-1].long().view(-1))

                        # compute resulting loss
                        # print("pre-loss curr_task_id == ", curr_task_id)
                        # print("pre-loss preds.size() == ", preds.size())
                        # print("pre-loss curr_sample[-1].view(-1).size() == ", curr_sample[-1].view(-1).size())
                        loss = loss_fn(preds, curr_sample[-1].long().view(-1))
                        # print("loss == ", loss)

                        # update parameters
                        loss.backward()
                        optimizer.step()

                        # compute accuracy/stats
                        preds = (
                            preds.detach()
                        )  # https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
                        # print("preds after loss update and detach == ", preds)
                        (
                            new_num_correct,
                            new_num_total,
                            new_pred_stats_by_class,
                        ) = get_accuracy_numerator_and_denominator_for_cdisn_preds(
                            preds, curr_sample[-1]
                        )
                        num_correct_train_preds += new_num_correct
                        total_num_train_preds += new_num_total
                        for label_id in new_pred_stats_by_class.keys():
                            num_correct_train_preds_by_class[
                                label_id
                            ] += new_pred_stats_by_class[label_id]["num_correct"]
                            total_num_train_preds_by_class[
                                label_id
                            ] += new_pred_stats_by_class[label_id]["num_total"]

                        # track roc-auc score
                        _, _, curr_roc_auc = get_roc_metrics_from_preds(
                            torch.argmax(preds, dim=1).view(curr_sample[-1].size()).cpu(), # preds.cpu(), 
                            curr_sample[-1].cpu()
                        )
                        running_train_roc_auc += curr_roc_auc
                        total_num_train_batches += 1.0

                        # track loss
                        running_train_loss += loss.item()

                        # free up cuda memory
                        del curr_sample
                        del preds
                        del embeds
                        del loss
                        torch.cuda.empty_cache()

                        # if batch_ind == 3:
                        #     print("BREAKING FOR DEBUGGING PURPOSES")
                        #     break # FOR DEBUGGING PURPOSES
                        pass  # end of current train-batch iteration
                    pass
                # LOOP THROUGH VALIDATION FILES
                with torch.no_grad():
                    for model in cdisn_model_ensemble:
                        model.eval()



                    #LOOP THROUGH FILES 
                    val_subset_foldernames = val_set_foldernames[curr_task_ind]
                    for val_subset_foldername in val_subset_foldernames:
                        curr_val_set = CachedDatasetForTraining(
                            root_data_dirs[curr_task_ind], 
                            val_subset_foldername, 
                            max_data_samp_ratios_by_task_and_split[curr_task_id]["validation"]
                        )
                        curr_val_loader = torch.utils.data.DataLoader(
                            curr_val_set, batch_size=batch_size, shuffle=shuffle
                        )


                        # LOOP THROUGH BATCHES IN CURR_VAL_FILE
                        for batch_ind, curr_sample in enumerate(curr_val_loader):
                            # VALIDATE
                            # transfer to GPU
                            curr_sample = list(curr_sample)
                            for item_index, item in enumerate(curr_sample):
                                # curr_sample[item_index] = item.to(device)
                                if item_index < len(curr_sample)-1:
                                    curr_sample[item_index] = item.float().view(-1, window_len, num_channels).to(device)
                                else:
                                    curr_sample[item_index] = item.float().view(-1).to(device)

                            # make predictions
                            preds, embeds = cdisn_model_ensemble[curr_task_ind](
                                curr_task_id,
                                curr_sample[:-1]
                                + [
                                    cdisn_model_ensemble[:curr_task_ind]
                                    + cdisn_model_ensemble[curr_task_ind + 1 :]
                                ],
                            )

                            # compute accuracy/stats
                            (
                                new_num_correct,
                                new_num_total,
                                new_pred_stats_by_class,
                            ) = get_accuracy_numerator_and_denominator_for_cdisn_preds(
                                preds, curr_sample[-1]
                            )
                            num_correct_val_preds += new_num_correct
                            total_num_val_preds += new_num_total
                            for label_id in new_pred_stats_by_class.keys():
                                num_correct_val_preds_by_class[
                                    label_id
                                ] += new_pred_stats_by_class[label_id]["num_correct"]
                                total_num_val_preds_by_class[
                                    label_id
                                ] += new_pred_stats_by_class[label_id]["num_total"]

                            # track roc-auc score
                            # print("<<< GETTING VAL ROC METRICS >>>")
                            # print("torch.argmax(preds, dim=1).view(curr_sample[-1].size()).cpu().size() == ", torch.argmax(preds, dim=1).view(curr_sample[-1].size()).cpu().size())
                            # print("curr_sample[-1].cpu().size() == ", curr_sample[-1].cpu().size())
                            # print("torch.argmax(preds, dim=1).view(curr_sample[-1].size()).cpu() == ", torch.argmax(preds, dim=1).view(curr_sample[-1].size()).cpu())
                            # print("curr_sample[-1].cpu() == ", curr_sample[-1].cpu())
                            _, _, curr_roc_auc = get_roc_metrics_from_preds(
                                torch.argmax(preds, dim=1).view(curr_sample[-1].size()).cpu(), # preds.cpu(), 
                                curr_sample[-1].cpu()
                            )
                            running_val_roc_auc += curr_roc_auc
                            total_num_val_batches += 1.0

                            # free up cuda memory
                            del curr_sample
                            del preds
                            del embeds
                            torch.cuda.empty_cache()

                            # if batch_ind == 3:
                            #     print("BREAKING FOR DEBUGGING PURPOSES")
                            #     break # FOR DEBUGGING PURPOSES
                            pass  # end of current val-batch loop

                        pass  # end of validation "with" statement
                    pass

                # record averages/stats
                avg_train_losses[curr_task_id].append(
                    running_train_loss / total_num_train_preds
                )
                avg_train_accs[curr_task_id].append(
                    num_correct_train_preds / total_num_train_preds
                )
                avg_train_roc_aucs[curr_task_id].append(
                    running_train_roc_auc / total_num_train_batches
                )
                avg_val_accs[curr_task_id].append(
                    num_correct_val_preds / total_num_val_preds
                )
                avg_val_roc_aucs[curr_task_id].append(
                    running_val_roc_auc / total_num_val_batches
                )
                for label_id in range(num_classes[curr_task_ind]):
                    if total_num_train_preds_by_class[label_id] > 0:
                        avg_train_accs_by_class[curr_task_id][label_id].append(
                            num_correct_train_preds_by_class[label_id]
                            / total_num_train_preds_by_class[label_id]
                        )
                    else:
                        avg_train_accs_by_class[curr_task_id][label_id].append(-1)
                    if total_num_val_preds_by_class[label_id] > 0:
                        avg_val_accs_by_class[curr_task_id][label_id].append(
                            num_correct_val_preds_by_class[label_id]
                            / total_num_val_preds_by_class[label_id]
                        )
                    else:
                        avg_val_accs_by_class[curr_task_id][label_id].append(-1)

                # check stopping criterion / save model
                if epoch >= 10:
                    incorrect_val_percentage = 1.0 - (
                        num_correct_val_preds / total_num_val_preds
                    )
                    if incorrect_val_percentage < min_val_inaccuracy:
                        num_evaluations_since_model_saved = 0
                        min_val_inaccuracy = incorrect_val_percentage
                        saved_ensemble = copy.deepcopy(
                            cdisn_model_ensemble
                        )  # see https://docs.python.org/3/library/copy.html
                    else:
                        num_evaluations_since_model_saved += 1
                        if num_evaluations_since_model_saved >= max_evals_after_saving:
                            print(
                                "train.train_taskwise_cdisn_ensemble.train_cdisn_ensemble_taskwise:: EARLY STOPPING on epoch ",
                                epoch,
                            )
                            stopped_early = True
                            break

                # save intermediate state_dicts just in case
                if epoch % save_freq == 0:
                    temp_model_save_path = os.path.join(
                        curr_save_dirs_for_tasks[curr_task_id],
                        "temp_cdisn_model_ensemble_epoch" + str(epoch) + ".pkl",
                    )
                    with open(temp_model_save_path, "wb") as outfile:
                        pkl.dump(saved_ensemble, outfile)

                    plot_cdisn_ensemble_training_avgs(
                        avg_train_losses[curr_task_id],
                        avg_train_accs[curr_task_id],
                        avg_val_accs[curr_task_id],
                        avg_train_accs_by_class[curr_task_id],
                        avg_val_accs_by_class[curr_task_id],
                        avg_train_roc_aucs[curr_task_id],
                        avg_val_roc_aucs[curr_task_id],
                        "epoch" + str(epoch),
                        curr_save_dirs_for_tasks[curr_task_id],
                    )

                # if epoch == 10:
                #     print("BREAKING FOR DEBUGGING PURPOSES")
                #     break # FOR DEBUGGING PURPOSES
                pass  # end of current epoch iteration

            # SAVE FINAL TRAINED MODELS
            if stopped_early:
                cdisn_model_ensemble = saved_ensemble  # cdisn_model_ensemble.load_state_dict(saved_ensemble)

            ensemble_save_path = os.path.join(
                curr_save_dirs_for_tasks[curr_task_id],
                models_root_file_name + "_cdisn_model_ensemble_vFinal.pkl",
            )
            with open(ensemble_save_path, "wb") as outfile:
                pkl.dump(cdisn_model_ensemble, outfile)

            # LOOP THROUGH TESTING FILES
            with torch.no_grad():
                for model in cdisn_model_ensemble:
                    model.eval()

                num_correct_test_preds = 0
                total_num_test_preds = 0
                num_correct_test_preds_by_class = {
                    i: 0 for i in range(num_classes[curr_task_ind])
                }
                total_num_test_preds_by_class = {
                    i: 0 for i in range(num_classes[curr_task_ind])
                }
                running_test_pred_record = []
                running_test_label_record = []

                # ACCESS TEST DATA
                curr_test_set = CachedDatasetForTraining(
                    root_data_dirs[curr_task_ind], 
                    test_set_foldernames[curr_task_ind], 
                    max_data_samp_ratios_by_task_and_split[curr_task_id]["test"]
                )
                curr_test_loader = torch.utils.data.DataLoader(
                    curr_test_set, batch_size=batch_size, shuffle=shuffle
                )

                # LOOP THROUGH BATCHES IN CURR_TEST_FILE
                for batch_ind, curr_sample in enumerate(curr_test_loader):
                    # TEST
                    # transfer to GPU
                    curr_sample = list(curr_sample)
                    for item_index, item in enumerate(curr_sample):
                        # curr_sample[item_index] = item.to(device)
                        if item_index < len(curr_sample)-1:
                            curr_sample[item_index] = item.float().view(-1, window_len, num_channels).to(device)
                        else:
                            curr_sample[item_index] = item.float().view(-1).to(device)

                    # make predictions
                    preds, embeds = cdisn_model_ensemble[curr_task_ind](
                        curr_task_id,
                        curr_sample[:-1]
                        + [
                            cdisn_model_ensemble[:curr_task_ind]
                            + cdisn_model_ensemble[curr_task_ind + 1 :]
                        ],
                    )

                    # compute accuracy
                    (
                        new_num_correct,
                        new_num_total,
                        new_pred_stats_by_class,
                    ) = get_accuracy_numerator_and_denominator_for_cdisn_preds(
                        preds, curr_sample[-1]
                    )
                    num_correct_test_preds += new_num_correct
                    total_num_test_preds += new_num_total
                    for label_id in new_pred_stats_by_class.keys():
                        num_correct_test_preds_by_class[
                            label_id
                        ] += new_pred_stats_by_class[label_id]["num_correct"]
                        total_num_test_preds_by_class[
                            label_id
                        ] += new_pred_stats_by_class[label_id]["num_total"]

                    # record test preds
                    running_test_pred_record.append(
                        torch.argmax(preds, dim=1).view(curr_sample[-1].size()).cpu() # preds.cpu()
                    )  # see https://discuss.pytorch.org/t/what-is-the-cpu-in-pytorch/15007
                    running_test_label_record.append(curr_sample[-1].cpu())

                    # free up cuda memory
                    del curr_sample
                    del preds
                    del embeds
                    torch.cuda.empty_cache()

                    # if batch_ind == 3:
                    #     print("BREAKING FOR DEBUGGING PURPOSES")
                    #     break # FOR DEBUGGING PURPOSES
                    # print("NOT BREAKING TEST LOOP FOR DEBUGGING PURPOSES") # FOR DEBUGGING PURPOSES
                    pass  # end of current test-batch loop

                pass  # end of testing "with" statement

            # Compute Test Accuracies/Stats
            avg_test_acc = num_correct_test_preds / total_num_test_preds
            avg_test_accs[curr_task_id] = avg_test_acc
            historical_avg_test_accs[curr_task_id].append(avg_test_acc)
            for label_id in range(num_classes[curr_task_ind]):
                if total_num_test_preds_by_class[label_id] == 0:
                    print(
                        "WARNING: CLASS ",
                        label_id,
                        " MISSING FROM TEST CASES - SETTING ACCURACY TO -1 FOR THIS CLASS TO DENOTE NO TEST MADE.",
                    )
                    num_correct_test_preds_by_class[label_id] = 1
                    total_num_test_preds_by_class[label_id] = -1
                if total_num_test_preds_by_class[label_id] > 0:
                    historical_avg_test_accs_by_class[curr_task_id][label_id].append(
                        num_correct_test_preds_by_class[label_id]
                        / total_num_test_preds_by_class[label_id]
                    )
                else:
                    historical_avg_test_accs_by_class[curr_task_id][label_id].append(-1)

            # print("<<< GETTING TEST ROC METRICS >>>")
            # see https://discuss.pytorch.org/t/stacking-a-list-of-tensors-whose-dimensions-are-unequal/31888
            # as well as https://pytorch.org/docs/stable/generated/torch.cat.html
            print("torch.cat(running_test_pred_record).view(-1).size() == ", torch.cat(running_test_pred_record).view(-1).size())
            print("torch.cat(running_test_label_record).view(-1).size() == ", torch.cat(running_test_label_record).view(-1).size())
            print("torch.cat(running_test_pred_record).view(-1) == ", torch.cat(running_test_pred_record).view(-1))
            print("torch.cat(running_test_label_record).view(-1) == ", torch.cat(running_test_label_record).view(-1))
            # test_fpr, test_tpr, test_roc_auc = get_roc_metrics_from_preds(
            #     torch.stack(running_test_pred_record).view(-1),
            #     torch.stack(running_test_label_record).view(-1),
            # )
            test_fpr, test_tpr, test_roc_auc = get_roc_metrics_from_preds(
                torch.cat(running_test_pred_record).view(-1),
                torch.cat(running_test_label_record).view(-1),
            )
            test_roc_metrics[curr_task_id]["fpr"] = test_fpr
            test_roc_metrics[curr_task_id]["tpr"] = test_tpr
            test_roc_metrics[curr_task_id]["roc_auc"] = test_roc_auc
            historical_test_roc_aucs[curr_task_id].append(test_roc_auc)

            # Store current training stats / hyper-params
            meta_data_save_path = os.path.join(
                curr_save_dirs_for_tasks[curr_task_id],
                "meta_data_and_hyper_parameters.pkl",
            )
            with open(meta_data_save_path, "wb") as outfile:
                pkl.dump(
                    {
                        "retrain_iter_num": retrain_iter_num,
                        "curr_task_id": curr_task_id,
                        "avg_train_losses": avg_train_losses[curr_task_id],
                        "avg_train_accs": avg_train_accs[curr_task_id],
                        "avg_train_roc_aucs": avg_train_roc_aucs[curr_task_id],
                        "avg_val_accs": avg_val_accs[curr_task_id],
                        "avg_val_roc_aucs": avg_val_roc_aucs[curr_task_id],
                        "avg_train_accs_by_class": avg_train_accs_by_class[
                            curr_task_id
                        ],
                        "avg_val_accs_by_class": avg_val_accs_by_class[curr_task_id],
                        "avg_test_acc": avg_test_acc,
                        "test_roc_metrics": test_roc_metrics[curr_task_id],
                        "save_dir_for_models": curr_save_dirs_for_tasks[curr_task_id],
                        "root_data_dirs": root_data_dirs,  # list of datasets for each task (ordered according to requested_tasks)
                        "train_set_foldernames": train_set_foldernames,
                        "val_set_foldernames": val_set_foldernames,
                        "test_set_foldernames": test_set_foldernames,
                        "requested_tasks": requested_tasks,
                        "cv_split_nums": cv_split_nums,
                        "max_num_retrain_iterations": max_num_retrain_iterations,
                        "batch_size": batch_size,
                        "shuffle": shuffle,  # "num_workers": num_workers,
                        "max_epochs": max_epochs,
                        "max_evals_after_saving": max_evals_after_saving,
                        "save_freq": save_freq,
                        "models_root_file_name": models_root_file_name,
                        "dropout_rate": dropout_rate,
                        "embed_dim": embed_dim,
                        "num_channels": num_channels,
                        "window_len": window_len,
                        "num_classes": num_classes,
                        "beta_vals": beta_vals,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "former_state_dict_file": former_state_dict_file,
                        "pretrained_upstream_cdisn_file_path": pretrained_upstream_cdisn_file_path, 
                        "max_data_samp_ratios_by_task_and_split": max_data_samp_ratios_by_task_and_split, 
                    },
                    outfile,
                )
                pass  # end of meta-data saving 'with' statement

            plot_cdisn_ensemble_training_avgs(
                avg_train_losses[curr_task_id],
                avg_train_accs[curr_task_id],
                avg_val_accs[curr_task_id],
                avg_train_accs_by_class[curr_task_id],
                avg_val_accs_by_class[curr_task_id],
                avg_train_roc_aucs[curr_task_id],
                avg_val_roc_aucs[curr_task_id],
                "final_train_val",
                curr_save_dirs_for_tasks[curr_task_id],
            )

            plot_roc_auc_curve(
                test_fpr,
                test_tpr,
                test_roc_auc,
                "final_test",
                curr_save_dirs_for_tasks[curr_task_id],
            )

            # if curr_task_ind > 0:
            #     print("BREAKING FOR DEBUGGING PURPOSES")
            #     break # FOR DEBUGGING PURPOSES
            pass  # end of current task retraining iteration

        # Store current retrain training stats / hyper-params
        meta_data_save_path = os.path.join(
            save_dir, "retrainIteration" + str(retrain_iter_num) + "_summary_stats.pkl"
        )
        with open(meta_data_save_path, "wb") as outfile:
            pkl.dump(
                {
                    "retrain_iter_num": retrain_iter_num,
                    "avg_train_losses": avg_train_losses,
                    "avg_train_accs": avg_train_accs,
                    "avg_train_roc_aucs": avg_train_roc_aucs,
                    "avg_val_accs": avg_val_accs,
                    "avg_val_roc_aucs": avg_val_roc_aucs,
                    "avg_train_accs_by_class": avg_train_accs_by_class,
                    "avg_val_accs_by_class": avg_val_accs_by_class,
                    "avg_test_accs": avg_test_accs,
                    "test_roc_metrics": test_roc_metrics,
                },
                outfile,
            )
            pass

        plot_cdisn_ensemble_retrain_summary(
            avg_train_losses,
            avg_train_accs,
            avg_val_accs,
            avg_train_roc_aucs,
            avg_val_roc_aucs,
            test_roc_metrics,
            "retrainIteration" + str(retrain_iter_num),
            save_dir,
        )

        if (
            retrain_iter_num == max_num_retrain_iterations - 1
            and curr_task_ind == len(requested_tasks) - 1
        ):
            final_test_roc_metrics = test_roc_metrics

        # print("BREAKING FOR DEBUGGING PURPOSES")
        # break # FOR DEBUGGING PURPOSES
        pass  # end of meta retrain over tasks iteration

    # Store current retrain training stats / hyper-params
    meta_data_save_path = os.path.join(save_dir, "historical_test_summary_stats.pkl")
    with open(meta_data_save_path, "wb") as outfile:
        pkl.dump(
            {
                "historical_avg_test_accs": historical_avg_test_accs,
                "historical_avg_test_accs_by_class": historical_avg_test_accs_by_class,
                "historical_test_roc_aucs": historical_test_roc_aucs,
                "final_test_roc_metrics": final_test_roc_metrics,
            },
            outfile,
        )
        pass
    
    for i, task_name in enumerate(requested_tasks):
        if "Behavioral" in task_name:
            with open(
                os.path.join(
                    save_dir,
                    str(task_name)
                    + "_downstream_cross_val_sample_throttle_test_acc_points.pkl",
                ),
                "wb",
            ) as outfile:
                pkl.dump(
                    {
                        "final_test_acc": historical_avg_test_accs[task_name][-1],
                        "test_accs_over_all_retrain_iters": historical_avg_test_accs[
                            task_name
                        ],
                        "cv_split_num": cv_split_nums[i],
                        "root_data_dirs": root_data_dirs,
                        "requested_tasks": requested_tasks,
                    },
                    outfile,
                )
                pass
            pass
        pass
    pass

    plot_cdisn_ensemble_historical_test_summary(
        historical_avg_test_accs, final_test_roc_metrics, save_dir
    )  # TO-DO: add auc plots to this function

