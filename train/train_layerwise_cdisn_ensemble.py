# train_taskwise_cdisn_ensemble.py: a file defining functions for training a CDISN Ensemble layer-wise
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
from data_loading.data_loading_utils import extract_necessary_elements_from_layerwise_sample


def train_cdisn_ensemble_layerwise(
    save_dir,
    root_data_dir,  # dataset for training psi functions
    train_set_foldername,
    val_set_foldername,
    test_set_foldername,
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
    pretrained_upstream_cdisn_file_paths=None, # list of file paths for pretrained networks to use (ordered according to requested_tasks)
    max_data_samp_ratios_by_split = {
        "train":1.0, # may be 0.1 or 0.01, depending on how much data is in original dataset
        "validation":1.0, 
        "test":1.0
    },
    embedder_type="ShallowNet",
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
    START_LAYER_INDEX = 0
    supported_tasks = [
        "BehavioralTUAB",
        "RP",
        "TS",
    ]
    for requested_task in requested_tasks:
        assert requested_task in supported_tasks
    
    assert requested_tasks == sorted(requested_tasks)

    # cuda setup if allowed
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Pytorch v0.4.0
    # # due to issues with cuda (suspected memory problems), just run on cpu
    # device = torch.device("cpu") # Pytorch v0.4.0

    assert embedder_type in ["ShallowNet", "CorrelatedShallowNet"]
    NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN = 5
    save_dirs_for_layers = {
        k: save_dir + os.sep + "layer_"+str(k) for k in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)
    }
    for key in save_dirs_for_layers.keys():
        os.mkdir(
            save_dirs_for_layers[key]
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
                    embedder_type=embedder_type
                )
            )
            if pretrained_upstream_cdisn_file_paths is not None:
                print("LOADING PRETRAINED WEIGHTS FOR TASK == ", requested_task)
                assert requested_task in pretrained_upstream_cdisn_file_paths[i]
                cdisn_model_ensemble[i].load_pretrained_upstream_params(
                    pretrained_upstream_cdisn_file_paths[i]
                )
            else:
                print("WARNING: NOT LOADING ANY PRETRAINED WEIGHTS FOR TASK == ", requested_task)
    cdisn_model_ensemble = [x.to(device) for x in cdisn_model_ensemble]

    # load dataset(s)
    samp_type = None
    if "AnchoredBTUABRPTS" in root_data_dir:
        samp_type = "AnchoredBTUABRPTS"
        assert embedder_type == "CorrelatedShallowNet"
    elif "NonAnchoredBTUABRPTS" in root_data_dir:
        samp_type = "NonAnchoredBTUABRPTS"
        assert embedder_type == "ShallowNet"
    else:
        raise ValueError("train_layerwise_cdisn_ensemble.train_warm_start_cdisn_ensemble_layerwise: Sample type is not named in root_data_dir - please rename to include either AnchoredBTUABRPTS or NonAnchoredBTUABRPTS")

    # ACCESS TRAIN/VAL DATASETS
    curr_train_set = CachedDatasetForTraining(
        root_data_dir, 
        train_set_foldername, 
        max_data_samp_ratios_by_split["train"]
    )
    curr_train_loader = torch.utils.data.DataLoader(
        curr_train_set, batch_size=batch_size, shuffle=shuffle
    )

    curr_val_set = CachedDatasetForTraining(
        root_data_dir, 
        val_set_foldername, 
        max_data_samp_ratios_by_split["validation"]
    )
    curr_val_loader = torch.utils.data.DataLoader(
        curr_val_set, batch_size=batch_size, shuffle=shuffle
    )

    # ACCESS TEST DATA
    curr_test_set = CachedDatasetForTraining(
        root_data_dir, 
        test_set_foldername, 
        max_data_samp_ratios_by_split["test"]
    )
    curr_test_loader = torch.utils.data.DataLoader(
        curr_test_set, batch_size=batch_size, shuffle=shuffle
    )

    # begin training
    print(
        "train.train_layerwise_cdisn_ensemble.train_cdisn_ensemble_layerwise: beginning training at START_LAYER_INDEX == ",
        START_LAYER_INDEX,
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
        curr_save_dirs_for_layers = {
            k: save_dirs_for_layers[k]
            + os.sep
            + "retrainIter"
            + str(retrain_iter_num)
            for k in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)
        }
        for save_dir_key in curr_save_dirs_for_layers.keys():
            os.mkdir(curr_save_dirs_for_layers[save_dir_key])

        avg_combined_train_losses = {k: [] for k in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)}
        avg_train_losses = [{task_name: [] for task_name in requested_tasks} for _ in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)]
        avg_train_accs = [{task_name: [] for task_name in requested_tasks} for _ in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)]
        avg_train_roc_aucs = [{task_name: [] for task_name in requested_tasks} for _ in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)]
        avg_val_accs = [{task_name: [] for task_name in requested_tasks} for _ in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)]
        avg_val_roc_aucs = [{task_name: [] for task_name in requested_tasks} for _ in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)]
        avg_train_accs_by_class = [{task_name: {label_id: [] for label_id in range(num_classes[i])} for i, task_name in enumerate(requested_tasks)} for _ in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)]
        avg_val_accs_by_class = [{task_name: {label_id: [] for label_id in range(num_classes[i])} for i, task_name in enumerate(requested_tasks)} for _ in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)]
        avg_test_accs = [{task_name: None for task_name in requested_tasks} for _ in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)]
        test_roc_metrics = [{task_name: {"fpr": None, "tpr": None, "roc_auc": None} for task_name in requested_tasks} for _ in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN)]

        for curr_layer_ind in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN):
            # prep cdisn ensemble for current retraining run
            psi_params_to_optimize = []
            for model_ind, model in enumerate(cdisn_model_ensemble):
                model.train()
                for p in model.parameters():
                    p.requires_grad = False
                curr_psi_params = model.embedder.unfreeze_psi_update_parameters_for_given_layer(curr_layer_ind)
                psi_params_to_optimize = psi_params_to_optimize + curr_psi_params

            # initialize training state
            min_val_inaccuracy = float("inf")
            num_evaluations_since_model_saved = 0
            saved_ensemble = None
            stopped_early = False

            # see https://neptune.ai/blog/pytorch-loss-functions
            base_loss_fn = nn.CrossEntropyLoss() # each model is assumed to be optimizing some form of cross entropy loss, all of which will be summed for ensemble-wide optimization

            # define optimizer: see https://discuss.pytorch.org/t/train-only-part-of-variable-in-the-network/16425/3
            optimizer = None
            if len(requested_tasks) > 1:
                optimizer = torch.optim.Adam(
                    psi_params_to_optimize, # see https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603
                    betas=beta_vals,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            else:
                optimizer = torch.optim.Adam(
                    cdisn_model_ensemble[0].parameters(),
                    betas=beta_vals,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )

            # LOOP THROUGH EPOCHS
            for epoch in range(max_epochs):
                running_train_loss = 0
                running_train_loss_by_task = {x:0 for x in requested_tasks}
                num_correct_train_preds_by_task = {x:0 for x in requested_tasks}
                total_num_train_preds_by_task = {x:0 for x in requested_tasks}
                num_correct_train_preds_by_task_and_class = {x:{i: 0 for i in range(num_classes[x_ind])} for x_ind, x in enumerate(requested_tasks)}
                total_num_train_preds_by_task_and_class = {x:{i: 0 for i in range(num_classes[x_ind])} for x_ind, x in enumerate(requested_tasks)}
                running_train_roc_auc_by_task = {x:0 for x in requested_tasks}
                total_num_train_batches_by_task = {x:0 for x in requested_tasks}

                num_correct_val_preds_by_task = {x:0 for x in requested_tasks}
                total_num_val_preds_by_task = {x:0 for x in requested_tasks}
                num_correct_val_preds_by_task_and_class = {x:{i: 0 for i in range(num_classes[x_ind])} for x_ind, x in enumerate(requested_tasks)}
                total_num_val_preds_by_task_and_class = {x:{i: 0 for i in range(num_classes[x_ind])} for x_ind, x in enumerate(requested_tasks)}
                running_val_roc_auc_by_task = {x:0 for x in requested_tasks}
                total_num_val_batches_by_task = {x:0 for x in requested_tasks}

                # LOOP THROUGH BATCHES IN TRAIN LOADER
                for batch_ind, full_sample in enumerate(curr_train_loader):
                    # TRAIN
                    # transfer to GPU
                    # curr_sample = list(full_sample)
                    curr_sample = extract_necessary_elements_from_layerwise_sample(full_sample, samp_type, requested_tasks)
                    for item_index, item in enumerate(curr_sample):
                        # curr_sample[item_index] = item.to(device)
                        if item_index < len(curr_sample)-3: # for anything that isn't a label
                            curr_sample[item_index] = item.float().view(-1, window_len, num_channels).to(device)
                        else:
                            curr_sample[item_index] = item.float().view(-1).to(device)

                    # zero out any existing gradients
                    optimizer.zero_grad()

                    # make predictions
                    preds, embeds = cdisn_model_ensemble[0](samp_type, [curr_sample[:-1], requested_tasks, cdisn_model_ensemble[1:]])
                    # print("preds after model forward pass == ", preds)
                    # print("curr_sample[-1].long().view(-1) == ", curr_sample[-1].long().view(-1))

                    # compute resulting loss
                    # print("pre-loss curr_task_id == ", curr_task_id)
                    # print("pre-loss preds.size() == ", preds.size())
                    # print("pre-loss curr_sample[-1].view(-1).size() == ", curr_sample[-1].view(-1).size())
                    loss = None
                    for i, task_id in enumerate(requested_tasks):
                        curr_task_pred_ind = supported_tasks.index(task_id)
                        if loss is None:
                            # print("curr_sample[-1].size() == ", curr_sample[-1].size()) # FOR DEBUGGING
                            # print("curr_sample[-1].size() == ", curr_sample[-1][i].size())
                            # see https://discuss.pytorch.org/t/what-is-loss-item/61218/22
                            # loss = base_loss_fn(preds[curr_task_pred_ind], curr_sample[-1][curr_task_pred_ind].long().view(-1))
                            loss = base_loss_fn(preds[curr_task_pred_ind], curr_sample[-(3-curr_task_pred_ind)].long().view(-1))
                            running_train_loss_by_task[task_id] = loss.item()
                        else:
                            former_loss_val = loss.item()
                            # loss = loss + base_loss_fn(preds[curr_task_pred_ind], curr_sample[-1][curr_task_pred_ind].long().view(-1))
                            loss = loss + base_loss_fn(preds[curr_task_pred_ind], curr_sample[-(3-curr_task_pred_ind)].long().view(-1))
                            running_train_loss_by_task[task_id] += loss.item() - former_loss_val
                    # print("loss == ", loss)

                    # update parameters
                    loss.backward()
                    optimizer.step()

                    # compute accuracy/stats
                    # preds = (
                    #     preds.detach()
                    # )  # https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
                    # print("preds after loss update and detach == ", preds)

                    for i, task_id in enumerate(requested_tasks):
                        curr_task_pred_ind = supported_tasks.index(task_id)
                        (
                            new_num_correct,
                            new_num_total,
                            new_pred_stats_by_class,
                        ) = get_accuracy_numerator_and_denominator_for_cdisn_preds(
                            preds[curr_task_pred_ind].detach(), curr_sample[-(3-curr_task_pred_ind)] # curr_sample[-1][curr_task_pred_ind]
                        )
                        num_correct_train_preds_by_task[task_id] += new_num_correct
                        total_num_train_preds_by_task[task_id] += new_num_total
                        for label_id in new_pred_stats_by_class.keys():
                            num_correct_train_preds_by_task_and_class[task_id][label_id] += new_pred_stats_by_class[label_id]["num_correct"]
                            total_num_train_preds_by_task_and_class[task_id][label_id] += new_pred_stats_by_class[label_id]["num_total"]
                            pass

                        # track roc-auc score
                        # highest_weighted_preds = torch.argmax(preds[curr_task_pred_ind], dim=1).view(curr_sample[-1][curr_task_pred_ind].size()).cpu()
                        highest_weighted_preds = torch.argmax(preds[curr_task_pred_ind].detach(), dim=1).view(curr_sample[-(3-curr_task_pred_ind)].size()).cpu()
                        # _, _, curr_roc_auc = get_roc_metrics_from_preds(highest_weighted_preds, curr_sample[-1][curr_task_pred_ind].cpu())
                        _, _, curr_roc_auc = get_roc_metrics_from_preds(highest_weighted_preds, curr_sample[-(3-curr_task_pred_ind)].cpu())
                        running_train_roc_auc_by_task[task_id] += curr_roc_auc
                        total_num_train_batches_by_task[task_id] += 1.0
                        pass

                    # track loss
                    running_train_loss += loss.item()

                    # free up cuda memory
                    del curr_sample
                    del preds
                    del embeds
                    del loss
                    # torch.cuda.empty_cache() # COMMENTED OUT FOR DEBUGGING PURPOSES

                    if batch_ind == 3:
                        print("BREAKING FOR DEBUGGING PURPOSES")
                        break # FOR DEBUGGING PURPOSES
                    pass  # end of current train-batch iteration

                # LOOP THROUGH VALIDATION FILES
                with torch.no_grad():
                    for model in cdisn_model_ensemble:
                        model.eval()

                    # LOOP THROUGH BATCHES IN CURR_VAL_FILE
                    for batch_ind, full_sample in enumerate(curr_val_loader):
                        # VALIDATE
                        # transfer to GPU
                        # curr_sample = list(full_sample)
                        curr_sample = extract_necessary_elements_from_layerwise_sample(full_sample, samp_type, requested_tasks)
                        for item_index, item in enumerate(curr_sample):
                            # curr_sample[item_index] = item.to(device)
                            if item_index < len(curr_sample)-3:
                                curr_sample[item_index] = item.float().view(-1, window_len, num_channels).to(device)
                            else:
                                curr_sample[item_index] = item.float().view(-1).to(device)

                        # make predictions
                        preds, embeds = cdisn_model_ensemble[0](samp_type, [curr_sample[:-1], requested_tasks, cdisn_model_ensemble[1:]])
                        # print("preds after model forward pass == ", preds)
                        # print("curr_sample[-1].long().view(-1) == ", curr_sample[-1].long().view(-1))

                        # compute accuracy/stats
                        # preds = (
                        #     preds.detach()
                        # )  # https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
                        # print("preds after loss update and detach == ", preds)

                        for i, task_id in enumerate(requested_tasks):
                            curr_task_pred_ind = supported_tasks.index(task_id)
                            (
                                new_num_correct,
                                new_num_total,
                                new_pred_stats_by_class,
                            ) = get_accuracy_numerator_and_denominator_for_cdisn_preds(
                                preds[curr_task_pred_ind].detach(), curr_sample[-(3-curr_task_pred_ind)] # curr_sample[-1][curr_task_pred_ind]
                            )
                            num_correct_val_preds_by_task[task_id] += new_num_correct
                            total_num_val_preds_by_task[task_id] += new_num_total
                            for label_id in new_pred_stats_by_class.keys():
                                num_correct_val_preds_by_task_and_class[task_id][label_id] += new_pred_stats_by_class[label_id]["num_correct"]
                                total_num_val_preds_by_task_and_class[task_id][label_id] += new_pred_stats_by_class[label_id]["num_total"]
                                pass

                            # track roc-auc score
                            # highest_weighted_preds = torch.argmax(preds[curr_task_pred_ind], dim=1).view(curr_sample[-1][curr_task_pred_ind].size()).cpu()
                            highest_weighted_preds = torch.argmax(preds[curr_task_pred_ind].detach(), dim=1).view(curr_sample[-(3-curr_task_pred_ind)].size()).cpu()
                            # _, _, curr_roc_auc = get_roc_metrics_from_preds(highest_weighted_preds, curr_sample[-1][curr_task_pred_ind].cpu())
                            _, _, curr_roc_auc = get_roc_metrics_from_preds(highest_weighted_preds, curr_sample[-(3-curr_task_pred_ind)].cpu())
                            running_val_roc_auc_by_task[task_id] += curr_roc_auc
                            total_num_val_batches_by_task[task_id] += 1.0
                            pass

                        # free up cuda memory
                        del curr_sample
                        del preds
                        del embeds
                        # torch.cuda.empty_cache() # COMMENTED OUT FOR DEBUGGING PURPOSES

                        if batch_ind == 3:
                            print("BREAKING FOR DEBUGGING PURPOSES")
                            break # FOR DEBUGGING PURPOSES
                        pass  # end of current val-batch loop

                    pass  # end of validation "with" statement

                # record averages/stats
                avg_combined_train_losses[curr_layer_ind].append(running_train_loss / total_num_train_preds_by_task[requested_tasks[0]]) # note: this is okay because each prediction is made in-parallel in a CDISN ensemble
                for i, task_id in enumerate(requested_tasks):
                    avg_train_losses[curr_layer_ind][task_id].append(
                        running_train_loss_by_task[task_id] / total_num_train_preds_by_task[task_id]
                    )
                    avg_train_accs[curr_layer_ind][task_id].append(
                        num_correct_train_preds_by_task[task_id] / total_num_train_preds_by_task[task_id]
                    )
                    avg_train_roc_aucs[curr_layer_ind][task_id].append(
                        running_train_roc_auc_by_task[task_id] / total_num_train_batches_by_task[task_id]
                    )
                    avg_val_accs[curr_layer_ind][task_id].append(
                        num_correct_val_preds_by_task[task_id] / total_num_val_preds_by_task[task_id]
                    )
                    avg_val_roc_aucs[curr_layer_ind][task_id].append(
                        running_val_roc_auc_by_task[task_id] / total_num_val_batches_by_task[task_id]
                    )
                    
                    for label_id in range(num_classes[i]):
                        if total_num_train_preds_by_task_and_class[task_id][label_id] > 0:
                            avg_train_accs_by_class[curr_layer_ind][task_id][label_id].append(
                                num_correct_train_preds_by_task_and_class[task_id][label_id]
                                / total_num_train_preds_by_task_and_class[task_id][label_id]
                            )
                        else:
                            avg_train_accs_by_class[curr_layer_ind][task_id][label_id].append(-1)
                        if total_num_val_preds_by_task_and_class[task_id][label_id] > 0:
                            avg_val_accs_by_class[curr_layer_ind][task_id][label_id].append(
                                num_correct_val_preds_by_task_and_class[task_id][label_id]
                                / total_num_val_preds_by_task_and_class[task_id][label_id]
                            )
                        else:
                            avg_val_accs_by_class[curr_layer_ind][task_id][label_id].append(-1)

                # check stopping criterion / save model
                if epoch >= 10:
                    avg_incorrect_val_percentage = sum([1. - avg_val_accs[curr_layer_ind][t] for t in requested_tasks])/len(requested_tasks)
                    if avg_incorrect_val_percentage < min_val_inaccuracy:
                        num_evaluations_since_model_saved = 0
                        min_val_inaccuracy = avg_incorrect_val_percentage
                        saved_ensemble = copy.deepcopy(
                            cdisn_model_ensemble
                        )  # see https://docs.python.org/3/library/copy.html
                    else:
                        num_evaluations_since_model_saved += 1
                        if num_evaluations_since_model_saved >= max_evals_after_saving:
                            print(
                                "train.train_layerwise_cdisn_ensemble.train_cdisn_ensemble_layerwise: EARLY STOPPING on epoch ",
                                epoch,
                            )
                            stopped_early = True
                            break

                # save intermediate state_dicts just in case
                if epoch % save_freq == 0:
                    temp_model_save_path = os.path.join(
                        curr_save_dirs_for_layers[curr_layer_ind],
                        "temp_cdisn_model_ensemble_epoch" + str(epoch) + ".pkl",
                    )
                    with open(temp_model_save_path, "wb") as outfile:
                        pkl.dump(saved_ensemble, outfile)

                    for i, task_id in enumerate(requested_tasks):
                        plot_cdisn_ensemble_training_avgs(
                            avg_train_losses[curr_layer_ind][task_id],
                            avg_train_accs[curr_layer_ind][task_id],
                            avg_val_accs[curr_layer_ind][task_id],
                            avg_train_accs_by_class[curr_layer_ind][task_id],
                            avg_val_accs_by_class[curr_layer_ind][task_id],
                            avg_train_roc_aucs[curr_layer_ind][task_id],
                            avg_val_roc_aucs[curr_layer_ind][task_id],
                            "epoch" + str(epoch) + "task" + str(task_id),
                            curr_save_dirs_for_layers[curr_layer_ind],
                        )

                if epoch == 10:
                    print("BREAKING FOR DEBUGGING PURPOSES")
                    break # FOR DEBUGGING PURPOSES
                pass  # end of current epoch iteration

            # SAVE FINAL TRAINED MODELS
            if stopped_early:
                cdisn_model_ensemble = saved_ensemble  # cdisn_model_ensemble.load_state_dict(saved_ensemble)

            ensemble_save_path = os.path.join(
                curr_save_dirs_for_layers[curr_layer_ind],
                models_root_file_name + "_cdisn_model_ensemble_vFinal.pkl",
            )
            with open(ensemble_save_path, "wb") as outfile:
                pkl.dump(cdisn_model_ensemble, outfile)

            # LOOP THROUGH TESTING FILES
            with torch.no_grad():
                for model in cdisn_model_ensemble:
                    model.eval()
                
                num_correct_test_preds_by_task = {x:0 for x in requested_tasks}
                total_num_test_preds_by_task = {x:0 for x in requested_tasks}
                num_correct_test_preds_by_task_and_class = {x:{i: 0 for i in range(num_classes[x_ind])} for x_ind, x in enumerate(requested_tasks)}
                total_num_test_preds_by_task_and_class = {x:{i: 0 for i in range(num_classes[x_ind])} for x_ind, x in enumerate(requested_tasks)}
                running_test_pred_record_by_task = {x:[] for x in requested_tasks}
                running_test_label_record_by_task = {x:[] for x in requested_tasks}

                # LOOP THROUGH BATCHES IN CURR_TEST_FILE
                for batch_ind, full_sample in enumerate(curr_test_loader):
                    # TEST
                    # transfer to GPU
                    # curr_sample = list(full_sample)
                    curr_sample = extract_necessary_elements_from_layerwise_sample(full_sample, samp_type, requested_tasks)
                    for item_index, item in enumerate(curr_sample):
                        # curr_sample[item_index] = item.to(device)
                        if item_index < len(curr_sample)-3:
                            curr_sample[item_index] = item.float().view(-1, window_len, num_channels).to(device)
                        else:
                            curr_sample[item_index] = item.float().view(-1).to(device)

                    # make predictions
                    preds, embeds = cdisn_model_ensemble[0](samp_type, [curr_sample[:-1], requested_tasks, cdisn_model_ensemble[1:]])
                    # print("preds after model forward pass == ", preds)
                    # print("curr_sample[-1].long().view(-1) == ", curr_sample[-1].long().view(-1))

                    # compute accuracy/stats
                    # preds = (
                    #     preds.detach()
                    # )  # https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
                    # print("preds after loss update and detach == ", preds)

                    for i, task_id in enumerate(requested_tasks):
                        curr_task_pred_ind = supported_tasks.index(task_id)
                        (
                            new_num_correct,
                            new_num_total,
                            new_pred_stats_by_class,
                        ) = get_accuracy_numerator_and_denominator_for_cdisn_preds(
                            preds[curr_task_pred_ind].detach(), curr_sample[-(3-curr_task_pred_ind)] # curr_sample[-1][curr_task_pred_ind]
                        )
                        num_correct_test_preds_by_task[task_id] += new_num_correct
                        total_num_test_preds_by_task[task_id] += new_num_total
                        for label_id in new_pred_stats_by_class.keys():
                            num_correct_test_preds_by_task_and_class[task_id][label_id] += new_pred_stats_by_class[label_id]["num_correct"]
                            total_num_test_preds_by_task_and_class[task_id][label_id] += new_pred_stats_by_class[label_id]["num_total"]
                            pass

                        # track roc-auc score
                        running_test_pred_record_by_task[task_id].append(highest_weighted_preds) # see https://discuss.pytorch.org/t/what-is-the-cpu-in-pytorch/15007
                        # running_test_label_record_by_task[task_id].append(curr_sample[-1][curr_task_pred_ind].cpu())
                        running_test_label_record_by_task[task_id].append(curr_sample[-(3-curr_task_pred_ind)].cpu())
                        pass

                    # free up cuda memory
                    del curr_sample
                    del preds
                    del embeds
                    # torch.cuda.empty_cache() # COMMENTED OUT FOR DEBUGGING PURPOSES

                    if batch_ind == 3:
                        print("BREAKING FOR DEBUGGING PURPOSES")
                        break # FOR DEBUGGING PURPOSES
                    pass  # end of current test-batch loop

                pass  # end of testing "with" statement

            # Compute Test Accuracies/Stats
            for i, task_id in enumerate(requested_tasks):
                avg_test_acc = num_correct_test_preds_by_task[task_id] / total_num_test_preds_by_task[task_id]
                avg_test_accs[curr_layer_ind][task_id] = avg_test_acc
                historical_avg_test_accs[task_id].append(avg_test_acc)
                for label_id in range(num_classes[task_id]):
                    if total_num_test_preds_by_task_and_class[task_id][label_id] == 0:
                        print(
                            "WARNING: CLASS ",
                            label_id,
                            " MISSING FROM TEST CASES - SETTING ACCURACY TO -1 FOR THIS CLASS TO DENOTE NO TEST MADE.",
                        )
                        num_correct_test_preds_by_task_and_class[task_id][label_id] = 1
                        total_num_test_preds_by_task_and_class[task_id][label_id] = -1
                    if total_num_test_preds_by_task_and_class[task_id][label_id] > 0:
                        historical_avg_test_accs_by_class[task_id][label_id].append(
                            num_correct_test_preds_by_task_and_class[task_id][label_id]
                            / total_num_test_preds_by_task_and_class[task_id][label_id]
                        )
                    else:
                        historical_avg_test_accs_by_class[task_id][label_id].append(-1)

                # print("<<< GETTING TEST ROC METRICS >>>")
                # print("torch.stack(running_test_pred_record).view(-1).size() == ", torch.stack(running_test_pred_record).view(-1).size())
                # print("torch.stack(running_test_label_record).view(-1).size() == ", torch.stack(running_test_label_record).view(-1).size())
                # print("torch.stack(running_test_pred_record).view(-1) == ", torch.stack(running_test_pred_record).view(-1))
                # print("torch.stack(running_test_label_record).view(-1) == ", torch.stack(running_test_label_record).view(-1))
                test_fpr, test_tpr, test_roc_auc = get_roc_metrics_from_preds(
                    torch.stack(running_test_pred_record_by_task[task_id]).view(-1),
                    torch.stack(running_test_label_record_by_task[task_id]).view(-1),
                )
                test_roc_metrics[curr_layer_ind][task_id]["fpr"] = test_fpr
                test_roc_metrics[curr_layer_ind][task_id]["tpr"] = test_tpr
                test_roc_metrics[curr_layer_ind][task_id]["roc_auc"] = test_roc_auc
                historical_test_roc_aucs[task_id].append(test_roc_auc)

            # Store current training stats / hyper-params
            meta_data_save_path = os.path.join(
                curr_save_dirs_for_layers[curr_layer_ind],
                "meta_data_and_hyper_parameters.pkl",
            )
            with open(meta_data_save_path, "wb") as outfile:
                pkl.dump(
                    {
                        "retrain_iter_num": retrain_iter_num,
                        "curr_layer_ind": curr_layer_ind,
                        "avg_train_losses": avg_train_losses[curr_layer_ind],
                        "avg_train_accs": avg_train_accs[curr_layer_ind],
                        "avg_train_roc_aucs": avg_train_roc_aucs[curr_layer_ind],
                        "avg_val_accs": avg_val_accs[curr_layer_ind],
                        "avg_val_roc_aucs": avg_val_roc_aucs[curr_layer_ind],
                        "avg_train_accs_by_class": avg_train_accs_by_class[
                            curr_layer_ind
                        ],
                        "avg_val_accs_by_class": avg_val_accs_by_class[curr_layer_ind],
                        "avg_test_accs": avg_test_accs[curr_layer_ind],
                        "test_roc_metrics": test_roc_metrics[curr_layer_ind],
                        "save_dir_for_models": curr_save_dirs_for_layers[curr_layer_ind],
                        "root_data_dir": root_data_dir,  # list of datasets for each task (ordered according to requested_tasks)
                        "train_set_foldername": train_set_foldername,
                        "val_set_foldername": val_set_foldername,
                        "test_set_foldername": test_set_foldername,
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
                        "pretrained_upstream_cdisn_file_paths": pretrained_upstream_cdisn_file_paths, 
                        "max_data_samp_ratios_by_split": max_data_samp_ratios_by_split, 
                        "embedder_type": embedder_type,
                    },
                    outfile,
                )
                pass  # end of meta-data saving 'with' statement
            
            for i, task_id in enumerate(requested_tasks):
                plot_cdisn_ensemble_training_avgs(
                    avg_train_losses[curr_layer_ind][task_id],
                    avg_train_accs[curr_layer_ind][task_id],
                    avg_val_accs[curr_layer_ind][task_id],
                    avg_train_accs_by_class[curr_layer_ind][task_id],
                    avg_val_accs_by_class[curr_layer_ind][task_id],
                    avg_train_roc_aucs[curr_layer_ind][task_id],
                    avg_val_roc_aucs[curr_layer_ind][task_id],
                    "final_train_val_"+str(task_id),
                    curr_save_dirs_for_layers[curr_layer_ind],
                )

                plot_roc_auc_curve(
                    test_roc_metrics[curr_layer_ind][task_id]["fpr"], # test_fpr,
                    test_roc_metrics[curr_layer_ind][task_id]["tpr"], # test_tpr,
                    test_roc_metrics[curr_layer_ind][task_id]["roc_auc"], # test_roc_auc,
                    "final_test_"+str(task_id),
                    curr_save_dirs_for_layers[curr_layer_ind],
                )

            if curr_layer_ind > 0:
                print("BREAKING FOR DEBUGGING PURPOSES")
                break # FOR DEBUGGING PURPOSES
            pass  # end of current layer retraining iteration

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
        
        # for layer_id in range(NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN):
        plot_cdisn_ensemble_retrain_summary(
            avg_train_losses[-1],
            avg_train_accs[-1],
            avg_val_accs[-1],
            avg_train_roc_aucs[-1],
            avg_val_roc_aucs[-1],
            test_roc_metrics[-1],
            "retrainIteration" + str(retrain_iter_num),
            save_dir,
        )

        if (
            retrain_iter_num == max_num_retrain_iterations - 1
            and curr_layer_ind == NUM_PSI_LAYERS_IN_SHALLOWNET_BASED_CDISN - 1
        ):
            final_test_roc_metrics = test_roc_metrics

        print("BREAKING FOR DEBUGGING PURPOSES")
        break # FOR DEBUGGING PURPOSES
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
                        "root_data_dir": root_data_dir,
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


