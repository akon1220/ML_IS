# cdisn_models.py: a file defining classes and functions for CDISN Ensemble training
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle as pkl


class CDISNCompatibleStagerNet(nn.Module):
    """
    In [25]: x_11chan_out = stager_11chan_instance(x_11chan_in)
    - stager: orig x shape ==  torch.Size([1, 3000, 11])
    - stager: after first operation ==  torch.Size([1, 1, 3000, 11])
    - stager: after conv1 ==  torch.Size([1, 11, 3000, 1])
    X stager: after permute1 ==  torch.Size([1, 1, 3000, 11])
    X stager: after conv2 ==  torch.Size([1, 16, 2951, 11])
    - stager: after relu/maxpool 1 ==  torch.Size([1, 16, 227, 11])
    X stager: after batchnorm1 ==  torch.Size([1, 16, 227, 11])
    X stager: after conv3 ==  torch.Size([1, 16, 178, 11])
    - stager: after relu/max_pool 2 ==  torch.Size([1, 16, 13, 11])
    X stager: after batchnorm2 ==  torch.Size([1, 16, 13, 11])
    - stager: after flatten1 ==  torch.Size([1, 2288])
    - stager: after dropout 1 ==  torch.Size([1, 2288])
    X stager: after linear1 ==  torch.Size([1, 100])
    Hidden Representation Set(s)
    Set 1
        torch.Size([1, 1, 3000, 11]) # after permute1
        total_params_in_dim_2 = 3000
    Set 2
        torch.Size([1, 16, 2951, 11]) # after conv2
        torch.Size([1, 16, 227, 11]) # after batchnorm1
        torch.Size([1, 16, 178, 11]) # after conv3
        torch.Size([1, 16, 13, 11]) # after batchnorm2
        total_params_in_dim_2 = 3369
    Set 3
        torch.Size([1, 100]) # after linear 1
        total_params_in_dim_1 = 100
    """

    def __init__(
        self, channels, dropout_rate=0.5, embed_dim=100, num_tandem_nets=2, device="cpu"
    ):
        super(CDISNCompatibleStagerNet, self).__init__()
        self.SELF_REFERENCE_INDEX = 0

        # sanity check
        assert num_tandem_nets > 0

        # see https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
        self.conv1_layer = nn.Conv2d(1, channels, (1, channels), stride=(1, 1))
        self.conv1_update_layers = nn.ModuleList()

        self.conv2_layer = nn.Conv2d(1, 16, (50, 1), stride=(1, 1))
        self.conv2_update_pad_tuple = (
            0,
            0,
            24,
            25,
            0,
            0,
            0,
            0,
        )  # pad by (0, 0), (24, 25), (0, 0), and (0, 0) in reverse-dimensional order
        self.conv2_update_layers = nn.ModuleList()

        self.batchnorm1_layer = nn.BatchNorm2d(16)
        self.batchnorm1_update_pad_tuple = (
            0,
            0,
            14,
            15,
            0,
            0,
            0,
            0,
        )  # pad by (0, 0), (14, 15), (0, 0), and (0, 0)
        self.batchnorm1_update_layers = nn.ModuleList()

        self.conv3_layer = nn.Conv2d(16, 16, (50, 1), stride=(1, 1))
        self.conv3_update_pad_tuple = (
            0,
            0,
            14,
            15,
            0,
            0,
            0,
            0,
        )  # pad by (0, 0), (14, 15), (0, 0), and (0, 0)
        self.conv3_update_layers = nn.ModuleList()

        self.batchnorm2_layer = nn.BatchNorm2d(16)
        self.batchnorm2_update_pad_tuple = (
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
        )  # pad by (0, 0), (1, 1), (0, 0), and (0, 0)
        self.batchnorm2_update_layers = nn.ModuleList()

        self.linear1_layer = nn.Linear(208 * channels, embed_dim)
        self.linear1_update_layers = nn.ModuleList()

        for _ in range(
            num_tandem_nets - 1
        ):  # add psi functions at each layer for all tandem networks (excluding self)
            self.conv1_update_layers.append(
                nn.Conv2d(1, channels, (1, channels), stride=(1, 1))
            )
            self.conv2_update_layers.append(nn.Conv2d(16, 16, (50, 1), stride=(1, 1)))
            self.batchnorm1_update_layers.append(
                nn.Conv2d(16, 16, (30, 1), stride=(1, 1))
            )
            self.conv3_update_layers.append(nn.Conv2d(16, 16, (30, 1), stride=(1, 1)))
            self.batchnorm2_update_layers.append(
                nn.Conv2d(16, 16, (3, 1), stride=(1, 1))
            )
            self.linear1_update_layers.append(nn.Linear(embed_dim, embed_dim))

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        self.num_tandem_nets = num_tandem_nets
        self.device = device

        self.BATCH_DIM = 0
        self.cdisn_layer_output_shapes = [
            [None, 1, 3000, 11],
            [None, 16, 2951, 11],
            [None, 16, 227, 11],
            [None, 16, 178, 11],
            [None, 16, 13, 11],
            [None, 100],
        ]
        pass

    def get_phi_output_for_current_layer(self, z, cdisn_layer_index):
        """
        z: tensor of concatenated hidden layer inputs
        cdisn_layer_index: current layer in the cdisn network (i.e. first conv layer in each net, 2nd batchnorm, etc)
        """
        hiddens = None
        # get phi_i output
        if cdisn_layer_index == 0:
            hiddens = self.conv1_layer(z)
            # permute to (batch_num, C, T, 1)
            hiddens = hiddens.permute(0, 3, 2, 1)
        elif cdisn_layer_index == 1:
            hiddens = self.conv2_layer(z)
        elif cdisn_layer_index == 2:
            hiddens = self.batchnorm1_layer(z)
        elif cdisn_layer_index == 3:
            hiddens = self.conv3_layer(z)
        elif cdisn_layer_index == 4:
            hiddens = self.batchnorm2_layer(z)
        elif cdisn_layer_index == 5:
            hiddens = self.linear1_layer(z)
        else:
            raise ValueError(
                "Unrecognized layer index "
                + str(cdisn_layer_index)
                + " for Stager Net."
            )

        return hiddens

    def get_updates_for_current_hidden_layer(
        self, cdisn_layer_index, other_hiddens, curr_pad_tuple=None
    ):
        """
        cdisn_layer_index: current layer in the cdisn network (i.e. first conv layer in each net, 2nd batchnorm, etc)
        other_hiddens: output of get_phi_output_for_current_layer corresponding to cdisn_layer_index from other networks in CDISN Ensemble
        curr_pad_tuple=None: tuple describing the shape of the padding necessary for current run
        """
        # compute updates psi_{i,k,p}(phi_{p,k}) for z

        # see https://discuss.pytorch.org/t/runtimeerror-element-0-of-variables-does-not-require-grad-and-does-not-have-a-grad-fn/11074
        d_i = (
            torch.from_numpy(
                np.zeros(self.cdisn_layer_output_shapes[cdisn_layer_index])
            )
            .to(self.device)
            .float()
        )
        d_i.requires_grad = True  # see https://pytorch.org/docs/stable/autograd.html

        for j, hidden in enumerate(
            other_hiddens
        ):  # compute updates for nets besides ith net activation in list of hidden activations
            if curr_pad_tuple is not None:
                hidden = F.pad(hidden, curr_pad_tuple, "constant", 0)

            curr_d = None  # will become output of call to curr_psi_functions[j](hidden)
            if cdisn_layer_index == 0:
                curr_d = self.conv1_update_layers[j](hidden)
                curr_d = curr_d.permute(0, 3, 2, 1)
            elif cdisn_layer_index == 1:
                curr_d = self.conv2_update_layers[j](hidden)
            elif cdisn_layer_index == 2:
                curr_d = self.batchnorm1_update_layers[j](hidden)
            elif cdisn_layer_index == 3:
                curr_d = self.conv3_update_layers[j](hidden)
            elif cdisn_layer_index == 4:
                curr_d = self.batchnorm2_update_layers[j](hidden)
            elif cdisn_layer_index == 5:
                curr_d = self.linear1_update_layers[j](hidden)
            else:
                raise ValueError(
                    "Unrecognized layer index "
                    + str(cdisn_layer_index)
                    + " for Stager Net."
                )

            d_i = d_i + curr_d

        return d_i

    def get_updated_hidden_layer_activations(
        self, all_hiddens, cdisn_layer_index, frozen_cdisn_nets, curr_pad_tuple=None
    ):
        for i in range(len(frozen_cdisn_nets) + 1):
            if i == self.SELF_REFERENCE_INDEX:
                all_hiddens[i] = self.get_phi_output_for_current_layer(
                    all_hiddens[i], cdisn_layer_index
                )
            else:
                all_hiddens[i] = frozen_cdisn_nets[
                    i - 1
                ].embedder.get_phi_output_for_current_layer(
                    all_hiddens[i], cdisn_layer_index
                )

        all_updates = []
        for i in range(len(frozen_cdisn_nets) + 1):
            if i == self.SELF_REFERENCE_INDEX:
                all_updates.append(
                    self.get_updates_for_current_hidden_layer(
                        cdisn_layer_index,
                        all_hiddens[:i] + all_hiddens[i + 1 :],
                        curr_pad_tuple=curr_pad_tuple,
                    )
                )
            else:
                all_updates.append(
                    frozen_cdisn_nets[
                        i - 1
                    ].embedder.get_updates_for_current_hidden_layer(
                        cdisn_layer_index,
                        all_hiddens[:i] + all_hiddens[i + 1 :],
                        curr_pad_tuple=curr_pad_tuple,
                    )
                )

        for i, (phi_output, psi_outputs) in enumerate(zip(all_hiddens, all_updates)):
            all_hiddens[i] = phi_output + psi_outputs

        return all_hiddens

    def forward(self, x, frozen_cdisn_nets):
        # input assumed to be of shape (batch_num,window_len,channels)
        print("<<< BEGINNING STAGER FORWARD PASS >>>")
        print("stager: original input x.size() == ", x.size())
        curr_batch_size = x.size()[0]
        for i in range(len(self.cdisn_layer_output_shapes)):
            self.cdisn_layer_output_shapes[i][self.BATCH_DIM] = curr_batch_size
            for j in range(len(frozen_cdisn_nets)):
                frozen_cdisn_nets[j].embedder.cdisn_layer_output_shapes[i][
                    self.BATCH_DIM
                ] = curr_batch_size
        x = torch.unsqueeze(x, 1)
        print("stager: after 1st squeeze x.size() == ", x.size())

        all_hiddens = self.get_updated_hidden_layer_activations(
            [x for _ in range(len(frozen_cdisn_nets) + 1)],
            0,
            frozen_cdisn_nets,
            curr_pad_tuple=None,
        )
        print("stager: after 1st update all_hiddens[0].size() == ", all_hiddens[0].size())
        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens,
            1,
            frozen_cdisn_nets,
            curr_pad_tuple=self.conv2_update_pad_tuple,
        )
        print("stager: after 2nd update all_hiddens[0].size() == ", all_hiddens[0].size())

        for i in range(len(frozen_cdisn_nets) + 1):
            all_hiddens[i] = F.relu(F.max_pool2d(all_hiddens[i], (13, 1)))
        print("stager: after 1st maxpool all_hiddens[0].size() == ", all_hiddens[0].size())

        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens,
            2,
            frozen_cdisn_nets,
            curr_pad_tuple=self.batchnorm1_update_pad_tuple,
        )
        print("stager: after 3rd update all_hiddens[0].size() == ", all_hiddens[0].size())
        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens,
            3,
            frozen_cdisn_nets,
            curr_pad_tuple=self.conv3_update_pad_tuple,
        )
        print("stager: after 4th update all_hiddens[0].size() == ", all_hiddens[0].size())

        for i in range(len(frozen_cdisn_nets) + 1):
            all_hiddens[i] = F.relu(F.max_pool2d(all_hiddens[i], (13, 1)))
        print("stager: after 2nd maxpool all_hiddens[0].size() == ", all_hiddens[0].size())

        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens,
            4,
            frozen_cdisn_nets,
            curr_pad_tuple=self.batchnorm2_update_pad_tuple,
        )
        print("stager: after 5th update all_hiddens[0].size() == ", all_hiddens[0].size())
        for i in range(len(frozen_cdisn_nets) + 1):
            all_hiddens[i] = F.dropout(all_hiddens[i], p=self.dropout_rate)
            all_hiddens[i] = all_hiddens[i].view(
                curr_batch_size, -1
            )  # see https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
        print("stager: after final view all_hiddens[0].size() == ", all_hiddens[0].size())

        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens, 5, frozen_cdisn_nets, curr_pad_tuple=None
        )
        print("stager: after final update all_hiddens[0].size() == ", all_hiddens[0].size())
        print("stager: END OF FORWARD PASS")
        return all_hiddens[self.SELF_REFERENCE_INDEX], all_hiddens


class CDISNCompatibleShallowNet(nn.Module):
    """
    x: (batch_size, 1, 600, 21)
    x: (batch_size, 40, 576, 21)
    x: (batch_size, 40, 576, 1)
    x: (batch_size, 40, 34, 1)
    x: (batch_size, 1360)
    x: (batch_size, 100)
    """
    def __init__(
        self, channels=21, dropout_rate=0.5, embed_dim=100, num_tandem_nets=2, device="cpu"
    ):
        super(CDISNCompatibleShallowNet, self).__init__()
        self.SELF_REFERENCE_INDEX = 0

        # sanity check
        assert num_tandem_nets > 0

        # see https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
        self.conv1_layer = nn.Conv2d(1, 40, (25, 1), stride=(1, 1)) # Temporal Conv
        self.conv1_update_pad_tuple = (
            0,
            0,
            12,
            12,
            0,
            0,
            0,
            0,
        )  # pad by (0, 0), (12, 12), (0, 0), and (0, 0) in reverse-dimensional order
        self.conv1_update_layers = nn.ModuleList()

        self.batchnorm1_layer = nn.BatchNorm2d(40)
        self.batchnorm1_update_pad_tuple = (
            0,
            0,
            12,
            12,
            0,
            0,
            0,
            0,
        )  # pad by (0, 0), (14, 15), (0, 0), and (0, 0)
        self.batchnorm1_update_layers = nn.ModuleList()

        # self.conv2_layer = nn.Conv2d(40, 40, (1, 21), stride=(1, 1)) # Spacial Conv
        # self.conv2_update_pad_tuple = (
        #     10,
        #     10,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        # )  # pad by (0, 0), (24, 25), (0, 0), and (0, 0) in reverse-dimensional order
        self.conv2_layer = nn.Conv2d(40, 40, (1, channels), stride=(1, 1)) # Spacial Conv
        self.conv2_update_pad_tuple = (
            int((channels-1)//2),
            int(((channels-1)//2)+((channels-1)%2)),
            0,
            0,
            0,
            0,
            0,
            0,
        )  # pad by (0, 0), (24, 25), (0, 0), and (0, 0) in reverse-dimensional order
        self.conv2_update_layers = nn.ModuleList()

        self.avgPool1_layer = nn.AvgPool2d((75, 1), stride=(15, 1)) # Mean Pool
        self.avgPool1_update_pad_tuple = (
            0,
            0,
            271,
            271,
            0,
            0,
            0,
            0,
        )  # pad by (0, 0), (14, 15), (0, 0), and (0, 0)
        self.avgPool1_update_layers = nn.ModuleList()

        self.linear1_layer = nn.Linear(1360, embed_dim) # Fully Connected
        self.linear1_update_layers = nn.ModuleList()

        for _ in range(
            num_tandem_nets - 1
        ):  # add psi functions at each layer for all tandem networks (excluding self)
            self.conv1_update_layers.append(
                nn.Conv2d(40, 40, (25, 1), stride=(1, 1))
            )
            self.batchnorm1_update_layers.append(
                nn.Conv2d(40, 40, (25, 1), stride=(1, 1))
            )
            # self.conv2_update_layers.append(nn.Conv2d(40, 40, (1, 21), stride=(1, 1)))
            self.conv2_update_layers.append(nn.Conv2d(40, 40, (1, channels), stride=(1, 1)))
            self.avgPool1_update_layers.append(
                nn.Conv2d(40, 40, (75, 1), stride=(15, 1))
            )
            self.linear1_update_layers.append(nn.Linear(embed_dim, embed_dim))

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        self.num_tandem_nets = num_tandem_nets
        self.device = device

        self.BATCH_DIM = 0
        self.cdisn_layer_output_shapes = [
            [None, 40, 576, channels], # [None, 40, 576, 21],
            [None, 40, 576, channels], # [None, 40, 576, 21],
            [None, 40, 576, 1],
            [None, 40, 34, 1],
            [None, 100],
        ]
        pass

    def unfreeze_psi_update_parameters_for_given_layer(self, k):
        params_to_optimize = []
        
        if k == 0:
            for j in range(len(self.conv1_update_layers)):
                for p in self.conv1_update_layers[j].parameters():
                    p.requires_grad = True
                    params_to_optimize.append(p)
        elif k == 1: 
            for j in range(len(self.batchnorm1_update_layers)):
                for p in self.batchnorm1_update_layers[j].parameters():
                    p.requires_grad = True
                    params_to_optimize.append(p)
        elif k == 2: 
            for j in range(len(self.conv2_update_layers)):
                for p in self.conv2_update_layers[j].parameters():
                    p.requires_grad = True
                    params_to_optimize.append(p)
        elif k == 3: 
            for j in range(len(self.avgPool1_update_layers)):
                for p in self.avgPool1_update_layers[j].parameters():
                    p.requires_grad = True
                    params_to_optimize.append(p)
        elif k == 4: 
            for j in range(len(self.linear1_update_layers)):
                for p in self.linear1_update_layers[j].parameters():
                    p.requires_grad = True
                    params_to_optimize.append(p)
        else:
            raise ValueError("CDISNCompatibleShallowNet.unfreeze_psi_update_parameters_for_given_layer: CDISNCompatibleShallowNet only has 5 layers, meaning requested layer index k=="+str(k)+" is out-of-bounds.")
        
        return params_to_optimize

    def get_phi_output_for_current_layer(self, z, cdisn_layer_index):
        """
        z: tensor of concatenated hidden layer inputs
        cdisn_layer_index: current layer in the cdisn task model (e.g. first conv layer in each net)
        """
        hiddens = None
        # get phi_i output
        if cdisn_layer_index == 0:
            hiddens = self.conv1_layer(z)
        elif cdisn_layer_index == 1:
            hiddens = self.batchnorm1_layer(z)
        elif cdisn_layer_index == 2:
            hiddens = self.conv2_layer(z)
        elif cdisn_layer_index == 3:
            hiddens = self.avgPool1_layer(z)
        elif cdisn_layer_index == 4:
            hiddens = self.linear1_layer(z)
        else:
            raise ValueError(
                "Unrecognized layer index "
                + str(cdisn_layer_index)
                + " for ShallowNet."
            )

        return hiddens

    def get_updates_for_current_hidden_layer(
        self, cdisn_layer_index, other_hiddens, curr_pad_tuple=None
    ):
        """
        cdisn_layer_index: current layer in the cdisn network (e.g., first conv layer in each net)
        other_hiddens: output of get_phi_output_for_current_layer corresponding to cdisn_layer_index from other networks in CDISN Ensemble
        curr_pad_tuple=None: tuple describing the shape of the padding necessary for current run
        """
        # compute updates psi_{i,k,p}(phi_{p,k}) for z

        # see https://discuss.pytorch.org/t/runtimeerror-element-0-of-variables-does-not-require-grad-and-does-not-have-a-grad-fn/11074
        d_i = (
            torch.from_numpy(
                np.zeros(self.cdisn_layer_output_shapes[cdisn_layer_index])
            )
            .to(self.device)
            .float()
        )
        d_i.requires_grad = True  # see https://pytorch.org/docs/stable/autograd.html

        for j, hidden in enumerate(
            other_hiddens
        ):  # compute updates for nets besides ith net activation in list of hidden activations
            if curr_pad_tuple is not None:
                hidden = F.pad(hidden, curr_pad_tuple, "constant", 0)

            curr_d = None  # will become output of call to curr_psi_functions[j](hidden)
            if cdisn_layer_index == 0:
                curr_d = self.conv1_update_layers[j](hidden)
            elif cdisn_layer_index == 1:
                curr_d = self.batchnorm1_update_layers[j](hidden)
            elif cdisn_layer_index == 2:
                curr_d = self.conv2_update_layers[j](hidden)
            elif cdisn_layer_index == 3:
                curr_d = self.avgPool1_update_layers[j](hidden)
            elif cdisn_layer_index == 4:
                curr_d = self.linear1_update_layers[j](hidden)
            else:
                raise ValueError(
                    "Unrecognized layer index "
                    + str(cdisn_layer_index)
                    + " for ShallowNet."
                )

            d_i = d_i + curr_d

        return d_i

    def get_updated_hidden_layer_activations(
        self, all_hiddens, cdisn_layer_index, frozen_cdisn_nets, curr_pad_tuple=None
    ):
        for i in range(len(frozen_cdisn_nets) + 1):
            if i == self.SELF_REFERENCE_INDEX:
                all_hiddens[i] = self.get_phi_output_for_current_layer(
                    all_hiddens[i], cdisn_layer_index
                )
            else:
                all_hiddens[i] = frozen_cdisn_nets[
                    i - 1
                ].embedder.get_phi_output_for_current_layer(
                    all_hiddens[i], cdisn_layer_index
                )

        all_updates = []
        for i in range(len(frozen_cdisn_nets) + 1):
            if i == self.SELF_REFERENCE_INDEX:
                all_updates.append(
                    self.get_updates_for_current_hidden_layer(
                        cdisn_layer_index,
                        all_hiddens[:i] + all_hiddens[i + 1 :],
                        curr_pad_tuple=curr_pad_tuple,
                    )
                )
            else:
                all_updates.append(
                    frozen_cdisn_nets[
                        i - 1
                    ].embedder.get_updates_for_current_hidden_layer(
                        cdisn_layer_index,
                        all_hiddens[:i] + all_hiddens[i + 1 :],
                        curr_pad_tuple=curr_pad_tuple,
                    )
                )

        for i, (phi_output, psi_outputs) in enumerate(zip(all_hiddens, all_updates)):
            all_hiddens[i] = phi_output + psi_outputs

        return all_hiddens

    def forward(self, x, frozen_cdisn_nets):
        # input assumed to be of shape (batch_num,window_len,channels)
        # print("<<< BEGINNING SHALLOWNET FORWARD PASS >>>")
        # print("shallownet: original input x.size() == ", x.size())
        # print("shallownet: original input torch.sum(x) == ", torch.sum(x))
        # print("shallownet: original input x == ", x)
        curr_batch_size = x.size()[0]
        for i in range(len(self.cdisn_layer_output_shapes)):
            self.cdisn_layer_output_shapes[i][self.BATCH_DIM] = curr_batch_size
            for j in range(len(frozen_cdisn_nets)):
                frozen_cdisn_nets[j].embedder.cdisn_layer_output_shapes[i][
                    self.BATCH_DIM
                ] = curr_batch_size
        x = torch.unsqueeze(x, 1)
        # print("shallownet: after 1st squeeze x.size() == ", x.size())
        # print("shallownet: after 1st squeeze torch.sum(x) == ", torch.sum(x))
        # print("shallownet: after 1st squeeze x == ", x)

        all_hiddens = self.get_updated_hidden_layer_activations(
            [x for _ in range(len(frozen_cdisn_nets) + 1)],
            0,
            frozen_cdisn_nets,
            curr_pad_tuple=self.conv1_update_pad_tuple,
        )
        # print("shallownet: after 1st layer conv update all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("shallownet: after 1st layer conv update torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("shallownet: after 1st layer conv update all_hiddens[0] == ", all_hiddens[0])
        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens,
            1,
            frozen_cdisn_nets,
            curr_pad_tuple=self.batchnorm1_update_pad_tuple,
        )
        # print("shallownet: after 2nd layer batchnorm update all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("shallownet: after 2nd layer batchnorm update torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("shallownet: after 2nd layer batchnorm update all_hiddens[0] == ", all_hiddens[0])

        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens,
            2,
            frozen_cdisn_nets,
            curr_pad_tuple=self.conv2_update_pad_tuple,
        )
        # print("shallownet: after 3rd layer conv update and all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("shallownet: after 3rd layer conv update and torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("shallownet: after 3rd layer conv update and all_hiddens[0] == ", all_hiddens[0])

        for i in range(len(frozen_cdisn_nets) + 1):
            all_hiddens[i] = torch.square(all_hiddens[i]) # F.relu(all_hiddens[i])
        # print("shallownet: after square activation all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("shallownet: after square activation torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("shallownet: after square activation all_hiddens[0] == ", all_hiddens[0])

        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens,
            3,
            frozen_cdisn_nets,
            curr_pad_tuple=self.avgPool1_update_pad_tuple,
        )
        # print("shallownet: after 4th layer avg pool update all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("shallownet: after 4th layer avg pool update torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("shallownet: after 4th layer avg pool update all_hiddens[0] == ", all_hiddens[0])

        for i in range(len(frozen_cdisn_nets) + 1):
            all_hiddens[i] =  F.relu(all_hiddens[i]) # torch.log(all_hiddens[i])
        # print("shallownet: after log activation all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("shallownet: after log activation torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("shallownet: after log activation all_hiddens[0] == ", all_hiddens[0])

        for i in range(len(frozen_cdisn_nets) + 1):
            all_hiddens[i] = F.dropout(all_hiddens[i], p=self.dropout_rate)
            all_hiddens[i] = all_hiddens[i].view(
                curr_batch_size, -1
            )  # see https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
        # print("shallownet: after final view all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("shallownet: after final view torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("shallownet: after final view all_hiddens[0] == ", all_hiddens[0])

        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens, 4, frozen_cdisn_nets, curr_pad_tuple=None
        )
        # print("shallownet: after final update all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("shallownet: after final update torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("shallownet: after final update all_hiddens[0] == ", all_hiddens[0])
        # print("shallownet: END OF FORWARD PASS")
        return all_hiddens[self.SELF_REFERENCE_INDEX], all_hiddens


class CDISNCompatibleShallowNetWithCorrelatedMatching(CDISNCompatibleShallowNet):
    """
    x: (batch_size, 1, 600, 21)
    x: (batch_size, 40, 576, 21)
    x: (batch_size, 40, 576, 1)
    x: (batch_size, 40, 34, 1)
    x: (batch_size, 1360)
    x: (batch_size, 100)
    """
    def __init__(
        self, channels=21, dropout_rate=0.5, embed_dim=100, num_tandem_nets=2, device="cpu"
    ):
        super().__init__(
            channels=channels, 
            dropout_rate=dropout_rate, 
            embed_dim=embed_dim, 
            num_tandem_nets=num_tandem_nets, 
            device=device
        )
        pass

    def forward(self, x, adjacent_cdisn_nets): # overwrite CDISNCompatibleShallowNet.forward with different handling of input x
        # input assumed to be of shape [(batch_num,window_len,channels) for _ in range(len(adjacent_cdisn_nets)+1)]
        # print("<<< BEGINNING CORR-SHALLOWNET FORWARD PASS >>>")
        # print("corr-shallownet: original input sizes x_i.size() == ", [x_i.size() for x_i in x])
        # print("corr-shallownet: original input sums torch.sum(x_i) == ", [torch.sum(x_i) for x_i in x])
        # print("corr-shallownet: original input x == ", x)
        curr_batch_size = x[0].size()[0]
        for i in range(len(self.cdisn_layer_output_shapes)):
            self.cdisn_layer_output_shapes[i][self.BATCH_DIM] = curr_batch_size
            for j in range(len(adjacent_cdisn_nets)):
                adjacent_cdisn_nets[j].embedder.cdisn_layer_output_shapes[i][
                    self.BATCH_DIM
                ] = curr_batch_size
        x = [torch.unsqueeze(x_i, 1) for x_i in x]
        # print("corr-shallownet: after 1st squeeze x_i.size() == ", [x_i.size() for x_i in x])
        # print("corr-shallownet: after 1st squeeze torch.sum(x_i) == ", [torch.sum(x_i) for x_i in x])
        # print("corr-shallownet: after 1st squeeze x == ", x)

        all_hiddens = self.get_updated_hidden_layer_activations(
            x, # [x for _ in range(len(adjacent_cdisn_nets) + 1)],
            0,
            adjacent_cdisn_nets,
            curr_pad_tuple=self.conv1_update_pad_tuple,
        )
        # print("corr-shallownet: after 1st layer conv update all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("corr-shallownet: after 1st layer conv update torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("corr-shallownet: after 1st layer conv update all_hiddens[0] == ", all_hiddens[0])
        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens,
            1,
            adjacent_cdisn_nets,
            curr_pad_tuple=self.batchnorm1_update_pad_tuple,
        )
        # print("corr-shallownet: after 2nd layer batchnorm update all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("corr-shallownet: after 2nd layer batchnorm update torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("corr-shallownet: after 2nd layer batchnorm update all_hiddens[0] == ", all_hiddens[0])

        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens,
            2,
            adjacent_cdisn_nets,
            curr_pad_tuple=self.conv2_update_pad_tuple,
        )
        # print("corr-shallownet: after 3rd layer conv update and all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("corr-shallownet: after 3rd layer conv update and torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("corr-shallownet: after 3rd layer conv update and all_hiddens[0] == ", all_hiddens[0])

        for i in range(len(adjacent_cdisn_nets) + 1):
            all_hiddens[i] = torch.square(all_hiddens[i]) # F.relu(all_hiddens[i])
        # print("corr-shallownet: after square activation all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("corr-shallownet: after square activation torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("corr-shallownet: after square activation all_hiddens[0] == ", all_hiddens[0])

        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens,
            3,
            adjacent_cdisn_nets,
            curr_pad_tuple=self.avgPool1_update_pad_tuple,
        )
        # print("corr-shallownet: after 4th layer avg pool update all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("corr-shallownet: after 4th layer avg pool update torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("corr-shallownet: after 4th layer avg pool update all_hiddens[0] == ", all_hiddens[0])

        for i in range(len(adjacent_cdisn_nets) + 1):
            all_hiddens[i] =  F.relu(all_hiddens[i]) # torch.log(all_hiddens[i])
        # print("corr-shallownet: after log activation all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("corr-shallownet: after log activation torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("corr-shallownet: after log activation all_hiddens[0] == ", all_hiddens[0])

        for i in range(len(adjacent_cdisn_nets) + 1):
            all_hiddens[i] = F.dropout(all_hiddens[i], p=self.dropout_rate)
            all_hiddens[i] = all_hiddens[i].view(
                curr_batch_size, -1
            )  # see https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
        # print("corr-shallownet: after final view all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("corr-shallownet: after final view torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("corr-shallownet: after final view all_hiddens[0] == ", all_hiddens[0])

        all_hiddens = self.get_updated_hidden_layer_activations(
            all_hiddens, 4, adjacent_cdisn_nets, curr_pad_tuple=None
        )
        # print("corr-shallownet: after final update all_hiddens[0].size() == ", all_hiddens[0].size())
        # print("corr-shallownet: after final update torch.sum(all_hiddens[0]) == ", torch.sum(all_hiddens[0]))
        # print("corr-shallownet: after final update all_hiddens[0] == ", all_hiddens[0])
        # print("corr-shallownet: END OF FORWARD PASS")
        return all_hiddens[self.SELF_REFERENCE_INDEX], all_hiddens


# rp net for Relative Positioning Task
class CDISNCompatibleRPNetDecoder(nn.Module):
    def __init__(self, embed_dim=100):
        super(CDISNCompatibleRPNetDecoder, self).__init__()
        self.linear = nn.Linear(embed_dim, 2)
        self.embed_dim = embed_dim

    def forward(self, x1, x2):
        # the torch.abs() is able to emulate the grp function in RP
        out = self.linear(torch.abs(x1 - x2))
        return out


# ts net for Temporal Shuffling Task
class CDISNCompatibleTSNetDecoder(nn.Module):
    def __init__(self, embed_dim=100):
        super(CDISNCompatibleTSNetDecoder, self).__init__()
        self.linear = nn.Linear(2 * embed_dim, 2)
        self.embed_dim = embed_dim

    def forward(self, x1, x2, x3):
        # the torch.abs() is able to emulate the grp function in RP
        out = self.linear(torch.cat((torch.abs(x1 - x2), torch.abs(x2 - x3)), dim=-1))
        return out


class CDISNCompatibleLinearDecoder(nn.Module):
    def __init__(self, num_classes, embed_dim=100):
        super(CDISNCompatibleLinearDecoder, self).__init__()
        self.linear = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def forward(self, x):
        out = self.linear(x)
        return out


class FullCDISNTaskModel(nn.Module):
    def __init__(
        self,
        requested_task_id,
        num_tandem_nets,
        channels=22, # 21,
        num_classes=None,
        dropout_rate=0.5,
        embed_dim=100,
        device="cpu",
        embedder_type="ShallowNet",
    ):
        super(FullCDISNTaskModel, self).__init__()
        self.BATCH_DIM = 0

        self.supported_task_ids = [
            "RP",
            "TS",
            "BehavioralTST",
            "BehavioralFluoxetine",
            "BehavioralTUAB",
        ]
        assert requested_task_id in self.supported_task_ids
        self.requested_task_id = requested_task_id

        self.channels = channels
        self.unfrozen_network_id = None
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        if embedder_type == "ShallowNet":
            self.embedder = CDISNCompatibleShallowNet( # used in taskwise and NonAnchored layerwise training
                channels=channels,
                dropout_rate=dropout_rate,
                embed_dim=embed_dim,
                num_tandem_nets=num_tandem_nets,
                device=device,
            )
        elif embedder_type == "CorrelatedShallowNet":
            self.embedder = CDISNCompatibleShallowNetWithCorrelatedMatching( # used in Anchored layerwise training
                channels=channels,
                dropout_rate=dropout_rate,
                embed_dim=embed_dim,
                num_tandem_nets=num_tandem_nets,
                device=device,
            )
        elif embedder_type == "StagerNet": # decremented (used in pre-TUAB experiments)
            self.embedder = CDISNCompatibleStagerNet(
                channels,
                dropout_rate=dropout_rate,
                embed_dim=embed_dim,
                num_tandem_nets=num_tandem_nets,
                device=device,
            )
        else:
            raise NotImplementedError("FullCDISNTaskModel does not currently support the following embedder type: "+str(embedder_type))

        if "Behavioral" in requested_task_id:
            self.decoder = CDISNCompatibleLinearDecoder(
                num_classes, embed_dim=embed_dim
            )
        elif requested_task_id == "RP":
            self.decoder = CDISNCompatibleRPNetDecoder(embed_dim=embed_dim)
        elif requested_task_id == "TS":
            self.decoder = CDISNCompatibleTSNetDecoder(embed_dim=embed_dim)
        else:
            raise ValueError("Unrecognized task_id == " + str(requested_task_id))

        # putting it all together
        self.forward_functions_by_id = {
            "Behavioral": self.behavioral_forward,
            "BehavioralTUAB": self.behavioral_forward,
            "RP": self.rp_forward,
            "TS": self.ts_forward,
            "AnchoredBTUABRPTS": self.anchoredBRPTS_forward,
            "NonAnchoredBTUABRPTS": self.nonanchoredBRPTS_forward,
        }
        pass

    def load_pretrained_upstream_params(self, pretrained_upstream_cdisn_file_path):
        print("load_pretrained_upstream_params: ATTEMPTING TO LOAD WARM-START PARAMS")
        with open(pretrained_upstream_cdisn_file_path, "rb") as infile:
            upstream_cdisn_model_ensemble = pkl.load(infile)
            assert len(upstream_cdisn_model_ensemble) == 1
            self.embedder.load_state_dict(
                upstream_cdisn_model_ensemble[0].embedder.state_dict()
            )  # https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492
        print("load_pretrained_upstream_params: SUCCESSFULLY LOADED WARM-START PARAMS")
        pass

    def forward(self, forward_func_id, forward_inputs):
        return self.forward_functions_by_id[forward_func_id](*forward_inputs)

    def behavioral_forward(self, x, frozen_cdisn_nets):
        _, x_embeddings = self.embedder(x, frozen_cdisn_nets)
        out = self.decoder(x_embeddings[self.embedder.SELF_REFERENCE_INDEX])
        return out, [x_embeddings]

    def rp_forward(self, x1, x2, frozen_cdisn_nets):
        _, x1_embeddings = self.embedder(x1, frozen_cdisn_nets)
        _, x2_embeddings = self.embedder(x2, frozen_cdisn_nets)
        out = self.decoder(
            x1_embeddings[self.embedder.SELF_REFERENCE_INDEX],
            x2_embeddings[self.embedder.SELF_REFERENCE_INDEX],
        )
        return out, [x1_embeddings, x2_embeddings]

    def ts_forward(self, x1, x2, x3, frozen_cdisn_nets):
        _, x1_embeddings = self.embedder(x1, frozen_cdisn_nets)
        _, x2_embeddings = self.embedder(x2, frozen_cdisn_nets)
        _, x3_embeddings = self.embedder(x3, frozen_cdisn_nets)
        out = self.decoder(
            x1_embeddings[self.embedder.SELF_REFERENCE_INDEX],
            x2_embeddings[self.embedder.SELF_REFERENCE_INDEX],
            x3_embeddings[self.embedder.SELF_REFERENCE_INDEX],
        )
        return out, [x1_embeddings, x2_embeddings, x3_embeddings]
    
    def anchoredBRPTS_forward(self, x, requested_tasks, adjacent_cdisn_task_models):
        """
        Inputs:
         - x: list containing an anchor window and one or more of an rp other, ts anchor2, and/or ts other window
         - requested_tasks: a list of task_ids ordered as [self.requested_task_id]+[other_id1, other_id2,...]
         - adjacent_cdisn_task_models: a list of cdisn task models (ordered according to requested_tasks[1:])
        """
        # perform sanity checks
        if len(requested_tasks) == 2:
            if "BehavioralTUAB" in requested_tasks:
                assert requested_tasks[0] == "BehavioralTUAB"
            else:
                assert requested_tasks[0] == "RP"
        elif len(requested_tasks) == 3:
            assert requested_tasks == ["BehavioralTUAB", "RP", "TS"]
        
        # perform forward computation
        if requested_tasks == ["BehavioralTUAB"]:
            pred_label, embeds = self.behavioral_forward([x[0]], adjacent_cdisn_task_models)
            return [pred_label, None, None], [[embeds], None, None]
        elif requested_tasks == ['RP']:
            pred_label, embeds = self.rp_forward([x[0]], [x[1]], adjacent_cdisn_task_models)
            return [None, pred_label, None], [None, [embeds], None]
        elif requested_tasks == ['TS']:
            pred_label, embeds = self.ts_forward([x[0]], [x[1]], [x[2]], adjacent_cdisn_task_models)
            return [None, None, pred_label], [None, None, [embeds]]
        elif requested_tasks == sorted(["BehavioralTUAB", "RP"]):
            assert self.requested_task_id == "BehavioralTUAB"
            assert self.embedder.SELF_REFERENCE_INDEX == 0
            _, embeds = self.embedder(x, adjacent_cdisn_task_models)
            out_behavioral = self.decoder(embeds[self.embedder.SELF_REFERENCE_INDEX])
            out_rp = adjacent_cdisn_task_models.decoder(
                embeds[self.embedder.SELF_REFERENCE_INDEX], 
                embeds[1]
            )
            return [out_behavioral, out_rp, None], [[embeds], [embeds], None]
        elif requested_tasks == sorted(["BehavioralTUAB", "TS"]):
            assert self.requested_task_id == "BehavioralTUAB"
            assert self.embedder.SELF_REFERENCE_INDEX == 0
            _, embeds1 = self.embedder([x[0], x[1]], adjacent_cdisn_task_models)
            _, embeds2 = self.embedder([x[0],x[2]], adjacent_cdisn_task_models)
            behavioral_embeds = (embeds1[self.embedder.SELF_REFERENCE_INDEX] + embeds2[self.embedder.SELF_REFERENCE_INDEX] ) / 2.
            out_behavioral = self.decoder(behavioral_embeds)
            out_ts = adjacent_cdisn_task_models.decoder(
                behavioral_embeds[self.embedder.SELF_REFERENCE_INDEX], 
                embeds1[1], 
                embeds2[1]
            )
            return [out_behavioral, None, out_ts], [[behavioral_embeds], None, [embeds1, embeds2]]
        elif requested_tasks == sorted(["RP", "TS"]):
            assert self.requested_task_id == "RP"
            assert self.embedder.SELF_REFERENCE_INDEX == 0
            _, embeds1 = self.embedder([x[0], x[0]], adjacent_cdisn_task_models)
            _, embeds2 = self.embedder([x[1],x[2]], adjacent_cdisn_task_models)
            anchor_embed = (embeds1[self.embedder.SELF_REFERENCE_INDEX] + embeds1[1] ) / 2.
            out_rp = self.decoder(anchor_embed, embeds2[self.embedder.SELF_REFERENCE_INDEX])
            out_ts = adjacent_cdisn_task_models.decoder(
                anchor_embed, 
                embeds2[self.embedder.SELF_REFERENCE_INDEX], 
                embeds2[1]
            )
            return [None, out_rp, out_ts], [None, [anchor_embed, embeds2], [anchor_embed, embeds2]]
        elif requested_tasks == sorted(["BehavioralTUAB", "RP", "TS"]):
            assert self.requested_task_id == "BehavioralTUAB"
            assert self.embedder.SELF_REFERENCE_INDEX == 0
            _, embeds = self.embedder(x, adjacent_cdisn_task_models)
            out_behavioral = self.decoder(embeds[self.embedder.SELF_REFERENCE_INDEX])
            out_rp = adjacent_cdisn_task_models.decoder(
                embeds[self.embedder.SELF_REFERENCE_INDEX], 
                embeds[1]
            )
            out_ts = adjacent_cdisn_task_models.decoder(
                embeds[self.embedder.SELF_REFERENCE_INDEX], 
                embeds[1], 
                embeds[2]
            )
            return [out_behavioral, out_rp, out_ts], [[embeds], [embeds], [embeds]]
        else:
            raise ValueError("anchoredBRPTS_forward: requested_tasks is not sorted properly, leading to unhandled case")
        pass
    
    def nonanchoredBRPTS_forward(self, x, requested_tasks, adjacent_cdisn_task_models):
        # perform sanity checks
        if len(requested_tasks) == 2:
            if "BehavioralTUAB" in requested_tasks:
                assert requested_tasks[0] == "BehavioralTUAB"
            else:
                assert requested_tasks[0] == "RP"
        elif len(requested_tasks) == 3:
            assert requested_tasks == ["BehavioralTUAB", "RP", "TS"]
        
        # perform forward computation
        if requested_tasks == ["BehavioralTUAB"]:
            pred_label, embeds = self.behavioral_forward(x[0], adjacent_cdisn_task_models)
            return [pred_label, None, None], [[embeds], None, None]
        elif requested_tasks == ['RP']:
            pred_label, embeds = self.rp_forward(x[0], x[1], adjacent_cdisn_task_models)
            return [None, pred_label, None], [None, [embeds], None]
        elif requested_tasks == ['TS']:
            pred_label, embeds = self.ts_forward(x[0], x[1], x[2], adjacent_cdisn_task_models)
            return [None, None, pred_label], [None, None, [embeds]]
        elif requested_tasks == sorted(["BehavioralTUAB", "RP"]):
            assert self.requested_task_id == "BehavioralTUAB"
            assert self.embedder.SELF_REFERENCE_INDEX == 0
            behavioral_label, behavioral_embeds = self.behavioral_forward(x[0], adjacent_cdisn_task_models)
            rp_label, rp_embeds = self.rp_forward(x[1], x[2], adjacent_cdisn_task_models)
            return [behavioral_label, rp_label, None], [[behavioral_embeds], [rp_embeds], None]
        elif requested_tasks == sorted(["BehavioralTUAB", "TS"]):
            assert self.requested_task_id == "BehavioralTUAB"
            assert self.embedder.SELF_REFERENCE_INDEX == 0
            behavioral_label, behavioral_embeds = self.behavioral_forward(x[0], adjacent_cdisn_task_models)
            ts_label, ts_embeds = self.ts_forward(x[1], x[2], x[3], adjacent_cdisn_task_models)
            return [behavioral_label, None, ts_label], [[behavioral_embeds], None, [ts_embeds]]
        elif requested_tasks == sorted(["RP", "TS"]):
            assert self.requested_task_id == "RP"
            assert self.embedder.SELF_REFERENCE_INDEX == 0
            rp_label, rp_embeds = self.rp_forward(x[0], x[1], adjacent_cdisn_task_models)
            ts_label, ts_embeds = self.ts_forward(x[2], x[3], x[4], adjacent_cdisn_task_models)
            return [None, rp_label, ts_label], [None, [rp_embeds], [ts_embeds]]
        elif requested_tasks == sorted(["BehavioralTUAB", "RP", "TS"]):
            assert self.requested_task_id == "BehavioralTUAB"
            assert self.embedder.SELF_REFERENCE_INDEX == 0
            behavioral_label, behavioral_embeds = self.behavioral_forward(x[0], adjacent_cdisn_task_models)
            rp_label, rp_embeds = self.rp_forward(x[1], x[2], adjacent_cdisn_task_models)
            ts_label, ts_embeds = self.ts_forward(x[3], x[4], x[5], adjacent_cdisn_task_models)
            return [behavioral_label, rp_label, ts_label], [[behavioral_embeds], [rp_embeds], [ts_embeds]]
        else:
            raise ValueError("anchoredBRPTS_forward: requested_tasks is not sorted properly, leading to unhandled case")
        pass




