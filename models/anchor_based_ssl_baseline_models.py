# anchor_based_ssl_baseline_models.py: a file containing SSL pretext task baselines presented by Banville et al (https://arxiv.org/pdf/2007.16104.pdf),
#                                      and adapted from Jason Stranne's github repository https://github.com/jstranne/mouse_self_supervision

# dependency import statements
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.general_models import ShallowNet # StagerNet

# rp net for Relative Positioning Task


class RPNet(nn.Module):
    """
    RPNet Description:
    - General Purpose: implements the RP baseline SelfSL pretext task/model presented by Banville et al (see https://arxiv.org/pdf/2007.16104.pdf)
    - Usage:
        * not implemented
    """

    def __init__(self, channels=21, dropout_rate=0.5, embed_dim=100):
        """
        RPNet.__init__: initializes model weights (both embedding and decoding) and hyperparameters
         - Inputs:
            * channels (int): the number of channels expected in each biosignal input into the self.forward() method
            * dropout_rate (float): the dropout rate to be applied to embeder weights
            * embed_dim (int): the number of features to be output by the embedder
         - Outputs:
            * n/a
         - Usage:
            * not implemented
        """
        super(RPNet, self).__init__()
        self.embed_model = ShallowNet( # StagerNet(
            channels, dropout_rate=dropout_rate, embed_dim=embed_dim
        )
        self.linear = nn.Linear(embed_dim, 1)
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim

    def forward(self, x1, x2):
        """
        RPNet.forward: defines the forward-pass functionality of the RP model
         - Inputs:
            * x1 (torch tensor): the 'anchor window' for the RP task
            * x2 (torch tensor): the 'other window' to be compared against the 'anchor' in the RP task
         - Outputs:
            * out (torch 1D tensor): a torch tensor containing a single value scoring how close the x1 and x2 windows are
         - Usage:
            * not implemented
        """
        x1_embedded = self.embed_model(x1)
        x2_embedded = self.embed_model(x2)
        # the torch.abs() is able to emulate the grp function in RP
        out = self.linear(torch.abs(x1_embedded - x2_embedded))
        return out


# ts net for Temporal Shuffling Task
class TSNet(nn.Module):
    """
    TSNet Description:
    - General Purpose: implements the TS baseline SelfSL pretext task/model presented by Banville et al (see https://arxiv.org/pdf/2007.16104.pdf)
    - Usage:
        * not implemented
    """

    def __init__(self, channels=21, dropout_rate=0.5, embed_dim=100):
        """
        TSNet.__init__: initializes model weights (both embedding and decoding) and hyperparameters
         - Inputs:
            * channels (int): the number of channels expected in each biosignal input into the self.forward() method
            * dropout_rate (float): the dropout rate to be applied to embeder weights
            * embed_dim (int): the number of features to be output by the embedder
         - Outputs:
            * n/a
         - Usage:
            * not implemented
        """
        super(TSNet, self).__init__()
        self.embed_model = ShallowNet( # StagerNet(
            channels, dropout_rate=dropout_rate, embed_dim=embed_dim
        )
        self.linear = nn.Linear(2 * embed_dim, 1)
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim

    def forward(self, x1, x2, x3):
        """
        TSNet.forward: defines the forward-pass functionality of the TS model
         - Inputs:
            * x1 (torch tensor): the first 'anchor window' for the TS task
            * x2 (torch tensor): the second 'anchor window' for the TS task
            * x3 (torch tensor): the 'other window' to be compared against the 'anchor' windows in the TS task
         - Outputs:
            * out (torch 1D tensor): a torch tensor containing a single value scoring how likely it is that x3 occurs between x1 and x2
         - Usage:
            * not implemented
        """
        x1_embedded = self.embed_model(x1)
        x2_embedded = self.embed_model(x2)
        x3_embedded = self.embed_model(x3)
        # the torch.abs() is able to emulate the grp function in RP
        out = self.linear(
            torch.cat(
                (
                    torch.abs(x1_embedded - x2_embedded),
                    torch.abs(x2_embedded - x3_embedded),
                ),
                dim=-1,
            )
        )
        return out


# cpc net for Contrastive Predictive Coding Task
class CPCNet(nn.Module):
    """
    CPCNet Description:
    - General Purpose: implements the CPC baseline SelfSL pretext task/model presented by Banville et al (see https://arxiv.org/pdf/2007.16104.pdf)
    - Usage:
        * not implemented
    """

    def __init__(
        self, Np, channels=21, ct_dim=100, h_dim=100, dropout_rate=0.5, embed_dim=100
    ):
        """
        CPCNet.__init__: initializes model weights (both embedding and decoding) and hyperparameters
         - Inputs:
            * Np (int): the number of Bilinear modules used by the CPC decoder
            * channels (int): the number of channels expected in each biosignal input into the self.forward() method
            * ct_dim (int): the number of features to be input into the CPC GRU model
            * h_dim (int): the number of features to be encoded by the hidden layers of the CPC GRU model
            * dropout_rate (float): the dropout rate to be applied to embeder weights
            * embed_dim (int): the number of features to be output by the embedder
         - Outputs:
            * n/a
         - Usage:
            * not implemented
        """
        super(CPCNet, self).__init__()

        self.BATCH_DIM = 0
        self.ENTRY_DIM = 1
        self.PRED_VAL_DIM = 2
        self.NUM_ENTRIES = 16
        self.NUM_PREDS = 11

        self.Np = Np
        self.channels = channels
        self.ct_dim = ct_dim
        self.h_dim = h_dim
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim

        self.embed_model = ShallowNet( # StagerNet(
            channels, dropout_rate=dropout_rate, embed_dim=embed_dim
        )
        self.gru = nn.GRU(ct_dim, h_dim, 1, batch_first=True)
        self.bilinear_list = nn.ModuleList()

        for _ in range(Np):
            self.bilinear_list.append(
                nn.Bilinear(
                    in1_features=h_dim, in2_features=ct_dim, out_features=1, bias=False
                )
            )

    def forward(self, Xc, Xp, Xb):
        """
        CPCNet.forward: defines the forward-pass functionality of the CPC model
         - Inputs:
            * Xc (torch tensor): the 'context windows' for the CPC task
            * Xp (torch tensor): the 'future windows' for the CPC task
            * Xb (torch tensor): the 'random negative windows' for the CPC task (?)
         - Outputs:
            * out (torch 1D tensor): a torch tensor containing a single value scoring how likely it is that Xp occurs after the Xc context (?)
         - Usage:
            * not implemented

        *** Notes:
            1) This function involves some very convoluted reshaping of the inputs:
                Essentially, we need to get a 3d tensor with the indices as (batch, entry, predicted value) for
                positive and negative samples where the positive samples are always in the 0th index of the last
                dimension of the output
            2) in jstranne's original repo, the for-loop had the sample and predicted indices swapped - therefore
                  the naming convention may be off (it may be inappropriate to view 'prediction' indices as actual
                  predictions, for example). Unfortunately, I am unable to clarify this with jstranne at the moment
        """

        # embed and reshape Xb
        Xb = [
            [
                self.embed_model(torch.squeeze(Xb[:, i, j, :, :]))
                for i in range(Xb.shape[1])
            ]
            for j in range(Xb.shape[2])
        ]
        for i in range(len(Xb)):
            Xb[i] = torch.stack(Xb[i])
        Xb = torch.stack(Xb).permute(2, 1, 0, 3)

        # embed and reshape Xc
        Xc = [
            self.embed_model(torch.squeeze(Xc[:, i, :, :])) for i in range(Xc.shape[1])
        ]
        Xc = torch.stack(Xc).permute(1, 0, 2)

        # embed and reshape Xp
        Xp = [
            self.embed_model(torch.squeeze(Xp[:, i, :, :])) for i in range(Xp.shape[1])
        ]
        Xp = torch.stack(Xp).permute(1, 0, 2).unsqueeze(2)

        # combine Xp and Xb tensors
        Xp = torch.cat((Xp, Xb), 2)

        # initialize output tensor
        out = torch.empty(
            [Xb.shape[self.BATCH_DIM], self.NUM_ENTRIES, self.NUM_PREDS],
            dtype=Xp.dtype,
            device=Xp.device,
        )

        # process the inputs to make the final output
        _, hidden = self.gru(Xc)
        hidden = torch.squeeze(hidden)

        for batch in range(Xp.shape[self.BATCH_DIM]):
            for sample in range(Xp.shape[self.ENTRY_DIM]):
                for predicted in range(Xp.shape[self.PRED_VAL_DIM]):
                    out[batch, sample, predicted] = self.bilinear_list[sample](
                        hidden[batch, :], Xp[batch, sample, predicted, :]
                    )

        return out

    def custom_cpc_loss(self, input):
        """
        CPCNet.custom_cpc_loss: Runs a negative log softmax on the first column of the input's last index using the knowledge that this is where
                                all the positive samples are
         - Inputs:
            * input (torch tensor): Input (containing both predictions and answers) should be in the shape [batch, np, nb+1], the first index of
                                    nb+1 being the 'correct' one
         - Outputs:
            * out (torch 1D tensor): a torch tensor containing a single value defining the loss for the current input predictions (?)
         - Usage:
            * not implemented
        """
        NB_DIM = 2
        CORRECT_INDEX = 0
        loss_func = nn.LogSoftmax(dim=NB_DIM)
        log_soft = loss_func(input)[:, :, CORRECT_INDEX]
        return -torch.sum(log_soft)
