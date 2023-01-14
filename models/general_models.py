# general_models.py: a file containing models widely used by various methods used in this repo, with some adapted from
#                    Jason Stranne's github repository https://github.com/jstranne/mouse_self_supervision
# SEE LICENSE STATEMENT AT THE END OF THE FILE

# dependency import statements
import torch
import torch.nn as nn
import torch.nn.functional as F

# stager net
class StagerNet(nn.Module):
    """
    StagerNet Description:
    - General Purpose: implements the StagerNet architecture cited by Banville et al (see https://arxiv.org/pdf/2007.16104.pdf)
    - Usage:
        * N/A

    *** Notes:
        1) adapted from https://github.com/jstranne/mouse_self_supervision - License in this file may not apply to this class
    """

    def __init__(self, channels, dropout_rate=0.5, embed_dim=100):
        """
        StagerNet.__init__: initializes model weights (both embedding and decoding) and hyperparameters
         - Inputs:
            * channels (int): the number of channels expected in each biosignal input into the self.forward() method
            * dropout_rate (float): the dropout rate to be applied to embeder weights
            * embed_dim (int): the number of features to be output by the embedder
         - Outputs:
            * n/a
         - Usage:
            * anchor_based_ssl_baseline_models.*: use this __init__ function to define their embedder sub-models
        """
        super(StagerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, channels, (1, channels), stride=(1, 1))
        self.conv2 = nn.Conv2d(1, 16, (50, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, (50, 1), stride=(1, 1))
        self.linear1 = nn.Linear(208 * channels, embed_dim)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16)

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        pass

    def forward(self, x):
        """
        StagerNet.forward: defines the forward-pass functionality of the StagerNet model
         - Inputs:
            * x (torch tensor): a batch of biosignals to be embedded/convolved
         - Outputs:
            * out (torch tensor): a torch tensor represented the embedded signals contained in x
         - Usage:
            * not implemented
        """
        # input assumed to be of shape (C,T)=(2,3000)
        x = torch.unsqueeze(x, 1)

        # convolve x with C filters to 1 by T by C
        x = self.conv1(x)
        # permute to (C, T, I)
        x = x.permute(0, 3, 2, 1)

        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, (13, 1)))
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = F.relu(F.max_pool2d(x, (13, 1)))
        x = self.batchnorm2(x)

        x = torch.flatten(x, 1)  # flatten all but batch dim
        x = F.dropout(x, p=self.dropout_rate)
        x = self.linear1(x)
        return x


# shallow net
class ShallowNet(nn.Module):
    """
    ShallowNet Description:
    - General Purpose: implements the ShallowNet architecture cited by Banville et al (see https://arxiv.org/pdf/2007.16104.pdf)
    - Usage:
        * anchor_based_ssl_baseline_models.*: all classes/models in this module use ShallowNet for their embedder sub-models
    """

    def __init__(self, channels=21, dropout_rate=0.5, embed_dim=100):
        """
        ShallowNet.__init__: initializes model weights (both embedding and decoding) and hyperparameters
         - Inputs:
            * channels (int): the number of channels expected in each biosignal input into the self.forward() method
            * dropout_rate (float): the dropout rate to be applied to embeder weights
            * embed_dim (int): the number of features to be output by the embedder
         - Outputs:
            * n/a
         - Usage:
            * anchor_based_ssl_baseline_models.*: use this __init__ function to define their embedder sub-models
        """
        super(ShallowNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, channels, (1, channels), stride=(1, 1))
        self.conv1 = nn.Conv2d(1, 40, (25, 1), stride=(1, 1)) # Temporal Conv
        self.batchnorm1 = nn.BatchNorm2d(40)
        # self.conv2 = nn.Conv2d(1, 16, (50, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(40, 40, (1, 21), stride=(1, 1)) # Spacial Conv
        self.avgPool1 = nn.AvgPool2d((75, 1), stride=(15, 1)) # Mean Pool
        self.linear1 = nn.Linear(1360, embed_dim) # Fully Connected

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        pass

    def forward(self, x):
        """
        StagerNet.forward: defines the forward-pass functionality of the StagerNet model
         - Inputs:
            * x (torch tensor): a batch of biosignals to be embedded/convolved
         - Outputs:
            * out (torch tensor): a torch tensor represented the embedded signals contained in x
         - Usage:
            * not implemented
        """
        # input assumed to be of shape (C,T)=(2,3000)
        x = torch.unsqueeze(x, 1)

        # convolve x with C filters to 1 by T by C
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = torch.square(x)
        x = self.avgPool1(x)
        x = torch.log(x)
        x = torch.flatten(x, 1)  # flatten all but batch dim
        x = F.dropout(x, p=self.dropout_rate)
        x = self.linear1(x)
        return x


class DownstreamNet(nn.Module):
    """
    DownstreamNet Description:
    - General Purpose: implements linear regression (LR) for embeddings produced by (optionally multiple) pre-trained embedders
    - Usage:
        * not implemented
    """

    def __init__(self, embedders, classes, embed_dim=100):
        """
        DownstreamNet.__init__: initializes model weights (both embedding and decoding) and hyperparameters
         - Inputs:
            * embedders ((str, nn.Module) tuples): the embedders (nn.Modules) and their type-strings (str) to be used by the DownstreamNet
            * classes (int): the number of classes to make predictions for based on embeddings
            * embed_dim (int): the number of features output by each embedder
         - Outputs:
            * n/a
         - Usage:
            * not implemented
        """
        super(DownstreamNet, self).__init__()
        self.BATCH_DIM_INDEX = 0
        self.EMBED_DIM_INDEX = 1

        self.embedder_types = []
        self.embedders = nn.ModuleList()
        for embedder_type, embedder in embedders:
            if embedder_type not in ["RP", "TS", "CPC", "PS", "SQ", "SA"]:
                raise ValueError(
                    "Embedder type " + str(embedder_type) + " not supported"
                )
            self.embedder_types.append(embedder_type)
            self.embedders.append(embedder)

        self.num_embedders = len(embedders)
        self.linear = nn.Linear(self.num_embedders * embed_dim, classes)
        pass

    def forward(self, x):
        """
        DownstreamNet.forward: defines the forward-pass functionality of the DownstreamNet model
         - Inputs:
            * x (torch tensor): a batch of biosignals to be classified
         - Outputs:
            * out (torch tensor): a torch tensor represented the class predictions for x
         - Usage:
            * not implemented
        """
        x_embeds = []
        for i in range(self.num_embedders):
            embedded_x = None
            if self.embedder_types[i] in ["RP", "TS", "CPC", "SA"]:
                embedded_x = self.embedders[i](x)
            else:
                _, embedded_x = self.embedders[i](x)
            x_embeds.append(embedded_x)
        x = torch.cat(tuple(x_embeds), dim=self.EMBED_DIM_INDEX)
        return self.linear(x)


#