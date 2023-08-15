import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.ratio = args.ratio
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m

        self.hidR = args.hidRNN
        self.GRU1 = nn.GRU(self.m, self.hidR)
        self.residual_window = args.residual_window

        self.h = args.horizon

        self.mask_mat = nn.Parameter(torch.Tensor(self.m, self.m))
        self.adj = data.adj
        # print (self.adj)

        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, self.P)
            self.residual = nn.Linear(self.residual_window, 1);
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

        self.GRU2 = nn.GRU(1, self.hidR, batch_first = True)
        self.PredBeta = nn.Sequential(
                                nn.Linear(self.hidR, 5),
                                # nn.Sigmoid(),
                                nn.ReLU(),
                                nn.Linear(5, 1),
                                nn.Sigmoid(),
                            )

        self.GRU3 = nn.GRU(1, self.hidR, batch_first = True)
        self.PredGamma = nn.Sequential(
                                nn.Linear(self.hidR, 5),
                                # nn.Sigmoid(),
                                nn.ReLU(),
                                nn.Linear(5, 1),
                                nn.Sigmoid(),
                            )


    def forward(self, x):
        # x: batch x window (self.P) x #signal (m)
        xOriginal = x[:,:,:]
        # number of samples in a batch
        b = x.shape[0]

        # location number / location number
        # Nomalize to 0-1 / And sum of row = 0
        masked_adj_soft = F.softmax(self.mask_mat, dim=1)

        # -------- for epi component
        # sparse
        masked_adj_epi = self.adj * masked_adj_soft

        # -------- for deep component
        # sparse
        masked_adj_deep = self.adj * masked_adj_soft
        
        # --------------------------------------------------
        # ------------ calculate epidemiological parameters

        # #batch*location * window * 1
        x_for_Epi = x.permute(0, 2, 1).contiguous().view(-1, self.P, 1)

        # ROut: #batch*location / window / #hidden
        # HiddenBeta: 1 / #batch*location / #hidden
        ROut, HiddenBeta = self.GRU2(x_for_Epi)
        # HiddenBeta = torch.squeeze(HiddenBeta, 0)
        # RoutFinalStep: #batch*location / #hidden
        RoutFinalStep = ROut[:,-1,:]
        Beta = self.PredBeta(RoutFinalStep).view(b, self.m)

        ROut, HiddenGamma = self.GRU3(x_for_Epi)
        RoutFinalStep = ROut[:,-1,:]
        # HiddenGamma = torch.squeeze(HiddenGamma, 0)
        Gamma = self.PredGamma(RoutFinalStep).view(b, self.m)


        BetaDiag = torch.Tensor(b, self.m, self.m)
        GammaDiag = torch.Tensor(b, self.m, self.m)
        
        for batch in range(0, b):
            BetaDiag[batch] = torch.diag(Beta[batch])
            GammaDiag[batch] = torch.diag(Gamma[batch])

        # same for each sample
        masked_adj_diagValue = torch.diag(torch.diagonal(masked_adj_epi))
        W = torch.diag(torch.sum(masked_adj_epi,dim=0))-masked_adj_diagValue
        A = ((masked_adj_epi.T - masked_adj_diagValue) - W).repeat(b, 1).view(b, self.m, self.m)

        tmp1 = (GammaDiag - A)
        tmp1[tmp1>1] = 1

        NextGenerationMatrix = BetaDiag.bmm(tmp1.inverse())
        
        # #sample * 1 * #location
        X_vector_t = xOriginal[:,-1,:].view(b,1,self.m)
        # #sample * #location * #location
        NGMT = (NextGenerationMatrix).permute(0,2,1)
        outputNGMT = NextGenerationMatrix

        # #sample * #location
        y_vector_t = X_vector_t.bmm(NGMT).view(b,self.m)
        EpiOutput = y_vector_t

        # ----------------------------------------
        # CNN
        xTrans = x.matmul(masked_adj_deep)

        # RNN
        # r: window (self.P) x # sample x #Location (m)
        # x: Sample Number * window * Location
        r = xTrans.permute(1, 0, 2).contiguous()

        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # print (r.shape)
        res = self.linear1(r)

        #residual
        if (self.residual_window > 0):
            z = x[:, -self.residual_window:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window);
            z = self.residual(z);
            z = z.view(-1,self.m);
            res = res * self.ratio + z;

        if self.output is not None:
            res = self.output(res).float()

        return res, EpiOutput, Beta, Gamma, outputNGMT

