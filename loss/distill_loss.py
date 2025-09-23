from re import T
import torch.nn as nn
import torch.nn.functional as F
import torch


class FeatureLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 ):
        super(FeatureLoss, self).__init__()

        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W = preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map / temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
        C_attention = C * F.softmax(channel_map / temp, dim=1)

        return S_attention, C_attention


    def calculate_corrleation(self, feature0, feature1, maxdisp=192//4):
        # global correlation on horizontal direction
        b, c, h, w = feature0.shape

        feature0 = feature0.permute(0, 2, 3, 1)  # [B, H, W, C]
        feature1 = feature1.permute(0, 2, 1, 3)  # [B, H, C, W]

        correlation = torch.matmul(feature0, feature1) / (c ** 0.5)  # [B, H, W, W]

        # mask subsequent positions to make disparity positive
        mask = torch.triu(torch.ones((w, w)), diagonal=1).type_as(feature0)  # [W, W]
        if maxdisp > 0:
            mask = mask + torch.tril(torch.ones((w, w)), diagonal=-maxdisp+1).type_as(feature0) 

        valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)  # [B, H, W, W]

        correlation[~valid_mask] = -1e9

        bs, h, w, w_t = correlation.shape
        phi = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost
        similarity_matrix = torch.cat([correlation, phi.expand(bs, h, w, 1).to(correlation.device)], -1)
        similarity_matrix = torch.cat([similarity_matrix, phi.expand(bs, h, 1, w_t + 1).to(correlation.device)], -2)

        prob = F.softmax(similarity_matrix, dim=-1)  # [B, H, W, W]

        return prob  # feature resolution

    def get_fea_loss(self, preds_S, preds_T, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')
        
        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))

        attn_loss = loss_mse(fea_s, fea_t)

        return attn_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        mask_loss = torch.sum(torch.abs((C_s - C_t))) / len(C_s) + torch.sum(torch.abs((S_s - S_t))) / len(S_s)

        return mask_loss
    
    def get_corrleation_loss(self, preds_S_l, preds_S_r, preds_T_l, preds_T_r):

        left_preds_S, right_preds_S = preds_S_l, preds_S_r
        left_preds_T, right_preds_T = preds_T_l, preds_T_r

        corrleation_S = self.calculate_corrleation(left_preds_S, right_preds_S)
        corrleation_T = self.calculate_corrleation(left_preds_T, right_preds_T)

        b, h, w , _ = corrleation_T.shape
        corrleation_loss = corrleation_T * torch.log(corrleation_T + 1e-4) - corrleation_T * torch.log(corrleation_S + 1e-4)
        corrleation_loss = corrleation_loss.sum().mean() / len(preds_S_l)
        return corrleation_loss

     
    def forward(self,
                preds_S_l, preds_S_r,
                preds_T_l, preds_T_r,
                intra=True, mask=True, inter=False,
                ):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S_l.shape[-2:] == preds_T_l.shape[-2:],'the output dim of teacher and student differ'
        
        N, C, H, W = preds_T_l.shape
        losses = {}

        intra_attn_loss = torch.tensor(0.).cuda()
        intra_mask_loss =  torch.tensor(0.).cuda()
        inter_corrleation_loss = torch.tensor(0.).cuda()

        if intra:
            S_attention_t_l, C_attention_t_l = self.get_attention(preds_T_l, self.temp)
            S_attention_t_r, C_attention_t_r = self.get_attention(preds_T_r, self.temp)

            S_attention_s_l, C_attention_s_l = self.get_attention(preds_S_l, self.temp)
            S_attention_s_r, C_attention_s_r = self.get_attention(preds_S_r, self.temp)

            ### attn loss calculation
            attn_loss_l = self.get_fea_loss(preds_S_l, preds_T_l, 
                           C_attention_s_l, C_attention_t_l, S_attention_s_l, S_attention_t_l)

            attn_loss_r = self.get_fea_loss(preds_S_r, preds_T_r, 
                           C_attention_s_r, C_attention_t_r, S_attention_s_r, S_attention_t_r)
        
            intra_attn_loss = attn_loss_l + attn_loss_r

            ### mask loss calculation
            if mask:
                mask_loss_l = self.get_mask_loss(C_attention_s_l, C_attention_t_l, S_attention_s_l, S_attention_t_l)
                mask_loss_r = self.get_mask_loss(C_attention_s_r, C_attention_t_r, S_attention_s_r, S_attention_t_r)

                intra_mask_loss = mask_loss_l + mask_loss_r

        if inter:
            ### corrleation loss calculation
            inter_corrleation_loss = self.get_corrleation_loss(preds_S_l, preds_S_r, preds_T_l, preds_T_r)

        total_loss = self.alpha_fgd * intra_attn_loss  + \
                    self.beta_fgd * intra_mask_loss + self.gamma_fgd * inter_corrleation_loss
        
        losses ={
            "attn_loss": self.alpha_fgd * intra_attn_loss.item(),
            "mask_loss": self.beta_fgd * intra_mask_loss.item(),
            "corrleation_loss": self.gamma_fgd * inter_corrleation_loss.item(),
            "total_loss": total_loss,
        }
        return losses

