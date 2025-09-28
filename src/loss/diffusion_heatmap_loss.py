from . import BaseLoss
import torch


class Diffusion_Heatmap_Loss(BaseLoss):
    def __init__(self, args):
        super(Diffusion_Heatmap_Loss, self).__init__(args)

        self.loss_name = []

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output):
        loss_val = []

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue
            if loss_type in ['L1', 'L2', 'Sig']:
                if 'amp_pred' in output.keys():
                    pred = output['amp_pred']
                    gt = output['amp_heatmap']
                    loss_tmp = loss_func(pred, gt)

                    pred = output['phase_pred']
                    gt = output['phase_heatmap']
                    loss_tmp += loss_func(pred, gt)
                else:
                    pred = output['pred']
                    gt = sample['heatmap']
                    loss_tmp = loss_func(pred, gt)
            elif loss_type in ['DDIM']: 
                loss_tmp = output['ddim_loss']
            elif loss_type in ['BIN']: 
                # loss_bindepth = output['bin_losses']['loss_depth']
                # loss_binhamfer = output['bin_losses']['loss_chamfer']
                loss_tmp = 0
                for key, value in output['bin_losses'].items():
                    loss_tmp = loss_tmp + value
            elif loss_type in ['MSIG']:
                loss_tmp = output['msig_loss']
            else:
                raise NotImplementedError

            assert loss_tmp, loss_tmp
            assert loss['weight'], loss['weight']
            loss_tmp = loss['weight'] * loss_tmp

            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        assert not torch.isnan(loss_sum).any()
        assert not torch.isnan(loss_val).any()

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val
