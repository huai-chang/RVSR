import torch
import torch.nn.functional as F
from models.inference_arch import RVSR
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='pretrained/pretrained.ckpt')
    args = parser.parse_args()
    model = RVSR(sr_rate=4, N=16).cuda()
    rep_state_dict = model.state_dict()
    # pretrain_state_dict = torch.load(args.pretrained_path, map_location='cuda')
    pretrain_state_dict = torch.load(args.pretrained_path, 'cuda')['state_dict']
    pretrain_state_dict = {k.replace('network.',''): v for k,v in pretrain_state_dict.items()}

    for k, v in tqdm(rep_state_dict.items()):
        if 'rep_conv.weight' in k:
            k0 = pretrain_state_dict[k.replace('rep', 'expand')]
            k1 = pretrain_state_dict[k.replace('rep', 'fea')]
            k2 = pretrain_state_dict[k.replace('rep', 'reduce')]
            k3 = pretrain_state_dict[k.replace('rep_conv', 'res_conv3x3')]
            k4 = pretrain_state_dict[k.replace('rep_conv', 'res_conv1x1')]
            k5 = pretrain_state_dict[k.replace('rep', 'expand1')]
            k6 = pretrain_state_dict[k.replace('rep', 'fea1')]
            k7 = pretrain_state_dict[k.replace('rep', 'reduce1')]

            bias_str = k.replace('weight', 'bias')
            b0 = pretrain_state_dict[bias_str.replace('rep', 'expand')]
            b1 = pretrain_state_dict[bias_str.replace('rep', 'fea')]
            b2 = pretrain_state_dict[bias_str.replace('rep', 'reduce')]
            b3 = pretrain_state_dict[bias_str.replace('rep_conv', 'res_conv3x3')]
            b4 = pretrain_state_dict[bias_str.replace('rep_conv', 'res_conv1x1')]
            b5 = pretrain_state_dict[bias_str.replace('rep', 'expand1')]
            b6 = pretrain_state_dict[bias_str.replace('rep', 'fea1')]
            b7 = pretrain_state_dict[bias_str.replace('rep', 'reduce1')]

            mid_feats, n_feats = k0.shape[:2]
        
            # merge the firsr conv1x1-conv3x3-conv1x1
            merge_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merge_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3).cuda()
            merge_b0b1 = F.conv2d(input=merge_b0b1, weight=k1, bias=b1)
            merge_k0k1k2 = F.conv2d(input=merge_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merge_b0b1b2 = F.conv2d(input=merge_b0b1, weight=k2, bias=b2).view(-1)

            # merge the second conv1x1-conv3x3-conv1x1
            merge_k5k6 = F.conv2d(input=k6, weight=k5.permute(1, 0, 2, 3))
            merge_b5b6 = b5.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3).cuda()
            merge_b5b6 = F.conv2d(input=merge_b5b6, weight=k6, bias=b6)
            merge_k5k6k7 = F.conv2d(input=merge_k5k6.permute(1, 0, 2, 3), weight=k7).permute(1, 0, 2, 3)
            merge_b5b6b7 = F.conv2d(input=merge_b5b6, weight=k7, bias=b7).view(-1)

            # merge repconv
            merge_repconv_k = merge_k0k1k2 + k3 + F.pad(k4,(1,1,1,1)) + merge_k5k6k7
            merge_repconv_b = merge_b0b1b2 + b3 + b4 + merge_b5b6b7
            
            # remove the global identity
            for i in range(n_feats):
                merge_repconv_k[i, i, 1, 1] += 1.0

            rep_state_dict[k] = merge_repconv_k.float()
            rep_state_dict[bias_str] = merge_repconv_b.float()   

        elif 'rep_conv.bias' in k:
            pass

        elif k in pretrain_state_dict.keys():
            rep_state_dict[k] = pretrain_state_dict[k]

        else:
            raise NotImplementedError('{} is not found in pretrain_state_dict.'.format(k))

    torch.save(rep_state_dict, 'RVSR_rep.pth')
    print('Reparameterize successfully!')
