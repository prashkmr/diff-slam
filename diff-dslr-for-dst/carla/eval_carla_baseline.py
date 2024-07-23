#This is the code where we evaluate our method/model on CARLA on EMD/Chamfer / Result is 210 ans 1.

import argparse
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, Dataset
import torch
import sys
from torchsummary import summary
import numpy as np
import os
# import __init__
# from emd import EMD
from utils512 import * 
# from models256 import * 
# from model import PointNet, DGCNN
from models512 import *

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=128,            help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--ae_weight',          type=str,   default='',             help='Location of the weights')
parser.add_argument('--data',               type=str,   default='',             help='Loction of the dataset')
parser.add_argument('--dim',                type=int,   default=8,             help='Location of the weights')
parser.add_argument('--beam',               type=int,   default=16,             help='Loction of the dataset')
parser.add_argument('--emb_dims', type=int,          default=1024, metavar='N', help='Dimension of embeddings')


parser.add_argument('--debug', action='store_true')


'''
Expect two arguments: 
    1) path_to_model_folder
    2) epoch of model you wish to load
    3) metric to evaluate on 
e.g. python eval.py runs/test_baseline 149 emd
'''

#---------------------------------------------------------------
#Helper Function and classes
class Pairdata(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self,pairDynamic, pairStatic):
        super(Pairdata, self).__init__()
        
        self.pairDynamic       = pairDynamic
        self.pairStatic      = pairStatic

    def __len__(self):
        return self.pairDynamic.shape[0]

    def __getitem__(self, index):
        
        return index, self.pairDynamic[index], self.pairStatic[index]

#-------------------------------------------------------------------------------
args = parser.parse_args()



# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200
# out_dir = os.path.join(sys.argv[1], 'final_samples')
# maybe_create_dir(out_dir)
save_test_dataset = False

fast = True

# fetch metric
# # if 'emd' in sys.argv[3]: 
# #     loss = EMD
# # elif 'chamfer' in sys.argv[3]:
# #     loss = get_chamfer_dist
# else:
#     raise ValueError("{} is not a valid metric for point cloud eval. Either \'emd\' or \'chamfer\'"\
#             .format(sys.argv[2]))


model = VAE(args).cuda()

# print(summary(model, input_size=[(2,40,256),(2,40,256)]))
# exit(1)
summary(model, (2,40,128))
model = model.cuda()
network=torch.load(args.ae_weight)

model.load_state_dict(network['gen_dict'])
# model.load_state_dict(network['gen_dict'])
model.eval() 
print('its there')

# loss1 = EMD
# loss_fn1 = loss1()
loss = get_chamfer_dist
# size = 10 if 'emd' in sys.argv[3] else 5
# size = 2

npydata =[8,9,10,11,12,13,14,15]
tot = 0
with torch.no_grad():
    
  for i in npydata:

    # 1) load trained model
    # model = load_model_from_file(sys.argv[1], epoch=int(sys.argv[2]), model='gen')[0]
    # model = VAE(args).cuda()


    # if 'panos' in sys.argv[1] or 'atlas' in sys.argv[1] : model.args.no_polar = 1 
    
    # 2) load data
    # print('test set reconstruction')
    # dataset = np.load('../lidar_generation/kitti_data/lidar_test.npz')
    # if fast: dataset = dataset[:100]
    # dataset_test = preprocess(dataset).astype('float32')

    # dataset preprocessing
    # print('loading Testing data')
    # dataset_train = np.load('../lidar_generation/kitti_data/lidar.npz')
    
    lidarDy    = (np.load(args.data + "d{}.npy".format(str(i)))[:,:,5:45,::args.dim]).astype('float32')
    lidarSt  = (np.load(args.data + "s{}.npy".format(str(i)))[:,:,5:45,::args.dim]).astype('float32')
    # lidarDy    = (np.load(args.data + "{}.npy".format(str(i)))[:,:,5:45,::args.dim]).astype('float32')
    out      = np.ndarray(shape=(lidarDy.shape[0],2,args.beam,int(1024/args.dim)))
    # print("out", lidar.shape)
    # args.batch_size = 160
    test_loader    = Pairdata(lidarDy, lidarSt)
    loader = (torch.utils.data.DataLoader(test_loader, batch_size= args.batch_size,
                        shuffle=False, num_workers=1, drop_last=True)) #False))

    loss_fn = loss()
    # process_input = (lambda x : x) if model.args.no_polar else to_polar
    process_input = from_polar if args.no_polar else lambda x : x
    
    # noisy reconstruction
    for noise in [0]:#0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.][::(2 if fast else 1)]: 
        losses = []
        # losses1 = []
        ind = -1 
        for batch in loader:
            lidar = batch[1].cuda()
            # mask  = batch[2].cuda()
            lidarStat=batch[2].cuda()
       
            recon,_,_  = model(process_input(lidar))
            # print(recon.shape)
            
            ind+=1
            # print(recon.shape)
 
            # end = lidarDy.shape[0] if ((ind+1)*args.batch_size) > (lidarDy.shape[0]) else  ((ind+1)*args.batch_size)
            # print(end)
            # out[ind*args.batch_size:end]   =    recon[:,:,:40].detach().cpu().numpy().reshape(-1, 2, args.beam, int(1024/args.dim))
            losses += [loss_fn(recon[:,:,:args.beam].reshape(-1,2,args.beam,int(512/args.dim)), lidarStat.reshape(-1,2,args.beam,int(512/args.dim)))]

        losses = torch.stack(losses).mean().item()
        # losses1 += [loss_fn1.apply(recon, lidar)]
        # losses1 = torch.stack(losses1).mean().item()
        print('Chamfer Loss for {}: {:.4f}'.format(i, losses))
        tot += losses
        # np.save(str(i)+'.npy', out)
        # print('Saved ', i)
        # print('EMD Loss for {}: {:.4f}'.format(i, losses1))

        del recon, losses

print(tot/len(npydata))