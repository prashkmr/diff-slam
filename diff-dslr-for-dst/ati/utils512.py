import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import trange

# -------------------------------------------------------------------------
# Handy Utilities
# -------------------------------------------------------------------------
def to_polar_np(velo):
    if len(velo.shape) == 4:
        velo = velo.transpose(1, 2, 3, 0)

    if velo.shape[2] > 4:
        assert velo.shape[0] <= 4
        velo = velo.transpose(1, 2, 0, 3)
        switch=True
    else:
        switch=False
    
    # assumes r x n/r x (3,4) velo
    dist = np.sqrt(velo[:, :, 0] ** 2 + velo[:, :, 1] ** 2)
    # theta = np.arctan2(velo[:, 1], velo[:, 0])
    out = np.stack([dist, velo[:, :, 2]], axis=2)
    
    if switch:
        out = out.transpose(2, 0, 1, 3)

    if len(velo.shape) == 4: 
        out = out.transpose(3, 0, 1, 2)
    
    return out

def to_polar(velo):
    if len(velo.shape) == 4:
        velo = velo.permute(1, 2, 3, 0)

    if velo.shape[2] > 4:
        assert velo.shape[0] <= 4
        velo = velo.permute(1, 2, 0, 3)
        switch=True
    else:
        switch=False
    
    # assumes r x n/r x (3,4) velo
    dist = torch.sqrt(velo[:, :, 0] ** 2 + velo[:, :, 1] ** 2)
    # theta = np.arctan2(velo[:, 1], velo[:, 0])
    out = torch.stack([dist, velo[:, :, 2]], dim=2)
    
    if switch:
        out = out.permute(2, 0, 1, 3)

    if len(velo.shape) == 4: 
        out = out.permute(3, 0, 1, 2)
    
    return out

def from_polar(velo):  #takes a tensor of dimension [batch, 2,40,512], return a tensor of dimension [batch, 3,40,512]
    angles = np.linspace(0, np.pi * 2, velo.shape[-1])
    dist, z = velo[:, 0], velo[:, 1]
    x = torch.Tensor(np.cos(angles)).cuda().unsqueeze(0).unsqueeze(0) * dist
    y = torch.Tensor(np.sin(angles)).cuda().unsqueeze(0).unsqueeze(0) * dist
    out = torch.stack([x,y,z], dim=1)

    return out

def from_polar_np(velo):
    angles = np.linspace(0, np.pi * 2, velo.shape[-1])
    dist, z = velo[:, 0], velo[:, 1]
    x = np.cos(angles) * dist
    y = np.sin(angles) * dist
    out = np.stack([x,y,z], axis=1)
    return out.astype('float32')

def print_and_log_scalar(writer, name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return 
        value = torch.mean(torch.stack(value))
    zeros = 40 - len(name) 
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))
    writer.add_scalar(name, value, write_no)

def log_point_clouds(writer, data, name, step):
    if len(data.shape) == 3:
        data = [data]
    
    out = np.stack([from_polar(x.transpose(1, 2, 0)) for x in \
            data.cpu().data.numpy()])
    out = torch.tensor(out).float()

    for i, cloud in enumerate(out):
        cloud = cloud.view(-1, 3)
        writer.add_embedding(cloud, tag=name + '_%d' % i, global_step=step)

def print_and_save_args(args, path):
    print(args)
    # let's save the args as json to enable easy loading
    import json
    with open(os.path.join(path, 'args.json'), 'w') as f: 
        json.dump(vars(args), f)

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def remove_zeros(pc):
    xx = torch.cuda.FloatTensor(pc)
    if xx.dim() == 3: 
        xx = xx.unsqueeze(0)

    iters = 0
    pad = 2
    ks = 5
    while (xx[:, 0] == 0).sum() > 0 : 
        if iters  > 100:
            raise ValueError()
            ks += 2
            pad += 1
        
        mask = (xx[:, 0] == 0).unsqueeze(1).float()
        out_a = F.max_pool2d(xx[:, 0], ks, padding=pad, stride=1)
        out_b = F.max_pool2d(xx[:, 1], ks, padding=pad, stride=1)
        #out_b_ = (xx[:, 1]).min(dim=-1, keepdim=True)[0].expand_as(out_b)
        #out_b = torch.cat([out_b_[:, :10], out_b[:, 10:]], dim=1)
        out_b = out_b.expand_as(out_a)
        out = torch.stack([out_a, out_b], dim=1)
        mask = (xx[:, 0] == 0).unsqueeze(1)
        mask = mask.float()
        xx = xx * (1 - mask) + (mask) * out
        iters += 1

    return xx.cpu().data.numpy()




def add_noise(batch , noise):

    # This noise is added on the individual (x,y,z) coordinates
    batch_xyz = from_polar(batch)      #convert from polar to x,y,z
    noise_tensor = torch.zeros_like(batch_xyz).normal_(0, noise) 
    noise_tensor = torch.zeros_like(batch_xyz).normal_(0, noise)
    means = batch_xyz.transpose(1,0).reshape((3, -1)).mean(dim=-1)   #get mean across dimesion,x,y,z
    stds  = batch_xyz.transpose(1,0).reshape((3, -1)).std(dim=-1)     #get standard devaition across x,y,z
    means, stds = [x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) for x in [means, stds]]

    # normalize data
    norm_batch_xyz = (batch_xyz - means) / (stds + 1e-9)         #nomralize the x,y,z data with respective mena an std
    # add the noise
    input = norm_batch_xyz + noise_tensor   #add noiser

    # unnormalize
    input = input * (stds + 1e-9) + means    #move back to original dimension

    return to_polar(input)    #convert back to polar andreturn



# Hello

# I read this paper recently and wanted to try my hands on it. Thank you for providing the code for the same.
# I have one doubt with regards to the preprocessing step.

# The preprocessing step should return a tensor of shape (N, 60, 512, 3) but on trying the code, the tensor comes out to be of shape (N, 40, 256, 3)
# Can you clarify whether it's okay to have this output tensor? Also, can you shed some light on the following two lines mentioned in the preprocessing function defined in utils.py

# Line 151: dataset = dataset[:, 5:45]
# Line 180: return dataset[:, :, :, ::2]
# I think that line 151 is reducing the circles or clusters from 60 to 40 and then line 180 is reducing the point per circle/cluster to 256 from 512.

# If it's okay to have this as the output tensor, is there any reasoning for reducing clusters and points per cluster?

# Please help.

# Thanks
# Abhishek

# @pclucas14
 
# Owner
# pclucas14 commented 18 days ago
# Hi Abhishek,

# What you said is correct. I downsampled the input mainly to reduce the computational burden and improve training speed.
# I also discarded the outer circles because they usually are the ones that contain the most noise, and thus are the hardest to model. You can bypass these steps if you want, however you may need to make a few changes in the model architecture.

# Best,
# Lucas



#def randomly_remove_entries(originaldy,prob):
#    mask = np.random.choice([0, 1], size=originaldy.shape, p=[prob, 1 - prob])
#    original_temp = np.multiply(originaldy.detach().cpu() , mask)
#    original_temp = original_temp.cuda()
#    original_temp = original_temp.type(torch.FloatTensor)
#    return(original_temp.cuda())



def randomly_remove_entries(originaldy,prob):
    # print(originaldy.shape)                            #[128,2,40,512]

    #get a mask of 0's and 1's with shape of the original dataset 
    mask      = torch.Tensor(np.random.binomial(size=(originaldy.shape[0], 1, originaldy.shape[2], originaldy.shape[3]), n=1, p= 1 - prob)).cuda()
    #temp_mask = np.random.choice([0, 1], size = (originaldy.shape[0], 1, originaldy.shape[2], originaldy.shape[3]), p = [prob, 1 - prob])
    
    #mulriply thre mask  with the data
    original_temp = originaldy * mask
    original_temp = original_temp
    original_temp = original_temp.type(torch.FloatTensor)
    return(original_temp.cuda())





def preprocess(dataset):
    # remove outliers 
    min_a, max_a = np.percentile(dataset[:, :, :, [0]], 1), np.percentile(dataset[:, :, :, [0]], 99)
    min_b, max_b = np.percentile(dataset[:, :, :, [1]], 1), np.percentile(dataset[:, :, :, [1]], 99)
    min_c, max_c = np.percentile(dataset[:, :, :, [2]], 1), np.percentile(dataset[:, :, :, [2]], 99)

    # print(min_a,max_a,min_b,max_b,min_c,max_c)
    # exit(1)
    # min_a, max_a = -41.1245002746582,   36.833248138427734
    #min_a, max_a = -25.833599090576172, 30.474000930786133
    # min_c, max_c = -2.3989999294281006, 0.7383332848548889
    #changes
    # min_a, max_a = -71.1245002746582,   66.833248138427734
    # min_b, max_b = -55.833599090576172, 60.474000930786133
    # min_c, max_c = -30.3989999294281006, 10.7383332848548889


    #changed, commented this, now at phi instead of 40 getting all 60
    #dataset = dataset[:, 5:45]


    mask = np.maximum(dataset[:, :, :, 0] < min_a, dataset[:, :, :, 0] > max_a)
    # print(mask)
    # raise SystemError
    mask = np.maximum(mask, np.maximum(dataset[:, :, :, 1] < min_b, dataset[:, :, :, 1] > max_b))
    mask = np.maximum(mask, np.maximum(dataset[:, :, :, 2] < min_c, dataset[:, :, :, 2] > max_c))
    
    dist = dataset[:, :, :, 0] ** 2 + dataset[:, :, :, 1] ** 2
    mask = np.maximum(mask, dist < 7)

    dataset = dataset * (1 - np.expand_dims(mask, -1))

    #----------------------------------------------------------------------

    #Fix for an error where np.absolute(dataset).max()   goes to 0 causing failure
    npmaxError=0.0
    if np.absolute(dataset).max()==0.0:
        npmaxError=1.0

    #----------------------------------------------------------------------

    # dataset /= (np.absolute(dataset).max()+npmaxError)
    dataset /= 120

    dataset = to_polar_np(dataset).transpose(0, 3, 1, 2)
    previous = (dataset[:, 0] == 0).sum()

    remove = []
    for i in range(dataset.shape[0]):
        #print('processing {}/{}'.format(i, dataset.shape[0]))
        try:
            pp = remove_zeros(dataset[i]).squeeze(0)
            dataset[i] = pp
        except:
            #print('removing %d' % i)
            remove += [i]

    for i in remove:
        dataset = np.concatenate([dataset[:i-1], dataset[i+1:]], axis=0)
    #print(dataset[:, :, :, ::2].shape)


    #changed,  for every 2 entries it retuend 1 entry thereby reducing the 512 to 256
    # dataset = dataset[:, :, :, ::2]
    return dataset[:, :, :, :]
    # print('In preprocess')
    # print(dataset.shape)

    #get 384 entries from 512 entries
    # listt=np.random.choice(dataset.shape[3],384,replace=False)
    # listt=np.sort(listt)
    # return dataset[:,:,:,listt]
    #return dataset

    


def show_pc(velo, save=0, save_path=None):
    import mayavi.mlab

    fig = mayavi.mlab.figure(size=(1400, 700), bgcolor=(0,0,0)) 

    if len(velo.shape) == 3:
        if velo.shape[0] == 3 : 
            velo = velo.transpose(1,2,0)

        assert velo.shape[2] == 3
        velo = velo.reshape((-1, 3))

    max_ = np.absolute(velo[:, :2]).max()
    nodes = mayavi.mlab.points3d(
        velo[:, 0],   # x
        velo[:, 1],   # y
        velo[:, 2],   # z
        scale_factor=0.008, #0.022,     # scale of the points
        figure=fig) 
    
    nodes.glyph.scale_mode = 'scale_by_vector'
    color = (velo[:, 2] - velo[:, 2].min()) / (velo[:, 2].max() - velo[:, 2].min())
    color = (velo[:, 2] - -0.069667026) / ( 0.0041348818 - -0.069667026)
    
    nodes.mlab_source.dataset.point_data.scalars = color
    print('showing pc')
    aa, bb = -95, -40 #np.random.randint(-105, -85), np.random.randint(-55, -35)
    print(aa, bb)
    mayavi.mlab.view(azimuth=-87, elevation=-40, focalpoint=(0, 0, np.median(velo[:, -1])))
    f = mayavi.mlab.gcf()
    f.scene.camera.zoom(2.7)

    if save:
        print(save)
        mayavi.mlab.savefig('../inter_images_2/{}.png'.format(i))
        mayavi.mlab.close()
    elif save_path is not None:
        mayavi.mlab.savefig(save_path)
        mayavi.mlab.close()
    else:
        mayavi.mlab.show()

def show_pc_lite(velo, ind=1, show=True):
    velo=velo.cpu()
    # print(velo.shape)
    import matplotlib.pyplot as plt
    plt.scatter(velo[:, 0], velo[:, 1], s=0.7, color='k')
    plt.show() 


def to_attr(args_dict):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    return AttrDict(args_dict)


def load_model_from_file(path, epoch, model='dis'):
    from models import netD, netG, VAE
    import json
    with open(os.path.join(path, 'args.json'), 'r') as f: 
        old_args = json.load(f)

    old_args = to_attr(old_args)
    if 'gen' in model.lower():
        try:
            z_ = old_args.z_dim
            model_ = VAE(old_args)
        except:
            z_ = 100
            model_ = netG(old_args, nz=z_, nc= 3 if old_args.no_polar else 2)
    elif 'dis' in model.lower():
        model_ = netD(old_args)
    else: 
        raise ValueError('%s is not a valid model name' % model)

    model_.load_state_dict(torch.load(os.path.join(path, 'models/%s_%d.pth' % (model, epoch))))
    print('model successfully loaded')

    return model_, epoch 


def batch_pairwise_dist(A, B):
    # pa, pb are bs x points x 3
    r_A = (A * A).sum(dim=2, keepdim=True)
    r_B = (B * B).sum(dim=2, keepdim=True)
    m = torch.bmm(A, B.permute(0, 2, 1))
    D = r_A - 2 * m + r_B.permute(0, 2, 1)
    return D

def chamfer_quadratic(a,b):
    D = batch_pairwise_dist(a,b)
    return D.min(dim=-1)[0], D.min(dim=-2)[0]


# Utilities for baseline
def get_chamfer_dist(get_slow=False):
    try:
        if get_slow: raise ValueError

        import sys
        sys.path.insert(0, './nndistance')
        from modules.nnd import NNDModule
        dist = NNDModule()
    except:
        dist = chamfer_quadratic

    def loss(a, b):
        if a.dim() == 4:
            if a.size(1) == 2: 
                a = from_polar(a)

            assert a.size(1) == 3
            a = a.permute(0, 2, 3, 1).contiguous().reshape(a.size(0), -1, 3)
            
        if b.dim() == 4:
            if b.size(1) == 2: 
                b = from_polar(b)

            assert b.size(1) == 3
            b = b.permute(0, 2, 3, 1).contiguous().reshape(b.size(0), -1, 3)

        assert a.dim() == b.dim() == 3
        if a.size(-1) != 3: 
            assert a.size(-2) == 3
            a = a.transpose(-2, -1).contiguous()
        
        if b.size(-1) != 3: 
            assert b.size(-2) == 3
            b = a.transpose(-2, -1).contiguous()

        dist_a, dist_b = dist(a, b)
        return dist_a.sum(dim=-1) + dist_b.sum(dim=-1)

    return loss


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    # check if both chamfer implementations give the same results
    ch_fast = get_chamfer_dist()
    ch_slow = get_chamfer_dist(get_slow=True)

    for _ in range(10):
        x = torch.cuda.FloatTensor(32, 1000, 3).normal_()
        y = torch.cuda.FloatTensor(32, 1000, 3).normal_()
        
        out_fast = ch_fast(x,y)
        out_slow = ch_slow(x,y)

