import pandas as pd
import numpy as np
import os
import random
import numpy as np
from copy import deepcopy
from functools import reduce
import multiprocessing
from multiprocessing import Process, Manager
import time
import math
from scipy import signal
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
import pickle
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
from itertools import permutations
from torch.multiprocessing import Pool
from itertools import combinations,permutations,product
import pickle
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
data_path = '/home/cdsw/Data/'
import warnings
warnings.filterwarnings('ignore')


import torch
cuda=None

class EMS():
    '''
    Class to keep track of empty maximal spaces in box.
    self.spaces is a N x 2 x 3 tensor, where N is the current number of empty maximal spaces.
    Each space is reprsented by two coordinate triplets; the first triplet is the coordinates of the corner closest to
    (0,0,0) and the second is the opposite corner.
    '''
    def __init__(self, dims,full_support = False):
        '''
        Input: 
            dims: size of box as list of three numbers
        '''

        self.spaces = torch.tensor([[0,0,0],dims], device=cuda).unsqueeze(0).double()
        self.items = torch.tensor([[0,0,0],[0,0,0]], device=cuda).unsqueeze(0).double()
        self.box = torch.tensor(dims).double()
        self.full_support = full_support
    def cleave(self, item, sort=True):
        '''
        given an item to insert, identify all Maximal Empty Spaces that intersect it and cleave those spaces into
        new maximal empty spaces.
        
        Input: 
            item: 2 x 3 tensor w/ coordinates of the two corners of the item to be inserted
            sort: whether or not to sort resulting list of maximal spaces.
        '''
        if len(item.shape) == 2:
            item = item.unsqueeze(0)
        self.items = torch.cat([self.items, item],dim=0)
#         print('item',item)
#         print('self.items',self.items)
#         print('self.spaces',self.spaces)
        #high coords of item higher than low coords of spaces?
        diff_1 = item[:,1,:] > self.spaces[:,0,:]
#         print('diff_1',diff_1)
        #low coords of item lower than higher coords of spaces?
        diff_2 = item[:,0,:] < self.spaces[:,1,:]
#         print('diff_2',diff_2)
        #intersection occurs if both conditions are met on all 3 axes
        intersect_inds = (diff_1 * diff_2).all(dim=-1)
#         print('intersect_inds',intersect_inds)
        
        intersect_spaces = self.spaces[intersect_inds]
#         print('intersect_spaces',intersect_spaces)
        new_spaces = []
        for dim in range(3):
            low = intersect_spaces.clone()
            high = intersect_spaces.clone()
            
            low[:,1,dim] = item[:,0,dim]
            high[:,0,dim] = item[:,1,dim]
            
            new_spaces = new_spaces + [low, high]
        new_spaces = torch.cat(new_spaces, dim=0)
#         print('new_spaces',new_spaces)
        new_spaces = self.elim_zero(new_spaces)
#         print('new_spaces',new_spaces)
        old_spaces = self.spaces[~intersect_inds]
#         print('old_spaces',old_spaces)
        new_spaces = self.elim_insc(new_spaces, old_spaces)
#         print('new_spaces',new_spaces)
        all_spaces = torch.cat([new_spaces, old_spaces], dim=0)
#         print('all_spaces',all_spaces)
        if sort:
            all_spaces = all_spaces.flip(dims=(-1,)).unique(dim=0).flip(dims=(-1,))
#             all_spaces = all_spaces.unique(dim=0)
        self.spaces = all_spaces
        
        return self.spaces    
    
    def elim_zero(self, new_spaces):
        '''eliinate spaces with volume 0'''
        lengths = new_spaces[:,1,:]  - new_spaces[:,0,:]
        nonzero = (lengths > 0).all(dim=-1)
        return new_spaces[nonzero]
    
    def elim_insc(self, new_spaces, old_spaces):
        '''eliminate new spaces that are completely inscribed within another space (new or existing)'''
        all_spaces_m = torch.cat([new_spaces, old_spaces], dim=0).unsqueeze(0)
        new_spaces_m = new_spaces.unsqueeze(1)
        
        all_spaces_m = all_spaces_m.expand(new_spaces_m.shape[0],-1,-1,-1)
        new_spaces_m = new_spaces_m.expand(-1, all_spaces_m.shape[1],-1,-1)
        
        low = (new_spaces_m[...,0,:] >= all_spaces_m[...,0,:]).all(dim=-1)
        high = (new_spaces_m[...,1,:] <= all_spaces_m[...,1,:]).all(dim=-1)
        both = low * high
        diag = torch.arange(new_spaces_m.shape[0], device=cuda)
        both[diag,diag] = False
        keep = ~(both.any(dim=-1))
        return new_spaces[keep]
        
    def find_space(self, item,packed_items,packed_items_pos):
        '''
        Input:
            Item: length 3 tensor (l, w, h)
        Finds first empty space where item will fit
        '''
        item = item[...,:3]
#         print(self.spaces)
        space_sizes = (self.spaces[:,1,:] - self.spaces[:,0,:])
        
        valid = (space_sizes >= item).all(dim=-1).nonzero()
        if len(valid) == 0:
            return None
        if (self.spaces[valid[0].item()][0][2] == 0) | (self.full_support == False):
            out = self.spaces[valid[0].item()]
        elif (self.spaces[valid[0].item()][0][2] > 0) & (self.full_support == True):
            dim1 = packed_items[:,:3].unsqueeze(1)
            pos1 = packed_items_pos[:,0].unsqueeze(1)
            itemsize = item.unsqueeze(0).unsqueeze(0).clone().double()
            spaces = self.spaces[valid[:,0]].clone()
            pos2 = spaces[:,0].unsqueeze(1)
            dim2 = itemsize.expand(pos2.shape[0],1,3)
            v = self.intersection_volume3(dim1, pos1, dim2, pos2).squeeze(0)
            p1mat = pos1.expand(-1,pos2.shape[0],-1)
            p2mat = pos2.expand(-1,pos1.shape[0],-1).permute(1,0,2)

            z_below = p1mat[...,2] < p2mat[...,2]
            z_condition = v[...,2] == 0
            x_condition = v[...,0] > 0
            y_condition = v[...,1] > 0

            condition_mask = x_condition * y_condition * z_condition * z_below

            masked = v[...,:2] * condition_mask.unsqueeze(-1)
            ground_support_areas = masked.prod(dim=-1).sum(dim=0)

            ground_support_ratios = ground_support_areas / itemsize[...,:2].prod(dim=-1).squeeze()
            satisfied = ground_support_ratios==1
            spaces = spaces[satisfied]
            if spaces.shape[0] > 0:
                out = spaces[0]
            else:
                out = None
        else:
            out = None
        return out
    
    def intersection_volume3(self,dim1, pos1, dim2, pos2):
        '''
        Takes two pairs of tensors of shape N1 x B x 3 and N2 x B x 3
        dim1 and pos1 give the dimensions and positions of the first set of items
        dim2 and pos2 give the dimensions and positions of the second set of items
        returns B x N1 x N2 tensor containing volume of intersections 
        between each item in the first set and each item in the second set
        '''
        if dim1.shape[-1] != 3 or dim2.shape[-1] != 3:
            #print('last dim not 3')
            dim1 = dim1[...,:3]
            dim2 = dim2[...,:3]

        min1 = pos1
        min2 = pos2

        max1 = min1 + dim1
        max2 = min2 + dim2

        len1 = dim1.shape[0]
        len2 = dim2.shape[0]

        inter_dims = []

        #reshape N1 x B x 3 and N2 x B x 3 to N1 x N2 x B x 3 each.

        dim_i = dim1.unsqueeze(0).expand(len2,-1, -1, -1).transpose(0,1)
        dim_j = dim2.unsqueeze(0).expand(len1,-1, -1, -1)

        max_i = max1.unsqueeze(0).expand(len2,-1, -1, -1).transpose(0,1)
        max_j = max2.unsqueeze(0).expand(len1,-1, -1, -1)

        min_i = min1.unsqueeze(0).expand(len2,-1, -1, -1).transpose(0,1)
        min_j = min2.unsqueeze(0).expand(len1,-1, -1, -1)


        #max of obj i - min of obj j
        d1 = max_i - min_j
        #max of obj j - min of obj i
        d2 = max_j - min_i

        #intersection exists if d1 >= 0 and d2 >=0
        #size of intersection given by min(dim_, dim_j, d1, d2)

        inter_dim = torch.min(torch.min(torch.min(dim_i, dim_j), d1), d2)

        #if any of these is negative, no intersection, so clamp negatives to 0
        #inter_dim = torch.clamp(inter_dim, min=0)

        #return torch.prod(inter_dim, dim=-1).permute(-1,0,1)
        return inter_dim.permute(2,0,1,-1)
    
    
    
    

def pack(box, items):
    '''
    Inputs:
        boxes: length 4 tensor (l, w, h, weight capacity)
        items: Nx4 tensor -- N items x (l, w, h, weight capacity)
    Returns:
        ems object with packed items
        indices of items that were packed (some items may not have fit.)
    
    Packs all items into box. Items are assumed to have already been ordered and rotated 
    (and their dimensions should reflect that)
    '''
    items = items.float()
    box = box.float()
    ems = EMS([i for i in box[...,:3]],full_support=False)
    packed_indices = []
    total_weight = 0
    packed_items = torch.tensor([[0,0,0,0]], device=cuda,dtype=torch.double)
    packed_items_pos = torch.tensor([[0,0,0],[0,0,0]], device=cuda,dtype=torch.double).unsqueeze(0)
    for i in range(len(items)):
        item = items[i]
        if total_weight + item[...,3] > box[...,3]:
            continue
        else:
            total_weight = total_weight + item[...,3]

        space = ems.find_space(item,packed_items,packed_items_pos)
        
        if space is None:
            continue
            
        new_origin = space[0]
        new_item = torch.stack([new_origin, new_origin + item[...,:3]])
            
        ems.cleave(new_item)
        packed_indices.append(i)
        packed_items = torch.cat([packed_items,item.clone().double().unsqueeze(0)],dim=0)
        packed_items_pos = torch.cat([packed_items_pos,new_item.clone().unsqueeze(0)])
        if ems.spaces.shape[0] == 0:
#             print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
#             print('ems.spaces',ems.spaces)
            break
    return ems, packed_indices, packed_items, packed_items_pos



def break_list(sublst_ln,lst):
    """
    Divides given list into sublists of given length
    """
    all_parts = len(lst)//sublst_ln
    last_part = len(lst)%sublst_ln
    tmp = []
    for i in range(all_parts):
        tmp.append(lst[i*sublst_ln:(i+1)*sublst_ln])
    if last_part > 0:
        tmp.append(lst[sublst_ln*all_parts:])
    return tmp
def surface_area(dms):
    l,w,h = dms
    return 2*l*w+2*l*h+2*w*h

def uniques(a):
    x, ix = np.unique(a, return_index=True)
    return np.array(a)[sorted(ix)].tolist()


def simulate(inp):
#     inp = pickle.loads(inp)
    batch_id,items,boxes,items_seq,items_ori = inp
    total_items_vol = items[:,:3].prod(1).sum().item()
    items_seq = uniques(items_seq)
    items_ori = items_ori[items_seq,:]
    items = items[torch.LongTensor(items_seq).expand(4,-1).T,items_ori]
    box = boxes[0]
    if len(items_seq) == 0:
        print('new_items_seq',items_seq)
        print('size',items.size())
        print('batch_id',batch_id)
    ems, indices,packed_items, packed_items_pos = pack(box, items)
    mx = packed_items_pos[:,1].max(axis=0)
    x,y,z = tuple(mx.values.tolist())
    
    packed_items_vol = packed_items[:,:3].prod(dim=1).sum().item()
    portion = packed_items_vol/total_items_vol
    
    packed_items_surf = (2*packed_items[:,(0,1)].prod(dim=1) + 2*packed_items[:,(1,2)].prod(dim=1) + 2*packed_items[:,(0,2)].prod(dim=1)).sum().item()
    tot_items_surf = (2*items[:,(0,1)].prod(dim=1) + 2*items[:,(1,2)].prod(dim=1) + 2*items[:,(0,2)].prod(dim=1)).sum().item()
    portion_surf = packed_items_surf/tot_items_surf
    
    
    new_dim = (x,y,z,packed_items[:,3].sum().item())
    if z > 0:
        occupied = packed_items_vol/(x*y*z)
    else:
        occupied = 0
    
#     all_hus_T = torch.tensor([i[0]+[i[1]] for i in all_hus])
    
#     nearest_hu = all_hus_T[0]

    
    cubes_n = len(indices)
    cubes_packed = torch.LongTensor(items_seq)[indices].tolist()
    if len(cubes_packed) > 0:
        nearest_hu = boxes[(boxes >= torch.tensor(new_dim)).all(axis=1)][0]
        reality_score = sum(1/nearest_hu[:3]).sum().item()
        fill_for_nearest_hu = packed_items_vol/nearest_hu[:3].prod().item()
        nearest_hu = nearest_hu[:3].tolist()
        fill_rate = packed_items_vol/box[:3].prod().item()
        fill_rate_for_best_box = packed_items_vol/(x*y*z)
        
    else:
        nearest_hu = []
        reality_score = 0
        fill_for_nearest_hu = 0
        fill_rate = 0
        fill_rate_for_best_box = 0
    items_seq_next = [i for i in items_seq if i not in cubes_packed]
    weights_filled = packed_items[:,3].sum().item()
#     return packed_items_pos,items,cubes_n,occupied,cubes_packed,weights_filled,new_dim,portion,reality_score,fill_for_nearest_hu,portion_surf,inp[0],(nearest_hu.tolist()[:3],nearest_hu.tolist()[3])
    return batch_id,cubes_n,len(items_seq),cubes_packed,indices,items_seq_next,portion,fill_for_nearest_hu,nearest_hu,items.tolist(),box.tolist(),fill_rate,fill_rate_for_best_box,portion

def get_action(logits,max_prob=False):
    if max_prob == True:
        prob, indices = torch.max(logits, 2)
        return indices
    else:
        m = Categorical(logits=logits)
        return m.sample()
    
def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x

class PointerNetwork(nn.Module):
    def __init__(self,MODEL_INPUT_SIZE,hidden_size, weight_size,ITEMS_SEQ_LN):
        super().__init__()
        self.input_size = MODEL_INPUT_SIZE
        self.hidden_size = hidden_size
        self.weight_size = weight_size
        self.ITEMS_SEQ_LN = ITEMS_SEQ_LN
        
        RNN = nn.GRU
        RNNCell = nn.GRUCell

        self.encoder = RNN(hidden_size, hidden_size, batch_first=True)
        self.decoder_items = RNNCell(hidden_size, hidden_size)
        
        self.embedding = nn.Linear(self.input_size,hidden_size)

        self.W1 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.v1 = nn.Linear(weight_size, 1, bias=False)
        
        self.W3 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.W4 = nn.Linear(hidden_size, weight_size, bias=False)
        self.W5 = nn.Linear(hidden_size, weight_size, bias=False)
        self.v2 = nn.Linear(weight_size, 1, bias=False)
        
        self.W6 = nn.Linear(hidden_size,weight_size,bias=False)
        self.W7 = nn.Linear(hidden_size*2,weight_size,bias=False)
        self.v3 = nn.Linear(weight_size,1)
        
        self.W_ori = nn.Linear(hidden_size*3,6)
        
        
        self.W8 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.W9 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.v4 = nn.Linear(weight_size, 1, bias=False)
        
        self.W10 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.W11 = nn.Linear(hidden_size, weight_size, bias=False)
        self.W12 = nn.Linear(hidden_size, weight_size, bias=False)
        self.v5 = nn.Linear(weight_size, 1, bias=False)
        
        

    def forward(self, items):
        batch_size = items.shape[0]
        decoder_seq_len = items.shape[1]
        items = self.embedding(items)
        encoder_output, hc = self.encoder(items)

        # Decoding states initialization
        hidden_items = encoder_output[:, -1, :] #hidden state for decoder is the last timestep's output of encoder 
        decoder_items_input = to_cuda(torch.rand(batch_size, self.hidden_size))   #%%%%%%%%%%%% decoder input research
        
        # Decoding with attention             
        probs_seq = []
        probs_ori = []
        probs_boxes = []
        encoder_output = encoder_output.transpose(1, 0) #Transpose the matrix for mm
        decoder_output_items = torch.empty(batch_size,1,self.hidden_size)
        for i in range(decoder_seq_len):
            hidden_items = self.decoder_items(decoder_items_input, hidden_items) 
            
            if decoder_output_items.shape[1] == 1:
                decoder_output_items = hidden_items.unsqueeze(1)
            else:
                decoder_output_items = torch.cat((decoder_output_items,hidden_items.unsqueeze(1)),dim=1)
                
            # Computing Intra-attention
            sm_intra = torch.tanh(self.W1(decoder_output_items.transpose(1, 0)) + self.W2(hidden_items))
            out_intra = self.v1(sm_intra)
            attnd = torch.log_softmax(out_intra.transpose(0, 1).contiguous(), -1)
            hidden_intra = (attnd*decoder_output_items).sum(dim=1)
            
            # Computing attention
            sm = torch.tanh(self.W3(encoder_output) + self.W4(hidden_items) + self.W5(hidden_intra))
            out_sm = self.v2(sm).squeeze()
#             out = torch.log_softmax(out_sm.transpose(0, 1).contiguous(), -1)
            out = out_sm.transpose(0, 1).contiguous()
            probs_seq.append(out.clone())
            
            # Orientation Probs
            sm_ori = torch.tanh(self.W6(encoder_output) + self.W7(torch.cat((hidden_items.unsqueeze(1),hidden_intra.unsqueeze(1)),dim=2).transpose(1, 0)))
            out_ori = self.v3(sm_ori)
            attnc = torch.log_softmax(out_ori.transpose(0, 1).contiguous(), -1)
            he = (attnc*encoder_output.transpose(0, 1)).sum(dim=1)
            total_attention = torch.cat((he.unsqueeze(1),hidden_items.unsqueeze(1),hidden_intra.unsqueeze(1)),dim=2)
#             out_ori = torch.log_softmax(self.W_ori(total_attention.squeeze()),-1)
            out_ori = self.W_ori(total_attention.squeeze())
            probs_ori.append(out_ori.clone())

        probs_seq = torch.stack(probs_seq, dim=1)
        probs_ori = torch.stack(probs_ori, dim=1)
        return probs_seq,probs_ori
    
    
def rot_func(rot):
    rotation_indx = {0:(0,1,2,3),1:(0,2,1,3),2:(1,0,2,3),3:(1,2,0,3),4:(2,0,1,3),5:(2,1,0,3)}
    return rotation_indx[rot]

rot_func_v = np.vectorize(rot_func)





class Env():
    def __init__(self,BATCH_SIZE,ITEMS_SEQ_LN,INPUT_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.ITEMS_SEQ_LN = ITEMS_SEQ_LN
        self.INPUT_SIZE = INPUT_SIZE
        self.batch_indx = list(range(self.BATCH_SIZE))
        self.expected_items_n = [self.ITEMS_SEQ_LN] * BATCH_SIZE
        self.all_outs = {i:[] for i in range(self.BATCH_SIZE)}
        self.base_indx_items = torch.arange(self.BATCH_SIZE).expand(self.INPUT_SIZE,self.ITEMS_SEQ_LN,self.BATCH_SIZE).T
        self.base_indx_boxes = torch.arange(self.BATCH_SIZE).expand(self.ITEMS_SEQ_LN,self.BATCH_SIZE).T
        self.items_batch_alligned = None
        
    def reset(self):
        self.found = False
        self.items_batch = []
        self.boxes_batch = []
        for j in range(self.BATCH_SIZE):
            found = False
            while not found:
                boxes = torch.from_numpy(np.random.choice(range(300,1010,10),(3,))).expand(self.ITEMS_SEQ_LN,3).type(torch.FloatTensor)
                box_wt_capacities = boxes.sum(1).unsqueeze(1)/50
                boxes = torch.cat((boxes,box_wt_capacities),dim=1)
                cubes = torch.randint(10,300,(self.ITEMS_SEQ_LN,self.INPUT_SIZE-1)) + torch.rand(self.ITEMS_SEQ_LN,self.INPUT_SIZE-1)
                cubes_wts = torch.randint(1,4,(self.ITEMS_SEQ_LN,1)) + torch.rand(self.ITEMS_SEQ_LN,1)
                cubes = torch.cat((cubes,cubes_wts),dim=1)
                lookup_sum = boxes.sort(dim=1,descending=True)[0].expand(self.ITEMS_SEQ_LN,self.ITEMS_SEQ_LN,self.INPUT_SIZE) - cubes.sort(dim=1,descending=True)[0].unsqueeze(1)
                found = (lookup_sum >= 0).all(2).any(1).all().item()
                if found == True:
                    self.items_batch.append(cubes)
                    self.boxes_batch.append(boxes)
        self.items_batch = torch.stack(self.items_batch,0)
        self.boxes_batch = torch.stack(self.boxes_batch,0)
        
        self.batch_indx = list(range(self.BATCH_SIZE))
        self.expected_items_n = [self.ITEMS_SEQ_LN] * BATCH_SIZE
        self.all_outs = {i:[] for i in range(self.BATCH_SIZE)}
        self.current_level = 0
        self.items_batch_alligned = None
        self.boxes_batch_alligned = None
        
        return self.items_batch,self.boxes_batch
        
        
        self.base_indx_items = torch.arange(BATCH_SIZE).expand(self.INPUT_SIZE,self.ITEMS_SEQ_LN,self.BATCH_SIZE).T
        self.base_indx_boxes = torch.arange(BATCH_SIZE).expand(self.ITEMS_SEQ_LN,self.BATCH_SIZE).T
#         self.op = None
    
    
    
    
    def target_func(self,inputs):
        result = simulate(inputs)
        return pickle.dumps(result)
    
    def dict_update(self,data):
        key,value = data[0],data[1:]
        self.all_outs[key].append(value)
        
    def calc_reward(self,items_list,key,filled_items_indx,filled_items_HUs):
        if len(self.all_outs[key]) > 0:
            packed_items = items_list[filled_items_indx[key]]
            if packed_items.size()[0] > 0:
                total_vol = items_list[:,:3].prod(dim=1).sum().item()
                packed_vol = packed_items[:,:3].prod(dim=1).sum().item()

                total_surf = (2*items_list[:,(0,1)].prod(dim=1) + 2*items_list[:,(1,2)].prod(dim=1) + 2*items_list[:,(0,2)].prod(dim=1)).sum().item()
                packed_surf = (2*packed_items[:,(0,1)].prod(dim=1) + 2*packed_items[:,(1,2)].prod(dim=1) + 2*packed_items[:,(0,2)].prod(dim=1)).sum().item()

                portion_vol = packed_vol/total_vol
                portion_surf = packed_surf/total_surf

                used_boxes = torch.Tensor(filled_items_HUs[key])
                total_boxes_vol = used_boxes[:,:3].prod(dim=1).sum().item()
                total_boxes_surf = (2*used_boxes[:,(0,1)].prod(dim=1) + 2*used_boxes[:,(1,2)].prod(dim=1) + 2*used_boxes[:,(0,2)].prod(dim=1)).sum().item()
                return (packed_vol/total_boxes_vol)#*portion_vol  + (packed_surf/total_boxes_surf)*portion_surf
            else:
                return 0
        else:
            return 0
        
        
    def step(self,items_seq,items_ori,items_batch=None,boxes_batch=None):
        if (items_batch != None) & (boxes_batch != None):
            self.items_batch = items_batch
            self.boxes_batch = boxes_batch
            self.batch_indx = list(range(self.BATCH_SIZE))
            self.expected_items_n = [self.ITEMS_SEQ_LN] * BATCH_SIZE
            self.all_outs = {i:[] for i in range(self.BATCH_SIZE)}
            self.current_level = 0
            self.items_batch_alligned = None
            self.boxes_batch_alligned = None
        
        items_seq_ = torch.LongTensor(items_seq).transpose(1,0).expand(self.INPUT_SIZE,self.ITEMS_SEQ_LN,self.BATCH_SIZE).transpose(2,0)
        items_ori_ = items_ori[torch.arange(self.BATCH_SIZE).expand(self.ITEMS_SEQ_LN,self.BATCH_SIZE).transpose(1,0),torch.LongTensor(items_seq).expand(self.BATCH_SIZE,self.ITEMS_SEQ_LN)]
        self.items_batch_alligned = self.items_batch[[self.base_indx_items,items_seq_,items_ori_]]
        lookup_sm = self.boxes_batch.expand(self.ITEMS_SEQ_LN,self.BATCH_SIZE,self.ITEMS_SEQ_LN,self.INPUT_SIZE).transpose(1,0) - self.items_batch_alligned.unsqueeze(2)
        validities = (lookup_sm >= 0).all(3).any(2).tolist()
        new_seq = []
        for i,j in zip(items_seq,validities):
            new_seq.append([i[k] for k in range(len(i)) if j[k] == True])
        self.batch_indx = [i for i in self.batch_indx if len(new_seq[i]) > 0]
        items_seq = [i for i in new_seq if len(i) > 0]
        
        
        zp = list(zip(self.batch_indx,self.items_batch[self.batch_indx],self.boxes_batch[self.batch_indx],items_seq,items_ori[self.batch_indx]))
        p = Pool(10)
        out = p.map(self.target_func,zp)
        p.close()
        p.join()
        out = [pickle.loads(i) for i in out]

        out_series = pd.Series(out)
        _ = out_series.apply(lambda x: self.dict_update(x))
#         out = [i for i in out if i[1] < i[2]]

        self.batch_indx = [i[0] for i in out]

        self.current_level += 1

        items_seq = [i[5] for i in out]
        all_rewards = [i[-1]*i[-2] for i in out]
            
#         filled_items_indx = {i:[i[2] for i in j] for i,j in self.all_outs.items() if len(j) > 0}
#         filled_items_HUs = {i:[i[7] for i in j if len(i[7]) > 0] for i,j in self.all_outs.items()}
#         all_rewards = [self.calc_reward(self.items_batch[i],i,filled_items_indx,filled_items_HUs) for i in range(self.BATCH_SIZE)]
        return all_rewards
    
    
    
gamma = 0.99
lamda = 0.95
clip_ratio = 0.2
lr_pi = 1e-4
target_kl = 0.01
train_pi_iter = 2
accs = []
EPOCH = 2000
BATCH_SIZE = 128
INPUT_SIZE = 4
HIDDEN_SIZE = 256
WEIGHT_SIZE = 10
ITEMS_SEQ_LN = 10
MODEL_INPUT_SIZE = 8
alpha = 0.5


ptr_net = PointerNetwork(MODEL_INPUT_SIZE,HIDDEN_SIZE,WEIGHT_SIZE,ITEMS_SEQ_LN)
if torch.cuda.is_available():
    ptr_net.cuda()
optim_actor = torch.optim.Adam(ptr_net.parameters(), lr=lr_pi)

env = Env(BATCH_SIZE, ITEMS_SEQ_LN, INPUT_SIZE)

traj_n = 50000
losses_n = []
rews_n = []
best_reward = 0
for t in range(traj_n):
    print('&&&&&&&&&&&&&&&',t,'&&&&&&&&&&&&&&&&')
    
#     if t % 500 == 0:
#         lr_pi *= 0.96
#         print('lr changed to',lr_pi)
#         optim_actor = torch.optim.Adam(ptr_net.parameters(), lr=lr_pi)
    
    env = Env(BATCH_SIZE, ITEMS_SEQ_LN, INPUT_SIZE)
    items_batch,boxes_batch = env.reset()
    inp = torch.cat((items_batch,boxes_batch),2)
    
    probs_seq,probs_ori = ptr_net.forward(inp)
    
    m_ori = Categorical(logits=probs_ori)
    m_seq = Categorical(logits=probs_seq)
    
#     ori = get_action(probs_ori)
    ori = m_ori.sample()
    ori_log_p = m_ori.log_prob(ori)
    items_ori = rot_func_v(ori)
    items_ori = [np.expand_dims(i,2) for i in items_ori]
    items_ori = np.concatenate(items_ori,2)
#     items_seq = get_action(probs_seq).tolist()
    items_seq = m_seq.sample()
    seq_log_p = m_seq.log_prob(items_seq)
    rewards = env.step(items_seq,items_ori)
    rewards = torch.Tensor(rewards)
    print('reward',rewards.mean().item())
    if rewards.mean().item() >= best_reward:
        torch.save(ptr_net,'ptr_net_best.pt')
        best_reward = rewards.mean().item()
    if t % 10 == 0:
        torch.save(ptr_net,'ptr_net_regular.pt')
    policies = [probs_seq,probs_ori]
    learn_indx = random.choice([0,1])
#     print('learn_indx',learn_indx)
#     old_policy = policies[learn_indx].detach()
    rews = []
    losses_coll = {0:[],1:[],2:[]}
    losses_indx = random.choice([0,1,2])
    env_outs = []
    for k in range(train_pi_iter):
        print('k',k)
        probs_seq_,probs_ori_ = ptr_net.forward(inp)
        m_ori_ = Categorical(logits=probs_ori_)
        m_seq_ = Categorical(logits=probs_seq_)
        
        ori_log_p_ = m_ori_.log_prob(ori)
        seq_log_p_ = m_seq_.log_prob(items_seq)
        
#         policies_ = [probs_seq_,probs_ori_]
#         new_policy = policies_[learn_indx]
        
#         ratio = torch.exp(new_policy-old_policy).mean(2).mean(1)
#         clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * rewards
#         loss_pi = -(torch.min(ratio * rewards, clip_adv)).mean()
        ratio_seq = torch.exp(seq_log_p_ - seq_log_p.detach())
        ratio_ori = torch.exp(ori_log_p_ - ori_log_p.detach())
        
        

        clip_adv_seq = torch.clamp(ratio_seq, 1-clip_ratio, 1+clip_ratio) * rewards.unsqueeze(1)
        loss_seq = -(torch.min(ratio_seq * rewards.unsqueeze(1), clip_adv_seq)).mean()
        
        clip_adv_ori = torch.clamp(ratio_ori, 1-clip_ratio, 1+clip_ratio) * rewards.unsqueeze(1)
        loss_ori = -(torch.min(ratio_ori * rewards.unsqueeze(1), clip_adv_ori)).mean()
        
        print('loss',loss_seq.item(),loss_ori.item())
#         losses.append((loss_seq.item(),loss_ori.item()))
        optim_actor.zero_grad()
        losses = [loss_seq,loss_ori,alpha*loss_seq+alpha*loss_ori]
        
        print('losses_indx',losses_indx)
        loss_selected = losses[losses_indx]
        losses_coll[losses_indx].append(loss_selected.item())
        
        loss_selected.backward()
        optim_actor.step()

        
        
        probs_seq_dummy,probs_ori_dummy = ptr_net.forward(inp)
        ori_dummy = get_action(probs_ori_dummy)
        items_ori_dummy = rot_func_v(ori_dummy)
        items_seq_dummy = get_action(probs_seq_dummy).tolist()
        items_ori_dummy = [np.expand_dims(i,2) for i in items_ori_dummy]
        items_ori_dummy = np.concatenate(items_ori_dummy,2)

        rewards_dummy = env.step(items_seq_dummy,items_ori_dummy,items_batch,boxes_batch)
        rewards_dummy = torch.Tensor(rewards_dummy)
        rews.append(rewards_dummy.mean().item())
        print(rewards_dummy.mean())
        
    losses_n.append(losses_coll)
    rews_n.append(rews)
    env_outs.append(env.all_outs)
    pickle.dump(losses_n,open('losses_n.pkl','wb'))
    pickle.dump(rews_n,open('rews_n.pkl','wb'))
    pickle.dump(env_outs,open('env_all_outs.pkl','wb'))
