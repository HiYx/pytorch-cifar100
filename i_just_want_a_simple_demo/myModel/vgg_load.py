## 测算权重文件
import torch
import numpy as np

weights = torch.load('./iter_10000.pth', map_location='cpu')

for k, v in weights.items():
        print("Layer: {}".format(k))

for k, v in weights['state_dict'].items():
        print("Layer: {}".format(k))

with open('model.txt', 'w') as w:
    w.write(str(weights['state_dict']))

# ## 把torch加载的权重转换成 dict字典
# import pickle

# def save_dict(obj, name):
    # with open(name + '.pkl', 'wb') as f:
        # pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# def load_dict(name ):
    # with open(name + '.pkl', 'rb') as f:
        # return pickle.load(f)


# dict={}
# for k, v in weights.items():
    # dict[k] = v 
# #print(dict['0.weight'])
# save_dict(dict, "a")
# # dicr_pickle=load_dict("a")
# # print(dicr_pickle['0.weight'])

# ## 把torch加载的权重转换成 列表

# # x_train1 = []
# # for k, v in weights.items():
    # # x_train1.append([k ,v])
# # x_train2 = np.array(x_train1)
# # np.save('save_x',weights) 
# # weights=np.load('save_x.npy',allow_pickle=True)

