import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbeddingSelf(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """
    def __init__(self, d_model: int = 256):
        super().__init__()

        # MLP to transform the tensor
        self.mlp = nn.Linear(10, d_model)
        

    def forward(self, points_xyz, index, all_points_xyz):
        pos_emd = []
        B = len(points_xyz)
        
        for i in range(B):
            # 得到当前批次的原始点的xyz
            points = points_xyz[i] # N, 3
            # 得到当前批次的原始点的邻居的index
            index_tmp = index[i] # N, k
            # 得到当前批次的原始点的邻居的x,y,z 
            points_neig_xyz = all_points_xyz[index_tmp] # N, K, 3
            # 得到N,K
            N, K, _ = points_neig_xyz.shape
            # 将原始点的x,y,z复制k次
            expanded_points = points.unsqueeze(1).expand(N, K, 3) # N, K, 3
            #d 得到原始点与邻居点的差值
            Point_i = expanded_points - points_neig_xyz # N, K, 3
            # 得到原始点与邻居点的欧氏距离
            distance = torch.norm(expanded_points - points_neig_xyz, dim=-1, keepdim=True) # N, K, 1
            # 将他们cat在一起
            all_feats = torch.cat([expanded_points, points_neig_xyz, Point_i, distance], dim=-1) # N, K, 10
            # 将其转换为N, K, d
            transformed_tensor = self.mlp(all_feats)
            # 在第2个维度做max，得到N，1，d
            final_tensor, _ = transformed_tensor.max(1)
            # 加入到list中
            pos_emd.append(final_tensor)

        return pos_emd

# 测试代码
if __name__ == '__main__':

    B, N , K, d = 2, 101, 40, 3

    Points1 = torch.rand(N, 3)
    Points2 = torch.rand(N+20, 3)

    Point = []
    Point.append(Points1)
    Point.append(Points2)

    index1 = torch.randint(0, 10000, (N, K))
    index2 = torch.randint(0, 10000, (N+20, K))

    index = []
    index.append(index1)
    index.append(index2)


    all_points_xyz = torch.rand(10000,3)
    # test = torch.tensor([[1,2], [3,4]])
    # test2 = torch.tensor([[2,3], [4,5]])
    # # expanded_points = test.unsqueeze(2).expand(2,2,2)
    # # print(expanded_points)

    # distance = torch.norm(test.float() - test2.float(), dim=-1, keepdim=True)
    # print(distance)

    test = PositionEmbeddingSelf(256)
    result = test(Point, index, all_points_xyz)
    print(result)