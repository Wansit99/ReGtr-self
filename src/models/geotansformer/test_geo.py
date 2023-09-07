from geotransformers import GeometricTransformer
import torch 



transformer = GeometricTransformer(
            2048,
            128,
            256,
            4,
            ['self', 'cross', 'self', 'cross', 'self', 'cross'],
            4.8,
            15,
            3,
            reduction_a = 'max',
        )

# 3. Conditional Transformer
# 定义张量的维度大小
B, N, M, C = 10, 20, 30, 2048

# 使用随机数据生成张量
ref_points = torch.randn(B, N, 3)
src_points = torch.randn(B, M, 3)
ref_feats = torch.randn(B, N, C)
src_feats = torch.randn(B, M, C)

ref_feats_c, src_feats_c = transformer(
    ref_points,
    src_points,
    ref_feats,
    src_feats,
)

print(ref_feats.shape, src_feats.shape)
print(ref_feats_c.shape, src_feats_c.shape)