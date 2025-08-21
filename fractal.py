import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im

N = 4127
K = 88
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def farey_sequence(n):
    """
    Print the n'th Farey sequence. Allow for either ascending or descending.

    >>> print(*farey_sequence(5), sep=' ')
    0 1/5 1/4 1/3 2/5 1/2 3/5 2/3 3/4 4/5 1

    Source: https://en.wikipedia.org/wiki/Farey_sequence#Next_term
    """
    a, b, c, d = 0, 1, 1, n
    yield (a, b)
    while 0 <= c <= n:
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        yield (a, b)

def slope_positions(slopes):
    """
    input: tensor of slopes with shape (K*8, 2)
    output: tensor of x, y coords crossed by the slopes (2, N*K*8)
    """
    i = torch.arange(0, N).view(1, N)

    i = i.to(device)
    
    # i * dx (N, K*8)
    dx = slopes[:, 0].view(slopes.shape[0], 1)
    dy = slopes[:, 1].view(slopes.shape[0], 1)

    dx_ = dx * i + N // 2
    dy_ = dy * i + N // 2

    rows = torch.remainder(dy_, N).flatten()
    cols = torch.remainder(dx_, N).flatten()
    
    # ps = []
    # for i in range(0, N):
    #     x = i*slope[0] + N // 2
    #     y = i*slope[1] + N // 2
    #     ps.append((x % N, y % N))
    
    return rows, cols

farey = list(farey_sequence(N))
l2_farey = sorted(farey, key=lambda frac: frac[0] * frac[0] + frac[1] * frac[1])
top_k_farey = l2_farey[:K]

fig = plt.figure(figsize=(10,10))
# (K, 2) -> (K*8, 2)
slopes = torch.tensor(top_k_farey)

x, y = slopes[:, 0], slopes[:, 1]
all_slopes = torch.cat([
    torch.stack([ x,  y], dim=1),
    torch.stack([-x,  y], dim=1),
    torch.stack([ x, -y], dim=1),
    torch.stack([-x, -y], dim=1),
    torch.stack([ y,  x], dim=1),
    torch.stack([-y,  x], dim=1),
    torch.stack([ y, -x], dim=1),
    torch.stack([-y, -x], dim=1)
])

unique_slopes = torch.unique(all_slopes, dim=0)

all_slopes = unique_slopes
p = torch.ones((N, N))

p = p.to(device)
all_slopes = all_slopes.to(device)

rows, cols = slope_positions(all_slopes)
p[rows, cols] = 0

# for i in range(K):
#     for s in octant_symmetries(*l2_farey[i]):
#         coords = np.array(slope_positions(s))
#         rows = coords[:, 1]
#         cols = coords[:, 0]
#         p[rows, cols] = 1

plt.imshow(p.cpu().numpy(), cmap='Greys')
im.imsave('fractal.png', p.cpu().numpy(), cmap='Greys')