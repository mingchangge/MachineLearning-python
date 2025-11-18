# 微积分：瞬时斜率
import numpy as np, matplotlib.pyplot as plt
x = np.linspace(-2,2,400); y = x**2
plt.plot(x,y)
dx = 0.01; slope = (1.01**2 - 1**2)/dx
plt.axline((1,1), slope=slope, color='r', label=f'slope={slope:.2f}')
plt.legend(); plt.show()