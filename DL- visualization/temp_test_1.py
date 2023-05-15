import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

# 生成随机图像数组
img_array = np.random.randint(0, 255, size=(5, 5))

# 绘制图像
cmap = 'jet'
plt.subplot(231)
plt.imshow(img_array, cmap=cmap, interpolation='nearest')
plt.title('nearest')

plt.subplot(232)
plt.imshow(img_array, cmap=cmap, interpolation='bilinear')
plt.title('bilinear')

plt.subplot(233)
plt.imshow(img_array, cmap=cmap, interpolation='bicubic')
plt.title('bicubic')

plt.subplot(234)
plt.imshow(img_array, cmap=cmap, interpolation='spline16')
plt.title('spline16')

plt.subplot(235)
plt.imshow(img_array, cmap=cmap)
plt.title('oo')
# 关闭坐标轴
plt.axis('off')

# 显示图像
plt.show()

plt.close()


