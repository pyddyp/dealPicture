import  cv2
import matplotlib.pylab as plt
import numpy as np

myimg ="C:/Users/23368/Desktop/yangyang/2.jpg"

img = cv2.imread(myimg)
imgSize=cv2.resize(img,(100,100))
img_gray = cv2.cvtColor(imgSize, cv2.COLOR_BGR2GRAY)

def get_binary_img(img):
    bin_img = np.zeros(shape=(img.shape),dtype=np.uint8)
    h=img.shape[0]
    w=img.shape[1]
    for i in range(h):
        for j in range(w):
            bin_img[i][j]=255 if img[i][j]>160 else 0
    return bin_img

bin_img = get_binary_img(img_gray)

def get_contour(bin_img):
    contour_img = np.zeros(shape=(bin_img.shape), dtype=np.uint8)
    contour_img += 255
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (bin_img[i][j] == 0):
                contour_img[i][j] = 0
                sum = 0
                sum += bin_img[i - 1][j + 1]
                sum += bin_img[i][j + 1]
                sum += bin_img[i + 1][j + 1]
                sum += bin_img[i - 1][j]
                sum += bin_img[i + 1][j]
                sum += bin_img[i - 1][j - 1]
                sum += bin_img[i][j - 1]
                sum += bin_img[i + 1][j - 1]
                if sum == 0:
                    contour_img[i][j] = 255

    return contour_img

contour_img = get_contour(bin_img)
print("contour:",contour_img)
plt.figure(figsize=(10,10))
plt.imshow(contour_img,cmap='gray')
plt.show()

#获取某一行中出现0最多的个数
def pixel_num(contour):
    h=contour.shape[0]
    w=contour.shape[1]
    num=[0 for x in range(h)]
    for i in range(1,h-1):
        print("num0:",num)
        for j in range(1,w-1):
            if(contour[i][j]==0):
                num[i]=num[i]+1;
    max=num[0]
    for k in range(1,h-1):
        print("num[k]:",num[k])
        if(num[k]>max):
            max=num[k]
    return max

#统计出现0的下降趋势
def define_cat(contour):
    h=contour.shape[0]
    w=contour.shape[1]
    num=[0 for x in range(h)]
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(contour[i][j]==0):
                num[i]=num[i]+1;
    print(num)
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(contour[i][j]==0):
                line=3;
                threshold=5;
                k=0;
                while(num[i]>threshold):
                    if(num[i]<num[i-1]):
                        k=k+1;
                        if(k>line):
                            print("cat")
                        else:
                            print("this is maybe a handsome man!")




# def cat_ear_dect(contour):
#     h=contour.shape[0]
#     w=contour.shape[1]
#     for i in range(1,h-1):
#         for j in range(1,w-1):
#             if(contour[i][j]==0):
#                 wL=contour.shape[1]-j
#                 for k in wL:
#                     if(contour[i][k])

class Solution:
    def __init__(self, m):
        self.row = m[0][0]  # 行
        self.column = m[0][1]  # 列
        m.pop(0)
        self.m = m
        self.areas = []

    def count_islands(self):
        count = 0
        # 遍历二维数组，遇到为1的点，就调用感染函数
        for i in range(self.row):
            for j in range(self.column):
                if self.m[i][j] == 1:
                    self.areas.append(0)
                    count += 1
                    self.infect(i, j, count + 1)

    def infect(self, i, j, flag):
        if i < 0 or i >= self.row or j < 0 or j >= self.column or self.m[i][j] != 1:
            return
        self.m[i][j] = flag  # 将上下左右为1的值全部都改写为某一特定数字，属于同一个小岛的，数字相同
        self.areas[-1] += 1
        self.infect(i, j - 1, flag)
        self.infect(i - 1, j, flag)
        self.infect(i + 1, j, flag)
        self.infect(i, j + 1, flag)
        
        
       
num=pixel_num(contour_img);     

print("num:",num)
