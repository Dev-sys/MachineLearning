import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# def getData(filename):
#     with open(filename,'r') as f:
#         numOfLine = len(f.readlines())
#         matData = np.zeros((numOfLine, 3))
#         label = []
#         index = 0
#         for line in f.readlines():
#             line = line.strip()
#             listFromLine = line.split('\t')
#             matData[index, :] = listFromLine[:3]
#             label.append(int(listFromLine[-1]))
#             index += 1
#     return matData, label

def getData(filename):
    with open(filename, 'r') as f:
        mm = f.readlines()
    matrix = list(map(lambda x: x.strip().split('\t'), mm))
    dataSet = np.asarray(matrix).astype('float32')
    return dataSet

file = 'I:/JupyterNotebookCase/MachineLearning/input/2.KNN/datingTestSet2.txt'
dataSet = getData(file)
# 数据按label分类
dataSet1 = dataSet[dataSet[:,-1]==1]
dataSet2 = dataSet[dataSet[:,-1]==2]
dataSet3 = dataSet[dataSet[:,-1]==3]

fig = plt.figure(figsize=(13,7))
ax = fig.add_subplot(111)
# 用不同颜色标记不同label的数据
ax.scatter(dataSet1[:, 0], dataSet1[:, 2],c='r',marker='.',label='one')
ax.scatter(dataSet2[:, 0], dataSet2[:, 2],c='b',marker='.',label='two')
ax.scatter(dataSet3[:, 0], dataSet3[:, 2],c='g',marker='.',label='three')
plt.xlabel('飞行里程数',fontsize=20)
plt.ylabel('玩游戏的时间',fontsize=20)
ax.legend()
# plt.show()


def autoNorm(dataSet):
    '''
    Desc:
        归一化特征值，消除特征之间量级不同导致的影响
    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到
    
    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    '''
    # 每种属性的最大值和最小值
    dataMax = dataSet.max(0)
    dataMin = dataSet.min(0)
    # 极差矩阵
    ranges = dataMax-dataMin
    ranges_mart = np.zeros(dataSet.shape)
    ranges_mart[:] = ranges
    # 生成当前值与最小值的差
    dataFromMin = dataSet - np.tile(dataMin,(dataSet.shape[0],1))
    normDataSet = dataFromMin[:,:3] / ranges_mart[:,:3]
    return normDataSet

# $$\sqrt{(0-67)^2 + (20000-32000)^2 + (1.1-0.1)^2 }$$  样本距离计算
def classify0(inX, dataSet, labels, k):
    """
    inx[1,2,3]
    DS=[[1,2,3],[1,2,0]]
    inX: 用于分类的输入向量
    dataSet: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.
    预测数据所在分类可在输入下列命令
    kNN.classify0([0,0], group, labels, 3)
    """

    # -----------实现 classify0() 方法的第一种方式----------------------------------------------------------------------------------------------------------------------------
    # 1. 距离计算
    dataSetSize = dataSet.shape[0]
    # tile生成和训练样本对应的矩阵，并与训练样本求差
    """
    tile: 列-3表示复制的行数， 行-1／2表示对inx的重复的次数
    In [8]: tile(inx, (3, 1))
    Out[8]:
    array([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
    In [9]: tile(inx, (3, 2))
    Out[9]:
    array([[1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3]])
    """
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    """
    欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 dataSet的第一个点的距离。
       第二行： 同一个点 到 dataSet的第二个点的距离。
       ...
       第N行： 同一个点 到 dataSet的第N个点的距离。
    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    """
    # 取平方
    sqDiffMat = diffMat ** 2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：y=array([3,0,2,1,4,5]) 则，x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
    # print 'distances=', distances
    sortedDistIndicies = distances.argsort()
    # print('distances.argsort()=', sortedDistIndicies)

    # 2. 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 找到该样本的类型
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典中将该类型加一
        # 字典的get方法
        # 如：list.get(k,d) 其中 get相当于一条if...else...语句,参数k在字典中，字典将返回list[k];如果参数k不在字典中则返回参数d,如果K在字典中则返回k对应的value值
        # l = {5:2,3:4}
        # print l.get(3,0)返回的值是4；
        # Print l.get（1,0）返回值是0；
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 3. 排序并返回出现最多的那个类型
    # 字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
    # 例如：dict = {'Name': 'Zara', 'Age': 7}   print "Value : %s" %  dict.items()   Value : [('Age', 7), ('Name', 'Zara')]
    # sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    # 例如：a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2进行排序的，而不是a,b,c
    # b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
    # b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序。
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)   python3.x没有这个属性?
    sortedClassCount = sorted(map(lambda x: (x, classCount[x]), classCount), key=lambda x: x[1], reverse=True)
    # print(sortedClassCount)
    return sortedClassCount[0][0]

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # 实现 classify0() 方法的第二种方式

    # """
    # 1. 计算距离

    # 欧氏距离： 点到点之间的距离
    #    第一行： 同一个点 到 dataSet的第一个点的距离。
    #    第二行： 同一个点 到 dataSet的第二个点的距离。
    #    ...
    #    第N行： 同一个点 到 dataSet的第N个点的距离。

    # [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    # (A1-A2)^2+(B1-B2)^2+(c1-c2)^2

    # inx - dataset 使用了numpy broadcasting，见 https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
    # np.sum() 函数的使用见 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sum.html
    # """


#   dist = np.sum((inx - dataset)**2, axis=1)**0.5

# """
# 2. k个最近的标签

# 对距离排序使用numpy中的argsort函数， 见 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sort.html#numpy.sort
# 函数返回的是索引，因此取前k个索引使用[0 : k]
# 将这k个标签存在列表k_labels中
# """
# k_labels = [labels[index] for index in dist.argsort()[0 : k]]
# """
# 3. 出现次数最多的标签即为最终类别

# 使用collections.Counter可以统计各个标签的出现次数，most_common返回出现次数最多的标签tuple，例如[('lable1', 2)]，因此[0][0]可以取出标签值
# """
# label = Counter(k_labels).most_common(1)[0][0]
# return label

# ------------------------------------------------------------------------------------------------------------------------------------------

def dating(file):
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本 10%
    # 从文件中加载数据
    datingDataMat = getData(file)  # load data setfrom file
    # 归一化数据
    normMat = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i, :3], normMat[numTestVecs:m, :3], datingDataMat[numTestVecs:m, 3].astype('int32'), 20)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingDataMat[numTestVecs:m, 3].astype('int32')[i]))
        if (classifierResult != normMat[:,-1][i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


# 执行测试函数
# dating(file)























