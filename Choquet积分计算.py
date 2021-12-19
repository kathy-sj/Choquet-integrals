import numpy as np

def fuzzyfun(alpha,q_m,prob_num):
    '''
    # :param alpha: array, (stu_m, knowledge_k)，学生的知识点掌握度
    # :param mu: array, (2^(knowledge_k),)，单个试题的模糊测度
    # :param mu: array, (2^(knowledge_k),)，单个试题的模糊测度
    # :return:prob_num,表示试题的数量
    # '''

    stu_num=alpha.shape[0]
    eta=np.zeros((stu_num, prob_num))
    for i in range(0, prob_num):
        a = alpha * q_m[i]
        idx = np.argwhere(np.all(a[..., :] == 0, axis=0))
        alpha1 = np.delete(a, idx, axis=1)#处理后的alpha，选择本题对应的知识点计算
        inputs=alpha1
        nVars=2**alpha1.shape[1]-2
        mu = np.loadtxt("./data/Math2/FM_math2_p{}.csv".format(i + 1), delimiter=",", skiprows=0)  # 每道题目对应一个模糊测度问津
        # mu = mu[1:, int(sum(q_m[i])):]  # 适用于q矩阵几个知识点就几个k
        sortInputs= -np.sort(-inputs,axis=1)  # 对输入的值降序排序sortInputs对应值, sortInd对应序号
        sortInd=np.argsort(-inputs,axis=1)
        M= inputs.shape[0]
        N = inputs.shape[1]
        sortInputs = np.concatenate((sortInputs, np.zeros((M, 1))), axis=1)
        sortInputs1 = np.concatenate((sortInputs, np.zeros((M, 1))), axis=1)
        sortInputs = (sortInputs[:, :-1] - sortInputs[:, 1:])
        out = np.cumsum(np.power(2, sortInd), 1) - np.ones((1))
        out=out.astype('int64')# cumsum返回维度dim中输入元素的累计和。

        data = np.zeros((M, nVars + 1))
        for j in range(0,stu_num):
            data[j, out[j, :]] = sortInputs[j, :]


        eta[:, i]= np.matmul(data, mu).reshape(-1)

    print(eta)


    return eta   # (stu_m,)


# Convert decimal to binary string
def sources_and_subsets_nodes(N):
    str1 = "{0:{fill}" + str(N) + "b}"
    a = []
    for i in range(1, 2 ** N):
        print(str1.format(i, fill='0'))  # 1-001，2-010.3-011
        a.append(str1.format(i, fill='0'))

    sourcesInNode = []
    sourcesNotInNode = []
    subset = []
    sourceList = list(range(N))

    # find subset nodes of a node
    def node_subset(node, sourcesInNodes):
        return [node - 2 ** (i) for i in sourcesInNodes]


    # convert binary encoded string to integer list
    def string_to_integer_array(s, ch):
        N = len(s)
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]  # i为索引，ltr为对应的取值



    for j in range(len(a)):
        # index from right to left
        idxLR = string_to_integer_array(a[j], '1')
        # print(idxLR)
        sourcesInNode.append(idxLR)
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))  # 变为集合set对象
        subset.append(node_subset(j, idxLR))

    return sourcesInNode, subset


def subset_to_indices(indices):
    return [i for i in indices]

if __name__ == '__main__':
    # alpha=np.loadtxt("./data/Math1/alpha_final.csv", delimiter=",", skiprows=0)
    alpha=np.array([[0.1,0.3,0.1,1,0.1,0.5,0.8,0.9,0.1,0.4,0.9,0.4,0.3,0.1,1,0.7],[0.4,0.3,0.1,1,0.7,0.5,0.8,0.9,0.1,0.1,0.9,0.8,0.9,0.1,0.4,0.9]])
    q_m=np.loadtxt("./data/Math2/q.txt")
    prob_num=20
    eta=fuzzyfun(alpha,q_m,prob_num)



