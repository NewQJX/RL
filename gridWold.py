import numpy as np

#初始化策略 4*4*4 的矩阵，a[i][j][k]表示在第i行第j列下的第k个动作的概率
#k = 0,1,2,3代表上下左右的概率
def init_policy(n = 4):
    policy = np.ones((4,4,int(n)))
    policy = policy / n
    policy[0][0] = 0
    policy[3][3] = 0
    return policy

#初始化状态值矩阵为0
def init_v0():
    v0 = np.zeros((4,4))
    return v0
#action动作能到达下一个状态的值
def get_value(v, i, j, action):
    m, n = v.shape
    if action == "left":
        if j - 1 < 0:
            return v[i][j]
        return v[i][j - 1]
    elif action == "up":
        if i - 1 < 0:
            return v[i][j]
        return v[i - 1][j]
    elif action == "right":
        if j + 1 >= n:
            return v[i][j]
        return v[i][j+1]
    elif action == "buttom":
        if i + 1 >= m:
            return v[i][j]
        return v[i + 1][j]
    else:
        raise Exception("Invalid action!", action)
#计算新的状态值矩阵   
def updata_v(v, policy, gama = 1):
    new_v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            v_left = get_value(v, i, j, "left")
            v_up = get_value(v, i, j, "up")
            v_right = get_value(v, i, j, "right")
            v_buttom = get_value(v, i, j, "buttom")
            reward = -np.ones(4)   #即时回报-1
            temp = (np.array([v_left, v_up, v_right, v_buttom]) * gama + reward) * policy[i][j]
            new_v[i][j] = np.sum(temp)
    return new_v      
def get_subPolicy(v, i, j):
    v_left = get_value(v, i, j, "left")
    v_up = get_value(v, i, j, "up")
    v_right = get_value(v, i, j, "right")
    v_buttom = get_value(v, i, j, "buttom")
    temp = np.array([v_left, v_up, v_right, v_buttom])
    max_value = temp.max()
    max_index = []
    for index in range(4):
        if temp[index] == max_value:
            max_index.append(index)

    subPolicy = np.zeros(4)
    prob = 1 / len(max_index)
    for index in max_index:
        subPolicy[index] = prob
    return subPolicy
#用贪心的思想得到新策略
def new_policy(v,n = 4):
    new_policy = np.zeros((4,4,int(n)))
    for i in range(4):
        for j in range(4):
            new_policy[i][j] = get_subPolicy(v, i, j)
    new_policy[0][0] = 0
    new_policy[3][3] = 0
    return new_policy
#迭代测试
def test(v, policy, iterator_time = 1):
    time = 0  #记录迭代的次数
    
    while time < int(iterator_time):            #迭代
        v = updata_v(v,policy)
        # print("第%d次迭代的矩阵为："%(time + 1))
        # print(v) 
        time = time + 1
    return v

#程序入口

#policy updata
'''
if __name__ == "__main__":     
    v = init_v0()           #初始化状态值矩阵
    policy = init_policy()          #策略
    final_v = test(v, policy,408)       #得到在策略policy下收敛后的状态值矩阵      #迭代407次才收敛
    ################################
    new_policy1 = new_policy(final_v)        #算出新策略
    print("新的策略为：")
    print(new_policy1)
    print("——————————————————————新策略下迭代：——————————————————————————")
    final_v1 = test(final_v, new_policy1,4)     #迭代3次就收敛
    #############################
    new_policy2 = new_policy()
    print("新的策略为：")
    print(new_policy2)
    print("——————————————————————新策略下迭代：——————————————————————————")
    final_v1 = test(final_v1, new_policy2,4)     #状态值矩阵保持不变，所以目前的策略为最优的策略
'''  

#value iterator

if __name__ == "__main__":
    policy = init_policy()
    status_value_matrix = init_v0()
    times = 0
    while times < 4:
        new_status_value_matrix = test(status_value_matrix, policy, 1)
        print("第%d次迭代产生的矩阵为:"%(times + 1))
        print(new_status_value_matrix)
        policy = new_policy(new_status_value_matrix)
        print("第%d次迭代产生的策略为:"%(times + 1))
        print(policy)
        status_value_matrix = new_status_value_matrix
        times += 1







