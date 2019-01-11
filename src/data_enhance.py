# -*- coding: utf-8 -*-  # encoding:utf-8
# @File  : mdata_enhance.py
# @Author: liushuaipeng
# @Date  : 2018/12/5 18:35
# @Desc  :数据增强
import random
import sys

import itertools
import pandas as pd

basedir = '../data/all/'


class Graph(object):
    def __init__(self, *args, **kwargs):
        self.node_neighbors = {}
        self.visited = {}

    def add_nodes(self, nodelist):
        for node in nodelist:
            self.add_node(node)

    def add_node(self, node):
        if not node in self.nodes():
            self.node_neighbors[node] = []

    def add_edge(self, edge):
        u, v = edge
        if (v not in self.node_neighbors[u]) and (u not in self.node_neighbors[v]):
            self.node_neighbors[u].append(v)

            if (u != v):
                self.node_neighbors[v].append(u)

    def nodes(self):
        return self.node_neighbors.keys()

    def depth_first_search(self, root=None):
        order = []

        def dfs(node):
            self.visited[node] = True
            order.append(node)
            for n in self.node_neighbors[node]:
                if not n in self.visited:
                    dfs(n)

        if root:
            dfs(root)

        # for node in self.nodes():
        #     if not node in self.visited:
        #         dfs(node)

        # print(order)
        return order

    def breadth_first_search(self, root=None):
        queue = []
        order = []

        def bfs():
            while len(queue) > 0:
                node = queue.pop(0)

                self.visited[node] = True
                for n in self.node_neighbors[node]:
                    if (not n in self.visited) and (not n in queue):
                        queue.append(n)
                        order.append(n)

        if root:
            queue.append(root)
            order.append(root)
            bfs()

        # for node in self.nodes():
        #     if not node in self.visited:
        #         queue.append(node)
        #         order.append(node)
        #         bfs()
        # print(order)

        return order


if __name__ == '__main__':
    # g = Graph()
    # g.add_nodes([str(i + 1) for i in range(8)])
    # g.add_edge(('1', '2'))
    # # g.add_edge(('1', '3'))
    # g.add_edge(('2', '4'))
    # g.add_edge(('2', '5'))
    # g.add_edge(('4', '8'))
    # g.add_edge(('5', '8'))
    # g.add_edge(('3', '6'))
    # g.add_edge(('3', '7'))
    # g.add_edge(('6', '7'))
    # print("nodes:", g.nodes())
    # order = g.breadth_first_search('3')
    # order = g.depth_first_search('5')

    # agreed 数据节点
    nodes = set()
    # 已有agreed数据，形式sent1==sent2
    agreed_set = set()

    # 生成的agreed数据
    agreed_zh1 = []
    agreed_zh2 = []
    agreed_en1=[]
    agreed_en2=[]

    train_data = pd.read_csv(basedir + 'train_dataset.csv', sep=',')
    train_data.fillna(' ', inplace=True)
    for index in train_data.index:
        label = train_data.loc[index].values[0]
        zh1 = train_data.loc[index].values[1]
        zh2 = train_data.loc[index].values[2]
        if label != 'agreed':
            continue
        nodes.add(zh1)
        nodes.add(zh2)
        # print(zh1,zh2)
        agreed_set.add(zh1 + '--' + zh2)
        agreed_set.add(zh2 + '--' + zh1)
    print('nodes.size=', len(nodes))
    g = Graph()
    g.add_nodes(list(nodes))
    print('add nodes successed!')

    for index in train_data.index:
        label = train_data.loc[index].values[0]
        zh1 = train_data.loc[index].values[1]
        zh2 = train_data.loc[index].values[2]
        # print(label)
        if label == 'agreed':
            g.add_edge((zh1, zh2))
    print('add edges successed')

    # 连通子图节点结合序列
    paths = []
    for n in nodes:
        path = g.depth_first_search(n)
        if len(path) > 1:
            paths.append(set(path))

    cluster_num = 0
    file = open(basedir+'/agreed.txt', 'w')
    agree_num = 0
    for path in paths:
        if len(path) < 2:
            continue
        cluster_num += 1
        n = len(path)
        agree_num += n * (n - 1) / 2
        file.write('**************************************\n')
        file.write('{}\n'.format(len(path)))
        for item in path:
            file.write('{}\n'.format(item))

        agreedPairsGenerator = itertools.combinations(path, 2)
        for pair in agreedPairsGenerator:
            if len(pair) != 2:
                # print('pair 长度不为2，请检查程序')
                continue
            a = pair[0] + '--' + pair[1]
            b = pair[1] + '--' + pair[0]
            if a in agreed_set or b in agreed_set:
                continue
            agreed_zh1.append(pair[0])
            agreed_zh2.append(pair[1])

    file.close()
    print('lager than 2 cluster', cluster_num)
    print('total cluster num', len(paths))
    print('agree total num：', agree_num)
    print('agreed can construct', len(agreed_zh1), len(agreed_zh2))

    agreedlables = ['agreed' for i in (agreed_zh1)]
    random.Random(666).shuffle(agreedlables)
    random.Random(666).shuffle(agreed_zh1)
    random.Random(666).shuffle(agreed_zh2)


    enhanced_agreed = pd.DataFrame()
    enhanced_agreed[0] = agreedlables[:200000]
    enhanced_agreed[1] = agreed_zh1[:200000]
    enhanced_agreed[2] = agreed_zh2[:200000]
    enhanced_agreed.to_csv(basedir+'/enhanced_agreed_20w.csv', sep=',', index=False, header=['label', 't1_zh', 't2_zh'])
    print('enhanced_agreed.csv')

    # 构造disagreed数据
    lables = []
    t_zh1 = []
    t_zh2 = []

    dis_count = 0
    exist = set()
    new_dis = set()
    for index in train_data.index:
        label = train_data.loc[index].values[0]
        zh1 = train_data.loc[index].values[1]
        zh2 = train_data.loc[index].values[2]
        if label != 'disagreed':
            continue
        dis_count += 1
        exist.add(zh1 + "--" + zh2)
        exist.add(zh2 + "--" + zh1)
    print('exist ab type dis num=', dis_count)
    print('exist ab-ba type dis num = ', len(exist))

    for index in train_data.index:
        label = train_data.loc[index].values[0]
        zh1 = train_data.loc[index].values[1]
        zh2 = train_data.loc[index].values[2]
        if label != 'disagreed':
            continue
        for path in paths:
            if zh1 in path:
                for item in path:
                    a = item + '--' + zh2
                    b = zh2 + '--' + item
                    if a not in exist:
                        lables.append('disagreed')
                        t_zh1.append(item)
                        t_zh2.append(zh2)
                        new_dis.add(a)
                    # else:
                    #     print(a)
                    if b not in exist:
                        new_dis.add(b)
                        # else:
                        #     print(b)
            if zh2 in path:
                for item in path:
                    a = item + '--' + zh1
                    b = zh1 + '--' + item
                    if a not in exist:
                        new_dis.add(a)
                    # else:
                    #     print(a)
                    if b not in exist:
                        new_dis.add(b)
                        lables.append('disagreed')
                        t_zh1.append(zh1)
                        t_zh2.append(item)
                        # else:
                        #     print(b)
    print('new dis num = ', len(new_dis))

    enhanced_disagree = pd.DataFrame()
    enhanced_disagree[0] = lables
    enhanced_disagree[1] = t_zh1
    enhanced_disagree[2] = t_zh2
    enhanced_disagree.to_csv(basedir+'/enhanced_disagreed.csv', sep=',', index=False, header=['label', 't1_zh', 't2_zh'])
    print('enhanced_disagreed.csv')


print('1 data enhance done')