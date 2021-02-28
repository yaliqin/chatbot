#!/usr/bin/env python3
# coding: utf-8
# File: MedicalGraph.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-10-3

import os
import json
from py2neo import Graph,Node

class ChatbotGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'test.json')
        self.g = Graph(
            host="127.0.0.1",  # neo4j 搭载服务器的ip地址，ifconfig可获取到
            http_port=7474,  # neo4j 服务器监听的端口号
            user="neo4j",  # 数据库user name，如果没有更改过，应该是neo4j
            password="hhhkg")

    '''读取文件'''
    def read_nodes(self):
        # 共７类节点
        terminologies = [] # 检查
        subjects = [] #科室
        operations = [] #疾病
        aliass = []#症状

        operation_infos = []#疾病信息

        # 构建节点实体关系
        rels_subject = [] #　科室－科室关系
        rels_terminology = [] # 疾病－检查关系
        rels_alias = [] #疾病症状关系


        count = 0
        for data in open(self.data_path):
            operation_dict = {}
            count += 1
            print(count)
            data_json = json.loads(data)
            operation = data_json['name']
            operation_dict['name'] = operation
            operations.append(operation)
            operation_dict['method'] = ''


            if 'alias' in data_json:
                aliass += data_json['alias']
                for alias in data_json['alias']:
                    rels_alias.append([operation, alias])

            if 'subject' in data_json:
                subject = data_json['subject']
                for _subject in subject:
                    rels_subject.append([operation, _subject])
                operation_dict['subject'] = subject
                subjects += subject

            if 'method' in data_json:
                operation_dict['method'] = data_json['method']

            if 'terminology' in data_json:
                terminology = data_json['terminology']
                for _terminology in terminology:
                    rels_terminology.append([operation, _terminology])
                terminologies += terminology

            operation_infos.append(operation_dict)
        return set(terminologies), set(subjects), set(aliass), set(operations), operation_infos,rels_alias,rels_subject,rels_terminology

    '''建立节点'''
    def create_node(self, label, nodes):
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.g.create(node)
            count += 1
            print(count, len(nodes))
        return

    '''创建知识图谱中心疾病的节点'''
    def create_operation_nodes(self, operation_infos):
        count = 0
        for operation_dict in operation_infos:
            node = Node("Operation", name=operation_dict['name'], method=operation_dict['method'])
            self.g.create(node)
            count += 1
            print(count)
        return

    '''创建知识图谱实体节点类型schema'''
    def create_graphnodes(self):
        Terminologies, Subjects, Aliass, Operations, operation_infos,rels_alias, rels_subject,rels_terminology = self.read_nodes()
        self.create_operation_nodes(operation_infos)
        self.create_node('Terminology', Terminologies)
        print(Terminologies)
        self.create_node('Subject', Subjects)
        print(Subjects)
        self.create_node('Alias', Aliass)
        print(Aliass)
        return


    '''创建实体关系边'''
    def create_graphrels(self):
        Terminologies, Subjects, Aliass, Operations, operation_infos,rels_alias, rels_subject,rels_terminology = self.read_nodes()
        self.create_relationship('Operation', 'Terminology', rels_terminology, 'contain', 'contain')
        self.create_relationship('Operation', 'Alias', rels_alias, 'has_alias', 'has name of')
        self.create_relationship('Operation', 'Subject', rels_subject, 'belongs_to', 'belong to')

    '''创建实体关联边'''
    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        # 去重处理
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return

    '''导出数据'''
    def export_data(self):
        Terminologies, Subjects, Aliass, Operations, operation_infos,rels_alias, rels_subject,rels_terminology = self.read_nodes()
        f_terminology = open('terminology.txt', 'w+')
        f_subject = open('subject.txt', 'w+')
        f_alias = open('alias.txt', 'w+')
        f_operation = open('operation.txt', 'w+')


        f_terminology.write('\n'.join(list(Terminologies)))
        f_subject.write('\n'.join(list(Subjects)))
        f_alias.write('\n'.join(list(Aliass)))
        f_operation.write('\n'.join(list(Operations)))


        f_terminology.close()
        f_subject.close()
        f_alias.close()
        f_operation.close()

        return



if __name__ == '__main__':
    handler = ChatbotGraph()
    print("step1:导入图谱节点中")
    handler.create_graphnodes()
    print("step2:导入图谱边中")      
    handler.create_graphrels()
    handler.export_data()
    
