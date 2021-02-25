import csv
import torch
import transformers
from transformers import pipeline
import nlpaug.augmenter.word as naw

with open("input_classification_test_data.csv","r") as input:
    with open("/Users/wenyaxie/Downloads/positive_data_ppdb.csv","w") as output:
        ppdb = naw.SynonymAug(aug_src='ppdb', model_path='/Users/wenyaxie/Downloads/Chatbot_bilstm/ppdb-2.0-tldr')
  
        csv_reader = csv.reader(input)
        csv_writer = csv.writer(output)

        rows = []
        for row in csv_reader:
            rows.append(row)

        for row in list(rows): #sym_ppdb
            if row[1] != "flag" and row[1] != "":
                row[2] = str(ppdb.augment(row[2]))
            csv_writer.writerow(row)

with open("input_classification_test_data.csv","r") as input:
    with open("/Users/wenyaxie/Downloads/positive_data_bert_substitution.csv","w") as output:
        bert_sub = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

        csv_reader = csv.reader(input)
        csv_writer = csv.writer(output)

        rows = []
        for row in csv_reader:
            rows.append(row)

        for row in list(rows): #sub_bert
            if row[1] != "flag" and row[1] != "":
                row[2] = str(bert_sub.augment(row[2]))
            csv_writer.writerow(row)

with open("input_classification_test_data.csv","r") as input:
    with open("/Users/wenyaxie/Downloads/positive_data_distilbert_substitution.csv","w") as output:
        distilbert_sub = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")

        csv_reader = csv.reader(input)
        csv_writer = csv.writer(output)

        rows = []
        for row in csv_reader:
            rows.append(row)

        for row in list(rows): #sub_distillbert
            if row[1] != "flag" and row[1] != "":
                row[2] = str(distilbert_sub.augment(row[2]))
            csv_writer.writerow(row)

with open("input_classification_test_data.csv","r") as input:
    with open("/Users/wenyaxie/Downloads/positive_data_roberta_substitution.csv","w") as output:
        roberta_sub = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute")

        csv_reader = csv.reader(input)
        csv_writer = csv.writer(output)

        rows = []
        for row in csv_reader:
            rows.append(row)

        for row in list(rows): #sub_roberta
            if row[1] != "flag" and row[1] != "":
                row[2] = str(roberta_sub.augment(row[2]))
            csv_writer.writerow(row)

with open("input_classification_test_data.csv","r") as input:
    with open("/Users/wenyaxie/Downloads/positive_data_bert_insertion.csv","w") as output:
        bert_insert = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

        csv_reader = csv.reader(input)
        csv_writer = csv.writer(output)

        rows = []
        for row in csv_reader:
            rows.append(row)

        for row in list(rows): #insert_bert
            if row[1] != "flag" and row[1] != "":
                row[2] = str(bert_insert.augment(row[2]))
            csv_writer.writerow(row)