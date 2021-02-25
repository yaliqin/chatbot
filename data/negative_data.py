import csv
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

with open("input_classification_test_data.csv","r") as input:
    with open("/Users/wenyaxie/Downloads/negative_data_character_substitute.csv","w") as output:
        keyboard_dis = nac.KeyboardAug()

        csv_reader = csv.reader(input)
        csv_writer = csv.writer(output)
        counter = 0

        rows = []
        for row in csv_reader:
            rows.append(row)

        for row in list(rows): # Substitute character by keyboard distance
            if row[1] != "flag" and row[1] !='R' and row[1]!="":
                counter += 1
                if counter != 2:
                    row[2] = str(keyboard_dis.augment(row[2]))
                if counter == 3:
                    counter = 0
            csv_writer.writerow(row)

with open("input_classification_test_data.csv","r") as input:
    with open("/Users/wenyaxie/Downloads/negative_data_character_insertion.csv","w") as output:
        random_insert = nac.RandomCharAug(action="insert")

        csv_reader = csv.reader(input)
        csv_writer = csv.writer(output)
        counter = 0

        rows = []
        for row in csv_reader:
            rows.append(row)

        for row in list(rows): # Insert character randomly
            if row[1] != "flag" and row[1] !='R' and row[1]!="":
                counter += 1
                if counter != 2:
                    row[2] = str(random_insert.augment(row[2]))
                if counter == 3:
                    counter = 0
            csv_writer.writerow(row)

with open("input_classification_test_data.csv","r") as input:
    with open("/Users/wenyaxie/Downloads/negative_data_character_swap.csv","w") as output:
        character_swap = nac.RandomCharAug(action="swap")

        csv_reader = csv.reader(input)
        csv_writer = csv.writer(output)
        counter = 0

        rows = []
        for row in csv_reader:
            rows.append(row)

        for row in list(rows): # Swap character randomly
            if row[1] != "flag" and row[1] !='R' and row[1]!="":
                counter += 1
                if counter != 2:
                    row[2] = str(character_swap.augment(row[2]))
                if counter == 3:
                    counter = 0
            csv_writer.writerow(row)

with open("input_classification_test_data.csv","r") as input:
    with open("/Users/wenyaxie/Downloads/negative_data_character_delete.csv","w") as output:
        character_delete = nac.RandomCharAug(action="delete")

        csv_reader = csv.reader(input)
        csv_writer = csv.writer(output)
        counter = 0

        rows = []
        for row in csv_reader:
            rows.append(row)

        for row in list(rows): # Delete character randomly
            if row[1] != "flag" and row[1] !='R' and row[1]!="":
                counter += 1
                if counter != 2:
                    row[2] = str(character_delete.augment(row[2]))
                if counter == 3:
                    counter = 0
            csv_writer.writerow(row)

with open("input_classification_test_data.csv","r") as input:
    with open("/Users/wenyaxie/Downloads/negative_data_split_word.csv","w") as output:
        split_words = naw.SplitAug()

        csv_reader = csv.reader(input)
        csv_writer = csv.writer(output)
        counter = 0

        rows = []
        for row in csv_reader:
            rows.append(row)

        for row in list(rows): # Split word to two tokens randomly
            if row[1] != "flag" and row[1] !='R' and row[1]!="":
                counter += 1
                if counter != 2:
                    row[2] = str(split_words.augment(row[2]))
                if counter == 3:
                    counter = 0
            csv_writer.writerow(row)




