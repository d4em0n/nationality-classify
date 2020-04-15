from random import Random
import string
import unidecode

MAX_LEN = 42
class NameDataset:
    def __init__(self, train_file, test_file, ratio):
        self.datas = []
        self.train_file = train_file
        self.test_file = test_file
        self.ratio = ratio

    def add_data_country(self, loc_file, id, max_row=-1):
        with open(loc_file) as f:
            fdatas = f.read().strip().split("\n")
            print(loc_file)
            Random(1337).shuffle(fdatas)
            datas = []
            for fdata in fdatas[:max_row]:
                data = unidecode.unidecode(fdata) # clear unicode char to ascii
                data = [data]
                data.append(id)
                datas.append(data)
            self.datas.extend(datas)

    def write(self):
        Random(1337).shuffle(self.datas)
        print(len(self.datas))
        n = int(round(len(self.datas)*self.ratio))
        print(n)
        test_data = self.datas[:n]
        train_data = self.datas[n:]
        with open(self.train_file, "w+") as f:
            f.write("nama,country\n")
            for data in train_data:
                f.write(",".join(map(str, data)) + "\n")

        with open(self.test_file, "w+") as f:
            f.write("nama,country\n")
            for data in test_data:
                f.write(",".join(map(str, data)) + "\n")

rp = "datasets/"
len_row = 9800 # max row
dataset = NameDataset("train.csv", "evaluation.csv", 0.2)
dataset.add_data_country(rp + "russian_name_dataset.txt", 0, len_row)
dataset.add_data_country(rp + "chinese_name_dataset.txt", 1, len_row)
dataset.add_data_country(rp + "arabic_name_dataset.txt", 2, len_row)
dataset.add_data_country(rp + "german_name_dataset.txt", 3, len_row)
dataset.add_data_country(rp + "korean_name_dataset.txt", 4, len_row)
dataset.add_data_country(rp + "polish_name_dataset.txt", 5, len_row)
dataset.add_data_country(rp + "scottish_name_dataset.txt", 6, len_row)
dataset.add_data_country(rp + "italian_name_dataset.txt", 7, len_row)
dataset.add_data_country(rp + "english_name_dataset.txt", 8, len_row)
dataset.add_data_country(rp + "french_name_dataset.txt", 9, len_row)
dataset.add_data_country(rp + "japanese_name_dataset.txt", 10, len_row)
dataset.add_data_country(rp + "greece_name_dataset.txt", 11, len_row)
dataset.add_data_country(rp + "spanish_name_dataset.txt", 12, len_row)
dataset.add_data_country(rp + "india_name_dataset.txt", 13, len_row)
dataset.add_data_country(rp + "turkish_name_dataset.txt", 14, len_row)
dataset.add_data_country(rp + "indonesia_name_dataset.txt", 15, len_row)
dataset.add_data_country(rp + "vietname_name_dataset.txt", 16, len_row)
dataset.add_data_country(rp + "czech_name_dataset.txt", 17, len_row)
dataset.write()
