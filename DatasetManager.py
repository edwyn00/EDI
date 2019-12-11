import json
import numpy as np
import random
class DatasetManager:

    def __init__(self, batch_size=20, fields=None, jsonPath="./fixedFinancialDataset.json", n_min=1, n_max=2, seq_var=1):
        self.batch_size = batch_size
        self.seq_var = seq_var
        if fields == 1:
            self.fields = [\
            "Total staff (HC)",\
            "Total staff (FTE)",\
            "Total academic staff (HC)",\
            "Total academic staff (FTE)",\
            "Total graduates at ISCED 5",\
            "Total graduates at ISCED 6",\
            "Total graduates at ISCED 7",\
            "Total graduates at ISCED 8",\
            "Total students enrolled at ISCED 5",\
            "Total students enrolled at ISCED 6",\
            "Total students enrolled at ISCED 7",\
            "Total students enrolled at ISCED 8",\
            ]
        elif fields == None:
            self.fields = ['open', 'high', 'low', 'close', 'percent_change_price', 'percent_change_volume_over_last_wk', 'next_weeks_open', 'next_weeks_close', 'percent_change_next_weeks_price', 'days_to_next_dividend', 'percent_return_next_dividend']
            #self.fields = ['open', 'high', 'low', 'close', 'volume', 'percent_change_price', 'percent_change_volume_over_last_wk', 'previous_weeks_volume', 'next_weeks_open', 'next_weeks_close', 'percent_change_next_weeks_price', 'days_to_next_dividend', 'percent_return_next_dividend']

        else:
            self.fields = fields

        with open(jsonPath) as json_file:
            self.dataset = json.load(json_file)

        if seq_var != 1:
            self.num_entries = 10000
        else:
            self.num_entries = len(self.dataset.keys())
        self.tensorDim = len(self.fields)
        self.n_min = n_min
        self.n_max = n_max

        self.new_epoch()


    def max_value(self):
        self.max = 0.0
        for index in self.dataset.keys():
            for feature in self.dataset[index]:
                if float(feature) > self.max:
                    self.max = feature
        return self.max

    def next_batch(self):
        aux_list = []
        if self.dataPointer + self.batch_size > len(self.dataIndices):
            return None
        for i in range(self.dataPointer, self.dataPointer + self.batch_size):
            if self.seq_var != 1:
                aux_index = random.randint(2, 2 + self.seq_var)
                aux_list.append(np.array(list(range(aux_index, aux_index + self.tensorDim)), dtype=np.float32))#np.array(self.dataset[i]))
            else:
                aux_list.append(np.array(self.dataset[str(i)]))
        ret_np_stacked = np.stack(aux_list)
        self.dataPointer += self.batch_size

        masks = []
        for i in range(self.batch_size):
            mask = np.ones(self.tensorDim)
            num_hidden = random.randint(self.n_min, self.n_max)
            hidden_indices = np.random.randint(0, self.tensorDim, size=num_hidden)
            mask[hidden_indices] -= 1
            masks.append(mask)
        masks = np.stack(masks)

        return ret_np_stacked, masks

    def test_batch(self, dim):
        aux_list = []
        if self.dataPointer + self.batch_size > len(self.dataIndices):
            return None
        for i in range(dim):
            if self.seq_var == 1:
                random_index = random.randint(0, self.num_entries -1)
                aux_list.append(np.array(self.dataset[str(i)]))
            else:
                aux_index = random.randint(2, 2 + self.seq_var)
                aux_list.append(np.array(list(range(aux_index, aux_index + self.tensorDim)), dtype=np.float32))#np.array(self.dataset[i]))

        ret_np_stacked = np.stack(aux_list)

        masks = []
        for i in range(dim):
            mask = np.ones(self.tensorDim)

            num_hidden = random.randint(self.n_min, self.n_max)
            hidden_indices = np.random.randint(0, self.tensorDim, size=num_hidden)
            mask[hidden_indices] -= 1
            masks.append(mask)
        masks = np.stack(masks)

        return ret_np_stacked, masks

    def new_epoch(self):
        self.dataPointer = 0
        self.dataIndices = list(range(self.num_entries))#len(self.dataset.keys())))
        random.shuffle(self.dataIndices)
