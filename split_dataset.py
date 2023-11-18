import json
import random

if __name__ == "__main__":
    random.seed(0)

    data = [None for _ in range(100)]
    data = json.load(open('data/h3wb/annotations/RGBto3D_train.json'))
    num_total_data = len(data)
    print("total data: ", num_total_data)
    keys = list(data.keys())

    indice_list = [i for i in range(num_total_data)]

    random.shuffle(indice_list)
    train_indice = indice_list[: int(num_total_data * 0.8)]
    dev_indice = indice_list[int(num_total_data * 0.8) : int(num_total_data * 0.9)]
    test_indice = indice_list[int(num_total_data * 0.9) :]
    print("train data: ", len(train_indice))
    print("dev data: ", len(dev_indice))
    print("test data: ", len(test_indice))

    # save each data to json file
    train_data = {}
    for i in train_indice:
        train_data[keys[i]] = data[keys[i]]
    dev_data = {}
    for i in dev_indice:
        dev_data[keys[i]] = data[keys[i]]
    test_data = {}
    for i in test_indice:
        test_data[keys[i]] = data[keys[i]]
    
    json.dump(train_data, open("data/h3wb/annotations/train.json", "w"), indent=4)
    json.dump(dev_data, open("data/h3wb/annotations/dev.json", "w"), indent=4)
    json.dump(test_data, open("data/h3wb/annotations/test.json", "w"), indent=4)
    print("done")
