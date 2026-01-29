import os
import yaml

def test_list_loading():
    list_path = "data/RoNIN/lists/list_train.txt"
    if not os.path.exists(list_path):
        print(f"File not found: {list_path}")
        return

    with open(list_path) as f:
        lines = f.readlines()
        print(f"Total lines in file: {len(lines)}")
        data_list = [s.strip().split(',' or ' ')[0] for s in lines if len(s) > 0 and s[0] != '#']
        print(f"Loaded data_list length: {len(data_list)}")
        print(f"First 5 items: {data_list[:5]}")

if __name__ == "__main__":
    test_list_loading()
