import csv
import os
import pickle
import random
import sys

def LP(option_name, sample_list, cp_type, perf_list):

    trend = ""
    flag = False
    test_count = 0

    def result_analysis(p1, p2):
        nonlocal trend, test_count
        if option_name == "query_cache_size2" or option_name == "query_cache_size3" or option_name == "query_cache_size1" or option_name == "read_buffer_size1" or option_name == "query_cache_size4":
            trend_analysissp(p1,p2)
        else:
            trend_analysis(p1, p2)
        
        if cp_type == "Optimization":
            if trend == "left":
                return True
        elif cp_type == "Tradeoff":
            if trend == "right":
                if p1 / p2 > 3 and p1 - p2 > 5:
                    print("sp!!!!!!!!!!!!")
                    return True
            elif trend == "left":
                if p1 / p2 > 3 and p1 - p2 > 5:
                    print("sp!!!!!!!!!!!!")
                    return True
        elif cp_type == "Resource":
            if trend == "left":
                return True
        elif cp_type == "Function":
            if trend == "left":
                return True
            elif trend == "right":
                # print(p1, p2)
                if p1 / p2 > 3 and p1 - p2 > 5:
                    print("sp!!!!!!!!!!!!")
                    return True
        elif cp_type == "NONE":
            if trend != "":
                return True

        return False

    def trend_analysis(p1, p2):
        nonlocal trend
        rel_tol = 0.05
        if abs(p1) < abs(p2):
            max_val = abs(p2)
        else:
            max_val = abs(p1)
        if abs(p1 - p2) >= (rel_tol * max(1.8, max_val)):
            if trend == "":
                if p1 > p2:
                    trend = "right"
                else:
                    trend = "left"
        # if trend == "":
        #     print("None")

    def trend_analysissp(p1, p2):
        nonlocal trend
        rel_tol = 0.05
        if abs(p1) < abs(p2):
            max_val = abs(p2)
        else:
            max_val = abs(p1)
        if abs(p1 - p2) >= (rel_tol * max(1.0, max_val)):
            if trend == "":
                num_trend = 1
            
                if p1 < p2:
                    trend = "right"
                else:
                    trend = "left"
            elif trend == "left":
                if p1 < p2:
                    num_trend = 2
            elif trend == "right":
                if p1 > p2:
                    num_trend = 2
        if trend == "":
            print("None")

    max1 = len(sample_list) - 1 


    for i in range(1, len(sample_list)):
            test_count = test_count + 1
            for perf in perf_list:
                flag = result_analysis(perf[0], perf[i])
                if flag:
                    return flag, test_count, (len(sample_list)-1)
    return flag, test_count, (len(sample_list)-1)

def read_from_csv(filename='data.csv'):
    csv_data = {}
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            cat, data = row[0], eval(row[1])
            if cat in csv_data:
                csv_data[cat].append(data)
            else:
                csv_data[cat] = [data]
    return csv_data
  
        

# Main function to start GA
def main():
    # option_name = "sort_buffer_size"
    # sample_list = [32768, 33792, 34816, 36864, 40960, 49152, 65536, 98304, 163840, 294912, 557056, 1081344, 2129920, 4227072, 8421376, 16809984, 33587200, 67141632, 134238208, 268389376, 536600064, 1073208320]
    # cp_type = "Tradeoff"
    # se_type = "pre"
    # perf_list = [[1.223, 9.296, 14.715, 20.188, 35.183, 29.862, 53.651, 42.741, 47.361, 53.563, 89.962, 91.156, 89.33, 97.579, 100.551, 83.185, 141.709, 130.368, 120.874, 120.741, 121.809, 121.964, 121.118, 122.031], [1.3, 10.1, 11.955, 19.256, 30.57, 41.825, 38.44, 59.847, 67.279, 74.284, 83.636, 97.114, 108.88, 120.908, 85.871, 109.685, 147.909, 148.928, 122.356, 117.113, 118.607, 121.946, 118.796, 118.459], [1.225, 10.118, 18.544, 19.784, 34.804, 28.302, 43.295, 50.363, 70.542, 55.132, 79.985, 70.779, 78.772, 100.383, 115.705, 84.389, 118.964, 135.08, 118.264, 122.099, 122.102, 120.406, 118.547, 119.27]]

    # print(LP(option_name, sample_list, cp_type, perf_list))
    vars = []
    cptype = []
    with open("./ndp_predictions.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                text, label = parts
                vars.append(text)
                cptype.append(label)
    

    cpbug_num_list = []
    test_num_list = []
    samp_num_list = []

    for i in range(1, 11):
        filen = "data_mariadb_"
        csv_content = read_from_csv(f"{filen}{i}.csv")

        cpbug_num = 0
        test_num = 0
        samp_num = 0

        for cat in csv_content:
            cp_type = ""
            for j in range(len(vars)):
                if vars[j].lower() in cat.lower():
                    cp_type = cptype[j]

            if cp_type == "":
                print(cat)
            
            sample_list = csv_content[cat][0]
            perf_list = csv_content[cat][1:]

            

            print()
            print(cat)
            print(cp_type)
            print()

            flag, n1, n2 = LP(cat, sample_list, cp_type, perf_list)
            if flag:
                cpbug_num = cpbug_num + 1
            test_num = test_num + n1
            samp_num = samp_num + n2

            cpbug_num_list.append(cpbug_num)
            test_num_list.append(test_num)
            samp_num_list.append(samp_num)

            print(f"cpn:{cpbug_num_list}, tn:{test_num_list}, sn:{samp_num_list}")
            print()
            
        folder_name = f"lp_ndp_mariadb_output_folder_{i}"
        os.makedirs(folder_name, exist_ok=True)

            
        with open(os.path.join(folder_name, "cpbug_num_list.pkl"), 'wb') as f:
                pickle.dump(cpbug_num_list, f)
        with open(os.path.join(folder_name, "test_num_list.pkl"), 'wb') as f:
                pickle.dump(test_num_list, f)
        with open(os.path.join(folder_name, "samp_num_list.pkl"), 'wb') as f:
                pickle.dump(samp_num_list, f)

        
        cpbug_num_list.clear()
        test_num_list.clear()
        samp_num_list.clear()
        
if __name__ == "__main__":
    main()
