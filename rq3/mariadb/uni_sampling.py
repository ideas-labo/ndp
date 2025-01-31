import csv
import os
import pickle
import random
import sys

def US(option_name, sample_list, cp_type, perf_list):

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

    max1 = (len(sample_list) - 1) * len(sample_list) / 2 

    for i in range(1, len(sample_list)):
        for j in range(i):
            test_count = test_count + 1
            for perf in perf_list:
                flag = result_analysis(perf[j], perf[i])
                if flag:
                    print(option_name, flag, len(sample_list))
                    return flag, test_count, int(len(sample_list)*(len(sample_list)-1)/2)
    print(option_name, flag, len(sample_list))
    return flag, test_count, int(len(sample_list)*(len(sample_list)-1)/2)

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

        


def main():
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

            

            # print()
            # print(cat)
            # print(cp_type)
            # print()
            # # print(sample_list)
            # # print(perf_list)

            flag, n1, n2 = US(cat, sample_list, cp_type, perf_list)
            if flag:
                cpbug_num = cpbug_num + 1
            test_num = test_num + n1
            samp_num = samp_num + n2

            cpbug_num_list.append(cpbug_num)
            test_num_list.append(test_num)
            samp_num_list.append(samp_num)

            print(f"cpn:{cpbug_num_list}, tn:{test_num_list}, sn:{samp_num_list}")
            print()


        folder_name = f"us_ndp_mariadb_output_folder_{i}"
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
