import os
import pickle
import numpy as np


def filter_lists(cpbug_num_list, test_num_list, samp_num_list):

    last_occurrence = {}
    for i, num in enumerate(cpbug_num_list):
        last_occurrence[num] = i
    
    
    filtered_cpbug_num_list = []
    filtered_test_num_list = []
    filtered_samp_num_list = []
    
   
    for i, num in enumerate(cpbug_num_list):
        if i == last_occurrence[num] and num != 0:
            filtered_cpbug_num_list.append(num)
            filtered_test_num_list.append(test_num_list[i])
            filtered_samp_num_list.append(samp_num_list[i])
    
    return filtered_cpbug_num_list, filtered_test_num_list, filtered_samp_num_list

def read_data(folder_name):
    with open(os.path.join(folder_name, "cpbug_num_list.pkl"), 'rb') as f:
        cpbug_num_list = pickle.load(f)
    with open(os.path.join(folder_name, "test_num_list.pkl"), 'rb') as f:
        test_num_list = pickle.load(f)
    with open(os.path.join(folder_name, "samp_num_list.pkl"), 'rb') as f:
        samp_num_list = pickle.load(f)
    
    return cpbug_num_list, test_num_list, samp_num_list



def get_cpbug_list(cpbug_num_list, test_num_list, unique_percentages):
    new_list = []
    
    for value in unique_percentages:
        if value < test_num_list[0]:
            new_list.append(0)
        elif value > test_num_list[-1]:
            new_list.append(cpbug_num_list[-1])
        else:
            for i in range(len(test_num_list) - 1):
                if test_num_list[i] <= value < test_num_list[i + 1]:
                    new_list.append(cpbug_num_list[i])
                    break
                elif value == test_num_list[i]:
                    new_list.append(cpbug_num_list[i])
                    break
            else:
                if value == test_num_list[-1]:
                    new_list.append(cpbug_num_list[-1])
    
    return new_list


cpbug_num_all = []
test_num_all = []
samp_num_all = []

system = "apache"
pred = "ndp"
sw = "ndp"


for i in range(1, 11):
    ndpfolder_name = f"{sw}_{pred}_{system}_output_folder_{i}"
    lpfolder_name = f"lp_{pred}_{system}_output_folder_{i}"
    usfolder_name = f"us_{pred}_{system}_output_folder_{i}"

    folder_name = ndpfolder_name
    
    if os.path.exists(folder_name):
        cpbug_num_list, test_num_list, samp_num_list = read_data(folder_name)
        cpbug_num_list, test_num_list, samp_num_list = filter_lists(cpbug_num_list, test_num_list, samp_num_list)
        cpbug_num_all.append(cpbug_num_list)
        test_num_all.append(test_num_list)
        samp_num_all.append(samp_num_list)

    
last_elements = [sublist[-1] for sublist in test_num_all]


max_value = max(last_elements)


percentages = [int(max_value * i / 100) for i in range(10, 101, 10)]
percentages = [x for x in percentages if x != 0]

unique_percentages = list(set(percentages))

unique_percentages.sort()  

print(unique_percentages)

new_list_all = []

for i in range(len(cpbug_num_all)):
    # print(cpbug_num_all[i])
    # print(test_num_all[i])
    new_list = get_cpbug_list(cpbug_num_all[i], test_num_all[i], unique_percentages)
    new_list_all.append(new_list)
    # print()

a_np = np.array(new_list_all)


means = np.mean(a_np, axis=0)
std_devs = np.std(a_np, axis=0)


mean_plus_std = means + std_devs


mean_minus_std = means - std_devs


for i in range(len(unique_percentages)):
    print(f"({unique_percentages[i]}, {means[i]:.2f})")
    
print("Sum of mean and standard deviation:")
for i in range(len(unique_percentages)):
    print(f"({unique_percentages[i]}, {mean_plus_std[i]:.2f})")

print("Difference between mean and standard deviation:")
for i in range(len(unique_percentages)):
    print(f"({unique_percentages[i]}, {mean_minus_std[i]:.2f})")