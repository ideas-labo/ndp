import os
import pickle
import random
import sys
import csv

def GA(option_name, sample_list, cp_type, se_type, perf_list):
    # Ensure global variables are not used and replace with parameters

    # Parameters for Genetic Algorithm
    mutation_rate = 0.1
    crossover_rate = 0.9
    max_generations = 100
    flag = False
    
    # Initialize global variables inside the function
    fitness_values = {}
    pre_test_count = 0
    fur_test_count = 0
    comp_test_count = 0
    num_trend = 0
    trend = ""


    def calculate_fitness(individual):
        nonlocal trend, num_trend, pre_test_count, fur_test_count, comp_test_count, flag
        if str(individual) in fitness_values.keys():
            return fitness_values[str(individual)]

        if se_type == "pre":
            pre_test_count += 1
            print(f"pre_test_count:{pre_test_count}")
        elif se_type == "fur":
            fur_test_count += 1
            print(f"fur_test_count:{fur_test_count}")
        else:
            comp_test_count += 1
            print(f"comp_test_count:{comp_test_count}")
        fitness = 0

        for perf in perf_list:
            print(individual)
            print(perf[individual[0]], perf[individual[1]])
            print(trend)
            flag = result_analysis(perf[individual[0]], perf[individual[1]])
            fitness = max(abs(perf[individual[0]] - perf[individual[1]]), fitness)
            fitness_values[str(individual)] = fitness
            if flag:                
                return fitness
            else:
                fitness = max(abs(perf[individual[0]] - perf[individual[1]]), fitness)

        fitness_values[str(individual)] = fitness
        return fitness

    def result_analysis(p1, p2):
        nonlocal trend, num_trend
        if option_name == "query_cache_size2" or option_name == "query_cache_size3" or option_name == "query_cache_size1" or option_name == "read_buffer_size1" or option_name == "query_cache_size4":
            trend_analysissp(p1,p2)
        else:
            trend_analysis(p1, p2)
        if num_trend == 2:
            return True
        
        if cp_type == "Optimization":
            if trend == "left":
                return True
        elif cp_type == "Tradeoff":
            if trend == "right":
                return True
            elif trend == "left":
                return True
        elif cp_type == "Resource":
            if trend == "left":
                return True
        elif cp_type == "Function":
            if trend == "left":
                return True
            elif trend == "right":
                if p1 / p2 > 3 and p1 - p2 > 5:
                    return True
        elif cp_type == "NONE":
            if trend != "":
                return True

        return False

    def trend_analysis(p1, p2):
        nonlocal trend, num_trend
        rel_tol = 0.05
        if abs(p1) < abs(p2):
            max_val = abs(p2)
        else:
            max_val = abs(p1)
        if abs(p1 - p2) >= (rel_tol * max(1.0, max_val)):
            if trend == "":
                num_trend = 1
            
                if p1 > p2:
                    trend = "right"
                else:
                    trend = "left"
            elif trend == "left":
                if p1 > p2:
                    num_trend = 2
            elif trend == "right":
                if p1 < p2:
                    num_trend = 2
        if trend == "":
            print("None")

    def trend_analysissp(p1, p2):
        nonlocal trend, num_trend
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

    def rank_selection(population_size):  
         
        total_rank = population_size * (population_size + 1) / 2 
       
        probabilities = [i / total_rank for i in range(1, population_size + 1)] 
        
        selected_index = random.choices(range(population_size), weights=probabilities, k=1)[0]  
        return population[selected_index]

    def uniform_crossover(parent1, parent2):
        child = []
        flag = 1
        while flag:
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            if child[0] != child[1]:
                flag = 0
                return child
            else:
                child = []

        return child

    # def mutation_normal(individual, sample_list, mutation_rate):
    #     mutated_individual = []
    #     flag = 1
    #     while flag:
    #         for gene in individual:
    #             if random.random() < mutation_rate:
    #                 mutated_gene = random.randint(0, len(sample_list) - 1)
    #                 mutated_individual.append(mutated_gene)
    #             else:
    #                 mutated_individual.append(gene)
    #         if mutated_individual[0] != mutated_individual[1]:
    #             flag = 0
    #             return mutated_individual
    #         else:
    #             mutated_individual = []

    #     return mutated_individual
    def mutation_special(individual, sample_list, se_type):
        if len(sample_list) == 0 or len(sample_list) == 1:
            return individual  
        
        bound1 = max(int(len(sample_list) / 10), 1) - 1
        bound2 = max(int(len(sample_list) / 10), 1)
        bound3 = len(sample_list) - max(int(len(sample_list) / 10), 1) - 1
        bound4 = len(sample_list) - max(int(len(sample_list) / 10), 1)
        bound5 = len(sample_list) - 1

        mutated_individual = []

        flag = 1
        mutation = [random.random() < mutation_rate, random.random() < mutation_rate]
        while flag:  
            if mutation[0] == False and mutation[1] == False:
                break
            mutated_individual = []
            for i,gene in enumerate(individual):
                  
                if mutation[i]:                   
                    mutated_gene = gene 
                    print(se_type)
                    if se_type == "pre":
                        if gene < individual[1]:
                            mutated_gene = random.randint(0, bound1)
                        else :
                            mutated_gene = random.randint(bound4, bound5)
                    elif se_type == "fur":
                        if len(mutated_individual) == 0:
                            if random.random() < 0.5:
                                if random.random() < 0.5:
                                    mutated_gene = random.randint(0, bound1)
                                else:
                                    mutated_gene = random.randint(bound4, bound5)
                            else :
                                mutated_gene = random.randint(bound2, bound3)
                        elif mutated_individual[0] <= bound1 or  mutated_individual[0] >= bound4:
                            mutated_gene = random.randint(bound2, bound3)
                        elif random.random() < 0.5:
                            mutated_gene = random.randint(0, bound1)
                        else:
                            mutated_gene = random.randint(bound4, bound5)                       
                    elif se_type == "comp":
                        mutated_gene = random.randint(0, bound5)

                    mutated_individual.append(mutated_gene)  
                else:
                    mutated_individual.append(gene)
                    

            print(mutated_individual)
            if mutated_individual[0] != mutated_individual[1]:
                if mutated_individual[0] > mutated_individual[1]:
                    mutated_individual[0], mutated_individual[1] = mutated_individual[1], mutated_individual[0]
                flag = 0

            else:
                mutated_individual = []

        if mutated_individual == []:
            return individual
        else:
            return mutated_individual 

    def initialize_population_special(sample_list, se_type):
        population = []
        if len(sample_list) == 0 or len(sample_list) == 1:
            return population
        elif len(sample_list) == 2:
            population.append([0, 1])
            return population
        
        size1 = (max(int(len(sample_list) / 10), 1)) ** 2
        size2 = 2 * max(int(len(sample_list) / 10), 1) * (len(sample_list) - 2 * max(int(len(sample_list) / 10), 1))
        size3 = int((len(sample_list) * (len(sample_list) - 1)) / 2)

        bound1 = max(int(len(sample_list) / 10), 1) - 1
        bound2 = max(int(len(sample_list) / 10), 1)
        bound3 = len(sample_list) - max(int(len(sample_list) / 10), 1) - 1
        bound4 = len(sample_list) - max(int(len(sample_list) / 10), 1)
        bound5 = len(sample_list) - 1

        if se_type == "pre":
            population_size = min(size1, 10)
            while True:
                gene1 = random.randint(0, bound1)
                gene2 = random.randint(bound4, bound5)

                individual = [gene1, gene2]
                if individual in population:
                    continue
                population.append(individual)
                if len(population) >= population_size:
                    return population

        elif se_type == "fur":
            population_size = min(10, size2)
            while True:
                if random.random() < 0.5:
                    gene1 = random.randint(0, bound1)
                    gene2 = random.randint(bound2, bound3)
                else:
                    gene1 = random.randint(bound2, bound3)
                    gene2 = random.randint(bound4, bound5)

                individual = [gene1, gene2]
                if individual in population:
                    continue
                population.append(individual)
                if len(population) >= population_size:
                    return population

        elif se_type == "comp":
            population_size = min(10, size3)
            while True:
                gene1 = random.randint(0, bound5)
                gene2 = random.randint(0, bound5)
                while gene1 == gene2:
                    gene2 = random.randint(0, bound5)
                if gene1 > gene2:
                    gene1, gene2 = gene2, gene1
                individual = [gene1, gene2]
                if individual in population:
                    continue
                population.append(individual)
                if len(population) >= population_size:
                    return population
        

    population = initialize_population_special(sample_list, se_type)
    if len(sample_list) == 0 or len(sample_list) == 1:
        print(f"pop only {len(population)}, no pairs")
        return
    
    limit1 = max(int(len(sample_list) / 10), 1) ** 2
    limit2 = max(int(len(sample_list) / 10), 1) * (len(sample_list) - 2 * max(int(len(sample_list) / 10), 1))
    limit3 = len(sample_list) * (len(sample_list)-1) / 2

    # Main GA loop
    for generation in range(max_generations):
        if pre_test_count+fur_test_count+comp_test_count >=limit3:
            return limit3, limit3
        print(f"gen: {generation}")
        mutated_population = []
        for i in population:
            calculate_fitness(i)
            if flag:
                print("return")
                test_num = pre_test_count+fur_test_count+comp_test_count
                if se_type == "pre" :
                    if test_num > 10:
                        return flag, test_num, test_num
                    elif test_num <= 10 and limit1 <= 10 :
                        return flag, test_num, max(limit1, test_num)
                    elif test_num <= 10 and limit1 > 10:
                        return flag, test_num, 10
                elif se_type == "fur" :
                    if test_num > 10:
                        return flag, test_num, test_num
                    elif test_num <= 10 and limit2 <= 10 :
                        return flag, test_num, max(limit2, test_num)
                    elif test_num <= 10 and limit2 > 10:
                        return flag, test_num, 10
                elif se_type == "comp":
                    if test_num > 10:
                        return flag, test_num, test_num
                    elif test_num <= 10 and limit3 <= 10 :
                        return flag, test_num, max(limit3, test_num)
                    elif test_num <= 10 and limit3 > 10:
                        return flag, test_num, 10

        population = sorted(population, key=lambda x: calculate_fitness(x), reverse=True)

        


        if se_type == "pre":
            if pre_test_count >= limit1:
                if fur_test_count == 0:
                    se_type = "fur"
                else:
                    se_type = "comp"
        elif se_type == "fur":
            if fur_test_count >= limit2:
                if pre_test_count == 0:
                    se_type = "pre"
                else:
                    se_type = "comp"
        
        for _ in range(int(len(population)/2)):  
             
            parent1 = rank_selection(len(population))  
            parent2 = rank_selection(len(population))  

            while True:
    
                  
                if random.random() < crossover_rate:  
                    child = uniform_crossover(parent1, parent2)  
                else:  
                    child = parent1  
        
                
                mutated_child = mutation_special(child,sample_list, se_type)


                if mutated_child[0] > mutated_child[1]:  
                    mutated_child[0], mutated_child[1] = mutated_child[1], mutated_child[0]

                if mutated_child not in mutated_population:
                    break  

            mutated_population.append(mutated_child)

        for i in mutated_population:
            calculate_fitness(i)
            if flag:
                print("return")
                test_num = pre_test_count+fur_test_count+comp_test_count
                if se_type == "pre" :
                    if test_num > 10:
                        return flag, test_num, test_num
                    elif test_num <= 10 and limit1 <= 10 :
                        return flag, test_num, max(limit1, test_num)
                    elif test_num <= 10 and limit1 > 10:
                        return flag, test_num, 10
                elif se_type == "fur" :
                    if test_num > 10:
                        return flag, test_num, test_num
                    elif test_num <= 10 and limit2 <= 10 :
                        return flag, test_num, max(limit2, test_num)
                    elif test_num <= 10 and limit2 > 10:
                        return flag, test_num, 10
                elif se_type == "comp":
                    if test_num > 10:
                        return flag, test_num, test_num
                    elif test_num <= 10 and limit3 <= 10 :
                        return flag, test_num, max(limit3, test_num)
                    elif test_num <= 10 and limit3 > 10:
                        return flag, test_num, 10

        mutated_population = sorted(mutated_population, key=lambda x: calculate_fitness(x), reverse=True)
    

        total_rank = len(population) * (len(population) + 1) / 2 
        probabilities = [(len(population) - i) / total_rank for i in range(1, len(population) + 1)] 
        selected_index = random.choices(range(len(population)), weights=probabilities, k=1)[0]

        for i in range(0, len(mutated_population)):
            if mutated_population[i] == population[selected_index]:
                break
            elif mutated_population[i] not in population:
                population[selected_index] = mutated_population[i]  

        population = sorted(population, key=lambda x: calculate_fitness(x), reverse=True) 


        best_individual = population[0]  
        best_fitness = calculate_fitness(best_individual)  
        

    return flag, pre_test_count+fur_test_count+comp_test_count, pre_test_count+fur_test_count+comp_test_count
        
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

def contains_substring(lst, string):
    for item in lst:
        if item in string:
            return True
    return False

# Main function to start GA
def main():
    # option_name = "sort_buffer_size"
    # sample_list = [32768, 33792, 34816, 36864, 40960, 49152, 65536, 98304, 163840, 294912, 557056, 1081344, 2129920, 4227072, 8421376, 16809984, 33587200, 67141632, 134238208, 268389376, 536600064, 1073208320]
    # cp_type = "Tradeoff"
    # se_type = "pre"
    # perf_list = [[1.223, 9.296, 14.715, 20.188, 35.183, 29.862, 53.651, 42.741, 47.361, 53.563, 89.962, 91.156, 89.33, 97.579, 100.551, 83.185, 141.709, 130.368, 120.874, 120.741, 121.809, 121.964, 121.118, 122.031], [1.3, 10.1, 11.955, 19.256, 30.57, 41.825, 38.44, 59.847, 67.279, 74.284, 83.636, 97.114, 108.88, 120.908, 85.871, 109.685, 147.909, 148.928, 122.356, 117.113, 118.607, 121.946, 118.796, 118.459], [1.225, 10.118, 18.544, 19.784, 34.804, 28.302, 43.295, 50.363, 70.542, 55.132, 79.985, 70.779, 78.772, 100.383, 115.705, 84.389, 118.964, 135.08, 118.264, 122.099, 122.102, 120.406, 118.547, 119.27]]

    # print(GA(option_name, sample_list, cp_type, se_type, perf_list))
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
    
    prev = []
    with open("./pre.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:       
            prev.append(line)

    furv = []
    with open("./fur.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:       
            furv.append(line)
    
    compv = []
    with open("./comp.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:       
            compv.append(line)

    cpbug_num_list = []
    test_num_list = []
    samp_num_list = []

    for i in range(1, 11):
        filen = "data_mysql_0.csv"
        csv_content = read_from_csv(filen)

        cpbug_num = 0
        test_num = 0
        samp_num = 0

        for cat in csv_content:
            cp_type = ""

            if contains_substring(prev, cat):
                se_type = "pre"
            elif contains_substring(furv, cat):
                se_type = "fur"
            elif contains_substring(compv, cat):
                se_type = "comp"
        
            for j in range(len(vars)):
                print(vars[j].lower(), cat.lower())
                if vars[j].lower() in cat.lower():
                    cp_type = cptype[j]
            
            if cp_type == "":
                print(cat)

            sample_list = csv_content[cat][0]
            perf_list = csv_content[cat][1:]

            

            # print()
            # print(cat)
            # print(cp_type)
            # print(se_type)
            # # print(sample_list)
            # # print(perf_list)
            # print()

            flag, n1, n2 = GA(cat, sample_list, cp_type, se_type, perf_list)
            if flag:
                cpbug_num = cpbug_num + 1
            test_num = test_num + n1
            samp_num = samp_num + n2

            print(cpbug_num, test_num, samp_num)

            cpbug_num_list.append(cpbug_num)
            test_num_list.append(test_num)
            samp_num_list.append(samp_num)

        print(f"cpn:{cpbug_num_list}, tn:{test_num_list}, sn:{samp_num_list}")


        folder_name = f"ndp_ndp_{system}_output_folder_{i}"
        os.makedirs(folder_name, exist_ok=True)

            # 保存状态到文件
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
