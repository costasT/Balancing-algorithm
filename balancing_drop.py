import csv
import ast
import copy
import sys
import itertools
from math import factorial, sqrt

def loadDataset(filename):
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader, None)
        list_dataset = list(spamreader)#
        return list_dataset

def init(config):
    dit = ast.literal_eval(config)
    structure = []
    for i in range(len(dit)):
        add_up_breakpoints = 0
        for k in dit[i]:
            format_dit = {}
            format_dit['column'] = k
            format_dit['labels'] = []
            for m in dit[i][k]:
                label = {}
                label['label'] = m
                label['breakpoint'] = dit[i][k][m]
                add_up_breakpoints += label['breakpoint']
                label['count'] = 0
                label['percentage'] = 0
                format_dit['labels'].append(label)
        structure.append(format_dit)
        if add_up_breakpoints < 99:
            print("Column's",k,"percentage to not add up to 100. Current sum:",add_up_breakpoints)
            sys.exit()
    print(structure)
    return structure

def calculatePercentages(dataset, structure):
    for row in dataset:
        for value in row:
            for s in structure:
                for label in s['labels']:
                    if value == label['label']:
                        label['count']+=1
                        break

    for s in structure:
        for label in s['labels']:
            if label['count'] == 0:
                print("Cannot find any entries for: ", label)
                sys.exit()
            label['percentage'] = (label['count'] * 100 / len(dataset))

    print("##########################################")

    print(structure)
    return structure

def find_optimal_entry_to_remove_min_first(final_structure):
    labels_to_subtract = []
    labels_to_inject = []
    convergence_distance = 0
    for s in final_structure:
        diff = 999 # -1 for biggest
        inject_diff = 999
        label_oi = ""
        for label in s['labels']:
            distance = abs(label['percentage'] - label['breakpoint'])
            convergence_distance+= distance
            per_diff = label['percentage'] - label['breakpoint']
            inject_per_diff = label['breakpoint'] - label['percentage']
            if per_diff > 0:
                if per_diff < diff: #Also test biggest
                    diff = per_diff
                    label_oi = label['label']

        if (label_oi is None):
            print("No suitable entries found for removal")
            print("Dataset cannot coverge")
            sys.exit()

        labels_to_subtract.append(label_oi)

    return labels_to_subtract, convergence_distance

def find_optimal_entry_to_remove_max_first(final_structure):
    labels_to_subtract = []
    label_deviations = []
    labels_to_inject = []
    convergence_distance = 0
    for s in final_structure:
        diff = -1
        inject_diff = -1
        label_oi = ""
        for label in s['labels']:
            distance = abs(label['percentage'] - label['breakpoint'])
            convergence_distance+= distance
            per_diff = label['percentage'] - label['breakpoint']
            inject_per_diff = label['breakpoint'] - label['percentage']
            if per_diff > 0:
                if per_diff > diff: #Also test biggest
                    diff = per_diff
                    label_oi = label['label']
                    label_deviation = per_diff

        if (label_oi is None):
            print("No suitable entries found for removal")
            print("Dataset cannot coverge")
            sys.exit()
        labels_to_subtract.append(label_oi)

    return labels_to_subtract, convergence_distance

def create_column_priotiry_matching_table(final_structure, label_convergence_margin):

    column_matching_table = []
    converged_columns = 0
    for i in range(len(final_structure)):
        column_convergence_distance = 0
        column_labels = final_structure[i]['labels']
        column_converged = True
        sum=0
        for label in column_labels:
            column_convergence_distance += abs(label['percentage'] - label['breakpoint'])
            sum += (label['percentage'] - label['breakpoint']) * (label['percentage'] - label['breakpoint'])
            # if abs(label['percentage'] - label['breakpoint']) > label_convergence_margin:
            #     column_converged = False
        sum = sqrt(sum)
        if(sum>label_convergence_margin):
            column_converged = False
        if column_converged:
            converged_columns += 1
            column_matching_table.append(0)
        else:
            column_matching_table.append(column_convergence_distance)
    return column_matching_table, converged_columns

def sort_indexes_combinations_according_to_weights(indexes, weights, reverse):
    if reverse:
        return [x for _,x in sorted(zip(weights,indexes), reverse=True)]
    else:
        return [x for _,x in sorted(zip(weights,indexes))]

def is_entry_suitable(column_matching_table, most_suitable_entry_to_remove, row, number_of_columns_to_match, current_permutation, final_structure, reverse):

    indexes = list(range(len(row)-1))
    # Ignore converged columns
    for i, e in reversed(list(enumerate(column_matching_table))):
        if e <= 0:
            del indexes[i]

    all_combinations = list(itertools.combinations(indexes, min(number_of_columns_to_match, len(indexes))))

    weights = []
    for i in all_combinations:
        weight = 0;
        for j in i:
            weight += column_matching_table[j]
        weights.append(weight)

    sorted_indexes = sort_indexes_combinations_according_to_weights(all_combinations, weights, reverse)
    for i in sorted_indexes[current_permutation]:
        # TODO the row[i+1] should be fixed. The column number should be extracted from final_structure. This assumes that always the first column is the ID and ignores it.
        if most_suitable_entry_to_remove[i] != row[final_structure[i]['column']]:
            return False
    return True

def even_out(dataset, final_structure, label_convergence_margin):
    iteration_counter = 0
    while True: # While not converged
        iteration_counter+=1
        print(iteration_counter)
        # find labels of entry that must be deleted by calculating biggest percentage difference
        labels_to_subtract, convergence_distance = find_optimal_entry_to_remove_min_first(final_structure)
        print(convergence_distance)

        column_matching_table, converged_columns = create_column_priotiry_matching_table(final_structure, label_convergence_margin)

        suitable_entry = False
        number_of_columns_not_converged = len(final_structure) - converged_columns
        number_of_columns_to_match = min(len(final_structure), number_of_columns_not_converged)
        # print(number_of_columns_not_converged)
        # print(number_of_columns_to_match)
        if number_of_columns_not_converged == 0:
            break;
        current_permutation = 0

        found_to_subtract = False
        # Loop until finding an entry to either drop or inject
        while not suitable_entry:
            number_of_permutations = round(factorial(number_of_columns_not_converged)/(factorial(number_of_columns_to_match)*factorial(number_of_columns_not_converged-number_of_columns_to_match)))
            has_been_injected = False
            # Loop - for each row in the dataset
            for i in range(len(dataset)):
                if len(dataset) <= 1:
                    print("Dataset exhausted without convergence.")
                    print("Please consider changing the papameters provided.")
                    sys.exit()
                # If found a suitable entry from removal, keep its index (line)
                if not found_to_subtract:
                    if is_entry_suitable(column_matching_table, labels_to_subtract, dataset[i], number_of_columns_to_match, current_permutation, final_structure, True):
                        found_to_subtract = True
                        index_subtract = i

                if found_to_subtract:
                    print("Drop(",number_of_columns_to_match,")")
                    # Update the structure that holds the entries per label per column, since one row will be deleted
                    for s in final_structure:
                        for label in s['labels']:
                            if label['label'] in dataset[index_subtract]:
                                label['count']-=1
                    del dataset[index_subtract]
                    suitable_entry = True
                    break
            # If no entry was found for either injection or removal, try to match an entry using n-1 (number_of_columns_to_match-1) columns
            if current_permutation+1 < number_of_permutations:
                current_permutation += 1
            else:
                current_permutation = 0
                number_of_columns_to_match -=1

        # Update the parcentages after the changes made to the dataset
        for s in final_structure:
            for label in s['labels']:
                label['percentage'] = label['count'] * 100 / len(dataset)

        outmost_counter = 0
        # Identify labels that have been converged
        for s in final_structure:
            s_length = len(s['labels'])
            sum = 0
            column_converged = True
            for label in s['labels']:
                sum += (label['percentage'] - label['breakpoint']) * (label['percentage'] - label['breakpoint'])
            sum = sqrt(sum)
            if(sum>label_convergence_margin):
                column_converged = False
            # If all labels of a column have been converged, mark the column as converged
            if column_converged:
                outmost_counter+=1
            else:
                break
        # If all column have been converged, break the loop
        if outmost_counter == len(final_structure):
            break

    print("##########################################")
    print(final_structure)
    return dataset

def calculate_new_distances(final_structure, entry_to_subtract, entry_to_inject, dataset_length):
    subtract_distance = 0
    inject_distance = 0
    counter = 0
    for s in final_structure:
        for label in s['labels']:
            if label['label'] == entry_to_subtract[counter]:
                new_percentage = (label['count'] - 1) / (dataset_length - 1) * 100
            else:
                new_percentage = (label['count']) / (dataset_length - 1) * 100
            subtract_distance += abs(new_percentage - label['breakpoint'])

            if label['label'] == entry_to_inject[counter]:
                new_percentage = (label['count'] + 1) / (dataset_length + 1) * 100
            else:
                new_percentage = (label['count']) / (dataset_length + 1) * 100
            inject_distance += abs(new_percentage - label['breakpoint'])

        counter+=1
    return subtract_distance, inject_distance

def write_to_csv(new_csv_name, final_set):
    with open(new_csv_name + '.csv', 'w', newline='') as csvfile:
         writer = csv.writer(csvfile, delimiter=',')
         for i in range(len(final_set)):
             writer.writerow(final_set[i])

def run(config, filename, new_csv_name, label_convergence_margin):
  structure = init(config)
  dataset = loadDataset(filename)
  final_structure = calculatePercentages(dataset, structure)
  final_set = even_out(dataset, final_structure, label_convergence_margin)
  write_to_csv(new_csv_name, final_set)

def main():
    # configuration for MyDataBug.csv
    # config = "{1: {'degreeHigh':9.9,'noDegreeHigh':39.3,'degreeLow':20.5,'noDegreeLow':30.2}},{2: {'interest':69.7,'LowInterest':30.3}},{3: {'Male':48.6,'Female':51.4}},{4: {'CONS':21.41,'LAB':21.64,'LD':6.02,'other':4.6,'UKIP':25.04,'GRN':6.89,'Undec':14.13}}"
    # configuration for MyData.csv
    # config = "{1: {'degreeHigh':10.2,'noDegreeHigh':38.3,'degreeLow':21.7,'noDegreeLow':29.8}},{2: {'interest':50.6,'LowInterest':49.4}},{3: {'Male':48.1,'Female':51.9}},{4: {'FG':36.1,'LAB':19.5,'FF':17.5,'SF':9.9,'IND':12.1,'other':4.9}}"
    # config = "{1: {'degreeHigh':4.6,'noDegreeHigh':44.7,'degreeLow':13.8,'noDegreeLow':36.9}},{2: {'interest':47.2,'LowInterest':52.8}},{3: {'Male':47.5,'Female':52.5}},{4: {'AP':23.78,'PS':27.01,'CDU':10.89,'MPT':10.66,'BE':3.91,'other':3.15,'Undec':20.6}}"
    # england
    config = "{1: {'degreeHigh':9.9,'noDegreeHigh':39.3,'degreeLow':20.5,'noDegreeLow':30.2}},{2: {'interest':69.7,'LowInterest':30.3}},{3: {'Male':48.6,'Female':51.4}},{4: {'CONS':22.70,'LAB':22.94,'LD':6.38,'UKIP':26.54,'GRN':7.30,'Undec':14.13}}"
    filename = 'random10K_MyData.csv'
    run(config, filename, "random10K_output", 3)

    # config = "{1: {'degreeHigh':10.2,'noDegreeHigh':38.3,'degreeLow':21.7,'noDegreeLow':29.8}},{2: {'interest':50.6,'LowInterest':49.4}},{3: {'Male':48.1,'Female':51.9}},{4: {'FG':18.92,'SF':16.57,'FF':18.95,'LAB':4.52,'GRN':4.18,'IND':16.86,'Undec':20.00}}"
    # filename = 'MyData_ireland.csv'
    # run(config, filename, "ireland", 2.5)

#main()
