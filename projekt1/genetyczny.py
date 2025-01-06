import numpy as np
import random
import time
import pandas as pd
import os


# funkcja licząca długosc trasy
def calculate_route(ordering):
    travel_route = 0
    for i in range(len(ordering)):
        if(i < len(ordering)-1):
            travel_route = travel_route + dist[ordering[i], ordering[i+1]]
        else:
            travel_route = travel_route + dist[ordering[0], ordering[i]]
    return travel_route




# funkcja do algorytmu najblizszego sasiada
def closest_neighbour_solutions(dist, n_solutions):
    return random.choices(solutions, k=n_solutions)
        


# initial_solution_method: "closest_neighbour" | "random"
# parent_selection_method: "sorted_fitness" | "probabilistic_tournament" | "tournament"
# population_count: int
# n_iterations: int
# probability_of_mutation: int in (0, 1)
# crossing_method: "OX" | "PMX" | "CX"

# Opcjonalne, dla turniejów:
# probability_of_winning_tournament: int in (0, 1)
# tournament_size: int

# Funkcja realizująca algorytm
def genetic_algorithm(dist, initial_solution_method, parent_selection_method, population_count, n_iterations, probability_of_mutation, crossing_method, probability_of_winning_tournament, tournament_size):
    start_gen = time.time()
    # inicjalizacja populacji
    if(initial_solution_method == "random"): # losowo
        ordering = list(range(len(dist)))
        population = []
        for i in range(population_count):
            random.shuffle(ordering)
            random_order = ordering[:]
            population.append(random_order)
    else:
        population = closest_neighbour_solutions(dist, population_count) # closest_neighbour
    
    for n in range(n_iterations):
        # ========================== WYBÓR RODZICÓW ==============================
        parent_pairs = []
        if(parent_selection_method == "sorted_fitness"):
            # ranking dopasowania
            for j in range(population_count):
                inverted_exponentiated_fitness = []
                for i in range(len(population)):
                    # skalujemy przez f(x) = e^(-x), tak by większa droga skutkowała mniejszym prawdopodobieństwem
                    inverted_exponentiated_fitness.append(np.exp(- calculate_route(population[i])/1000)) 
                sum_inv = sum(inverted_exponentiated_fitness)
                probabilities = []
                for i in range(len(population)):
                    probabilities.append(inverted_exponentiated_fitness[i]/sum_inv)
                parent_1 = random.choices(population, weights=probabilities)[0]
                parent_2 = random.choices(population, weights=probabilities)[0]
                parent_pairs.append((parent_1, parent_2))
        
        if (parent_selection_method == "probabilistic_tournament"):
            # turniej probabilistyczny
            for i in range(population_count): 
                tournament_participants = random.sample(population, tournament_size)
                p = probability_of_winning_tournament 
                probabilities_of_winning = []
                for j in range(tournament_size):
                    if j == 0:
                        probabilities_of_winning.append(p)
                    else:
                        probabilities_of_winning.append(p*(1-p)**(j)) # ustal prawdopodobieństwa wygranej kolejnych osobników 
                
                tournament_participants.sort(key=calculate_route, reverse=False)
                tournament_winner = random.choices(tournament_participants, weights=probabilities_of_winning)[0]
                parent_1 = tournament_winner
                tournament_participants = random.sample(population, tournament_size)
                tournament_participants.sort(key=calculate_route, reverse=False)
                tournament_winner = random.choices(tournament_participants, weights=probabilities_of_winning)[0]  
                parent_2 = tournament_winner
                parent_pairs.append((parent_1, parent_2))
        if (parent_selection_method == "tournament"):
            # turniej
            for i in range(population_count): 
                tournament_participants = random.sample(population, tournament_size)
                tournament_participants.sort(key=calculate_route)
                parent_1 = tournament_participants[0]
                tournament_participants = random.sample(population, tournament_size)
                tournament_participants.sort(key=calculate_route)
                parent_2 = tournament_participants[0]
                parent_pairs.append((parent_1, parent_2))
        # ========================================================================


        # ============================ KRZYŻOWANIE ===============================
        genomes_children =[]
        if (crossing_method == "OX"): # order crossover
            for i in range(len(parent_pairs)):
                # wybierz punkty przecięcia genomu
                cut_points = random.sample(range(0,len(dist)), 2)
                cut_points.sort()
                # inicjuj genom dziecka
                genome_child = [None] * len(dist)
                # część genomu od rodzica 1
                genome_child[cut_points[0]:cut_points[1]] = parent_pairs[i][0][cut_points[0]:cut_points[1]]
                #print("Genom po dodaniu rodzica 1: " + str(genome_child))
                not_filled_indices = np.where(np.array(genome_child) == None)[0]
                # print("Niewypełnione indeksy w genomie: " + str(not_filled_indices))
                filled_indices_count = 0
                for j in range(len(dist)):
                    if parent_pairs[i][1][j] in genome_child:
                        pass
                    else:
                        genome_child[not_filled_indices[filled_indices_count]] = parent_pairs[i][1][j]
                        filled_indices_count = filled_indices_count + 1
                # print("Genom po dodaniu rodzica 2: " + str(genome_child))
                genomes_children.append(genome_child)
        
        if (crossing_method == "PMX"): # partially mapped crossover
            for i in range(len(parent_pairs)):
                #print(f'Genom rodzica 1: {parent_pairs[i][0]}')
                #print(f'Genom rodzica 2: {parent_pairs[i][1]}')
                # wybierz punkty przecięcia genomu
                cut_points = random.sample(range(0,len(dist)), 2)
                cut_points.sort()
                #print(f'Pkty przecięcia: {cut_points}')
                # dziecko: 
                genome_child = [None]* len(dist)
                # skopiuj genom z rodzica 1
                genome_child[cut_points[0]:cut_points[1]] = parent_pairs[i][0][cut_points[0]:cut_points[1]]
                #print(f'Genom dziecka po skopiowaniu rodzica 1: {genome_child}')
                # wypełnij pozostałe miejsca
                for j in range(len(dist)):
                    if genome_child[j] is None:  # Jeśli pozycja jest pusta
                        candidate = parent_pairs[i][1][j]
                        # Sprawdzanie, czy kandydat jest już w dziecku
                        while candidate in genome_child[cut_points[0]:cut_points[1]]:
                            # Jeśli jest, znajdź jego mapowanie w rodzicu 1
                            candidate = parent_pairs[i][1][parent_pairs[i][0].index(candidate)]
                        # Wstaw kandydata w dziecku
                        genome_child[j] = candidate
                #print(f'Uzupełniony genom dziecka: {genome_child}\n')
                genomes_children.append(genome_child)

        if (crossing_method == "CX"): # cycle crossover
            for i in range(len(parent_pairs)):
                parent_1 = parent_pairs[i][0]
                parent_2 = parent_pairs[i][1]
                #print(f'Rodzic 1: {parent_1}')
                #print(f'Rodzic 2: {parent_2}')
                genome_child = [None] * len(dist)
                start_city = parent_1[0]
                current_city = parent_2[0]
                cycle = [start_city]
                while (start_city != current_city):
                    cycle.append(current_city)
                    current_city = parent_2[parent_1.index(current_city)]
                #print(f'Cykl: {cycle}')
                for j in range(len(parent_1)):
                    if (parent_1[j] in cycle):
                        genome_child[j] = parent_1[j]
                    else:
                        genome_child[j] = parent_2[j]
                #print(f'Genom dziecka: {genome_child}')
                genomes_children.append(genome_child)




        # Mutacja
        if(random.random()<probability_of_mutation):
            # print("Mutacja!")
            individual_for_mutation_idx = random.randint(0,len(genomes_children)-1)
            swap_points = random.sample(range(0,len(dist)), 2)
            swap_points.sort()
            genome_part_to_reverse = genomes_children[individual_for_mutation_idx][swap_points[0]:swap_points[1]]
            genome_part_to_reverse = genome_part_to_reverse[::-1]
            genomes_children[individual_for_mutation_idx][swap_points[0]:swap_points[1]] = genome_part_to_reverse

            # city_1 = genomes_children[individual_for_mutation_idx][swap_points[0]]
            # city_2 = genomes_children[individual_for_mutation_idx][swap_points[1]]
            # genomes_children[individual_for_mutation_idx][swap_points[0]] = city_2
            # genomes_children[individual_for_mutation_idx][swap_points[1]] = city_1

        
        # zapisz najlepsze rozwiązanie w iteracji
        distances_in_iteration = []
        for i in range(population_count):
            distances_in_iteration.append(calculate_route(genomes_children[i]))
        if (n == 0):
            shortest_distance = min(distances_in_iteration)
            best_solution = genomes_children[np.argmin(distances_in_iteration)]
        else:
            if(min(distances_in_iteration) < shortest_distance):
                shortest_distance = min(distances_in_iteration)
                best_solution = genomes_children[np.argmin(distances_in_iteration)]

        # dzieci jako nowa populacja
        population = genomes_children
        print("\r", end="")
        print(f'Iteracja {n+1} z {n_iterations}, czas wykonywania: {round(time.time() - start_gen, 2)} s', end="")
    
    #print(f'\nNajlepsza znaleziona trasa w iteracjach: {best_solution} \nDługość: {shortest_distance}')
    return best_solution


dist = np.loadtxt('data48.csv', delimiter=";")
# utwórz tablicę rozwiązań z najblizszego sasiada
ordering = np.empty((len(dist), len(dist)))
travel_distances = np.zeros(len(dist))
for i in range(len(dist)): # start z różnych miast
    start_city = i
    ordering[i, 0] = start_city
    for j in range(len(dist)):  
        current_city = int(ordering[i, j])
        visited_cities = np.take(ordering[i, ], range(j+1) ) 
        cities_left = np.setdiff1d(range(len(dist)), visited_cities)
        if(len(cities_left) != 0):
            distances = []
            for city in cities_left:
                distances.append(dist[int(ordering[i, j]), city])
            # Wybierz najbliższe miasto
            closest_city = cities_left[np.argmin(distances)]
            distance_to_shortest_city = min(distances)
            # Ustaw to miasto jako następne w kolejności
            ordering[i, j+1] = closest_city 
            travel_distances[i] = travel_distances[i] + distance_to_shortest_city
    travel_distances[i] = travel_distances[i] + dist[i, int(ordering[i, len(dist)-1])]
ordering = ordering.astype(int)
solutions = []
for i in range(len(dist)):
    solutions.append(list(ordering[i]))

initial_solution_methods = ["closest_neighbour", "random"]
population_counts = [20, 50, 100, 500]
n_iterations_counts = [100, 300, 500, 1500]
probabilities_of_mutation = [0.05, 0.1, 0.25, 0.5]
crossing_methods = ["OX", "PMX", "CX"]
probabilities_of_winning_tournament = [0.7, 0.85, 0.95]
tournament_sizes = [2, 3, 5, 10]

results_df = pd.DataFrame(columns=["distance", "combination","initial_solution_method", "parent_selection_method", "population_count", "n_iterations", "probability_of_mutation", "crossing_method", "probability_of_winning_tournament", "tournament_size"])


computation_time_start = time.time()
n_combinations = len(initial_solution_methods) * len(population_counts) * len(n_iterations_counts) * len(probabilities_of_mutation) * len(crossing_methods) + len(initial_solution_methods) * len(population_counts) * len(n_iterations_counts) * len(probabilities_of_mutation) * len(crossing_methods) * len(tournament_sizes) + len(initial_solution_methods) * len(population_counts) * len(n_iterations_counts) * len(probabilities_of_mutation) * len(crossing_methods) * len(probabilities_of_winning_tournament) * len(tournament_sizes) 


combination_number = 0
# best solution
best_solution = 1000000
# dla metody turnieju
for initial_sol in initial_solution_methods:
    for population_count in population_counts:
        for n_iterations in n_iterations_counts:
            for probability_of_mutation in probabilities_of_mutation:
                for crossing_method in crossing_methods:
                    for tournament_size in tournament_sizes:
                        combination_number = combination_number + 1
                        print(f'Kombinacja parametrów numer {combination_number} z {n_combinations}\n')
                        print(f'Initial solution method: {initial_sol} \nParent selection method: tournament \nPopulation count: {population_count} \nNumber ofiterations: {n_iterations} \nProbability of mutation: {probability_of_mutation} \nCrossing method: {crossing_method} \nTournament size: {tournament_size}\n')
                        solution = genetic_algorithm(dist, initial_sol, "tournament",population_count, n_iterations, probability_of_mutation, crossing_method,0.85, 3)
                        distance = calculate_route(solution)
                        if(distance < best_solution):
                            best_solution = distance
                        results_df.loc[len(results_df)] = [distance, str(solution), initial_sol,"tournament", population_count, n_iterations, probability_of_mutation, crossing_method, "None", tournament_size]                    
                        os.system('clear')               
                        print(f'Czas wykonywania: {round(time.time() - computation_time_start)} s')
                        print(f'Najlepsze znalezione rozwiązanie: {best_solution}') 
                        results_df.to_csv("results48.csv", index=False)        
# dla metody turnieju probabilistycznego
for initial_sol in initial_solution_methods:
    for population_count in population_counts:
        for n_iterations in n_iterations_counts:
            for probability_of_mutation in probabilities_of_mutation:
                for crossing_method in crossing_methods:
                    for tournament_size in tournament_sizes:
                        for probability_of_winning_tournament in probabilities_of_winning_tournament:
                            combination_number = combination_number + 1
                            print(f'Kombinacja parametrów numer {combination_number} z {n_combinations}\n')
                            print(f'Initial solution method: {initial_sol} \nParent selection method: probabilistic_tournament \nPopulation count: {population_count} \nNumber of iterations: {n_iterations} \nProbability of mutation: {probability_of_mutation} \nCrossing method: {crossing_method} \nProbability of winning tournament: {probability_of_winning_tournament}\nTournament size: {tournament_size}\n')
                            solution = genetic_algorithm(dist, initial_sol, "probabilistic_tournament", population_count, n_iterations, probability_of_mutation, crossing_method, 0.85, 3)
                            distance = calculate_route(solution)
                            if(distance < best_solution):
                                best_solution = distance
                            results_df.loc[len(results_df)] = [distance, str(solution), initial_sol, "probabilistic_tournament", population_count, n_iterations, probability_of_mutation, crossing_method, probability_of_winning_tournament, tournament_size]                    
                            os.system('clear')               
                            print(f'Czas wykonywania: {round(time.time() - computation_time_start)} s')     
                            print(f'Najlepsze znalezione rozwiązanie: {best_solution}')     
                            results_df.to_csv("results48.csv", index=False)          
# dla metody sorted fitness
for initial_sol in initial_solution_methods:
    for population_count in population_counts:
        for n_iterations in n_iterations_counts:
            for probability_of_mutation in probabilities_of_mutation:
                for crossing_method in crossing_methods:
                    combination_number = combination_number + 1

                    print(f'Najlepsze znalezione rozwiązanie: {best_solution}\n') 
                    print(f'Kombinacja parametrów numer {combination_number} z {n_combinations}\n')
                    print(f'Initial solution method: {initial_sol} \nParent selection method: sorted_fitness \nPopulation count: {population_count} \nNumber of iterations: {n_iterations} \nProbability of mutation: {probability_of_mutation} \nCrossing method: {crossing_method}\n')
                    solution = genetic_algorithm(dist, initial_sol, "sorted_fitness", population_count, n_iterations, probability_of_mutation, crossing_method, 0.85, 3)
                    distance = calculate_route(solution)
                    if(distance < best_solution):
                        best_solution = distance
                    results_df.loc[len(results_df)] = [distance, str(solution), initial_sol, "sorted_fitness", population_count, n_iterations, probability_of_mutation, crossing_method, "None", "None"]
                    os.system('clear')                    
                    print(f'Czas wykonywania: {round(time.time() - computation_time_start)} s')           
                    results_df.to_csv("results48.csv", index=False)  
                    os.system('clear')
# dla metody turnieju
for initial_sol in initial_solution_methods:
    for population_count in population_counts:
        for n_iterations in n_iterations_counts:
            for probability_of_mutation in probabilities_of_mutation:
                for crossing_method in crossing_methods:
                    for tournament_size in tournament_sizes:
                        combination_number = combination_number + 1
                        print(f'Kombinacja parametrów numer {combination_number} z {n_combinations}\n')
                        print(f'Initial solution method: {initial_sol} \nParent selection method: tournament \nPopulation count: {population_count} \nNumber ofiterations: {n_iterations} \nProbability of mutation: {probability_of_mutation} \nCrossing method: {crossing_method} \nTournament size: {tournament_size}\n')
                        solution = genetic_algorithm(dist, initial_sol, "tournament",population_count, n_iterations, probability_of_mutation, crossing_method,0.85, 3)
                        distance = calculate_route(solution)
                        if(distance < best_solution):
                            best_solution = distance
                        results_df.loc[len(results_df)] = [distance, str(solution), initial_sol,"tournament", population_count, n_iterations, probability_of_mutation, crossing_method, "None", tournament_size]                    
                        os.system('clear')               
                        print(f'Czas wykonywania: {round(time.time() - computation_time_start)} s')
                        print(f'Najlepsze znalezione rozwiązanie: {best_solution}') 
                        results_df.to_csv("results48.csv", index=False)        
print(results_df)
results_df.to_csv("results48.csv", index=False)








