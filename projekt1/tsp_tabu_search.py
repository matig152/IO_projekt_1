import pandas as pd
import numpy as np
import random
import time
from itertools import product

def calculate_route_distance(route, distance_matrix):
    # Obliczanie długość trasy na podstawie podanej macierzy odległości.
    distance = 0
    for i in range(len(route)):
        # Dodaje odległość między kolejnym miastem i następnym w trasie (modulo dla powrotu do początku).
        distance += distance_matrix[route[i]][route[(i + 1) % len(route)]]
    return distance

def generate_initial_solution(num_cities):
    # Generowanie losowej początkowej trasy, odwiedzając wszystkie miasta.
    route = list(range(num_cities))
    random.shuffle(route)  # Losowe przetasowanie miast
    return route

def generate_neighborhood(route, method):
    # Generuje sąsiedztwo tras za pomocą różnych metod:
    # - swap: Zamiana miejscami dwóch miast
    # - reversal: Odwrócenie kolejności miast na wybranym odcinku
    # - insertion: Przeniesienie wybranego miasta w inne miejsce trasy
    neighbors = []
    num_cities = len(route)
    if method == "swap":
        # Zamienia dwa miasta miejscami
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                neighbor = route[:]  # Kopia obecnej trasy
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
    elif method == "reversal":
        # Odwraca kolejność miast na wybranym odcinku
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                neighbor = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                neighbors.append(neighbor)
    elif method == "insertion":
        # Przenosi wybrane miasto w inne miejsce trasy
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    neighbor = route[:]  # Kopia obecnej trasy
                    city = neighbor.pop(i)  # Usuwa miasto z pozycji i
                    neighbor.insert(j, city)  # Wstawia je na pozycję j
                    neighbors.append(neighbor)
    return neighbors

def tabu_search(distance_matrix, tabu_length, max_iter, neighborhood_method, stop_criteria, dynamic_initial_quality=True):
    #Implementacja algorytmu Tabu Search: Szukanie najlepszej możliwej trasy z uwzględnieniem ograniczeń listy tabu.
    num_cities = len(distance_matrix)
    # Generowanie początkowego losowego rozwiązania
    current_solution = generate_initial_solution(num_cities)
    current_distance = calculate_route_distance(current_solution, distance_matrix)
    best_solution = current_solution  # Inicjalizacja najlepszego rozwiązania
    best_distance = current_distance  # Koszt najlepszego rozwiązania
    tabu_list = []  # Lista tabu
    iterations_without_improvement = 0  # Licznik iteracji bez poprawy rozwiązania

    # Dynamiczna modyfikacja parametrów na podstawie jakości początkowego rozwiązania
    if dynamic_initial_quality:
        tabu_length = max(tabu_length, int(tabu_length * (current_distance / (np.mean(distance_matrix) * num_cities))))
        stop_criteria = int(stop_criteria * (current_distance / (np.mean(distance_matrix) * num_cities)))

    # Główna pętla algorytmu
    for iteration in range(max_iter):
        neighbors = generate_neighborhood(current_solution, neighborhood_method)  # Generowanie sąsiedztwa
        candidate_solutions = []
        for neighbor in neighbors:
            distance = calculate_route_distance(neighbor, distance_matrix)  # Obliczanie kosztu trasy
            if neighbor not in tabu_list:  # Ignorowanie tras z listy tabu
                candidate_solutions.append((neighbor, distance))
        if candidate_solutions:
            # Wybór najlepszego kandydata
            candidate_solutions.sort(key=lambda x: x[1])  # Sortowanie po kosztach trasy
            current_solution, current_distance = candidate_solutions[0]  # Najlepszy kandydat
            if current_distance < best_distance:  # Aktualizacja najlepszego rozwiązania
                best_solution = current_solution
                best_distance = current_distance
                iterations_without_improvement = 0  # Reset licznika
            else:
                iterations_without_improvement += 1  # Inkrementacja licznika bez poprawy
            # Dodanie obecnego rozwiązania do listy tabu
            tabu_list.append(current_solution)
            if len(tabu_list) > tabu_length:  # Ograniczenie długości listy tabu
                tabu_list.pop(0)
        # Przerwanie, jeśli liczba iteracji bez poprawy przekracza próg
        if iterations_without_improvement >= stop_criteria:
            break
    return best_solution, best_distance

def load_distance_matrix_from_excel(filename, sheet_name=0):
    #Wczytuje macierz dystansów z pliku Excela.
    df = pd.read_excel(filename, sheet_name=sheet_name, index_col=0)  # Wczytanie Excela do DataFrame
    distance_matrix = df.to_numpy()  # Konwersja na macierz NumPy
    return distance_matrix

def run_experiments(filename, distance_matrix, tabu_lengths, max_iters, stop_criteria, neighborhood_methods, dynamic_qualities, num_repeats):
    #Uruchamianie eksperymentów dla różnych kombinacji parametrów:
    # - tabu_lengths: Długość listy tabu.
    # - max_iters: Maksymalna liczba iteracji.
    # - stop_criteria: Maksymalna liczba iteracji bez poprawy.
    # - neighborhood_methods: Metody generowania sąsiedztwa.
    # - dynamic_qualities: Czy stosować dynamiczną modyfikację parametrów.
    results = []
    # Tworzenie wszystkich kombinacji parametrów
    for params in product(tabu_lengths, max_iters, stop_criteria, neighborhood_methods, dynamic_qualities):
        tabu_length, max_iter, stop_criterion, neighborhood_method, dynamic_quality = params
        distances = []
        best_routes = []
        start_time = time.time()  # Start pomiaru czasu
        for _ in range(num_repeats):
            # Uruchomienie algorytmu Tabu Search
            best_solution, best_distance = tabu_search(
                distance_matrix, tabu_length, max_iter, neighborhood_method, stop_criterion, dynamic_quality
            )
            best_routes.append(best_solution)  # Zapis trasy
            distances.append(best_distance)  # Zapis odległości
        end_time = time.time()  # Koniec pomiaru czasu
        avg_distance = np.mean(distances)  # Średnia odległość
        min_distance = np.min(distances)  # Minimalna odległość
        best_route = best_routes[np.argmin(distances)]  # Najlepsza trasa
        execution_time = end_time - start_time  # Czas wykonania
        results.append({
            "Tabu Length": tabu_length,
            "Max Iterations": max_iter,
            "Stop Criterion": stop_criterion,
            "Neighborhood Method": neighborhood_method,
            "Dynamic Quality": dynamic_quality,
            "Min Distance": min_distance,
            "Average Distance": avg_distance,
            "Best Route": [city + 1 for city in best_route],  # Miasta ponumerowane od 1
            "Execution Time": execution_time
        })
    # Zapis wyników do pliku Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(filename, index=False)
    print(f"Wyniki zapisane do pliku {filename}")

# Parametry eksperymentu
filename = "Dane_TSP_127.xlsx"
distance_matrix = load_distance_matrix_from_excel(filename)  # Wczytanie danych
tabu_lengths = [10, 50, 100, 200]  # Długości listy tabu
max_iters = [100, 500, 1500, 4000]  # Maksymalna liczba iteracji
stop_criteria = [50, 100, 200, 400]  # Kryteria zatrzymania
neighborhood_methods = ["swap", "reversal", "insertion"]  # Metody sąsiedztwa
dynamic_qualities = [True, False]  # Dynamiczna jakość
num_repeats = 5  # Liczba powtórzeń

# Uruchomienie eksperymentów
output_filename = "Tabu_Search_Results_127.xlsx"
run_experiments(output_filename, distance_matrix, tabu_lengths, max_iters, stop_criteria, neighborhood_methods, dynamic_qualities, num_repeats)
