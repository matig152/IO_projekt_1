import pandas as pd
import numpy as np


def closest_neighbour_solver(dist):
    # kolejności przy startach z kolejnych miast
    ordering = np.zeros((len(dist), len(dist)))

    # dystanse przy startach z kolejnych miast
    travel_distances = np.zeros(len(dist))


    for i in range(len(dist)): # start z każdego miasta    
        start_city = i
        ordering[i, 0] = start_city
        #print("Miasto startowe:" + str(start_city))
        for j in range(len(dist)):  
            # ustaw aktualne miasto
            current_city = int(ordering[i, j])
            # print("   -> Aktualne miasto: " + str(current_city))

            # odwiedzone już miasta
            visited_cities = np.take(ordering[i, ], range(j+1) ) 

            # miasta jeszcze nie odwiedzone
            cities_left = np.setdiff1d(range(len(dist)), visited_cities)
            
            #print("   -> Miasta pozostałe do odwiedzenia: " + str(cities_left)) 

            if(len(cities_left) != 0):
                # dystanse do miast jeszcze nie odwiedzonych
                distances = []
                for city in cities_left:
                    distances.append(dist[int(ordering[i, j]), city])

                #print("   -> Dystanse do pozostałych miast: " + str(distances))

                # Wybierz najbliższe miasto
                closest_city = cities_left[np.argmin(distances)]
                distance_to_shortest_city = min(distances)
                #print("   -> Najbliższe miasto to " + str(closest_city) + " w odległości " + str(distance_to_shortest_city))
        
                # Ustaw to miasto jako następne w kolejności
                ordering[i, j+1] = closest_city 
                travel_distances[i] = travel_distances[i] + distance_to_shortest_city

        # dodaj drogę z końcowego punktu do początkowego
        travel_distances[i] = travel_distances[i] + dist[i, int(ordering[i, len(dist)-1])]
            
        #print("Całkowity dystans: " + str(travel_distances[i]))
        #print("Start z miasta: " + str(i+1) + ". Dystans: " + str(travel_distances[i]))


    shortest_dist = min(travel_distances)
    shortest_dist_combination = np.argmin(travel_distances)

    print("Najkrótszy dystans wynosi: " + str(shortest_dist) + " przy starcie z miasta: " +str(int(ordering[shortest_dist_combination, 0]) + 1))
    print("Kolejność: " + str(ordering[shortest_dist_combination, ]))
    print("\n")


# 29 danych
dist29 = np.loadtxt('data29.csv', delimiter=';')
closest_neighbour_solver(dist29)

# 48 danych
dist48 = np.loadtxt('data48.csv', delimiter=';')
closest_neighbour_solver(dist48)

# 76 danych
dist76 = np.loadtxt('data76.csv', delimiter=';')
closest_neighbour_solver(dist76)

# 127 danych
dist127 = np.loadtxt('data127.csv', delimiter=';')
closest_neighbour_solver(dist127)