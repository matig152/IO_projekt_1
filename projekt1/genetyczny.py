import numpy as np

dist = np.loadtxt('data_exemplary.csv', delimiter=';')

# parametry algorytmu
population_size = 10


# początkowa populacja
order = np.arange(0,7)
seed = np.random.randint(0, 1000)

np.random.shuffle(order) 
comb_1 = order

np.random.seed(50)
np.random.shuffle(order) 
comb_2 = order



print(comb_1, comb_2)


