import numpy as np

# wczytaj dane
data = np.loadtxt("house_prices.csv", skiprows=1, delimiter=",")

# normalizuj dane
col_min = np.min(data, axis=0, keepdims=True)
col_max = np.max(data, axis=0, keepdims=True)
data = (data - col_min)/ (col_max - col_min)
#print(data)

# zbiór uczący i testowy
data_train = data[0:800, ] 
data_test = data[800:1000, ] 

# zmienne zależne i niezależne
X = data_train[0:800,0:7].T
Y = data_train[0:800,7].T
X_test = data_test[0:200,0:7].T
Y_test = data_test[0:200,7].T

# parametry sieci
n_layers = 4 # włącznie z wejściową i wyjściową!
n_neurons_in_layer = [7, 5, 3, 1] # liczby neuronów w kolejnych warstwach
if(n_layers != len(n_neurons_in_layer)):
    print("Nieprawidłowe parametry sieci!")
    exit()
learning_rate = 0.00001 


# definicja funkcji ReLU
def ReLU(x):
    return np.maximum(0, x)

# propagacja w przód

# inicializuacja wag
W_1 = np.random.randn(n_neurons_in_layer[1], n_neurons_in_layer[0]) * 1
W_2 = np.random.randn(n_neurons_in_layer[2], n_neurons_in_layer[1]) * 1
W_3 = np.random.randn(n_neurons_in_layer[3], n_neurons_in_layer[2]) * 1
# inicjalizacja obciążeń
B_1 = np.random.randn(n_neurons_in_layer[1], 1) * 0.0
B_2 = np.random.randn(n_neurons_in_layer[2], 1) * 0.01
B_3 = np.random.randn(n_neurons_in_layer[3], 1) * 0.01


for i in range(100000):
    # warstwa 2
    Z_1 = np.dot(W_1, X) + B_1
    A_1 = ReLU(Z_1)
    # warstwa 3
    Z_2 = np.dot(W_2, A_1) + B_2
    A_2 = ReLU(Z_2)
    # warstwa 4
    Z_3 = np.dot(W_3, A_2) + B_3
    # przewidywania
    Y_hat = Z_3


    # pochodna relu (dla wektora danych)
    def d_ReLU(x):
        return np.where(x > 0, 1, 0)
    # funkcja błędu
    def error(y, y_hat):
        return 0.5 * ((y - y_hat) ** 2)
    # pochodna funkcji błędu
    def d_error(y, y_hat):
        return y_hat - y

    # propagacja wsteczna
    
    d_error_d_y_hat = d_error(Y, Y_hat) # gradient względem Y_hat
    d_y_hat_d_z_3 = 1  # Brak aktywacji na wyjściu
    d_error_d_z_3 = d_error_d_y_hat * d_y_hat_d_z_3

    d_error_d_W_3 = np.dot(d_error_d_z_3, A_2.T)
    #print(f'Gradient po W_3: {d_error_d_W_3[0:4]}')
    d_error_d_B_3 = np.sum(d_error_d_z_3, axis=1, keepdims=True)
    #print(f'Gradient po B_3: {d_error_d_B_3}')

    d_error_d_A_2 = np.dot(W_3.T, d_error_d_z_3)
    d_error_d_z_2 = d_error_d_A_2 * d_ReLU(Z_2)

    d_error_d_W_2 = np.dot(d_error_d_z_2, A_1.T)
    d_error_d_B_2 = np.sum(d_error_d_z_2, axis=1, keepdims=True)

    d_error_d_A_1 = np.dot(W_2.T, d_error_d_z_2)
    d_error_d_z_1 = d_error_d_A_1 * d_ReLU(Z_1)

    d_error_d_W_1 = np.dot(d_error_d_z_1, X.T)
    d_error_d_B_1 = np.sum(d_error_d_z_1, axis=1, keepdims=True)


    # wykonaj krok w stronę ujemnego gradientu, pomnożony przez learing rate
    W_1 = W_1 - learning_rate * d_error_d_W_1
    W_2 = W_2 - learning_rate * d_error_d_W_2 
    W_3 = W_3 - learning_rate * d_error_d_W_3 
    B_1 = B_1 - learning_rate * d_error_d_B_1
    B_2 = B_2 - learning_rate * d_error_d_B_2
    B_3 = B_3 - learning_rate * d_error_d_B_3
    total_error = np.sum(error(Y, Y_hat))
    
    print("\r", end="")
    print(f'Iteracja: {i+1}; Całkowity błąd w zbiorze uczącym: {total_error}', end="")


# print(f'\nPrzewidywania: {Y_hat[0][0:4]}')
# print(f'Rzeczywiste wartości: {Y[0:4]}')

print(f'\nMSE: {1/800 * np.sum((Y- Y_hat)**2)}')

max_house_price = col_max[0][7]
min_house_price = col_min[0][7]


house_price_real = min_house_price + (max_house_price - min_house_price) * Y
house_price_predictions = min_house_price + (max_house_price - min_house_price) * Y_hat

print(f'\nPrzewidywania: {house_price_predictions[0][0:5]}')
print(f'Rzeczywiste wartości: {house_price_real[0:5]}')
