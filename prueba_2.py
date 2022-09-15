import mnist_loader
import network_1
import pickle


training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

#net=network_1.Network([784,30,10])
#net.Momentum( training_data, 30, 10, 0.5, test_data=test_data)

#archivo = open("red_prueba_2.1.pkl",'wb')
#pickle.dump(net,archivo)
#archivo.close()

#leer el archivo
archivo_lectura = open("red_prueba_2.1.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()


net.Momentum( training_data, 30, 10, 0.01, test_data=test_data)

archivo = open("red_prueba2.2.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()