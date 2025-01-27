import numpy as np


array = np.array([1,2,3])

# vektorwertig
new_array = array + 1
print(new_array)

my_list = [1,2,3]
# my_list + 1

# schnell

two_dimensional_array = np.array([[1,2,3], [4,5,6]])
print(two_dimensional_array)

print(two_dimensional_array.shape)

# Zugriff auf Array
print(new_array)
new_array[0]
i = new_array[1]
print(i*2)
new_array[-1]
new_array[1:]
new_array[-2:]
new_array[:2]
new_array[:-2]
