import numpy as np

#Array aus einer Python-Liste:
a = np.array([10, 20, 30, 40])
print(a)
print(type(a))

print("----------------------")
#Mehrdimensionale Arrays
matrix = np.array([
[1, 2, 3],
[4, 5, 6]
])
print("----------------------")
print(matrix)

print(matrix.shape) # (2, 3)    Form des Arrays
print(matrix.ndim) # 2          Anzahl Dimensionen
print(matrix.size) # 6          Anzahl aller Elemente
print(matrix.dtype)#            Datentyp der Elemente

print("----------------------")
#Arrays erzeugen
print("Arrays erzeugen")
print(np.zeros(5)) # [0. 0. 0. 0. 0.]
print(np.ones(4)) # [1. 1. 1. 1.]
print(np.arange(0, 10, 2)) # [0 2 4 6 8]
print(np.linspace(0, 1, 5)) # 5 gleichmäßig verteilte Werte

print("----------------------")
#Zufallszahlen
print("Zufallszahlen")
rng = np.random.default_rng(42)
print(rng.integers(1, 11, size=5))
print(rng.random(5))
print("----------------------")

#Indexing und Slicing
print("Indexing und Slicing")
a = np.array([10, 20, 30, 40, 50])
print(a[0]) # 10
print(a[-1]) # 50
print(a[1:4]) # [20 30 40]
print(a[::2]) # [10 30 50]


print("----------------------")
print("Zeilen und Spalten aus Matrix")
m = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
print(m[1, 2]) # 6
print(m[0, :]) # erste Zeile
print(m[:, 1]) # zweite Spalte

print("----------------------")
#Boolean Indexing ist besonders wichtig: Du filterst Werte anhand einer Bedingung.
print("Boolean Indexing")
a = np.array([3, 8, 12, 5, 20])
print(a[a > 10]) # [12 20]

print("----------------------")
#Rechnen mit Arrays - ohne Schleifen
print("Rechnen mit Arrays - ohne Schleifen")
a = np.array([1, 2, 3, 4])
print(a + 10)
print(a * 2)
print(a ** 2)
print(np.sqrt(a))
#Auch zwei gleich große Arrays können direkt miteinander verrechnet werden
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print(a + b) # [11 22 33]
print(a * b) # [10 40 90]

print("----------------------")
#Aggregationen
print("Aggregationen")
a = np.array([4, 8, 15, 16, 23, 42])
print(a.sum())
print(a.mean())
print(a.min())
print(a.max())
print(a.std())
#Bei Matrizen kann die Berechnung entlang einer Achse erfolgen
m = np.array([[1, 2, 3],
[4, 5, 6]])
print(m.sum(axis=0)) # spaltenweise: [5 7 9]
print(m.sum(axis=1)) # zeilenweise: [6 15]

print("----------------------")
#Shape, Reshape und Flatten
print("Shape, Reshape und Flatten")
a = np.arange(1, 13)     # Erstellt ein Array mit den Zahlen 1 bis 12
print(a.shape)
m = a.reshape(3, 4)         # Formt das Array in 3 Zeilen und 4 Spalten um
print(m)
print(m.shape)              # Zeigt die Form der Matrix: (3, 4)
flat = m.flatten()          # Wandelt die 3x4-Matrix in ein eindimensionales Array um
print(flat)
m = np.array([[1, 2, 3],
              [4, 5, 6]])   # Erstellt eine 2x3-Matrix
print(m + 10)               # Addiert 10 zu jedem einzelnen Wert der Matrix

offset = np.array([100, 200, 300])
print(m + offset)