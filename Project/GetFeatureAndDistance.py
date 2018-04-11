"""
    В статье http://cs229.stanford.edu/proj2011/HaggbladeHongKao-MusicGenreClassification.pdf делается следующее:
    1. Качается датасет Marsyas (его ссылку скидывал ранее в телеграм)
    2. Берется сердцевина песни (50%)
    3. От каждой песни берется MFCC и обрезаются верхние частоты (5 порогов).
    Таким образом размерность MFCC.shape становится равной (15, N), где N -- количество фрагментов
    4. Для каждого фрагмента берется среднее, а затем считается матрица ковариаций для всех фрагментов
    5. После пункта 4 у нас есть фичи!!!
    6. В статье далее определяется метрика Kullback-Lieber (KL) Divergence
    (TODO: понять ее физико-статистический смысл. В статье есть ссылка на эту тематику)
    7. Предлагаю следующую организацию данных: каждой песне соответствует одномерный вектор.
    Первый элемент -- количество фрагментов N
    Следующие N элементов -- среднее по каждому фрагменту.
    Следующие N^2 элементов -- матрица ковариаций,
    записанная в один вектор по строкам.
    
    Функция getFeatureMFCC извлекает фичи и записывает их в вектор
    Функция Distance_KL(song_p, song_q) считает расстояние в соответствии со статьей
"""

import librosa
import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import math

def getFeatureMFCC(song):
    # MFCC c обрезанными верхними порогами
    MFCC = librosa.feature.mfcc(song)[:15, 1:]
    
    # средние и матрица ковариаций
    means = np.mean(MFCC, axis = 1)
    cov_matrix = np.cov(MFCC, rowvar = True)    

    return np.append(means, cov_matrix.ravel())

def QuadraticForm(A, x, y):
    # скопировал из stackoverflow
    return np.dot(np.dot(A, x), y)

def Divergence_KL(song_p, song_q):    
    # Берем фичи
    means_p = song_p[0 : 15]
    cov_p = song_p[15 :].reshape(15, 15)
    means_q = song_q[0 : 15]
    cov_q = song_q[15 :].reshape(15, 15)

    inv_cov_q = inv(cov_q)
    means_diff = means_p - means_q
    return 0.5 * (math.log(det(cov_q) / det(cov_p)) +\
                  np.trace(np.dot(inv_cov_q, cov_p)) +\
                  QuadraticForm(inv_cov_q, means_diff, means_diff) - 15)
                  

def Distance_KL(song_p, song_q):
    return Divergence_KL(song_p, song_q) + Divergence_KL(song_q, song_p)

def getData(where_to, genre, range_tuple):
    for i in range(range_tuple):
        if i < 10:
            path = "./" + genre + "/" + genre + ".0000" + str(i) + ".au"        
        else:
            path = "./" + genre + "/" + genre + ".000" + str(i) + ".au"

        song = librosa.load(path)[0]
        where_to.append(getFeatureMFCC(song))
        
data = list()
getData(data, "hiphop", 30)
getData(data, "rock", 30)
getData(data, "classical", 30)
data = np.array(data)
y = np.append([np.zeros(30), 1 + np.zeros(30)], 2 +  np.zeros(30))

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 5, metric=Divergence_KL)
knn.fit(X_train, y_train)
accuracy_score(knn.predict(X_test), y_test)
