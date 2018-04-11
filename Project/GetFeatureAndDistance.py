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

import math

def getFeatureMFCC(song):
    # MFCC c обрезанными верхними порогами
    MFCC = librosa.feature.mfcc(song)[:15]
    
    # средние и матрица ковариаций
    means = np.mean(MFCC, axis = 0)
    cov_martix = np.cov(MFCC, rowvar = False)
    
    # возвращаем вектор фич
    appended_dim_and_means = np.append(MFCC.shape[1], means)
    return np.append(appended_dim_and_means, cov_martix.ravel())

def QuadraticForm(A, x, y):
    # скопировал из stackoverflow
    return ((np.matrix(x).T * np.matrix(A)).A * y.T.A).sum(1)

def Divergence_KL(song_p, song_q):
    if song_p[0] != song_q[0]:
        raise
    
    # Берем фичи
    N = song_p[0]
    means_p = song_p[1 : N + 1]
    cov_p = song_p[N + 1 :].reshape(N,N)
    means_q = song_q[1 : N + 1]
    cov_q = song_q[N + 1 :].reshape(N,N)

    inv_cov_q = inv(cov_q)
    means_diff = means_p - means_q
    return 0.5 * (math.log(det(cov_q) / det(cov_p)) +\
                  np.trace(np.dot(inv_cov_q, cov_p)) +\
                  QuadraticForm(inv_cov_q, means_diff, means_diff) - N)


def Distance_KL(song_p, song_q):
    return Divergence_KL(song_p, song_q) + Divergence_KL(song_q, song_p)
