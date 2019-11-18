#!/usr/bin/env python
# coding: utf-8

# # kMeans
# 
# ### Офицеров Владислав 
# 
# 

# In[137]:


import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[162]:


def myKMeans(k, data, y, s):
#     Функция показа   
    def pltshow():
        plt.xlim(data[:,0].min(),data[:,0].max())
        plt.ylim(data[:,1].min(),data[:,1].max())
        plt.show()

#     Добавление на график центроидов      
    def cent():
        fig = plt.figure(figsize=(7, 7))
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i % 6 + 1], marker = '*')

#     Добавление на график объектов 
    def poin():
        plt.scatter(data[:,0], data[:,1], color=cl, alpha=0.6, edgecolors=cl, marker = '.')
        
#    Вспомогатильные переменные     
    np.random.seed(123)
#    Количество объектов
    n = data.shape[0]
#     Количество фичей
    m = data.shape[1]
#     Цветовая гамма:
    colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm', 6: 'y'} 
#     Массив для цвета кажого объекта
    cl = np.full(n,' ')


#     Визуализация данных
    if (y):
        print('\n\nNUMBER OF CLUSTERS:', k,'\n\n')
        print('\n\nВизуализация данных')
        fig = plt.figure(figsize=(7,7))
        plt.scatter(data[:,0], data[:,1], color='black', alpha = 0.8)
        pltshow()
    
#     Инициализация центроидов
#     Kmeans++ здесь, это первый центроид случайным образом остальные максимально удаленные от предыдущих
    if (s == 'KMeans++'):
        centroids = {1: data[np.random.randint(0, n - 1)].tolist()}
        for it in range(k - 1):
            distss=np.full(n, 0)
            for i in range(n):
                dist = np.inf
                for j in (centroids.keys()):
                    new_dist = 0
                    for jj in range(m):
                        new_dist += (centroids[j][jj] - data[i][jj]) ** 2
                    dist = min(np.sqrt(new_dist), dist)
                distss[i] = dist
            mx = np.max(distss)
            centroids[it + 2] = data[np.where(distss == mx)[0]][0].tolist()
    else:
#         Инициализация случайным образом всех центроидов
        centroids = { i+1: data[np.random.randint(0, n - 1)].tolist() for i in range(k) }
    
#     Визуализация центроидов
    if (y):
        print('\n\nВизуализация центроидов')
        cent()
        pltshow()
    
#     Функция для определния кластера для точки 
    distances = np.full((n, k), -1)
    def closest(data, centroids):
        for i in range(n):
            for j in range(k):
                sum = 0
                for l in range(m):
                    sum += (data[i][l] - centroids[j + 1][l])**2
                distances[i][j] = np.sqrt(sum)
        closest_center = np.full(n, -1)
        for i in range(n):
            closest_center[i] = np.argmin(distances[i]) + 1
            cl[i] = colmap[(np.argmin(distances[i]) + 1 )% 6 + 1]
        return closest_center

#    Первое присваивания объектов центроидам
    distribution = closest(data, centroids)


#     Визуализация к какому классу относятся объкты после инициализации центроидов
    if (y):
        print('\n\nВизуализация к какому классу относятся объкты после инициализации центроидов')
        cent()
        poin()
        pltshow()
    
#     Глубокая копия создает новый составной объект, и затем рекурсивно вставляет в него копии объектов, находящихся в оригинале. 
    old_centroids = copy.deepcopy(centroids)

#     Функция обновления центроидов
    def update(distr, data):
        for i in centroids.keys():
            centroids[i][0] = np.mean(data[np.where(distribution == i)[0]][:,0])
            centroids[i][1] = np.mean(data[np.where(distribution == i)[0]][:,1])
        return centroids

#     Обновление центроидов
    centroids = update(distribution, data)
    
#     Визуализация после одного обновления 
    if (y):
        print('\n\nВизуализация после одного обновления положения центроидов')
        cent()
        poin()
        pltshow()
    
#     Обновление точек(так как центроиды сдвинулись)  
   
    distribution = closest(data, centroids)

#     Обновление центроидов пока изменения не перстанут происходить или 100 итераций
    n_iter = 100
    while True or (n_iter != 0):
        closest_centroids = distribution
        centroids = update(distribution, data)
        distribution = closest(data, centroids)
        n_iter -= 1
        if np.array_equal(closest_centroids, distribution):
            break

#     Визуализация окончательного результата
    if (y):
        print('\n\nВизуализация окончательного результата:')
        cent()
        poin()
        pltshow()
    return distribution, cl


# In[186]:


# Метод логтя для нахождения количества кластеров
def elbow_method(data,a,b):
    silh_score = []
    for k in range(a, b):
        model, cl_model = myKMeans(k, data, 0,'KMeans++')
        print('Running comput. for', k,'cluster(s)')
        silh_score.append(silhouette_score(data, model))
    plt.plot(range(a, b), silh_score)


# In[155]:


# Функция упорядочевания data:
def order_data(data, model):
    data_clustered = np.array([0,0])
    for i in range(1,int(np.max(model))+1):
        for j in range(len(model)):
            if(model[j] == i):
                data_clustered = np.vstack((data_clustered, data[j]))
    data_clustered = np.delete(data_clustered, 0, axis = 0) 
    return data_clustered

# Функция для визуализации матрицы попарных состояний:
def visualization(data, model, cl_model):
    print('Матрица попарных расстояний до упорядочевания:')
    fig = plt.figure(figsize=(7,7))
    plt.scatter(data[:,0],data[:,1])
    plt.show()
    plt.matshow(pairwise_distances(data), aspect='auto')
    plt.colorbar()
    plt.show()
    fig = plt.figure(figsize=(7,7))
    plt.scatter(data[:,0],data[:,1], c = cl_model)
    plt.show()
    plt.matshow(pairwise_distances(order_data(data, model)), aspect='auto')
    plt.colorbar()
    plt.show()


# ### Первый dataset "s1" скачан отсюда: http://cs.joensuu.fi/sipu/datasets/ 

# In[187]:


data = np.loadtxt("/Users/ofirserovlad/Desktop/Kaggle/K-means/s1.txt", dtype="int")


# ### Одним из минусов данного метода является то, что мы должны знать количество кластеров заранее
# ### Для нахождения числа кластеров воспользуемся "Методом логтя". Метрика данного метода показывает отношение внутрикластерного расстояния к межкластерному расстоянию. Таким образом мы можем "увидеть" нужное количество кластеров.

# In[189]:


elbow_method(data, 3, 25)


# ### Центроиды обозначены символом *
# ##### (в случае плохой видимости центроидов, стоит уменьшить прозрачность, гиперпараметр alpha  в 16 строке функции myKMeans)
# ### В визуализации используются только 6 цветов, большего количества ярких цветов из matplotlib вынести я не смог. Не стоит думать, что все объекты одного цвета принадлежат одному кластеру. 
# ### На примере ниже только две нижних группы точек объеденены в один кластер неправильно и одна группа точек разбита на два кластера, это связано с тем, что изначально 3 центроида были инициалированы почти в одной точке. Можем устронить такие проблемы методом, с помощью которого цетроиды инициализируются на максимальном расстоянии друг от друга

# In[190]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(15, data, 1, 'KMeans')")


# In[191]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(15, data, 1, 'KMeans++')")


# ### На примере выше видно явное преимущество метода, c помощью которого центроиды инициализируются таким образом, чтобы каждый новый центроид находился на максимальном расстоянии от остальных

# ## ------------
# 
# ### Визуализируем матрицу попарных расстояний
# #### Dataset рассматриваемый нами в настоящий момент скачен из интернета, поэтому матрица упорядочена уже изначально

# In[192]:


visualization(data, model, cl_model)


# ### Рассмотрим еще несколько sklearn sampe'ов 
# ####  Посмотрим на матрицу попарных расстояний до упорядочивания и после 
# ####  Найдем зависимость времени обучения от объема и сложности задачи

# In[193]:


data, d =  make_blobs(n_samples= 1000, n_features= 2, centers = 3, random_state=123)


# In[194]:


elbow_method(data, 2, 10)


# In[199]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(5, data, 1, ' ')")


# In[200]:


visualization(data, model, cl_model)


# In[201]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(5, data, 1, 'KMeans++')")


# In[202]:


visualization(data, model, cl_model)


# In[206]:


data, d =  make_blobs(n_samples= 1000, n_features= 2, centers = 5, random_state=123)


# In[211]:


elbow_method(data, 2, 10)


# In[212]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(3, data, 1, ' ')")


# In[209]:


visualization(data, model, cl_model)


# In[210]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(5, data, 1, 'KMeans++')")


# In[213]:


visualization(data, model, cl_model)


# ### На данном примере при random.seed(123) получается, что kMeans работает лучше

# In[224]:


data, d =  make_blobs(n_samples= 5000, n_features= 2, centers = 5, random_state=123)


# In[227]:


get_ipython().run_cell_magic('time', '', 'elbow_method(data, 2, 10)')


# In[216]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(5, data, 1, ' ')")


# In[217]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(5, data, 1, 'KMeans++')")


# In[228]:


data, d =  make_blobs(n_samples= 30000, n_features= 2, centers = 5, random_state=123)


# In[229]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(5, data, 1, ' ')")


# In[230]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(5, data, 1, 'KMeans++')")


# In[238]:


data, d =  make_blobs(n_samples= 5000, n_features= 2, centers = 15, random_state=123)


# In[239]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(15, data, 1, ' ')")


# In[240]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(15, data, 1, 'KMeans++')")


# In[241]:


data, d =  make_blobs(n_samples= 30000, n_features= 2, centers = 15, random_state=123)


# In[244]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(15, data, 1, ' ')")


# In[243]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(15, data, 1, 'KMeans++')")


# ### dataset "birch2" скачан отсюда: http://cs.joensuu.fi/sipu/datasets/

# In[245]:


data = np.loadtxt("/Users/ofirserovlad/Downloads/birch2.txt", dtype="int")


# In[247]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(2, data, 1, ' ')")


# In[250]:


get_ipython().run_cell_magic('time', '', "model, cl_model = myKMeans(2, data, 1, 'KMeans++')")


# ## Вывод:
# ### Проанализировав все примеры, можно сказать о том, что "kMeans++"" в большинстве примеров работает быстрее (может зависеть от начальной инициализации и от распределенности данных)
# ###  Минусом kMeans-like алгоритмов является то, что они могут "не находить глобальный минимум". Также нужно находить количество кластеров самостоятельно.
# ###  На большем количестве кластеров при том же количестве объектов алгоритм ожидаемо работает дольше; количество объектов влияет также "по направлению", несколько сильнее "по времени" 
