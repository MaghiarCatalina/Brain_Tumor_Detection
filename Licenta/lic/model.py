import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam, Adadelta
import matplotlib.pyplot as plot
from PIL import Image
import os
import numpy as numpy

dataset_path = "D:\\Facultate\\Licenta\\Licenta\\dataset_equal_classes\\"


def rename_img_no_tumour():
    i = 0
    path = dataset_path + "validation\\no\\"
    for filename in os.listdir(path):
        dst = str(i) + ".jpg"
        src = path + filename
        dst = path + dst
        os.rename(src, dst)
        i += 1


def rename_img_yes_tumour():
    i = 0
    path = dataset_path + "validation\\yes\\"
    for filename in os.listdir(path):
        dst = str(i) + ".jpg"
        src = path + filename
        dst = path + dst
        os.rename(src, dst)
        i += 1


def data_preparation():
    # we don't load all images at once --> load in batches(ex. 32 img at a time)
    # epoch = a pass through the whole data set
    # load data from folders, normalize it and augment(only train data)

    train_data_generator = ImageDataGenerator(
        rotation_range=90,                      # Degree range for random rotations.
        brightness_range=[0.4, 1.4],            # <1.0 darken the img, >1.0 brighten the img [0,2]
        shear_range=0.1,                        # Shear angle in counter-clockwise direction in degrees
        zoom_range=0.3,                         # <1 zoom in, >1 zoom out [0,2]; 0.1 -> [0.9,1.1]
        preprocessing_function=preprocess_input # Function that will be applied on each input after the image is resized and augmented
    )

    train_generator = train_data_generator.flow_from_directory( # Takes the path to a directory & generates batches of augmented data
        dataset_path + 'train',
        batch_size=16,                  # how many images are loaded at once(32 is default)
        class_mode='categorical',       # It's a binary classification; categorical
        # save_to_dir='C:\\Users\\cata3\\Desktop\\Licenta\\Licenta\\brain_tumor_dataset\\preview_augmented',
        target_size=(224, 224))         # The dimensions to which all images found will be resized(default for ResNet is 224x224)

    validation_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    validation_generator = validation_data_generator.flow_from_directory(
        dataset_path + 'validation',
        batch_size=16,
        class_mode='categorical', # categorical
        target_size=(224, 224))

    return train_generator, validation_generator


def create_network():
    # load pre-trained network, cut last layers and freeze weights
    # add our layers at the end
    # set optimizer
    # and loss function
    model_init = ResNet50(weights='imagenet',
                          include_top=False)     # don't include the fully-connected layer at the top of the network
    for layer in model_init.layers:
        layer.trainable = False                 # freeze weights

    x = model_init.output
    x = GlobalMaxPooling2D()(x)                 # 2D because we work with images, // GlobalAveragePooling2D()
    x = Dropout(0.2)(x)                         # helps prevent overfitting
    x = Dense(128, activation='tanh')(x)
    predictions = Dense(2, activation='softmax')(x)     # 2=nr of classes, activation function -> in order to have a non linear func (otherwise our NN will behave as a single layer netw)
                                                        # softmax - output layers should use it; giver probabilities for different classes ( good for classifications); tanh(2 class clasif); for hidden layers - relu
    model_new = Model(inputs=model_init.input, outputs=predictions)

    opt = Adam(learning_rate=0.0001, amsgrad=True)
    model_new.compile(                                  # Configure the model for training
                    loss='categorical_crossentropy',    # 'categorical_crossentropy' ,binary cross entropy(measures the performance of a classif model whose output is a probability value between 0 and 1), hinge loss(SVM)-faster but less accurate, square hinge
                    optimizer=opt,                      # adam; sgd
                    metrics=['accuracy'])
    return model_new


def train(model, train_generator, validation_generator, model_name):
    # train model
    # compute accuracy
    # save model and weights
    history = model.fit_generator(                       # Trains the model on data generated batch-by-batch by a Python generator
                    generator=train_generator,
                    validation_data=validation_generator,
                    epochs=45,
    )
    # plot for accuracy
    plot.plot(history.history['accuracy'])
    plot.plot(history.history['val_accuracy'])
    plot.title('model accuracy')
    plot.ylabel('accuracy')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    plot.savefig("acc_plot" + model_name + ".png")
    plot.show()
    # plot for loss
    plot.plot(history.history['loss'])
    plot.plot(history.history['val_loss'])
    plot.title('model loss')
    plot.ylabel('loss')
    plot.xlabel('epoch')
    plot.legend(['train', 'test'], loc='upper left')
    plot.savefig("loss_plot" + model_name + ".png")
    plot.show()

    model.save('mymodels/'+model_name)


def test(model):
    # load model and verify with images from validation folder
    trained_model = load_model('mymodels/'+model)
    test_img = [
                dataset_path + 'validation\\yes\\0.jpg',
                dataset_path + 'validation\\yes\\1.jpg',
                dataset_path + 'validation\\yes\\2.jpg',
                dataset_path + 'validation\\yes\\3.jpg',
                dataset_path + 'validation\\yes\\4.jpg',
                dataset_path + 'validation\\yes\\5.jpg',
                dataset_path + 'validation\\yes\\6.jpg',
                dataset_path + 'validation\\yes\\7.jpg',
                ]
    img_list = [Image.open(path) for path in test_img]

    validation_batch = numpy.stack([preprocess_input(numpy.array(img.resize((224, 224)))) for img in img_list])

    predictions = trained_model.predict(validation_batch)        # returns Numpy array(s) of predictions

    fig, axs = plot.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        ax.set_title("{:.0f}% no, {:.0f}% yes".format(100 * predictions[i, 0],
                                                      100 * predictions[i, 1]))
        ax.imshow(img)
    plot.show()


def accuracy(model_name):       # view validation accuracy and loss
    trained_model = load_model('mymodels/'+model_name)
    validation_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    validation_generator = validation_data_generator.flow_from_directory(
        dataset_path + 'validation',
        class_mode='categorical',
        target_size=(224, 224))
    score = trained_model.evaluate_generator(validation_generator, 1, workers=1)
    print(model_name + ": Loss: ", score[0], "Accuracy: ", score[1])


def data_preparation_from_gui(rotation_range, brightness_low, brightness_high, zoom_range):
    train_data_generator = ImageDataGenerator(
        rotation_range=rotation_range,
        brightness_range=[brightness_low, brightness_high],
        zoom_range=zoom_range,
        preprocessing_function=preprocess_input
    )
    train_generator = train_data_generator.flow_from_directory(
        dataset_path + 'train',
        batch_size=16,
        class_mode='categorical',
        target_size=(224, 224)
    )
    validation_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input)
    validation_generator = validation_data_generator.flow_from_directory(
        dataset_path + 'validation',
        batch_size=16,
        class_mode='categorical',
        target_size=(224, 224))
    return train_generator, validation_generator


def create_network_from_gui(loss_function, optimizer):
    model_init = ResNet50(weights='imagenet',
                          include_top=False)
    for layer in model_init.layers:
        layer.trainable = False
    x = model_init.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='tanh')(x)
    predictions = Dense(2, activation='softmax')(x)

    model_new = Model(inputs=model_init.input, outputs=predictions)

    model_new.compile(
                    loss=loss_function,
                    optimizer=optimizer,
                    metrics=['accuracy'])
    return model_new


def train_from_gui(model, train_generator, validation_generator, model_name, epochs_number):
    model.fit_generator(
                    generator=train_generator,
                    validation_data=validation_generator,
                    epochs=epochs_number,
    )

    model.save('usermodels/' + model_name)
    return "Done!"


def test_from_gui(model_name, image_path):
    model = load_model(model_name)
    test_img = [image_path]
    img_list = [Image.open(path) for path in test_img]
    validation_batch = numpy.stack([preprocess_input(numpy.array(img.resize((224, 224)))) for img in img_list])
    predictions = model.predict(validation_batch)
    return str("{:.0f}% Tumor, {:.0f}% No tumor".format(100 * predictions[0, 1],
                                                        100 * predictions[0, 0]))


def accuracy_from_gui(model_name):
    trained_model = load_model('usermodels/'+model_name)
    validation_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    validation_generator = validation_data_generator.flow_from_directory(
        dataset_path + 'validation',
        class_mode='categorical',
        target_size=(224, 224))
    score = trained_model.evaluate_generator(validation_generator, 1, workers=1)
    return score[1]


if __name__ == '__main__':
    # train_gen, validation_gen = data_preparation()
    # nw = create_network()
    # train(nw, train_gen, validation_gen, 'test.h5')

    #accuracy('model20_gui.h5')
    #test('model20_gui.h5')
    # plot_model(model, to_file='test_plot.png')
    model = load_model('mymodels/'+'model20_gui.h5')
    model.summary()


