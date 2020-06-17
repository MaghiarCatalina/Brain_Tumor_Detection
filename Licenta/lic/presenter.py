import kivy
from kivy.config import Config
Config.set('graphics', 'window_state', 'maximized')  #visible
# Config.set('graphics', 'fullscreen', 0)
# Config.write()
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from lic.model import test_from_gui, data_preparation_from_gui, train_from_gui, create_network_from_gui,\
    accuracy_from_gui
from kivy.properties import ObjectProperty, ListProperty


dataset_path = "D:\\Facultate\\Licenta\\Licenta\\dataset\\"


class Tabs(TabbedPanel):
    def __init__(self, **kwargs):
        super(Tabs, self).__init__(**kwargs)
        arr = os.listdir('D:\\Facultate\\Licenta\\Licenta\\lic\\usermodels')
        self.ids.spinner_model.values = arr

    notselected_color = [.12, .10, .14, .5]
    selected_color = [.35, .35, .36, .5]
    model1 = ObjectProperty(None)
    model2 = ObjectProperty(None)
    color1 = ListProperty(selected_color)
    color2 = ListProperty(notselected_color)
    parameters_ok = True

    model_name = ''
    image_path = ''
    image_path_tab2 = ''

    def checkbox_on_active(self):
        if self.model1.active:
            self.color1 = self.selected_color
            self.color2 = self.notselected_color
        else:
            self.color1 = self.notselected_color
            self.color2 = self.selected_color

    def upload_btn_press(self):
        self.ids.warning_upload_tab1.text = ""
        root = Tk()
        root.withdraw()                                 # don't open tk gui
        image_path = askopenfilename()                  # open file explorer to select file; return its path
        if image_path != "":
            self.image_path = image_path                # save image path in the class variable
            reverse_name, rest = ((image_path[::-1]).split('/', 1))     # reverse path and split once at '/'
            img_name = reverse_name[::-1]               # now reverse back to obtain image name
            self.ids.label_img_name.text = img_name     # display image name on gui
        else:
            self.ids.label_img_name.text = ""

    def predict_btn_press(self):
        if self.model1.active:
            self.model_name = 'model20_gui.h5'
        elif self.model2.active:
            self.model_name = 'model20x_gui.h5'
        else:
            self.model_name = 'model20_gui.h5'
        if self.image_path != "":
            result = test_from_gui('mymodels/'+self.model_name, self.image_path)
            self.ids.label_predict_result.text = result
            self.ids.image_mri.source = self.image_path
        else:
            self.ids.warning_upload_tab1.text = "Please select an image."

    def rotation_down(self):
        self.rotation_focus()
        rotation_value = self.ids.text_rotation.text
        if rotation_value != "":
            if int(rotation_value) > 0:
                self.ids.text_rotation.text = str(int(rotation_value) - 1)
        else:
            self.ids.text_rotation.text = "0"

    def rotation_up(self):
        self.rotation_focus()
        rotation_value = self.ids.text_rotation.text
        if rotation_value != "":
            if int(rotation_value) < 180:
                self.ids.text_rotation.text = str(int(rotation_value) + 1)
        else:
            self.ids.text_rotation.text = "0"

    def brightness_low_down(self):
        self.brightness_focus()
        brightness_value = self.ids.text_brightness_low.text
        if brightness_value != "":
            if float(brightness_value) > 0.0:
                self.ids.text_brightness_low.text = str(round(float(brightness_value) - 0.1, 2))
        else:
            self.ids.text_brightness_low.text = "0.0"

    def brightness_low_up(self):
        self.brightness_focus()
        brightness_value = self.ids.text_brightness_low.text
        if brightness_value != "":
            if float(brightness_value) < 1.0:
                self.ids.text_brightness_low.text = str(round(float(brightness_value) + 0.1, 2))
        else:
            self.ids.text_brightness_low.text = "0.0"

    def brightness_high_down(self):
        self.brightness_focus()
        brightness_value = self.ids.text_brightness_high.text
        if brightness_value != "":
            if float(brightness_value) > 1.0:
                self.ids.text_brightness_high.text = str(round(float(brightness_value) - 0.1, 2))
        else:
            self.ids.text_brightness_high.text = "1.0"

    def brightness_high_up(self):
        self.brightness_focus()
        brightness_value = self.ids.text_brightness_high.text
        if brightness_value != "":
            if float(brightness_value) < 2.0:
                self.ids.text_brightness_high.text = str(round(float(brightness_value) + 0.1, 2))
        else:
            self.ids.text_brightness_high.text = "1.0"

    def zoom_down(self):
        self.zoom_focus()
        zoom_value = self.ids.text_zoom.text
        if zoom_value != "":
            if float(zoom_value) > 0.0:
                self.ids.text_zoom.text = str(round(float(zoom_value) - 0.1, 2))
        else:
            self.ids.text_zoom.text = "0.0"

    def zoom_up(self):
        self.zoom_focus()
        zoom_value = self.ids.text_zoom.text
        if zoom_value != "":
            if float(zoom_value) < 1.0:
                self.ids.text_zoom.text = str(round(float(zoom_value) + 0.1, 2))
        else:
            self.ids.text_zoom.text = "0.0"

    def name_focus(self):
        self.ids.train_warning.text = ""
        self.ids.percent_progress.text = "In progress"
        self.ids.training_progress.value = 0

    def rotation_focus(self):
        self.ids.warning_rotation.color = [1, 1, 1, 1]
        self.ids.percent_progress.text = "In progress"
        self.ids.training_progress.value = 0

    def brightness_focus(self):
        self.ids.warning_brightness.color = [1, 1, 1, 1]
        self.ids.percent_progress.text = "In progress"
        self.ids.training_progress.value = 0

    def zoom_focus(self):
        self.ids.warning_zoom.color = [1, 1, 1, 1]
        self.ids.percent_progress.text = "In progress"
        self.ids.training_progress.value = 0

    def epochs_focus(self):
        self.ids.epochs_warning.text = ""

    def spinner_on_text(self):
        self.ids.percent_progress.text = "In progress"
        self.ids.training_progress.value = 0

    def spinner_model_on_text(self):
        self.ids.warning_model_select.text = ""

    def verify_values(self):
        warning_color = [.81, .40, .47, 1]
        self.ids.text_rotation.focus = False
        self.ids.text_brightness_low.focus = False
        self.ids.text_brightness_high.focus = False
        self.ids.text_zoom.focus = False
        self.ids.text_name.focus = False
        self.ids.text_epochs.focus = False

        self.parameters_ok = True
        rotation = self.ids.text_rotation.text
        zoom = self.ids.text_zoom.text
        brightness_low = self.ids.text_brightness_low.text
        brightness_high = self.ids.text_brightness_high.text
        if rotation != "":
            if int(rotation) < 0 or int(rotation) > 180:
                self.ids.warning_rotation.color = warning_color
                self.parameters_ok = False
        else:
            self.ids.warning_rotation.color = warning_color
            self.parameters_ok = False

        if zoom != "":
            if float(zoom) < 0.0 or float(zoom) > 1.0:
                self.ids.warning_zoom.color = warning_color
                self.parameters_ok = False
        else:
            self.ids.warning_zoom.color = warning_color
            self.parameters_ok = False

        if brightness_low != "" and brightness_high != "":
            if float(brightness_low) < 0.0 or float(brightness_low) > 1.0 \
                    or float(brightness_high) < 1.0 or float(brightness_high) > 2.0:
                self.ids.warning_brightness.color = warning_color
                self.parameters_ok = False
        else:
            self.ids.warning_brightness.color = warning_color
            self.parameters_ok = False

        if str(self.ids.text_name.text) == "":
            self.parameters_ok = False
            self.ids.train_warning.text = "Please enter a name for your model"
        if str(self.ids.text_epochs.text) == "" or str(self.ids.text_epochs.text) == "0":
            self.parameters_ok = False
            self.ids.epochs_warning.text = "Please enter an integer number of epochs (recommended <50)."

    def train_btn_press(self):
        if self.parameters_ok:
            
            rotation = int(self.ids.text_rotation.text)
            brightness_low = float(self.ids.text_brightness_low.text)
            brightness_high = float(self.ids.text_brightness_high.text)
            zoom = float(self.ids.text_zoom.text)
            loss = str(self.ids.loss.text)
            optimizer = str(self.ids.optimizer.text)
            model_name = str(self.ids.text_name.text)
            epochs = int(self.ids.text_epochs.text)

            train_gen, validation_gen = data_preparation_from_gui(rotation, brightness_low, brightness_high, zoom)
            network = create_network_from_gui(loss, optimizer)
            self.ids.training_progress.value = 50
            train_from_gui(network, train_gen, validation_gen, model_name+".h5", epochs)
            arr = os.listdir('D:\\Facultate\\Licenta\\Licenta\\lic\\usermodels')
            self.ids.spinner_model.values = arr
            self.ids.percent_progress.text = "Done"
            self.ids.training_progress.value = 100

    def accuracy_btn_press(self):
        model_name = self.ids.spinner_model.text
        if model_name != "":
            acc_value = 100 * accuracy_from_gui(model_name)
            self.ids.text_accuracy.text = str(acc_value)
        else:
            self.ids.warning_model_select.text = "Please select a model"

    def upload_btn_press_tab2(self):
        self.ids.warning_upload.text = ""
        root = Tk()
        root.withdraw()
        image_path = askopenfilename()
        if image_path != "":
            self.image_path_tab2 = image_path
            reverse_name, rest = ((image_path[::-1]).split('/', 1))
            img_name = reverse_name[::-1]
            self.ids.label_img_name_tab2.text = img_name  # display image name on gui

    def predict_btn_press_tab2(self):
        model_name = self.ids.spinner_model.text
        if model_name != "" and self.image_path_tab2 != "":
            result = test_from_gui('usermodels/'+model_name, self.image_path_tab2)
            self.ids.label_result_tab2.text = result
            self.ids.image_mri_tab2.source = self.image_path_tab2
        if model_name == "":
            self.ids.warning_model_select.text = "Please select a model."
        if self.image_path_tab2 == "":
            self.ids.warning_upload.text = "Please select an image."


class TumorDetection(App):
    def build(self):
        return Tabs()


if __name__ == '__main__':
    TumorDetection().run()
