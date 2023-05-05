import ttkbootstrap as ttk
from ttkbootstrap.scrolled import ScrolledText
from ttkbootstrap.toast import ToastNotification
from tkinter.messagebox import showerror
import googletrans
from googletrans import Translator
import pyttsx3
import pyperclip

translator = Translator()

engine = pyttsx3.init()


class LanguageTranslator:
    def __init__(self, master):
        self.master = master
        # calling the UI method in the constructor
        self.MainWindow()
        # calling the Widgets method in the constructor
        self.Widgets()


    def MainWindow(self):
        self.master.geometry('600x430+300+150')
        self.master.title('Language Translator')
        self.master.resizable(width = 0, height = 0)
        # setting the icon for the application
        icon = ttk.PhotoImage(file='icon.png')
        self.master.iconphoto(False, icon)


    def Widgets(self):
        # the canvas for containing the other widgets
        self.canvas = ttk.Canvas(self.master, width = 600, height = 400)
        self.canvas.pack()
        # the logo for the application
        self.logo = ttk.PhotoImage(file='logo.png').subsample(5, 5)
        self.canvas.create_image(75, 70, image = self.logo)
        # getting all the languages 
        language_data = googletrans.LANGUAGES
        # getting all the language values using the values() function
        language_values = language_data.values()
        # converting the languages to a list
        languages = list(language_values)
        # first combobox for the source language
        self.from_language = ttk.Combobox(self.canvas, width = 36, bootstyle = 'primary', values = languages)
        self.from_language.current(0)
        self.canvas.create_window(150, 140, window = self.from_language)
        # loading the arrow icon
        self.arrow_icon = ttk.PhotoImage(file='arrows.png')
        self.resized_icon = self.arrow_icon.subsample(15, 15)
        self.image_label = ttk.Label(self.master, image = self.resized_icon)
        self.canvas.create_window(300, 140, window = self.image_label)
        # the second combobox for the destination language
        self.to_language = ttk.Combobox(self.canvas, width = 36, bootstyle = 'primary', values = languages)
        self.to_language.current(21)
        self.canvas.create_window(450, 140, window = self.to_language)
        # scrollable text for entering input
        self.from_text = ScrolledText(self.master, font=("Dotum", 10), width = 30, height = 10)
        self.canvas.create_window(150, 250, window = self.from_text)
        # scrollable text for output
        self.to_text = ScrolledText(self.master, font=("Dotum", 10), width = 30, height = 10)
        self.canvas.create_window(450, 250, window = self.to_text)
        # loading icons
        self.speaker_icon = ttk.PhotoImage(file = 'speaker.png').subsample(5, 4)
        self.copy_icon = ttk.PhotoImage(file = 'copy.png').subsample(5, 4)
        self.speak_button = ttk.Button(self.master, image = self.speaker_icon, bootstyle='secondary', state=ttk.DISABLED, command = self.speak)
        self.canvas.create_window(355, 355, window = self.speak_button)
        self.copy_button = ttk.Button(self.master, image = self.copy_icon, bootstyle='secondary', state=ttk.DISABLED, command = self.copy_to_clipboard)
        self.canvas.create_window(395, 355, window = self.copy_button)
        self.translate_button = ttk.Button(self.master, text = 'Translate', width = 20, bootstyle = 'primary', command = self.translate)
        self.canvas.create_window(300, 400, window = self.translate_button)

    def translate(self):
        try:
            # getting source language from first combobox via get() 
            self.source_language = self.from_language.get()
            # getting destination language from first combobox via get() 
            self.destination_language = self.to_language.get()
            # getting every content fronm the first scrolledtext
            self.text = self.from_text.get(1.0, ttk.END)
            # translating the language
            self.translation = translator.translate(self.text, src=self.source_language, dest=self.destination_language)
            # clearing the second scrolledtext
            self.to_text.delete(1.0, ttk.END)
            # inserting translation output in the second scroledtext  
            self.to_text.insert(ttk.END, self.translation.text)
            # activating the speak_button
            self.speak_button.configure(state = ttk.ACTIVE)
            # activating the copy_button 
            self.copy_button.configure(state = ttk.ACTIVE)
        # handle TypeError using except
        except TypeError as e:
            showerror(title='Invalid Input', message='Make sure you have entered valid input!') 
        # handle connection errors
        except Exception as e:
            showerror(title='Connection Error', message='Make sure you have internet connection!')

    def speak(self):
        # getting every content from the second scrolledtext
        self.text = self.to_text.get(1.0, ttk.END)
        # gets the speaking rate
        rate = engine.getProperty('rate')
        # setting the speaking rate
        engine.setProperty('rate', 125)
        # getting the available voices
        voices = engine.getProperty('voices')
        # setting the second voice, the female voice
        engine.setProperty('voice', voices[1].id)
        # saying the translated text
        engine.say(self.text)
        # running the speech
        engine.runAndWait()

    def copy_to_clipboard(self):
        # this will create a toast notification object
        toast = ToastNotification(
            title='Clip Board',
            message='Text has been copied to clip board!',
            duration=3000,
        )
        # this will show the notification
        toast.show_toast()
        # getting all the content from the second scrolledtext
        self.text = self.to_text.get(1.0, ttk.END)
        # copy to clip board
        pyperclip.copy(self.text)
           


root = ttk.Window(themename="cosmo")
application = LanguageTranslator(root)
root.mainloop()

