# execute this file to launch the translation app

from tkinter import *
from machine_translation import predict

window = Tk()
window.geometry('1080x500')
window.resizable(0, 0)
window.title("NLP - Machine Translation")
window.config(bg='#023e8a')

label_title = Label(window, text="Machine Translation", font="arial 20 bold", fg="white", bg='#023e8a')
label_title.place(x=430, y=10)
label_footer = Label(window, text="Mini projet en NLP pour la traduction automatique", font='arial 20 bold', fg="#90e0ef", bg='#023e8a')
label_footer.pack(side='bottom')

# INPUT AND OUTPUT TEXT WIDGET
input_label = Label(window, text="Entrez le texte à traduire", font='arial 15 bold', bg='white smoke', padx=5, pady=5)
input_label.place(x=100, y=80)
input_text = Text(window, font='arial 17', height=11, wrap=WORD, padx=5, pady=5, width=40)
input_text.place(x=30, y=150)

output_label = Label(window, text="Résultat de la traduction", font='arial 15 bold', bg='white smoke', padx=5, pady=5)
output_label.place(x=770, y=80)
output_text = Text(window, font='arial 17', height=11, wrap=WORD, padx=5, pady=5, width=40)
output_text.place(x=600, y=150)

################## List translation ##################

languages = ["Francais ==> Portugais", "Anglais ==> Francais"]

variable = StringVar(window)
variable.set(languages[0])

opt = OptionMenu(window, variable, *languages)
opt.config(width=25, font=('Helvetica', 15), bg="#023e8a")
opt.place(x=400, y=70)


########################################  Define function #######

def translate():
    text_input = input_text.get(1.0, END)
    print(text_input, len(text_input))
    lang_selected = variable.get()
    print("selected: ", lang_selected)
    if len(text_input) > 1:
        translated_text = predict.translate(text_input, lang_selected)
        output_text.delete(1.0, END)
        output_text.insert(END, translated_text)


##########  Translate Button ########
trans_btn = Button(window,
                   text='Traduire',
                   font='arial 18 bold',
                   padx=20,
                   pady=10,
                   command=translate
                   )
trans_btn.place(x=470, y=400)

window.mainloop()

