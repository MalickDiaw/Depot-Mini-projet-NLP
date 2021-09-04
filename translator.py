# execute this file to launch the translation app

from tkinter import *
from machine_translation import predict

window = Tk()
window.geometry('1080x500')
window.resizable(0, 0)
window.title("NLP - Machine Translation")
window.config(bg='#0077b6')

label_title = Label(window, text="Mini-projet en NLP sur la Traduction Automatique", font="arial 20 bold", fg="#f1faee", bg='#0077b6')
label_title.place(x=300, y=10)
label_footer = Label(window, text="Copyright © 2021 Malick Diaw - Vanda Martins", font='courier 16', fg="#caf0f8", bg='#0077b6')
label_footer.pack(side='bottom')

# INPUT AND OUTPUT TEXT WIDGET
input_label = Label(window, text="Entrez le texte à traduire", font='arial 15 bold', bg='#caf0f8', padx=5, pady=5)
input_label.place(x=100, y=100)
input_text = Text(window, font='arial 17', height=11, wrap=WORD, padx=5, pady=5, width=40)
input_text.place(x=30, y=150)

output_label = Label(window, text="Résultat de la traduction", font='arial 15 bold', bg='#caf0f8', padx=5, pady=5)
output_label.place(x=770, y=100)
output_text = Text(window, font='arial 17', height=11, wrap=WORD, padx=5, pady=5, width=40)
output_text.bind("<Key>", lambda a: "break")
output_text.place(x=600, y=150)

################## List translation ##################

languages = ["Francais ==> Portugais", "Anglais ==> Francais"]

variable = StringVar(window)
variable.set(languages[1])

opt = OptionMenu(window, variable, *languages)
opt.config(width=20, font=('Helvetica', 20, 'bold'), fg="#023047", bg="#0077b6")
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
        output_text.bind("<Key>", lambda a: "break")


##########  Translate Button ########
trans_btn = Button(window,
                   text='Traduire',
                   font='arial 18 bold',
                   padx=25,
                   pady=10,
                   fg="#03045e",
                   command=translate
                   )
trans_btn.place(x=460, y=400)

window.mainloop()

