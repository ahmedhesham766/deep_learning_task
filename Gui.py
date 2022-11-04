import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

import model
from model import Features
from model import Species
from model import Perceptron
from model import preprocess


def feature_1_changed(event):
    features[0] = Features(feature1_cmb['values'].index(feature1_cmb.get()) + 1)


def feature_2_changed(event):
    features[1] = Features(feature2_cmb['values'].index(feature2_cmb.get()) + 1)


def class_1_changed(event):
    goals[0] = Species(goal1_cmb['values'].index(goal1_cmb.get()))


def class_2_changed(event):
    goals[1] = Species(goal2_cmb['values'].index(goal2_cmb.get()))


def get_eta():
    eta = float(eta_txt.get())
    return eta


def get_epochs():
    epochs = int(epoch_txt.get())
    return epochs


def data_entry_error():
    if selected_feature1 == selected_feature2 or selected_class1 == selected_class2:
        messagebox.showerror("Error", "Please make sure that the entered features and classes are different.")

    if len(eta_txt.get()) == 0 or len(epoch_txt.get()) == 0 or selected_feature1 == '' or selected_feature2 == ''\
            or selected_class1 == '' or selected_class2 == '':
        messagebox.showerror("Error", "Missing data.")


def func():
    x_train, y_train, x_test, y_test = preprocess(features=features, goals=goals, dataset=model.dataset)

    per = Perceptron(features=features,
                     goals=goals,
                     x_train_data=x_train,
                     x_test_data=x_test,
                     y_train_data=y_train,
                     y_test_data=y_test,
                     eta=get_eta(),
                     epochs=get_epochs(),
                     with_bias=bias_checkbox_var.get())
    per.train()
    per.test()
    per.plot()


gui = Tk()
gui.title('Task_1 GUI')
gui.geometry('500x720+550+60')

Title = Label(gui, text="User Input", fg='black', font=("Times New Roman", 20))
Title.place(x=180, y=30)

feature1_label = Label(gui, text="Select first feature", fg='black', font=("Times New Roman", 14))
feature1_label.place(x=50, y=130)
selected_feature1 = tk.StringVar()
feature1_cmb = ttk.Combobox(gui, textvariable=selected_feature1, width=27)
feature1_cmb['values']: list = [e.name for e in Features]
feature1_cmb.place(x=290, y=130)

features: list = [None] * 2
goals: list = [None] * 2

feature1_cmb.bind('<<ComboboxSelected>>', feature_1_changed)

feature2_label = Label(gui, text="Select second feature", fg='black', font=("Times New Roman", 14))
feature2_label.place(x=50, y=200)
selected_feature2 = tk.StringVar()
feature2_cmb = ttk.Combobox(gui, textvariable=selected_feature2, width=27)
feature2_cmb['values'] = [e.name for e in Features]
feature2_cmb.place(x=290, y=200)
feature2_cmb.bind('<<ComboboxSelected>>', feature_2_changed)

goal1_lbl = Label(gui, text="Select first class", fg='black', font=("Times New Roman", 14))
goal1_lbl.place(x=50, y=270)
selected_class1 = tk.StringVar()
goal1_cmb = ttk.Combobox(gui, textvariable=selected_class1, width=27)
goal1_cmb['values'] = [e.name for e in Species]
goal1_cmb.place(x=290, y=270)
goal1_cmb.bind('<<ComboboxSelected>>', class_1_changed)

goal2_lbl = Label(gui, text="Select second class", fg='black', font=("Times New Roman", 14))
goal2_lbl.place(x=50, y=340)
selected_class2 = tk.StringVar()
goal2_cmb = ttk.Combobox(gui, textvariable=selected_class2, width=27)
goal2_cmb['values'] = [e.name for e in Species]
goal2_cmb.place(x=290, y=340)
goal2_cmb.bind('<<ComboboxSelected>>', class_2_changed)

eta_lbl = Label(gui, text="Enter learning rate", fg='black', font=("Times New Roman", 14))
eta_lbl.place(x=50, y=410)
eta_txt = Entry(gui, bg='white', fg='black', bd=9, width=27)
eta_txt.place(x=290, y=410)

epoch_lbl = Label(gui, text="Enter number of epochs", fg='black', font=("Times New Roman", 14))
epoch_lbl.place(x=50, y=480)
epoch_txt = Entry(gui, bg='white', fg='black', bd=9, width=27)
epoch_txt.place(x=290, y=480)

bias_lbl = Label(gui, text="Add bias or not", fg='black', font=("Times New Roman", 14))
bias_lbl.place(x=50, y=550)
bias_checkbox_var = tk.StringVar()
bias_cb = ttk.Checkbutton(gui,
                          text='Bias',
                          variable=bias_checkbox_var,
                          onvalue=True,
                          offvalue=False)
bias_cb.place(x=360, y=555)

btn = Button(gui,
             text="Run",
             fg='black',
             width=15,
             font=("Times New Roman", 14),
             command=lambda: [data_entry_error(), func()])
btn.place(x=180, y=620)

gui.mainloop()
