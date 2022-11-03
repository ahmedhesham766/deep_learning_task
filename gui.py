
#Features
import tkinter as tk
from tkinter import *
from tkinter.messagebox import showinfo
#from tkinter import StringVar
#from tkinter import callback
import enum as Enum
from tkinter import ttk
import pandas as pd
import main
gui = Tk()
gui.title('Task_1 GUI')
gui.geometry('500x700+550+70')
Title = Label(gui, text="User Input", fg='black', font=("Times New Roman", 20))
Title.place(x=180, y=30)

features= [Enum,Enum]
goals = [Enum,Enum]

lbl1 = Label(gui, text="Select first feature", fg='black', font=("Times New Roman", 14))
lbl1.place(x=50, y=130)
selected_feature1 =  tk.StringVar()
cmb1= ttk.Combobox(gui,textvariable = selected_feature1, width=27)
cmb1['values'] = (main.Features(1), main.Features(2), main.Features(3),  main.Features(4), main.Features(5))
cmb1.place(x=290, y=130)


def feature_changed(event):
    showinfo(
        title='Result',
        message=f'You selected {selected_feature1.get()}!',
    )
    features.append(main.Features(cmb1.get()))
    print(cmb1.get())
    print("feature1:"+str(main.features[0]))
    
cmb1.bind('<<ComboboxSelected>>', feature_changed)


lbl2 = Label(gui, text="Select second feature", fg='black', font=("Times New Roman", 14))
lbl2.place(x=50, y=200)
selected_feature2 =  tk.StringVar()
cmb2 = ttk.Combobox(gui,textvariable = selected_feature2, width=27)
cmb2['values'] = (main.Features(1), main.Features(2), main.Features(3), main.Features(4), main.Features(5))
cmb2.place(x=290, y=200)

def feature_changed(event):
    showinfo(
        title='Result',
        message=f'You selected {selected_feature2.get()}!',
    )
    features.append(main.Features(cmb2.get()))
    print("feature2:"+str(features[1]))
cmb2.bind('<<ComboboxSelected>>', feature_changed)


lbl3 = Label(gui, text="Select first class", fg='black', font=("Times New Roman", 14))
lbl3.place(x=50, y=270)
selected_class1 =  tk.StringVar()
cmb3 = ttk.Combobox(gui,textvariable = selected_class1 ,width=27)
cmb3['values'] = (main.Species(0), main.Species(1), main.Species(2))
cmb3.place(x=290, y=270)
def class_changed(event):
    showinfo(
        title='Result',
        message=f'You selected {selected_class1.get()}!',
    )
    goals.append(main.Species(cmb3.get()))
    print("goal1:"+str(goals[0]))
cmb3.bind('<<ComboboxSelected>>',class_changed)


lbl4 = Label(gui, text="Select second class", fg='black', font=("Times New Roman", 14))
lbl4.place(x=50, y=300)
selected_class2 =  tk.StringVar()
cmb4 = ttk.Combobox(gui,textvariable = selected_class2, width=27)
cmb4['values'] = (main.Species(0), main.Species(1), main.Species(2))
cmb4.place(x=290, y=300)
def class_changed(event):
    showinfo(
        title='Result',
        message=f'You selected {selected_class2.get()}!',
    )
    goals.append(main.Species(cmb4.get()))
    print("goal2:"+str(goals[1]))
cmb4.bind('<<ComboboxSelected>>',class_changed)


def get_eta():
    eta = float(txt1.get())
    return eta


def get_epochs():
    epochs = int(txt2.get())
    return epochs


lbl5 = Label(gui, text="Enter learning rate", fg='black', font=("Times New Roman", 14))
lbl5.place(x=50, y=340)
txt1 = Entry(gui, bg='white', fg='black', bd=9, width=27)
txt1.place(x=290, y=340)


lbl6 = Label(gui, text="Enter number of epochs", fg='black', font=("Times New Roman", 14))
lbl6.place(x=50, y=410)
txt2 = Entry(gui, bg='white', fg='black', bd=9, width=27)
txt2.place(x=290, y=410)

lbl7 = Label(gui, text="Add bias or not", fg='black', font=("Times New Roman", 14))
lbl7.place(x=50, y=480)
checkbox_var = tk.StringVar()



cb = ttk.Checkbutton(gui,
                text='<Bias>',
                variable=checkbox_var,
                onvalue=True,
                offvalue=False)
cb.place(x=360, y=485)

def prints():
    print(features)
    print(goals)
def func():
    per = main.Perceptron(features=features,
                                goals=goals,
                                x_train_data=X_Train,
                                x_test_data=X_Test,
                                y_train_data=Y_Train,
                                y_test_data=Y_Test,
                                eta= get_eta(),
                                epochs= get_epochs(),
                                with_bias= checkbox_var.get())
    per.train()
    per.test()
    per.plot()

X_Train, Y_Train, X_Test, Y_Test = main.preprocess(features = features ,goals = goals, dataset = main.dataset)

btn = Button(gui, text="Run", fg='black', width=15, font=("Times New Roman", 14),
             command= func,
            
       
             )
btn.place(x=180, y=580)
"""
lambda: [main.Perceptron(features=main.features,
                            goals=main.goals,
                            x_train_data=X_Train,
                            x_test_data=X_Test,
                            y_train_data=Y_Train,
                            y_test_data=Y_Test,
                            eta= get_eta(),
                            epochs= get_epochs(),
                            with_bias= checkbox_var.get())]
"""

gui.mainloop()
