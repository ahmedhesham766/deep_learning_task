

#Features
from tkinter import *
from tkinter import ttk

gui = Tk()
gui.title('Task_1 GUI')
gui.geometry('500x700+550+70')

Title = Label(gui, text="User Input", fg='black', font=("Times New Roman", 20))
Title.place(x=180, y=30)

lbl1 = Label(gui, text="Select first feature", fg='black', font=("Times New Roman", 14))
lbl1.place(x=50, y=130)
cmb1 = ttk.Combobox(gui, width=27)
cmb1['values'] = ('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g')
cmb1.place(x=290, y=130)

lbl2 = Label(gui, text="Select second feature", fg='black', font=("Times New Roman", 14))
lbl2.place(x=50, y=200)
cmb2 = ttk.Combobox(gui, width=27)
cmb2['values'] = ('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g')
cmb2.place(x=290, y=200)

lbl3 = Label(gui, text="Select 2 classes combination", fg='black', font=("Times New Roman", 14))
lbl3.place(x=50, y=270)
cmb3 = ttk.Combobox(gui, width=27)
cmb3['values'] = ('Adelie && Gentoo', 'Adelie && Chinstrap', 'Gentoo && Chinstrap')
cmb3.place(x=290, y=270)

lbl4 = Label(gui, text="Enter learning rate", fg='black', font=("Times New Roman", 14))
lbl4.place(x=50, y=340)
txt1 = Entry(gui, bg='white', fg='black', bd=9, width=27)
txt1.place(x=290, y=340)

lbl5 = Label(gui, text="Enter number of epochs", fg='black', font=("Times New Roman", 14))
lbl5.place(x=50, y=410)
txt2 = Entry(gui, bg='white', fg='black', bd=9, width=27)
txt2.place(x=290, y=410)

lbl6 = Label(gui, text="Add bias or not", fg='black', font=("Times New Roman", 14))
lbl6.place(x=50, y=480)
cb = Checkbutton(gui, text="Bias")
cb.place(x=360, y=485)

btn = Button(gui, text="Run", fg='black', width=15, font=("Times New Roman", 14))
btn.place(x=180, y=580)

gui.mainloop()
