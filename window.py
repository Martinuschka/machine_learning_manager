import tkinter as tk
from tkinter import ttk
from pandastable import Table


class Window(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.leftFrame = tk.Frame(self)
        self.leftFrame.grid(column=0, row=0, sticky="news")
        for i in range(18):
            self.leftFrame.rowconfigure(i, weight=1)
        self.leftFrame.columnconfigure(0, weight=1)
        self.leftFrame.columnconfigure(1, weight=1)

        self.rightFrame = tk.Frame(self)
        self.rightFrame.grid(column=1, row=0, sticky="news")

        self.csv_table = Table(self.rightFrame, showstatusbar=True)
        self.csv_table.show()

        self.labelCSV = tk.Label(self.leftFrame, text="File", font=("Arial", 15, "underline"), anchor="w")
        self.labelCSV.grid(column=0, row=0, padx=2, pady=10, sticky="news", columnspan=2)
        self.labelFile = tk.Label(self.leftFrame, text="No file selected", font=("Arial", 10), relief="sunken", width=70, anchor="w")
        self.labelFile.grid(column=0, row=1, padx=2, pady=2, sticky="news", columnspan=2)
        self.buttonFile = tk.Button(self.leftFrame, text="Select File", font=("Arial", 15), width=20)
        self.buttonFile.grid(column=0, row=2, padx=2, pady=2, sticky="news")
        self.buttonLoad = tk.Button(self.leftFrame, text="Load CSV", font=("Arial", 15), width=20, state="disabled")
        self.buttonLoad.grid(column=1, row=2, padx=2, pady=2, sticky="news")

        self.labelVisual = tk.Label(self.leftFrame, text="Visualization", font=("Arial", 15, "underline"), anchor="w")
        self.labelVisual.grid(column=0, row=3, padx=2, pady=10, sticky="news", columnspan=2)
        self.buttonHisto = tk.Button(self.leftFrame, text="Histogram", font=("Arial", 15), width=20, state="disabled")
        self.buttonHisto.grid(column=0, row=4, padx=2, pady=2, sticky="news")
        self.buttonScatter = tk.Button(self.leftFrame, text="Scatter Matrix", font=("Arial", 15), width=20, state="disabled")
        self.buttonScatter.grid(column=1, row=4, padx=2, pady=2, sticky="news")

        self.varLR = tk.IntVar(value=1)
        self.varLDA = tk.IntVar(value=1)
        self.varKNN = tk.IntVar(value=1)
        self.varGBC = tk.IntVar(value=1)
        self.varDT = tk.IntVar(value=1)
        self.varRF = tk.IntVar(value=1)
        self.varNB = tk.IntVar(value=1)
        self.varSVM = tk.IntVar(value=1)
        self.varList = [self.varLR, self.varLDA, self.varKNN, self.varGBC, self.varDT, self.varRF, self.varNB, self.varSVM]
        self.comboColor = "green"
        self.labelAlgorithms = tk.Label(self.leftFrame, text="Algorithms (red=computing)", font=("Arial", 15, "underline"), anchor="w")
        self.labelAlgorithms.grid(column=0, row=5, padx=2, pady=10, sticky="news", columnspan=2)
        self.checkLR = tk.Checkbutton(self.leftFrame, text="LR - Logistic Regression", font=("Arial", 10), variable=self.varLR, onvalue=1, offvalue=0, fg=self.comboColor)
        self.checkLR.grid(column=0, row=6, padx=2, pady=2, sticky="w", columnspan=2)
        self.checkLDA = tk.Checkbutton(self.leftFrame, text="LDA - Linear Discriminant Analyses", font=("Arial", 10), variable=self.varLDA, onvalue=1, offvalue=0, fg=self.comboColor)
        self.checkLDA.grid(column=0, row=7, padx=2, pady=2, sticky="w", columnspan=2)
        self.checkKNN = tk.Checkbutton(self.leftFrame, text="KNN - K-Nearest Neighbors Classifier", font=("Arial", 10), variable=self.varKNN, onvalue=1, offvalue=0, fg=self.comboColor)
        self.checkKNN.grid(column=0, row=8, padx=2, pady=2, sticky="w", columnspan=2)
        self.checkGBC = tk.Checkbutton(self.leftFrame, text="GBC - Gradient Boosting Classifier", font=("Arial", 10), variable=self.varGBC, onvalue=1, offvalue=0, fg=self.comboColor)
        self.checkGBC.grid(column=0, row=9, padx=2, pady=2, sticky="w", columnspan=2)
        self.checkDT = tk.Checkbutton(self.leftFrame, text="DT - Decision Tree Classifier", font=("Arial", 10), variable=self.varDT, onvalue=1, offvalue=0, fg=self.comboColor)
        self.checkDT.grid(column=1, row=6, padx=2, pady=2, sticky="w", columnspan=2)
        self.checkRF = tk.Checkbutton(self.leftFrame, text="RF - Random Forest Classifier", font=("Arial", 10), variable=self.varRF, onvalue=1, offvalue=0, fg=self.comboColor)
        self.checkRF.grid(column=1, row=7, padx=2, pady=2, sticky="w", columnspan=2)
        self.checkNB = tk.Checkbutton(self.leftFrame, text="NB - Gaussian Naive Bayes", font=("Arial", 10), variable=self.varNB, onvalue=1, offvalue=0, fg=self.comboColor)
        self.checkNB.grid(column=1, row=8, padx=2, pady=2, sticky="w", columnspan=2)
        self.checkSVM = tk.Checkbutton(self.leftFrame, text="SVM - Support Vector Machines", font=("Arial", 10), variable=self.varSVM, onvalue=1, offvalue=0, fg=self.comboColor)
        self.checkSVM.grid(column=1, row=9, padx=2, pady=2, sticky="w", columnspan=2)

        self.checkList = {"LR": self.checkLR, "LDA": self.checkLDA, "KNN": self.checkLDA, "GBC": self.checkGBC, "DT": self.checkDT, "RF": self.checkRF, "NB": self.checkNB, "SVM": self.checkSVM}

        self.varColumn = tk.StringVar()
        self.labelColumn = tk.Label(self.leftFrame, text="Column to predict", font=("Arial", 15, "underline"), anchor="w")
        self.labelColumn.grid(column=0, row=10, padx=2, pady=10, sticky="news", columnspan=2)
        # self.labelCombobox=tk.Label(self.leftFrame,text="Column for result: ",font=("Arial",10), anchor="e")
        # self.labelCombobox.grid(column=0,row=13,padx=2,pady=2,sticky="e")
        self.comboColumn = ttk.Combobox(self.leftFrame, textvariable=self.varColumn)
        self.comboColumn.grid(column=0, row=11, padx=2, pady=2, sticky="news", columnspan=2)
        self.comboColumn["state"] = "readonly"

        self.varThreads_Processes = tk.IntVar(value=1)  # Threads for GUI interaction
        self.labelAlgorithms = tk.Label(self.leftFrame, text="Computation", font=("Arial", 15, "underline"), anchor="w")
        self.labelAlgorithms.grid(column=0, row=12, padx=2, pady=10, sticky="news", columnspan=2)
        self.radioThreads = tk.Radiobutton(self.leftFrame, text="Threads (GUI)", variable=self.varThreads_Processes, value=1)
        # self.radioThreads.grid(column=0, row=13, padx=2, pady=2, sticky="w")
        self.radioProcesses = tk.Radiobutton(self.leftFrame, text="Processes (CPU)", variable=self.varThreads_Processes, value=2)
        # self.radioProcesses.grid(column=1, row=13, padx=2, pady=2, sticky="w")
        self.buttonTrain = tk.Button(self.leftFrame, text="Train", font=("Arial", 15), width=20, state="disabled")
        self.buttonTrain.grid(column=0, row=14, padx=2, pady=2, sticky="news", columnspan=2)
        self.buttonAbort = tk.Button(self.leftFrame, text="Abort Computation", font=("Arial", 15), width=20, state="disabled")
        # self.buttonAbort.grid(column=1,row=14,padx=2,pady=2,sticky="news")

        self.varEvaluation = tk.IntVar(value=1)
        self.labelResults = tk.Label(self.leftFrame, text="Evaluation", font=("Arial", 15, "underline"), anchor="w")
        self.labelResults.grid(column=0, row=15, padx=2, pady=10, sticky="news", columnspan=2)
        self.radioRegression = tk.Radiobutton(self.leftFrame, text="Classification", variable=self.varEvaluation, value=1)
        self.radioRegression.grid(column=0, row=16, padx=2, pady=2, sticky="w")
        self.radioClassification = tk.Radiobutton(self.leftFrame, text="Regression", variable=self.varEvaluation, value=2)
        self.radioClassification.grid(column=0, row=17, padx=2, pady=2, sticky="w")
        self.buttonResults = tk.Button(self.leftFrame, text="Show Results", font=("Arial", 15), width=20, state="disabled")
        self.buttonResults.grid(column=1, row=16, padx=2, pady=2, sticky="news", rowspan=2)

        self.pack(expand=True, fill=tk.BOTH)
