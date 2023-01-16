import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk
import multiprocessing  # high CPU usage
import threading  # GUI interaction
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from pandastable import Table
from queue import Queue


# Order: select_file, load_csv, train, show results
class LearningModel:
    def __init__(self):
        self.path = ""
        self.df = pd.DataFrame()
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.Y_train = pd.DataFrame()
        self.Y_test = pd.DataFrame()
        self.filetypes = (("CSV", ".csv"), ("All files", "*.*"))
        self.models = []
        self.names = []
        self.results_class = []
        self.results_estimate = []
        self.evaluation_type = 1
        self.threads_or_processes = 1  # threads for GUI interaction
        self.all_processes = []
        # self.queue = multiprocessing.Queue()
        self.queue = Queue()
        self.lock_thread = threading.Lock()
        # self.lock_process = multiprocessing.Lock()  # not used since processes don't share memory
        self.computeCount = 0

    def select_file(self):
        buttonLoad.config(state="disabled")
        self.path = fd.askopenfilename(filetypes=self.filetypes)
        labelFile.config(text=self.path)
        if self.path:
            buttonLoad.config(state="normal")
        print("Selecting File...", self.path)

    def load_csv(self):
        print("Loading CSV...")
        # self.df=pd.read_csv(self.path,parse_dates=["date"])
        self.df = pd.read_csv(self.path)
        # csv_table.model.df=self.df
        csv_table.importCSV(self.path)
        csv_table.show()
        print("Prepare Data...")
        self.transform_data()
        self.algorithm_selection(varList)
        buttonHisto.config(state="normal")
        buttonScatter.config(state="normal")
        buttonTrain.config(state="normal")
        buttonTrain.config(text="Train")
        buttonAbort.config(state="normal")
        buttonResults.config(state="disabled")
        comboColumn["values"] = self.df.columns.values.tolist()
        comboColumn.current(len(self.df.columns) - 1)

    def transform_data(self):
        print("Replacing NaN (if existing) with 0...")
        self.df = self.df.fillna(0)
        for column in self.df.columns:
            # print("First value of column ",column,": ",self.df.loc[0,column])
            # print("Type of column ",column,": ",type(self.df.loc[0,column]))
            print("Type of column ", column, ": ", type(self.df.loc[0, column]))
            if type(self.df.loc[0, column]) == str:
                if column == "date":
                    try:
                        print("Converting string to datetime...")
                        self.df[column] = pd.to_datetime(self.df[column])
                        # print("Now type: ",type(self.df.loc[0,column]))
                        print("Replacing date with month from 0 to 12...")
                        self.df[column] = self.df[column].dt.month
                    except:
                        print("No proper datetime format...")
                        print("Replacing column ", column, "with numbers...")
                        text_to_int = {}
                        count = 0
                        for item in self.df[column].unique():
                            text_to_int.update({item: count})
                            count += 1
                        self.df[column] = self.df[column].map(text_to_int)
                else:
                    print("Replacing column ", column, "with numbers...")
                    text_to_int = {}
                    count = 0
                    for item in self.df[column].unique():
                        text_to_int.update({item: count})
                        count += 1
                    self.df[column] = self.df[column].map(text_to_int)
        print("DF shape:", self.df.shape)
        print("DF head: ", self.df.head())

    # csv_table.model.df=self.df
    # csv_table.show()

    def define_variables(self):
        # self.X=self.df.iloc[:,:-1].values
        # self.Y=self.df.iloc[:,-1].values
        self.X = self.df.loc[:, self.df.columns != varColumn.get()].values
        self.Y = self.df.loc[:, varColumn.get()].values.astype(int)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        print("X:", self.X.shape, " Y:", self.Y.shape)
        print("X_train:", self.X_train.shape, " Y_train: ", self.Y_train.shape)
        print("X_test:", self.X_test.shape, " Y_test: ", self.Y_test.shape)

    def show_histo(self):
        print("Showing histogram...")
        self.df.hist()
        plt.show()

    def show_scatter(self):
        print("Showing scatter...")
        scatter_matrix(self.df)
        plt.show()

    def algorithm_selection(self, var_list):
        self.models.clear()
        if var_list[0].get() == 1:
            # self.models.append(("LR",LogisticRegression(solver="liblinear",multi_class="ovr")))
            self.models.append(("LR", LogisticRegression()))
        if var_list[1].get() == 1:
            self.models.append(("LDA", LinearDiscriminantAnalysis()))
        if var_list[2].get() == 1:
            self.models.append(("KNN", KNeighborsClassifier()))
        if var_list[3].get() == 1:
            self.models.append(("GBC", GradientBoostingClassifier()))
        if var_list[4].get() == 1:
            self.models.append(("DT", DecisionTreeClassifier()))
        if var_list[5].get() == 1:
            self.models.append(("RF", RandomForestClassifier()))
        if var_list[6].get() == 1:
            self.models.append(("NB", GaussianNB()))
        if var_list[7].get() == 1:
            # self.models.append(("SVM", SVC(gamma="auto")))
            self.models.append(("SVM", SVC()))
        print("Algorithms selected...", var_list[0].get(), var_list[1].get(), var_list[2].get(), var_list[3].get(),
              var_list[4].get(), var_list[5].get(), var_list[6].get(), var_list[7].get())

    def train(self):
        self.abort_computation()
        print("Selecting data...")
        self.define_variables()
        print("Cleaning up...")
        self.all_processes.clear()
        self.results_class.clear()
        self.results_estimate.clear()
        self.names.clear()
        self.computeCount = 0
        while not self.queue.empty():
            self.queue.get()
        if self.threads_or_processes == 2:
            buttonAbort.config(state="normal")
        else:
            buttonAbort.config(state="disabled")
        print("Training data...")

        buttonTrain.config(state="disabled")
        buttonTrain.config(text="Computing...")
        buttonResults.config(state="disabled")

        for name, model in self.models:
            self.computeCount += 1
            if self.threads_or_processes == 1:
                checkList[name].config(fg="red")
                thread = threading.Thread(target=self.computation, args=(name, model, self.queue, self.lock_thread))
                thread.daemon = True
                thread.start()
            else:
                process = multiprocessing.Process(target=self.computation, args=(name, model, self.queue))
                # process.daemon = True  # not necessary since terminated on_closing
                process.start()
                self.all_processes.append(process)

    def computation(self, name, model, queue, lock=None):
        print(name, " started...")
        model.fit(self.X_train, self.Y_train)
        Y_pred = model.predict(self.X_test)
        acc_score_class = accuracy_score(self.Y_test, Y_pred)
        if self.Y_test.mean() != 0:
            acc_score_estimate = 1 - abs(mean_absolute_error(self.Y_test, Y_pred) / self.Y_test.mean())
        else:
            acc_score_estimate = 0
        if acc_score_estimate < 0:
            acc_score_estimate = 0
        # acc_score_estimate=1-mean_absolute_percentage_error(self.Y_test,Y_pred)
        # acc_score_estimate=mean_absolute_error(self.Y_test,Y_pred)
        print(name, " result: ", acc_score_class, acc_score_estimate)
        queue.put((name, acc_score_class, acc_score_estimate))
        if lock is not None:
            lock.acquire()
            checkList[name].config(fg="green")
            # print("Queue Size: ",queue.qsize())
            if queue.qsize() == self.computeCount:
                buttonTrain.config(state="normal")
                buttonTrain.config(text="Train")
                buttonResults.config(state="normal")
            # print("Pred: ",Y_pred)
            # print("Real: ",self.Y_test)
            lock.release()

    def computation_selection(self, var):
        self.threads_or_processes = var.get()
        print("Computation chosen: ", self.threads_or_processes)

    def evaluation_selection(self, var):
        self.evaluation_type = var.get()
        print("Evaluation chosen: ", self.evaluation_type)

    def show_results(self):
        print("Showing results...")
        for each in range(self.queue.qsize()):
            (name, result_class, result_estimate) = self.queue.get()
            self.names.append(name)
            self.results_class.append(result_class)
            self.results_estimate.append(result_estimate)
            self.computeCount -= 1
        if self.evaluation_type == 1:
            plt.plot(self.names, self.results_class, "ro")
            plt.title("Classification-Score of " + varColumn.get())
        else:
            plt.plot(self.names, self.results_estimate, "ro")
            plt.title("Regression-Score of " + varColumn.get())
        plt.ylim([0, 1])
        plt.show()

    # only works when using processes instead of threads
    def abort_computation(self):
        print("Abort existing computation...")
        for process in self.all_processes:
            process.terminate()


if __name__ == "__main__":
    machine = LearningModel()

    window = tk.Tk()
    window.title("Machine Learning Manager")
    window.geometry("1040x700")
    # window.resizable(False,False)

    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=1)
    window.rowconfigure(0, weight=1)

    leftFrame = tk.Frame(window)
    leftFrame.grid(column=0, row=0, sticky="news")
    for i in range(18):
        leftFrame.rowconfigure(i, weight=1)
    leftFrame.columnconfigure(0, weight=1)
    leftFrame.columnconfigure(1, weight=1)

    rightFrame = tk.Frame(window)
    rightFrame.grid(column=1, row=0, sticky="news")

    csv_table = Table(rightFrame, showstatusbar=True)
    csv_table.show()

    labelCSV = tk.Label(leftFrame, text="File", font=("Arial", 15, "underline"), anchor="w")
    labelCSV.grid(column=0, row=0, padx=2, pady=10, sticky="news", columnspan=2)
    labelFile = tk.Label(leftFrame, text="No file selected", font=("Arial", 10), relief="sunken", width=70, anchor="w")
    labelFile.grid(column=0, row=1, padx=2, pady=2, sticky="news", columnspan=2)
    buttonFile = tk.Button(leftFrame, text="Select File", font=("Arial", 15), command=machine.select_file, width=20)
    buttonFile.grid(column=0, row=2, padx=2, pady=2, sticky="news")
    buttonLoad = tk.Button(leftFrame, text="Load CSV", font=("Arial", 15), command=machine.load_csv, width=20, state="disabled")
    buttonLoad.grid(column=1, row=2, padx=2, pady=2, sticky="news")

    labelVisual = tk.Label(leftFrame, text="Visualization", font=("Arial", 15, "underline"), anchor="w")
    labelVisual.grid(column=0, row=3, padx=2, pady=10, sticky="news", columnspan=2)
    buttonHisto = tk.Button(leftFrame, text="Histogram", font=("Arial", 15), command=machine.show_histo, width=20, state="disabled")
    buttonHisto.grid(column=0, row=4, padx=2, pady=2, sticky="news")
    buttonScatter = tk.Button(leftFrame, text="Scatter Matrix", font=("Arial", 15), command=machine.show_scatter, width=20, state="disabled")
    buttonScatter.grid(column=1, row=4, padx=2, pady=2, sticky="news")

    varLR = tk.IntVar(value=1)
    varLDA = tk.IntVar(value=1)
    varKNN = tk.IntVar(value=1)
    varGBC = tk.IntVar(value=1)
    varDT = tk.IntVar(value=1)
    varRF = tk.IntVar(value=1)
    varNB = tk.IntVar(value=1)
    varSVM = tk.IntVar(value=1)
    varList = [varLR, varLDA, varKNN, varGBC, varDT, varRF, varNB, varSVM]
    comboColor = "green"
    labelAlgorithms = tk.Label(leftFrame, text="Algorithms (red=computing)", font=("Arial", 15, "underline"), anchor="w")
    labelAlgorithms.grid(column=0, row=5, padx=2, pady=10, sticky="news", columnspan=2)
    checkLR = tk.Checkbutton(leftFrame, text="LR - Logistic Regression", font=("Arial", 10), variable=varLR, onvalue=1, offvalue=0, command=lambda: machine.algorithm_selection(varList), fg=comboColor)
    checkLR.grid(column=0, row=6, padx=2, pady=2, sticky="w", columnspan=2)
    checkLDA = tk.Checkbutton(leftFrame, text="LDA - Linear Discriminant Analyses", font=("Arial", 10), variable=varLDA, onvalue=1, offvalue=0, command=lambda: machine.algorithm_selection(varList), fg=comboColor)
    checkLDA.grid(column=0, row=7, padx=2, pady=2, sticky="w", columnspan=2)
    checkKNN = tk.Checkbutton(leftFrame, text="KNN - K-Nearest Neighbors Classifier", font=("Arial", 10), variable=varKNN, onvalue=1, offvalue=0, command=lambda: machine.algorithm_selection(varList), fg=comboColor)
    checkKNN.grid(column=0, row=8, padx=2, pady=2, sticky="w", columnspan=2)
    checkGBC = tk.Checkbutton(leftFrame, text="GBC - Gradient Boosting Classifier", font=("Arial", 10), variable=varGBC, onvalue=1, offvalue=0, command=lambda: machine.algorithm_selection(varList), fg=comboColor)
    checkGBC.grid(column=0, row=9, padx=2, pady=2, sticky="w", columnspan=2)
    checkDT = tk.Checkbutton(leftFrame, text="DT - Decision Tree Classifier", font=("Arial", 10), variable=varDT, onvalue=1, offvalue=0, command=lambda: machine.algorithm_selection(varList), fg=comboColor)
    checkDT.grid(column=1, row=6, padx=2, pady=2, sticky="w", columnspan=2)
    checkRF = tk.Checkbutton(leftFrame, text="RF - Random Forest Classifier", font=("Arial", 10), variable=varRF, onvalue=1, offvalue=0, command=lambda: machine.algorithm_selection(varList), fg=comboColor)
    checkRF.grid(column=1, row=7, padx=2, pady=2, sticky="w", columnspan=2)
    checkNB = tk.Checkbutton(leftFrame, text="NB - Gaussian Naive Bayes", font=("Arial", 10), variable=varNB, onvalue=1, offvalue=0, command=lambda: machine.algorithm_selection(varList), fg=comboColor)
    checkNB.grid(column=1, row=8, padx=2, pady=2, sticky="w", columnspan=2)
    checkSVM = tk.Checkbutton(leftFrame, text="SVM - Support Vector Machines", font=("Arial", 10), variable=varSVM, onvalue=1, offvalue=0, command=lambda: machine.algorithm_selection(varList), fg=comboColor)
    checkSVM.grid(column=1, row=9, padx=2, pady=2, sticky="w", columnspan=2)

    checkList = {"LR": checkLR, "LDA": checkLDA, "KNN": checkLDA, "GBC": checkGBC, "DT": checkDT, "RF": checkRF, "NB": checkNB, "SVM": checkSVM}

    varColumn = tk.StringVar()
    labelColumn = tk.Label(leftFrame, text="Column to predict", font=("Arial", 15, "underline"), anchor="w")
    labelColumn.grid(column=0, row=10, padx=2, pady=10, sticky="news", columnspan=2)
    # labelCombobox=tk.Label(window,text="Column for result: ",font=("Arial",10), anchor="e")
    # labelCombobox.grid(column=0,row=13,padx=2,pady=2,sticky="e")
    comboColumn = ttk.Combobox(leftFrame, textvariable=varColumn)
    comboColumn.grid(column=0, row=11, padx=2, pady=2, sticky="news", columnspan=2)
    comboColumn["state"] = "readonly"

    varThreads_Processes = tk.IntVar(value=1)  # Threads for GUI interaction
    labelAlgorithms = tk.Label(leftFrame, text="Computation", font=("Arial", 15, "underline"), anchor="w")
    labelAlgorithms.grid(column=0, row=12, padx=2, pady=10, sticky="news", columnspan=2)
    radioThreads = tk.Radiobutton(leftFrame, text="Threads (GUI)", variable=varThreads_Processes, value=1, command=lambda: machine.computation_selection(varThreads_Processes))
    # radioThreads.grid(column=0, row=13, padx=2, pady=2, sticky="w")
    radioProcesses = tk.Radiobutton(leftFrame, text="Processes (CPU)", variable=varThreads_Processes, value=2, command=lambda: machine.computation_selection(varThreads_Processes))
    # radioProcesses.grid(column=1, row=13, padx=2, pady=2, sticky="w")
    buttonTrain = tk.Button(leftFrame, text="Train", font=("Arial", 15), command=machine.train, width=20, state="disabled")
    buttonTrain.grid(column=0, row=14, padx=2, pady=2, sticky="news", columnspan=2)
    buttonAbort = tk.Button(leftFrame, text="Abort Computation", font=("Arial", 15), command=machine.abort_computation, width=20, state="disabled")
    # buttonAbort.grid(column=1,row=14,padx=2,pady=2,sticky="news")

    varEvaluation = tk.IntVar(value=1)
    labelResults = tk.Label(leftFrame, text="Evaluation", font=("Arial", 15, "underline"), anchor="w")
    labelResults.grid(column=0, row=15, padx=2, pady=10, sticky="news", columnspan=2)
    radioRegression = tk.Radiobutton(leftFrame, text="Classification", variable=varEvaluation, value=1, command=lambda: machine.evaluation_selection(varEvaluation))
    radioRegression.grid(column=0, row=16, padx=2, pady=2, sticky="w")
    radioClassification = tk.Radiobutton(leftFrame, text="Regression", variable=varEvaluation, value=2, command=lambda: machine.evaluation_selection(varEvaluation))
    radioClassification.grid(column=0, row=17, padx=2, pady=2, sticky="w")
    buttonResults = tk.Button(leftFrame, text="Show Results", font=("Arial", 15), command=machine.show_results, width=20, state="disabled")
    buttonResults.grid(column=1, row=16, padx=2, pady=2, sticky="news", rowspan=2)


    def on_closing():
        machine.abort_computation()  # only relevant using processes instead of threads
        window.destroy()


    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()
