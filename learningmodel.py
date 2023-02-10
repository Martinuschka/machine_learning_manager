import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
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
from queue import Queue


# Order: select_file, load_csv, train, show results
class LearningModel:
    def __init__(self, window):
        self.window = window
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
        self.window.buttonLoad.config(state="disabled")
        self.path = fd.askopenfilename(filetypes=self.filetypes)
        self.window.labelFile.config(text=self.path)
        if self.path:
            self.window.buttonLoad.config(state="normal")
        print("Selecting File...", self.path)

    def load_csv(self):
        print("Loading CSV...")
        # self.df=pd.read_csv(self.path,parse_dates=["date"])
        self.df = pd.read_csv(self.path)
        # csv_table.model.df=self.df
        self.window.csv_table.importCSV(self.path)
        self.window.csv_table.show()
        print("Prepare Data...")
        self.transform_data()
        self.algorithm_selection(self.window.varList)
        self.window.buttonHisto.config(state="normal")
        self.window.buttonScatter.config(state="normal")
        self.window.buttonTrain.config(state="normal")
        self.window.buttonTrain.config(text="Train")
        self.window.buttonAbort.config(state="normal")
        self.window.buttonResults.config(state="disabled")
        self.window.comboColumn["values"] = self.df.columns.values.tolist()
        self.window.comboColumn.current(len(self.df.columns) - 1)

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
        self.X = self.df.loc[:, self.df.columns != self.window.varColumn.get()].values
        self.Y = self.df.loc[:, self.window.varColumn.get()].values.astype(int)
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
            self.window.buttonAbort.config(state="normal")
        else:
            self.window.buttonAbort.config(state="disabled")
        print("Training data...")

        self.window.buttonTrain.config(state="disabled")
        self.window.buttonTrain.config(text="Computing...")
        self.window.buttonResults.config(state="disabled")

        for name, model in self.models:
            self.computeCount += 1
            if self.threads_or_processes == 1:
                self.window.checkList[name].config(fg="red")
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

        print("First 10 predictions (prediction<->test data):")
        for i in range(10 if len(Y_pred) >= 10 else len(Y_pred)):
            print(Y_pred[i], "<->", self.Y_test[i])

        queue.put((name, acc_score_class, acc_score_estimate))
        if lock is not None:
            lock.acquire()
            self.window.checkList[name].config(fg="green")
            # print("Queue Size: ",queue.qsize())
            if queue.qsize() == self.computeCount:
                self.window.buttonTrain.config(state="normal")
                self.window.buttonTrain.config(text="Train")
                self.window.buttonResults.config(state="normal")
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
            plt.title("Classification-Score of " + self.window.varColumn.get())
        else:
            plt.plot(self.names, self.results_estimate, "ro")
            plt.title("Regression-Score of " + self.window.varColumn.get())
        plt.ylim([0, 1])
        plt.show()

    # only works when using processes instead of threads
    def abort_computation(self):
        print("Abort existing computation...")
        for process in self.all_processes:
            process.terminate()
