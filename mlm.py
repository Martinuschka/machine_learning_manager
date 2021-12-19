print("Importiere Module...")
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import multiprocessing
import threading
from tkinter import filedialog as fd
from tkinter import ttk
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
#from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import Lasso
#from sklearn.linear_model import ElasticNet
#from sklearn.linear_model import Ridge
#from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from pandastable import Table

class machine:
	def __init__(self):
		self.path=""
		self.df=pd.DataFrame()
		self.X=pd.DataFrame()
		self.Y=pd.DataFrame()
		self.X_train=pd.DataFrame()
		self.X_test=pd.DataFrame()
		self.Y_train=pd.DataFrame()
		self.Y_test=pd.DataFrame()
		self.filetypes=(("CSV",".csv"),("All files","*.*"))
		self.models=[]
		self.names=[]
		self.results_class=[]
		self.results_estimate=[]
		self.evaluation_type=1
		self.processes_or_threads=2
		self.all_processes=[]
		self.queue=multiprocessing.Queue()
		self.lock=threading.Lock()
		self.computeCount=0

	def selectFile(self):
		buttonLoad.config(state="disabled")
		self.path=fd.askopenfilename(filetypes=self.filetypes)
		labelFile.config(text=self.path)
		if self.path:
			buttonLoad.config(state="normal")
		print("Selecting File...",self.path)

	def loadCSV(self):
		print("Loading CSV...")
		#self.df=pd.read_csv(self.path,parse_dates=["date"])
		self.df=pd.read_csv(self.path)
		#csv_table.model.df=self.df
		csv_table.importCSV(self.path)
		csv_table.show()
		print("Prepare Data...")
		self.transformData()
		self.algorithmSelection(varList)
		buttonHisto.config(state="normal")
		buttonScatter.config(state="normal")
		buttonTrain.config(state="normal")
		buttonAbort.config(state="normal")
		buttonResults.config(state="normal")
		comboColumn["values"]=self.df.columns.values.tolist()
		comboColumn.current(len(self.df.columns)-1)

	def transformData(self):
		print("Replacing NaN (if existing) with 0...")
		self.df=self.df.fillna(0)
		for column in self.df.columns:
			#print("First value of column ",column,": ",self.df.loc[0,column])
			#print("Type of column ",column,": ",type(self.df.loc[0,column]))
			print("Type of column ",column,": ",type(self.df.loc[0,column]))
			if type(self.df.loc[0,column])==str:
				if column=="date":
					try:
						print("Converting string to datetime...")
						self.df[column]=pd.to_datetime(self.df[column])
						#print("Now type: ",type(self.df.loc[0,column]))
						print("Replacing date with month from 0 to 12...")
						self.df[column]=self.df[column].dt.month
					except:
						print("No proper datetime format...")
						print("Replacing column ",column,"with numbers...")
						text_to_int={}
						count=0
						for i in self.df[column].unique():
							text_to_int.update({i:count})
							count+=1
						self.df[column]=self.df[column].map(text_to_int)
				else:
					print("Replacing column ",column,"with numbers...")
					text_to_int={}
					count=0
					for i in self.df[column].unique():
						text_to_int.update({i:count})
						count+=1
					self.df[column]=self.df[column].map(text_to_int)
		print("DF shape:",self.df.shape)
		print("DF head: ",self.df.head())
		#csv_table.model.df=self.df
		#csv_table.show()

	def defineVariables(self):
#		self.X=self.df.iloc[:,:-1].values
#		self.Y=self.df.iloc[:,-1].values
		self.X=self.df.loc[:,self.df.columns!=varColumn.get()].values
		self.Y=self.df.loc[:,varColumn.get()].values.astype(int)
		self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.2,random_state=42)
		print("X:",self.X.shape, " Y:",self.Y.shape)
		print("X_train:",self.X_train.shape," Y_train: ",self.Y_train.shape)
		print("X_test:",self.X_test.shape," Y_test: ",self.Y_test.shape)

	def showHisto(self):
		print("Showing histogram...")
		self.df.hist()
		plt.show()

	def showScatter(self):
		print("Showing scatter...")
		scatter_matrix(self.df)
		plt.show()

	def algorithmSelection(self,varList):
		self.models.clear()
		if varList[0].get()==1:
#			self.models.append(("LR",LogisticRegression(solver="liblinear",multi_class="ovr")))
			self.models.append(("LR",LogisticRegression()))
		if varList[1].get()==1:
			self.models.append(("LDA", LinearDiscriminantAnalysis()))
		if varList[2].get()==1:
			self.models.append(("KNN", KNeighborsClassifier()))
		if varList[3].get()==1:
			self.models.append(("GBC", GradientBoostingClassifier()))
		if varList[4].get()==1:
			self.models.append(("DT", DecisionTreeClassifier()))
		if varList[5].get()==1:
			self.models.append(("RF", RandomForestClassifier()))
		if varList[6].get()==1:
			self.models.append(("NB", GaussianNB()))
		if varList[7].get()==1:
#			self.models.append(("SVM", SVC(gamme="auto")))
			self.models.append(("SVM", SVC()))
		print("Algorithms selected...",varList[0].get(),varList[1].get(),varList[2].get(),varList[3].get(),varList[4].get(),varList[5].get(),varList[6].get(),varList[7].get())

	def train(self):
		self.abortComputation()
		print("Selecting data...")
		self.defineVariables()
		print("Cleaning up...")
		self.all_processes.clear()
		self.results_class.clear()
		self.results_estimate.clear()
		self.names.clear()
		self.computeCount=0
		while not self.queue.empty():
			self.queue.get()
		if self.processes_or_threads==1:
			buttonAbort.config(state="normal")
		else:
			buttonAbort.config(state="disabled")
		print("Training data...")
		buttonTrain.config(state="disabled")
		buttonTrain.config(text="Computing...")
		for name,model in self.models:
			checkList[name].config(fg="red")
			self.computeCount+=1
			if self.processes_or_threads==1:
				process=multiprocessing.Process(target=self.computation,args=(name,model,self.queue,self.lock))			
				process.start()
				self.all_processes.append(process)
			else:
				t=threading.Thread(target=self.computation,args=(name,model,self.queue,self.lock))
				t.daemon=True
				t.start()

	def computation(self,name,model,queue,lock):
		print(name," started...")
		model.fit(self.X_train,self.Y_train)
		Y_pred=model.predict(self.X_test)
		acc_score_class=accuracy_score(self.Y_test,Y_pred)
		if self.Y_test.mean()!=0:
			acc_score_estimate=1-abs(mean_absolute_error(self.Y_test,Y_pred)/self.Y_test.mean())
		else:
			acc_score_estimate=0
		if acc_score_estimate<0:
			acc_score_estimate=0
		#acc_score_estimate=1-mean_absolute_percentage_error(self.Y_test,Y_pred)
		#acc_score_estimate=mean_absolute_error(self.Y_test,Y_pred)
		print(name," result: ",acc_score_class,acc_score_estimate)
		queue.put((name,acc_score_class,acc_score_estimate))
		lock.acquire()
		checkList[name].config(fg="green")
		#print("Queue Size: ",queue.qsize())
		if queue.qsize()==self.computeCount:
			buttonTrain.config(state="normal")
			buttonTrain.config(text="Train")
			buttonResults.config(state="normal")
		#print("Pred: ",Y_pred)
		#print("Real: ",self.Y_test)
		lock.release()

	def computationSelection(self,varProcesses_Threads):
		self.processes_or_threads=varProcesses_Threads.get()
		print("Computation chosen: ",self.processes_or_threads)

	def evaluationSelection(self,varEvaluation):
		self.evaluation_type=varEvaluation.get()
		print("Evaluation chosen: ",self.evaluation_type)

	def showResults(self):
		print("Showing results...")
		for i in range(self.queue.qsize()):
			(name,result_class,result_estimate)=self.queue.get()
			self.names.append(name)
			self.results_class.append(result_class)
			self.results_estimate.append(result_estimate)
			self.computeCount-=1
		if self.evaluation_type==1:
			plt.plot(self.names,self.results_class, "ro")
			plt.title("Classification-Score of "+varColumn.get())
		else:
			plt.plot(self.names,self.results_estimate, "ro")
			plt.title("Regression-Score of "+varColumn.get())
		plt.ylim([0,1])
		plt.show()

	def abortComputation(self):
		print("Abort existing computation...")
		for process in self.all_processes:
			process.terminate()



if __name__=="__main__":
	machine=machine()

	window=tk.Tk()
	window.title("Machine Learning Manager")
	window.geometry("1040x700")
	#window.resizable(False,False)

	window.columnconfigure(0,weight=1)
	window.columnconfigure(1,weight=1)
	window.rowconfigure(0,weight=1)

	leftFrame=tk.Frame(window)
	leftFrame.grid(column=0,row=0,sticky="news")
	for i in range(18):
		leftFrame.rowconfigure(i,weight=1)
	leftFrame.columnconfigure(0,weight=1)
	leftFrame.columnconfigure(1,weight=1)

	rightFrame=tk.Frame(window)
	rightFrame.grid(column=1,row=0,sticky="news")

	csv_table=Table(rightFrame,showstatusbar=True)
	csv_table.show()

	labelCSV=tk.Label(leftFrame,text="File",font=("Arial",15,"underline"),anchor="w")
	labelCSV.grid(column=0,row=0,padx=2,pady=10,sticky="news",columnspan=2)
	labelFile=tk.Label(leftFrame,text="No file selected",font=("Arial",10),relief="sunken",width=70,anchor="w")
	labelFile.grid(column=0,row=1,padx=2,pady=2,sticky="news",columnspan=2)
	buttonFile=tk.Button(leftFrame, text="Select File",font=("Arial",15),command=machine.selectFile,width=20)
	buttonFile.grid(column=0,row=2,padx=2,pady=2,sticky="news")
	buttonLoad=tk.Button(leftFrame, text="Load CSV",font=("Arial",15),command=machine.loadCSV,width=20,state="disabled")
	buttonLoad.grid(column=1,row=2,padx=2,pady=2,sticky="news")

	labelVisual=tk.Label(leftFrame,text="Visualization",font=("Arial",15,"underline"), anchor="w")
	labelVisual.grid(column=0,row=3,padx=2,pady=10,sticky="news",columnspan=2)
	buttonHisto=tk.Button(leftFrame, text="Histogram",font=("Arial",15),command=machine.showHisto,width=20,state="disabled")
	buttonHisto.grid(column=0,row=4,padx=2,pady=2,sticky="news")
	buttonScatter=tk.Button(leftFrame, text="Scatter Matrix",font=("Arial",15),command=machine.showScatter,width=20,state="disabled")
	buttonScatter.grid(column=1,row=4,padx=2,pady=2,sticky="news")

	varLR=tk.IntVar(value=1)
	varLDA=tk.IntVar(value=1)
	varKNN=tk.IntVar(value=1)
	varGBC=tk.IntVar(value=1)
	varDT=tk.IntVar(value=1)
	varRF=tk.IntVar(value=1)
	varNB=tk.IntVar(value=1)
	varSVM=tk.IntVar(value=1)
	varList=[varLR,varLDA,varKNN,varGBC,varDT,varRF,varNB,varSVM]
	comboColor="green"
	labelAlgorithms=tk.Label(leftFrame,text="Algorithms (red=computing)",font=("Arial",15,"underline"), anchor="w")
	labelAlgorithms.grid(column=0,row=5,padx=2,pady=10, sticky="news",columnspan=2)
	checkLR=tk.Checkbutton(leftFrame, text="LR - Logistic Regression",font=("Arial",10),variable=varLR,onvalue=1,offvalue=0,command=lambda:machine.algorithmSelection(varList),fg=comboColor)
	checkLR.grid(column=0,row=6,padx=2,pady=2,sticky="w",columnspan=2)
	checkLDA=tk.Checkbutton(leftFrame, text="LDA - Linear Discriminant Analyses",font=("Arial",10),variable=varLDA,onvalue=1,offvalue=0,command=lambda:machine.algorithmSelection(varList),fg=comboColor)
	checkLDA.grid(column=0,row=7,padx=2,pady=2,sticky="w",columnspan=2)
	checkKNN=tk.Checkbutton(leftFrame, text="KNN - K-Nearest Neighbors Classifier",font=("Arial",10),variable=varKNN,onvalue=1,offvalue=0,command=lambda:machine.algorithmSelection(varList),fg=comboColor)
	checkKNN.grid(column=0,row=8,padx=2,pady=2,sticky="w",columnspan=2)
	checkGBC=tk.Checkbutton(leftFrame, text="GBC - Gradient Boosting Classifier",font=("Arial",10),variable=varGBC,onvalue=1,offvalue=0,command=lambda:machine.algorithmSelection(varList),fg=comboColor)
	checkGBC.grid(column=0,row=9,padx=2,pady=2,sticky="w",columnspan=2)
	checkDT=tk.Checkbutton(leftFrame, text="DT - Decision Tree Classifier",font=("Arial",10),variable=varDT,onvalue=1,offvalue=0,command=lambda:machine.algorithmSelection(varList),fg=comboColor)
	checkDT.grid(column=1,row=6,padx=2,pady=2,sticky="w",columnspan=2)
	checkRF=tk.Checkbutton(leftFrame, text="RF - Random Forest Classifier",font=("Arial",10),variable=varRF,onvalue=1,offvalue=0,command=lambda:machine.algorithmSelection(varList),fg=comboColor)
	checkRF.grid(column=1,row=7,padx=2,pady=2,sticky="w",columnspan=2)
	checkNB=tk.Checkbutton(leftFrame, text="NB - Gaussian Naive Bayes",font=("Arial",10),variable=varNB,onvalue=1,offvalue=0,command=lambda:machine.algorithmSelection(varList),fg=comboColor)
	checkNB.grid(column=1,row=8,padx=2,pady=2,sticky="w",columnspan=2)
	checkSVM=tk.Checkbutton(leftFrame, text="SVM - Support Vector Machines",font=("Arial",10),variable=varSVM,onvalue=1,offvalue=0,command=lambda:machine.algorithmSelection(varList),fg=comboColor)
	checkSVM.grid(column=1,row=9,padx=2,pady=2,sticky="w",columnspan=2)

	checkList={"LR":checkLR,"LDA":checkLDA,"KNN":checkLDA,"GBC":checkGBC,"DT":checkDT,"RF":checkRF,"NB":checkNB,"SVM":checkSVM}

	varColumn=tk.StringVar()
	labelColumn=tk.Label(leftFrame,text="Column to predict",font=("Arial",15,"underline"), anchor="w")
	labelColumn.grid(column=0,row=10,padx=2,pady=10, sticky="news",columnspan=2)
#	labelCombobox=tk.Label(window,text="Column for result: ",font=("Arial",10), anchor="e")
#	labelCombobox.grid(column=0,row=13,padx=2,pady=2,sticky="e")
	comboColumn=ttk.Combobox(leftFrame,textvariable=varColumn)
	comboColumn.grid(column=0,row=11,padx=2,pady=2,sticky="news",columnspan=2)
	comboColumn["state"]="readonly"

	varProcesses_Threads=tk.IntVar(value=2)
	labelAlgorithms=tk.Label(leftFrame,text="Computation",font=("Arial",15,"underline"), anchor="w")
	labelAlgorithms.grid(column=0,row=12,padx=2,pady=10,sticky="news",columnspan=2)
	radioProcesses=tk.Radiobutton(leftFrame,text="Processes (Linux)",variable=varProcesses_Threads,value=1,command=lambda:machine.computationSelection(varProcesses_Threads))
	#radioProcesses.grid(column=0,row=13,padx=2,pady=2, sticky="w")
	radioThreads=tk.Radiobutton(leftFrame,text="Threads (Windows)",variable=varProcesses_Threads,value=2,command=lambda:machine.computationSelection(varProcesses_Threads))
	#radioThreads.grid(column=0,row=14,padx=2,pady=2, sticky="w")
	buttonTrain=tk.Button(leftFrame,text="Train",font=("Arial",15),command=machine.train,width=20,state="disabled")
	buttonTrain.grid(column=0,row=13,padx=2,pady=2,sticky="news",columnspan=2)
	buttonAbort=tk.Button(leftFrame,text="Abort Computation",font=("Arial",15),command=machine.abortComputation,width=20,state="disabled")
	#buttonAbort.grid(column=1,row=14,padx=2,pady=2,sticky="news")

	varEvaluation=tk.IntVar(value=1)
	labelResults=tk.Label(leftFrame,text="Evaluation",font=("Arial",15,"underline"), anchor="w")
	labelResults.grid(column=0,row=15,padx=2,pady=10, sticky="news",columnspan=2)
	radioRegression=tk.Radiobutton(leftFrame,text="Classification",variable=varEvaluation,value=1,command=lambda:machine.evaluationSelection(varEvaluation))
	radioRegression.grid(column=0,row=16,padx=2,pady=2, sticky="w")
	radioClassification=tk.Radiobutton(leftFrame,text="Regression",variable=varEvaluation,value=2,command=lambda:machine.evaluationSelection(varEvaluation))
	radioClassification.grid(column=0,row=17,padx=2,pady=2, sticky="w")
	buttonResults=tk.Button(leftFrame, text="Show Results",font=("Arial",15),command=machine.showResults,width=20,state="disabled")
	buttonResults.grid(column=1,row=16,padx=2,pady=2,sticky="news",rowspan=2)

	def onClosing():
		machine.abortComputation()
		window.destroy()

	window.protocol("WM_DELETE_WINDOW",onClosing)

	window.mainloop()