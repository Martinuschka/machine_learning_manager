import tkinter as tk
import window
import learningmodel


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Machine Learning Manager")
    root.geometry("1000x700")
    # root.resizable(False,False)
    window = window.Window(root)
    machine = learningmodel.LearningModel(window)

    window.buttonFile.configure(command=machine.select_file)
    window.buttonLoad.configure(command=machine.load_csv)
    window.buttonHisto.configure(command=machine.show_histo)
    window.buttonScatter.configure(command=machine.show_scatter)
    window.checkLR.configure(command=lambda: machine.algorithm_selection(window.varList))
    window.checkLDA.configure(command=lambda: machine.algorithm_selection(window.varList))
    window.checkKNN.configure(command=lambda: machine.algorithm_selection(window.varList))
    window.checkGBC.configure(command=lambda: machine.algorithm_selection(window.varList))
    window.checkDT.configure(command=lambda: machine.algorithm_selection(window.varList))
    window.checkRF.configure(command=lambda: machine.algorithm_selection(window.varList))
    window.checkNB.configure(command=lambda: machine.algorithm_selection(window.varList))
    window.checkSVM.configure(command=lambda: machine.algorithm_selection(window.varList))
    window.radioThreads.configure(command=lambda: machine.computation_selection(window.varThreads_Processes))
    window.radioProcesses.configure(command=lambda: machine.computation_selection(window.varThreads_Processes))
    window.buttonTrain.configure(command=machine.train)
    window.buttonAbort.configure(command=machine.abort_computation)
    window.radioRegression.configure(command=lambda: machine.evaluation_selection(window.varEvaluation))
    window.radioClassification.configure(command=lambda: machine.evaluation_selection(window.varEvaluation))
    window.buttonResults.configure(command=machine.show_results)


    def on_closing():
        machine.abort_computation()  # only relevant using processes instead of threads
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
