import tkinter as tk
from tkinter import ttk


class Table:
    def __init__(self):
        self.scores = tk.Tk()
        self.scores.resizable(False, False)
        self.cols = ("No.", "Document Name", "Document Title", "Score")
        self.listBox = ttk.Treeview(
            self.scores, columns=self.cols, show="headings", height=22
        )

    def display_Table(self, answer_list):
        if not answer_list:
            self.listBox.insert("", "end", values=("-", "-", "-", "-"))
        for i, ans in enumerate(answer_list):
            file_name = "speech_" + str(ans[0]) + ".txt"
            f = open("speech/" + file_name, "r")
            title = f.readline()
            self.listBox.insert(
                "", "end", values=(i + 1, file_name, title, "%0.4f" % (ans[1]))
            )

    def create_Gui(self, indexer):
        self.scores.geometry("1020x600")
        self.scores.title("Vector Space Model")

        self.label = tk.Label(
            self.scores, text="Query Search", font=("Arial", 30)
        ).grid(row=0, columnspan=2)

        answer = tk.StringVar()
        searchQuery = tk.Entry(self.scores, width=94, textvariable=answer).place(
            x=10, y=52
        )

        self.label = tk.Label(self.scores, text="").grid(row=1, column=2, pady=5)
        self.label = tk.Label(self.scores, text="\u03B1", font=(20)).place(
            x=694 + 205, y=48 - 35
        )

        answer2 = tk.StringVar(value="0.0005")
        searchQuery2 = tk.Entry(
            self.scores, width=10, textvariable=answer2, justify="right"
        ).place(x=710 + 205, y=50 - 35)

        vsb = ttk.Scrollbar(self.scores, orient="vertical", command=self.listBox.yview)
        vsb.place(x=1004, y=79, height=460)
        vsb.configure(command=self.listBox.yview)
        self.listBox.configure(yscrollcommand=vsb.set)

        for col in self.cols:
            self.listBox.heading(col, text=col)
        self.listBox.grid(row=2, column=0, columnspan=2)
        self.listBox.column(self.cols[0], minwidth=40, width=40, stretch=tk.NO)
        self.listBox.column(self.cols[1], minwidth=120, width=120, stretch=tk.NO)
        self.listBox.column(self.cols[2], minwidth=120, width=785)
        self.listBox.column(self.cols[3], minwidth=55, width=55)

        showScores = tk.Button(
            self.scores,
            text="Search",
            width=22,
            command=lambda: [
                self.listBox.delete(*self.listBox.get_children()),
                self.display_Table(
                    indexer.calculate(answer.get(), float(answer2.get()))
                ),
            ],
        ).place(x=798, y=45)
        closeButton = tk.Button(self.scores, text="Close", width=15, command=exit).grid(
            row=4, column=0, columnspan=2
        )

        self.scores.mainloop()
