from indexer.index import Indexer
from indexer.gui import Table

if __name__ == "__main__":
    indexer = Indexer()
    indexer.read_file("speech/", "files/")

    tb = Table()
    tb.create_Gui(indexer)
