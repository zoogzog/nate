import plotly.plotly as py
import plotly.graph_objs as go

class ChartLossDrop():

    def generate(pathLOG):
        with open("file.txt", "r") as ins:
            array = []
            for line in ins:
                array.append(line)
