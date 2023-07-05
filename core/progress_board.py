'''
author: yingmuzhi
time: 20230704

intro: Core Components. Animation about progress bar and so on.
'''
import collections, IPython.display as display
import core


class ProgressBoard(core.hyper_parameters.HyperParameters):
    """
    intro:
        A board that plots the data points in animation.
    """
    def __init__(self,
                 xlabel=None, 
                 ylabel=None, 
                 xlim=None,
                 ylim=None,
                 xscale="linear",
                 yscale="linear",
                 ls=['-', "--", "-.", ":"],
                 colors=["C0", "C1", "C2", "C3"],
                 fig=None, 
                 axes=None, 
                 figsize=(3.5, 2.5),
                 display=True,
                 ) -> None:
        self.save_hyperparameters()
    
    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
    
    def draw(self, x, y, label, every_n=1):
        """
        intro:
            draw.
        """
        Point =  collections.namedtuple("Point", ['x', 'y'])
        if not hasattr(self, "raw_points"):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return 
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        core.use_svg_display()
        if self.fig is None:
            self.fig = core.plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(core.plt.plot([p.x for p in v], [p.y for p in v],
                                      linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else core.plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)