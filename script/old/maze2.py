import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import networkx as nx


class CognitiveMaze:
    def __init__(self) -> None:
        self.mappings = {'VISUAL': 0, 'GOAL': 1, 'IMAGINAL': 2, 'RETRIEVAL': 3, 'MOTOR': 4}
        self.points_list = self.generate_points()
        self.edges_list = self.generate_edges()

    def generate_points(self):
        return list(self.mappings.values())

    def generate_edges(self):
        VISUAL, GOAL, IMAGINAL, RETRIEVAL, MOTOR = self.mappings.values()
        edges = [(VISUAL, GOAL), (VISUAL, IMAGINAL), (GOAL, IMAGINAL), (IMAGINAL, GOAL), (GOAL, RETRIEVAL),
                 (RETRIEVAL, GOAL), (IMAGINAL, RETRIEVAL), (RETRIEVAL, IMAGINAL), (RETRIEVAL, RETRIEVAL),
                 (IMAGINAL, MOTOR)]
        edges.sort()
        return edges

    def draw_maze(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.points_list)
        G.add_edges_from(self.edges_list)
        pos = nx.circular_layout(G)

        nx.draw_networkx_nodes(G, pos, node_color='#444', alpha=.2, node_size=800)
        nx.draw_networkx_edges(G, pos, width=1, arrows=True, arrowstyle="->", connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(G, pos, labels=self.points_list, horizontalalignment='center', verticalalignment='center')
        plt.show()
        return G

    def draw_full_maze(self):
        dG = nx.DiGraph()
        dG.add_nodes_from(self.points_list)
        dG.add_edges_from(self.edges_list)
        G = nx.complete_graph(5, dG)

        pos = nx.circular_layout(G)

        nx.draw_networkx_nodes(G, pos, node_color='#444', alpha=.2, node_size=800)
        nx.draw_networkx_edges(G, pos, width=1, arrows=True, arrowstyle="->", connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(G, pos, self.mappings, horizontalalignment='center', verticalalignment='center')
        plt.show()

class Maze:

    def __init__(self):
        self.fig = None
        self.ax = None
        self.line = None

    def draw_maze(self):

        self.fig = plt.figure(figsize=(5, 5))
        self.ax = plt.gca()

        # Draw walls
        kwargs = {'color': 'black', 'linewidth': 3}
        plt.plot([1, 1], [1, 2], **kwargs)
        plt.plot([1, 0], [2, 2], **kwargs)
        plt.plot([0, 2], [3, 3], **kwargs)
        plt.plot([2, 2], [1, 2], **kwargs)
        plt.plot([2, 4], [2, 2], **kwargs)
        plt.plot([3, 3], [0, 1], **kwargs)
        plt.plot([3, 3], [3, 4], **kwargs)

        # Draw states of cells
        cell_num = 0
        cell_pos = [0.5, 1.5, 2.5, 3.5]
        for y in reversed(cell_pos):
            for x in cell_pos:
                plt.text(x, y, 'S' + str(cell_num), size=14, ha='center')
                cell_num += 1
        plt.text(0.5, 3.3, 'START', ha='center')
        plt.text(3.5, 0.3, 'GOAL', ha='center')

        self.ax.set_xlim(0, 4)
        self.ax.set_ylim(0, 4)
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False, labelleft=False)

        self.line, = self.ax.plot([0.5], [3.5], marker='o', color='g', markersize=60)

    def save_animation(self, file_name, state_history):

        self.draw_maze()

        def init():
            self.line.set_data([], [])
            return self.line,

        def animate(i):
            state = state_history[i][0]
            x = (state % 4) + 0.5
            y = 3.5 - int(state / 4)
            self.line.set_data(x, y)
            return self.line,

        if file_name.split('.')[1] == 'html':
            anim = animation.FuncAnimation(self.fig, animate, init_func=init, frames=len(state_history), repeat=False)
            html = HTML(anim.to_jshtml()).data

            with open(file_name, 'w') as f:
                f.write(html)
                print("Animation is saved at %s." % file_name)
        elif file_name.split('.')[-1] == 'gif':
            anim = animation.FuncAnimation(self.fig, animate, init_func=init, frames=len(state_history), repeat=False)
            anim.save(file_name, writer='imagemagick', fps=240)
        else:
            raise ValueError("unknown extension: %s" % file_name)