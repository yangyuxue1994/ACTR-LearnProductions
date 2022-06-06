import matplotlib.pyplot as plt
import networkx as nx

##########################
#  Maze 
#########################

class CogintiveMaze:
    def __init__(self) -> None:

        self.nodes = self.get_actr_modules()
        self.edges = self.get_actr_connections()
        self.mappings = self.get_actr_networks()
        
    def get_actr_modules(self):
        return ['VISUAL', 'GOAL', 'IMAGINAL', 'RETRIEVAL', 'MOTOR']
    
    def get_actr_networks(self):
        modules = self.get_actr_modules()
        mappings = dict(zip(modules, range(len(modules))))
        return mappings
    
    def get_actr_connections(self):
        VISUAL, GOAL, IMAGINAL, RETRIEVAL, MOTOR = self.get_actr_modules()
        edges = [(VISUAL,GOAL), (VISUAL,IMAGINAL), (GOAL,IMAGINAL), (IMAGINAL,GOAL), (GOAL, RETRIEVAL), (RETRIEVAL, GOAL), (IMAGINAL, RETRIEVAL), (RETRIEVAL, IMAGINAL), (RETRIEVAL, RETRIEVAL), (IMAGINAL, MOTOR)]
        edges.sort()
        return edges
    
    def draw_maze(self):
        G= nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        pos = nx.circular_layout(G) 

        nx.draw_networkx_nodes(G,pos, node_color='#444', alpha=.2, node_size=800)
        nx.draw_networkx_edges(G,pos, width=1, arrows=True, arrowstyle="->", connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(G,pos, dict(zip(self.nodes, self.nodes)), 
                                horizontalalignment='center', verticalalignment='center')
        plt.show()

    def draw_full_maze(self):
        dG=nx.DiGraph()
        dG.add_nodes_from(self.nodes)
        dG.add_edges_from(self.edges)
        G = nx.complete_graph(len(self.nodes), dG)

        pos = nx.circular_layout(G) 

        nx.draw_networkx_nodes(G,pos, node_color='#444', alpha=.2, node_size=800)
        nx.draw_networkx_edges(G,pos, width=1, arrows=True, arrowstyle="->", connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(G,pos, self.nodes, horizontalalignment='center', verticalalignment='center')
        plt.show()
        
        
    def draw_maze_q(self, edge_dict):
        G= nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        pos = nx.circular_layout(G)
        
        edge_width = []
        for e in G.edges():
            w = edge_dict[(e[0], e[1])]
            edge_width.append(w)
        scale_factor = 5
        norm_edges = [scale_factor * (float(i) - min(edge_width)) / (max(edge_width) - min(edge_width)) for i in edge_width]

        nx.draw_networkx_nodes(G,pos, node_color='#444', alpha=.2, node_size=800)
        nx.draw_networkx_edges(G,pos, width=norm_edges, 
                               edge_color=norm_edges, edge_cmap  = plt.cm.Blues, 
                               arrows=True, arrowstyle="->", connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(G,pos, dict(zip(self.nodes, self.nodes)), 
                                horizontalalignment='center', verticalalignment='center')
        nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_dict, 
                             label_pos=0.3, 
                             verticalalignment='center', 
                             horizontalalignment='center')
        plt.show()