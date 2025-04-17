import networkx as nx
from networkx.utils import graphs_equal

from persona2vec_gnn.algorithms.ego_splitting import EgoNetSplitter


class TestEgoNetSplitter:
    """
    Test class for EgoNetSplitter
    """

    def test_init(self):
        # Create a sample graph
        G = nx.Graph()
        G.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("C", "D"),
            ("C", "E"),
            ("C", "F"),
            ("D", "F"),
            ("E", "F"),
            ("F", "G"),
            ("F", "H"),
            ("G", "H")
        ])

        # Initialize EgoNetSplitter
        splitter = EgoNetSplitter(G)

        # Check if the splitter is initialized correctly
        assert graphs_equal(splitter.network, G)
        assert splitter.directed == False
        assert splitter.lambd == 0.1
        assert splitter.clustering_method == nx.connected_components
        assert len(splitter.components) == 8
        assert splitter.persona_network is not None
        print(splitter.persona_network.edges())
