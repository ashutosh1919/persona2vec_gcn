import logging
import networkx as nx
from tqdm import tqdm
from itertools import permutations
from typing import Callable, Dict, List, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

from persona2vec_gnn.utils import louvain_algorithm


logger = logging.getLogger(__file__)


class EgoNetSplitter(BaseModel):
    """
    An implementation of network ego-splitting procedure. For details, please
    check the "Persona2Vec" paper (https://arxiv.org/abs/2006.04941).
    """

    network: nx.Graph
    """Original Network."""

    directed: bool = Field(default=False)
    """Whether the graph is directed or not."""

    lambd: float = Field(default=0.1)
    """Weight of persona edges."""

    clustering_method: Union[Callable, str] = Field(
        default="connected_component"
    )
    """Clustering method to use for splitting personas."""

    components: Dict = Field(default_factory=dict, exclude=True)
    """Components of the ego networks. (Generated during initialization)"""

    node_to_persona: Dict = Field(default_factory=dict, exclude=True)
    """Mapping from nodes to personas. (Generated during initialization)"""

    persona_to_node: Dict = Field(default_factory=dict, exclude=True)
    """Mapping from personas to nodes. (Generated during initialization)"""

    original_edges: List = Field(default_factory=list, exclude=True)
    """Original edges of the network. (Generated during initialization)"""

    persona_edges: List = Field(default_factory=list, exclude=True)
    """Persona edges of the network. (Generated during initialization)"""

    persona_network: nx.Graph = Field(default=None, exclude=True)
    """Persona network. (Generated during initialization)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Pydantic model configuration to allow arbitrary types."""

    @model_validator(mode="after")
    def _validate_init_fields(self) -> "EgoNetSplitter":
        """
        Validate the initialization fields and set the clustering method
        """
        # Initialize clustering method
        if isinstance(self.clustering_method, str):
            self._init_clustering_method()
        # intialize unweighted edges with 1
        if not nx.is_weighted(self.network):
            for edge in self.network.edges():
                self.network[edge[0]][edge[1]]["weight"] = 1
        # Create ego networks
        self._create_egonets()
        # Map nodes to personas
        self._map_node_to_persona()
        # Create persona network
        self._create_persona_network()

    def __init__(
        self,
        network: nx.Graph,
        directed: bool = False,
        lambd: float = 0.1,
        clustering_method: bool = "connected_component",
        **kwargs
    ):
        super().__init__(
            network=network,
            directed=directed,
            lambd=lambd,
            clustering_method=clustering_method,
            **kwargs
        )

    def _init_clustering_method(self) -> None:
        """
        Initialize the clustering method based on the provided string
        or callable.
        """
        if self.clustering_method == "connected_component":
            if self.directed:
                self.clustering_method = nx.weakly_connected_components
            else:
                self.clustering_method = nx.connected_components
        elif self.clustering_method == "modularity":
            import networkx.algorithms.community.modularity_max as modularity
            self.clustering_method = modularity.greedy_modularity_communities
        elif self.clustering_method == "label_prop":
            import networkx.algorithms.community.label_propagation as label_prop  # noqa: E501
            self.clustering_method = label_prop.label_propagation_communities
        elif self.clustering_method == "modularity_louvain":
            self.clustering_method = louvain_algorithm
        else:
            raise ValueError(
                f"Clustering method '{self.clustering_method}' is not "
                "implemented."
            )

    def _create_egonets(self) -> None:
        """
        Create ego networks for each node in the original network.
        """
        self.components = {}
        self.node_to_persona = {}
        logger.info("Creating egonets.")
        for node in tqdm(self.network.nodes()):
            self._create_egonet(node)

    def _map_node_to_persona(self) -> None:
        """
        Mapping the personas to new nodes.
        """
        self.persona_to_node = {
            persona: node
            for node in self.network.nodes()
            for persona in self.node_to_persona[node]
        }

    def _create_persona_network(self) -> None:
        """
        Create a persona network using the egonet components.
        """
        logging.info("Creating the persona network.")

        # Add original edges
        self.original_edges = [
            (
                self.components[edge[0]][edge[1]],
                self.components[edge[1]][edge[0]],
                edge[2]["weight"],
            )
            for edge in tqdm(self.network.edges(data=True))
            if edge[0] != edge[1]
        ]
        if not self.directed:
            self.original_edges += [
                (
                    self.components[edge[1]][edge[0]],
                    self.components[edge[0]][edge[1]],
                    edge[2]["weight"],
                )
                for edge in tqdm(self.network.edges(data=True))
                if edge[0] != edge[1]
            ]

        G = nx.DiGraph()
        G.add_weighted_edges_from(self.original_edges)

        #  Add persona edges
        degree_dict = dict(G.out_degree())
        degree_dict = {k: 1 if v == 0 else v for k, v in degree_dict.items()}
        self.persona_edges = [
            (x, y, self.lambd * (degree_dict[x]))
            for _, personas in self.node_to_persona.items()
            if len(personas) > 1
            for x, y in permutations(personas, 2)
        ]

        G.add_weighted_edges_from(self.persona_edges)
        self.persona_network = G

    def _create_egonet(self, node: str) -> None:
        """
        Create an ego network for a given node.
        """
        neighbor_set = set(nx.all_neighbors(self.network, node)) - set([node])
        ego_net_minus_ego = self.network.subgraph(neighbor_set)

        try:
            components = {
                i: nodes for i, nodes in enumerate(
                    self.clustering_method(ego_net_minus_ego)
                )
            }
        except ZeroDivisionError as error:  # noqa: F841
            components = {
                i: [node] for i, node in enumerate(ego_net_minus_ego.nodes)
            }

        new_mapping = {}
        node_to_persona = []
        for i, (_, v) in enumerate(components.items()):
            name = "{}-{}".format(node, i + 1)
            node_to_persona.append(name)
            for other_node in v:
                new_mapping[other_node] = name
        self.components[node] = new_mapping
        self.node_to_persona[node] = node_to_persona
