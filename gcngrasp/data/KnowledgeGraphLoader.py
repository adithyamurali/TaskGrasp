import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle
from ordered_set import OrderedSet
from networkx.generators.ego import ego_graph
from networkx.algorithms.operators.binary import compose


class KGDataset(Dataset):
    """
    :ivar ent2id
    :ivar rel2id
    :ivar id2ent
    :ivar id2rel
    :ivar edge_index: stores edges between entities, has dim [2, num_edges], includes reverse edges
    :ivar edge_type: stores edge types for edges in edge_index, has dim [num_edges]
    """

    def __init__(
            self,
            graph_data_path,
            include_reverse_relations,
            subgraph_sampling=False,
            sampling_radius=3):
        super().__init__()

        # params
        self.include_reverse_relations = include_reverse_relations
        self.subgraph_sampling = subgraph_sampling
        self.sampling_raidus = sampling_radius

        # data
        self.seeds = []
        self.graph = None

        self.ent2id = {}
        self.rel2id = {}
        self.id2ent = {}
        self.id2rel = {}

        self.edge_index = None
        self.edge_type = None

        # Important: Adithya, this need to be initialized.
        self.task_object_grasp_triples = []

        # preprocess
        self.load_graph(graph_data_path)
        self.build_dicts()
        self.edge_index, self.edge_type = self.build_adjancy_matrix(
            self.graph, self.include_reverse_relations)

    def load_graph(self, graph_data_path):
        # load a networkx graph and seeds as a list of (object_id,
        # wordnet_synset, conceptnet_name)
        with open(graph_data_path, "rb") as fh:
            self.graph, self.seeds = pickle.load(fh)

    def build_dicts(self):

        # build dictionaries
        ent_set, rel_set = OrderedSet(), OrderedSet()
        for node in self.graph.nodes:
            ent_set.add(node)
        for edge in self.graph.edges:
            rel_set.add(self.graph[edge[0]][edge[1]]["relation"])
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        # reverse relation id is: idx+len(rel2id)
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id)
                            for idx, rel in enumerate(rel_set)})
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

    def build_adjancy_matrix(self, graph, include_reverser_relations=True):
        """
        This function builds an adjancy matrix for a networkx graph.

        :param graph:
        :return:
        """
        # convert the graph into triples
        data = []
        for edge in self.graph.edges:
            rel = self.graph[edge[0]][edge[1]]["relation"]
            sub = edge[0]
            obj = edge[1]
            rel_id = self.rel2id[rel]
            sub_id = self.ent2id[sub]
            obj_id = self.ent2id[obj]
            data.append((sub_id, rel_id, obj_id))

        # build adjancy matrix
        edge_index, edge_type = [], []
        for sub, rel, obj in data:
            edge_index.append((sub, obj))
            edge_type.append(rel)
            # Important: include reverse relations
            if include_reverser_relations:
                inv_rel_id = rel + len(self.rel2id)
                edge_index.append((obj, sub))
                edge_type.append(inv_rel_id)
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_type = torch.LongTensor(edge_type)

        return edge_index, edge_type

    def extract_subgraph(self, object, task, hop=3):
        """
        This function extract n-hop neighbors for an object and task pairs.

        ToDo: using the subgraph is slightly more involved since node index in the subgraph needs to be adjusted

        :param object:
        :param task:
        :param hop:
        :return:
        """
        object_subgraph = ego_graph(
            self.graph, object, radius=hop, undirected=True)
        task_subgraph = ego_graph(
            self.graph, task, radius=hop, undirected=True)
        subgraph = compose(object_subgraph, task_subgraph)
        return subgraph

    def __len__(self):
        return len(self.task_object_grasp_triples)

    def __getitem__(self, idx):
        """
        return task_idx, object_idx, node_features simply as node idx, edge_index, and edge_type
        :param idx:
        :return:
        """

        # Important: Adithya, I assume you have the function to map from idx to
        # the plain names of the task, object
        t, o, g = some_function(idx)

        t_id = torch.LongTensor(self.ent2id[t])
        o_id = torch.LongTensor(self.ent2id[o])

        # inputs for gnns
        node_feats = torch.LongTensor(
            torch.arange(len(self.ent2id))).reshape(-1, 1)
        return o_id, t_id, node_feats, self.edge_index.clone(
        ).detach(), self.edge_type.clone().detach()

    @staticmethod
    def collate_fn(data):
        """
        Adjacency matrices are stacked in a diagonal fashion (creating a giant graph that holds multiple
        isolated subgraphs), and node and target features are simply concatenated in the node dimension
        (https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)

        automatically increment the edge_index tensor by the cumulated number of nodes of graphs that got collated
        before the currently processed graph, and will concatenate edge_index tensors (that are of shape [2, num_edges])
        in the second dimension.

        :param data:
        :return:
        """
        o_ids = torch.stack([_[0] for _ in data], dim=0)
        t_ids = torch.stack([_[1] for _ in data], dim=0)

        # node features
        node_feats = torch.cat([_[2] for _ in data], dim=0)

        # edge_relations
        edge_types = torch.cat([_[4] for _ in data], dim=0)

        # adjacency matrices
        edge_indices = []
        counter = 0
        for _, _, x, edge_index, _ in data:
            edge_indices.append(edge_index + counter)
            counter += len(x)
        eedge_indices = torch.cat(edge_indices, dim=0)

        return o_ids, t_ids, node_feats, edge_indices, edge_types


if __name__ == "__main__":
    graph_data = '../../data/knowledge_graph/graph_data.pkl'
    embed()
    dset = KGDataset(
        graph_data_path=graph_data,
        include_reverse_relations=True,
        subgraph_sampling=True,
        sampling_radius=2)
