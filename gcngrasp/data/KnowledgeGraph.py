import argparse
import copy
import os
import sys
import pickle
import requests

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
from nltk.corpus.reader import wordnet as rwn
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from collections import defaultdict
from tqdm import tqdm
from ordered_set import OrderedSet
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../../'))
from visualize import mkdir


class KnowledgeGraph:
    """
    Construct a knowledge graph using WordNet, ConceptNet, and crowdsourced task-object pairs

    :ivar graph_name: name of the knowledge graph
    :ivar _base_dir: base directory for all knowledge graphs, also location for object definition file (storing
                     wordnet synsets and concept names for object instances), object file (storing all target object
                     instances), object task file (storing task-object pairs), and node embedding file
    :ivar _save_path: path to save the current knowledge graph
    :ivar seeds: a list storing target objects that the knowledge graph will be built on
    :ivar graph: the knowledge graph as a networkx.DiGraph object
    :ivar name2wn: a one to one mapping from an object name to a wordnet synset
    :ivar name2cn: a one to one mapping from an object name to a conceptnet name
    :ivar wn2name: a one to many mapping from a wordnet synset to object names
    :ivar wn2cn: a one to many mapping from a wordnet synset to conceptnet names
    """

    def __init__(self, graph_name, base_dir=""):

        # data
        self.graph_name = graph_name
        self._base_dir = base_dir

        if self._base_dir == "":
            self._base_dir = os.path.join(BASE_DIR, "../../data/knowledge_graph")

        self._save_path = os.path.join(self._base_dir, self.graph_name)
        mkdir(self._save_path)

        self.seeds = []
        self.graph = nx.DiGraph()

        self.wn2name = defaultdict(list)
        self.name2cn = {}
        self.wn2cn = defaultdict(list)
        self.name2wn = {}

    def populate_graph(self, object_definition_file, object_file, add_wordnet_hierarchy=True, add_wn_synset=False,
                       add_object_nodes=True, add_cn_name=False, include_cn_phrase=False, conceptnet_relations=None,
                       object_task_filename="", remove_task_edges=False, use_task1_grasps=False,
                       connect_tasks_and_obj_classes=True, sunburst=False, debug=False):
        """
        This is the main function for constructing the knowledge graph

        :param object_definition_file: a file storing wordnet synsets and concept names for object instances
        :param object_file: a file storing all target object instances
        :param add_wordnet_hierarchy: whether to add WordNet hypernym paths for each synset
        :param add_wn_synset: if set to true, a WordeNet synset node for each object instance will be added and linked
                              to its corresponding object instance node if the instance node exists
        :param add_object_nodes: if set to true, an object instance node will be added for each object instance
        :param add_cn_name: if set to true, a ConceptNet name node for each object instance will be added and linked
                            to its corresponding object instance node
        :param include_cn_phrase: if set to true, add ConceptNet concepts that are phrases
        :param conceptnet_relations: a list of ConceptNet relations to extract
        :param object_task_filename: a file storing storing task-object pairs
        :param remove_task_edges: if set to true, edges between task and objects will not be added
        :param use_task1_grasps: if set to true, add edges between task and objects with crowdsourced task-object pairs
        :param connect_tasks_and_obj_classes: if set to true, tasks are connected directly to object classes (which are
                                              object synsets) instead of object instances
        :param sunburst: visualize object distribution with sunburst plot
        :param debug: print and visualize graphs for debugging purpose
        :return:
        """

        object_definition_file = os.path.join(self._base_dir, object_definition_file)
        object_file = os.path.join(self._base_dir, object_file)
        object_task_filename = os.path.join(self._base_dir, object_task_filename)
        assert os.path.exists(object_definition_file)
        assert os.path.exists(object_file)
        self.read_objects(object_definition_file, object_file)

        if add_object_nodes:
            self.add_object_nodes()
        if debug:
            self.draw(title="KB-1", save_path=self._save_path)

        self.add_wn_relations(add_wn_synset=add_wn_synset, add_wordnet_hierarchy=add_wordnet_hierarchy,
                              add_object_nodes=add_object_nodes)
        if debug:
            self.draw(title="KB-2", save_path=self._save_path)

        if sunburst:
            self.plot_object_distribution(save_path=self._save_path)

        if conceptnet_relations:
            for rel_idx, rel in enumerate(conceptnet_relations):
                self.add_cn_relations(relation_type=rel, add_cn_name=add_cn_name, include_cn_phrase=include_cn_phrase)
                if debug:
                    self.draw(title="KB-3-{}-{}".format(rel_idx, rel), save_path=self._save_path)

        if object_task_filename:
            self.add_object_task_pairs(filename=object_task_filename,
                                       connect_tasks_and_obj_classes=connect_tasks_and_obj_classes,
                                       use_task1_grasps=use_task1_grasps, remove_task_edges=remove_task_edges)
            self.draw(title="KB-4-ot", save_path=self._save_path)

        self.print_graph_statistics()
        self.save_graph(save_path=self._save_path)
        if debug:
            self.draw(title="KB-full", save_path=self._save_path)

    def read_objects(self, object_definition_file, object_file):
        """
        This function reads WordNet synsets and ConceptNet names for object instances

        :param object_definition_file: a file storing wordnet synsets and concept names for object instances
        :param object_file: a file storing all target object instances
        :return:
        """
        with open(object_definition_file, "rb") as fh:
            df = pd.read_excel(fh)

        with open(object_file, "r") as fh:
            add_objects = [line.strip() for line in fh.readlines()]

        for obj_id, wn_syn, cn_name in zip(list(df["id"]), list(df["wordnet_synset"]), list(df["conceptnet_name"])):
            # remove objects that are not fully specified
            if pd.isna(obj_id) or pd.isna(wn_syn) or pd.isna(cn_name):
                print("Remove object {} with Wordnet Synset {} and ConceptNet {}".format(obj_id, wn_syn, cn_name))
                continue
            if obj_id not in add_objects:
                print("Remove object {} that are not in this data split".format(obj_id))
                continue
            self.seeds.append((obj_id, wn_syn, cn_name))
            self.wn2name[wn_syn].append(obj_id)
            self.wn2cn[wn_syn].append(cn_name)
            self.name2cn[obj_id] = cn_name
            self.name2wn[obj_id] = wn_syn

        self.wn2name = dict(self.wn2name)
        self.wn2cn = dict(self.wn2cn)

        print("Target object instances: {}".format(self.seeds))

    def draw(self, method="graphviz", title="knowledge_graph", save_path=""):
        """
        This function helps to visualize a network DiGraph

        :param method: which visualization method to use
        :param title: name of the graph
        :param save_path: save directory
        :return:
        """
        if method == "graphviz":
            # There are two ways to visualize networkx graph
            # 1. write dot file to use with graphviz
            # run "dot -Tpng test.dot > test.png"
            dot_path = os.path.join(save_path, '{}.dot'.format(title))
            png_path = os.path.join(save_path, '{}.png'.format(title))
            write_dot(self.graph, dot_path)
            cmd = 'dot -Tpng {} > {}'.format(dot_path, png_path)
            os.system(cmd)
        elif method == "matplotlib":
            # 2. same layout using matplotlib with no labels
            # Not so good
            plt.title(title)
            pos = graphviz_layout(self.graph, prog='dot')
            nx.draw(self.graph, pos, with_labels=False, arrows=True)
            plt.savefig(os.path.join(save_path, 'nx_test.png'))

    def add_object_nodes(self):
        """
        This function adds object_id node for each object instance to the knowledge graph

        :return:
        """
        for obj_id, wn_syn, cn_name in self.seeds:
            self.graph.add_node(obj_id, color="red", type="object_id")

    def add_wn_relations(self, add_wn_synset=False, add_wordnet_hierarchy=True, add_object_nodes=True):
        """
        This function adds IsA relation extracted from WordNet

        :param add_wn_synset: whether to add WordNet synset node for each object instance
        :param add_wordnet_hierarchy: whether to add WordNet hypernym paths for each synset
        :param add_object_nodes: whether object instance nodes exist or will be added
        :return:
        """
        print("\n\nAdding WordNet object hierarchy...")

        for obj_id, wn_syn, cn_name in self.seeds:
            synset = wn.synset(wn_syn)
            hypens = synset.hypernym_paths()

            if add_wn_synset:
                self.graph.add_node(wn_syn, color="blue", type="wordnet_synset")
                if add_object_nodes:
                    self.graph.add_edge(obj_id, wn_syn, relation='HasWordNetSynset')
            if add_wordnet_hierarchy:
                for hypen in hypens:
                    if not add_wn_synset:
                        # print(obj_id, hypen[-1].name(), hypen[-2].name())
                        assert add_object_nodes
                        self.graph.add_edge(obj_id, hypen[-2].name(), relation='IsA')
                        self.graph.add_node(hypen[-2].name(), color="orange", type="extracted_wordnet_synset")
                        for i in range(0, len(hypen) - 2):
                            parent = hypen[i].name()
                            child = hypen[i + 1].name()
                            self.graph.add_node(hypen[i].name(), color="orange", type="extracted_wordnet_synset")
                            self.graph.add_node(hypen[i + 1].name(), color="orange", type="extracted_wordnet_synset")
                            self.graph.add_edge(child, parent, relation='IsA')
                    else:
                        for i in range(0, len(hypen) - 1):
                            parent = hypen[i].name()
                            child = hypen[i + 1].name()
                            if parent not in self.graph.nodes:
                                self.graph.add_node(parent, color="orange", type="extracted_wordnet_synset")
                            if child not in self.graph.nodes:
                                self.graph.add_node(child, color="orange", type="extracted_wordnet_synset")
                            self.graph.add_edge(child, parent, relation='IsA')

        # reduce graph size
        if add_wordnet_hierarchy:
            self.compress_isa_graph()

    def compress_isa_graph(self, verbose=True):
        """
        This function is used to compress the extracted graph from WordNet by removing some of the nodes.
        The compression strategy follows paper 'Nearly-Automated Metadata Hierarchy Creation'

        :param verbose: whether to show compression steps for debugging
        :return:
        """
        print("\n\nCompressing WordNet object hierarchy...")

        graph1 = copy.deepcopy(self.graph)

        # Rule 1 - Remove all nodes with low information content
        brown = wnic.ic('ic-brown.dat')
        for node in list(self.graph.nodes()):
            if self.graph.nodes[node]["type"] != "object_id" and self.graph.nodes[node]["type"] != "wordnet_synset":
                if rwn.information_content(wn.synset(node), brown) < 3.0:
                    self.graph.remove_node(node)
        if verbose:
            diff = set(graph1.nodes()) - set(self.graph.nodes())
            print("Nodes removed by compression rule 1: {}".format(list(diff)))

        # Rule 2 - Remove all nodes with only a single child except the root
        if verbose:
            graph2 = copy.deepcopy(self.graph)
        # starting from leaf nodes
        nodes_sort = [node for node in self.graph if len(
            list(self.graph.predecessors(node))) == 0]
        while len(nodes_sort) > 0:
            node = nodes_sort.pop(0)
            if node not in self.graph:
                continue

            parents = list(self.graph.successors(node))
            children = list(self.graph.predecessors(node))
            for parent in parents:
                nodes_sort.append(parent)

            if len(children) == 1 and len(
                    parents) != 0 and self.graph.nodes[node]["type"] != "object_id" and self.graph.nodes[node]["type"] != "wordnet_synset":
                self.graph.remove_node(node)
                for parent in parents:
                    for child in children:
                        self.graph.add_edge(child, parent, relation='IsA')
        if verbose:
            diff = set(graph2.nodes()) - set(self.graph.nodes())
            print("Nodes removed by compression rule 2: {}".format(list(diff)))

        # Rule 3 - Remove all nodes whose name contains the name of the parent
        # (except seed)
        if verbose:
            graph3 = copy.deepcopy(self.graph)
        for node in list(self.graph.nodes()):
            if len(list(self.graph.predecessors(node))) == 0:
                continue
            if self.graph.nodes[node]["type"] == "object_id" or self.graph.nodes[node]["type"] == "wordnet_synset":
                continue
            parents = list(self.graph.successors(node))
            children = list(self.graph.predecessors(node))
            should_remove = True if len(parents) > 0 else False
            for parent in parents:
                pname = parent.split('.')[0]
                cname = node.split('.')[0]
                if pname not in cname:
                    should_remove = False
                    break
            if should_remove:
                self.graph.remove_node(node)
                for child in children:
                    for parent in parents:
                        self.graph.add_edge(child, parent, relation='IsA')
        if verbose:
            diff = set(graph3.nodes()) - set(self.graph.nodes())
            print("Nodes removed by compression rule 3: {}".format(list(diff)))

        # sanity check: make sure no initial object nodes are removed
        current_seeds = []
        for n in list(graph1.nodes()):
            if graph1.nodes[n]["type"] == "wordnet_synset" or graph1.nodes[n]["type"] == "object_id":
                assert n in self.graph.nodes

        # add a common parent to combine the isolated graphs created by
        # compression
        root_nodes = [
            (node,
             "entity.n.01") for node in self.graph if len(
                list(
                    self.graph.successors(node))) == 0]
        self.graph.add_node(
            "entity.n.01",
            color="orange",
            type="extracted_wordnet_synset")
        self.graph.add_edges_from(root_nodes, relation="IsA")

    def add_cn_relations(self, relation_type, add_cn_name=False, cn_threshold=0, include_cn_phrase=False):
        """
        This function extract ConceptNet relations

        :param relation_type: a list of ConceptNet relations to extract
        :param add_cn_name: if set to true, a ConceptNet name node for each object instance will be added and linked
                            to its corresponding object instance node
        :param cn_threshold: ConceptNet edges with edge weights less than the threshold will not be added
        :param include_cn_phrase: if set to true, add ConceptNet concepts that are phrases
        :return:
        """
        print("\n\nAdding ConceptNet relations...")

        for seed in tqdm(self.seeds, desc="Extract ConceptNet Relation {}".format(relation_type)):
            obj_id, wn_syn, cn_name = seed
            r = requests.get('http://api.conceptnet.io/query',
                             params={'rel': '/r/{}'.format(relation_type), 'start': '/c/en/{}'.format(cn_name)})
            edges = r.json()['edges']

            if add_cn_name:
                if cn_name in self.graph.nodes:
                    if self.graph.nodes[cn_name]["type"] != "conceptnet_name":
                        print("Trying to add a conceptnet name node {}. It will overwrite an existing node".format(
                            cn_name))
                self.graph.add_node(
                    cn_name, color="green", type="conceptnet_name")
                self.graph.add_edge(
                    obj_id, cn_name, relation='HasConceptNetName')

            for edge in edges:
                concept = edge['end']['term'].split("/")[-1]

                if float(edge['weight']) < cn_threshold:
                    continue

                # remove compound term (e.g., pouring_liquids_out)
                if not include_cn_phrase:
                    if concept.find('_') != -1:
                        continue

                self.graph.add_node(concept, color="yellow", type="extracted_conceptnet_concept")
                if concept in self.graph.nodes:
                    if self.graph.nodes[concept]["type"] != "extracted_conceptnet_concept":
                        print("Trying to add an extracted conceptnet concept node {}. It will overwrite an existing node".format(cn_name))

                if add_cn_name:
                    self.graph.add_edge(cn_name, concept, relation=relation_type)
                else:
                    self.graph.add_edge(obj_id, concept, relation=relation_type)

    def add_object_task_pairs(self, filename, connect_tasks_and_obj_classes=True, use_task1_grasps=False,
                              remove_task_edges=False):
        """
        This function adds UsedForTask relation from crowdsourced object-task pairs

        :param filename: the file storing storing task-object pairs
        :param connect_tasks_and_obj_classes: if set to true, tasks are connected directly to object classes (which are
                                              object synsets) instead of object instances
        :param use_task1_grasps: if set to true, add edges between task and objects with crowdsourced task-object pairs
        :param remove_task_edges: if set to true, edges between task and objects will not be added
        :return:
        """
        print("\n\nAdding task-object relationship edges from dataset...")

        with open(filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    object, task, label = line.split("-")
                    if object not in self.name2wn:
                        continue
                    if not remove_task_edges:
                        if label == "True" or label == "Weak True":
                            assert object in self.graph.nodes or self.name2wn[object] in self.graph.nodes
                            if task in self.graph.nodes:
                                if self.graph.nodes[task]["type"] != "task":
                                    print(
                                        "Trying to add an task node {}. It will overwrite an existing node with type {}".format(
                                            task, self.graph.nodes[task]["type"]))
                            self.graph.add_node(task, color="purple", type="task")

                            if connect_tasks_and_obj_classes:
                                wn_syn = self.name2wn[object]
                                self.graph.add_edge(wn_syn, task, relation="UsedForTask")
                            else:
                                # Directly connect the tasks to the object
                                # nodes
                                self.graph.add_edge(object, task, relation="UsedForTask")
                        else:
                            if use_task1_grasps:
                                if task in self.graph.nodes:
                                    if self.graph.nodes[task]["type"] != "task":
                                        print(
                                            "Trying to add an task node {}. It will overwrite an existing node with type {}".format(
                                                task, self.graph.nodes[task]["type"]))
                                self.graph.add_node(task, color="purple", type="task")
                    else:
                        if task in self.graph.nodes:
                            if self.graph.nodes[task]["type"] != "task":
                                print(
                                    "Trying to add an task node {}. It will overwrite an existing node with type {}".format(
                                        task, self.graph.nodes[task]["type"]))
                        self.graph.add_node(task, color="purple", type="task")

    def print_graph_statistics(self):
        print("\n\nPrinting graph statistics...")
        print("Graph has", self.graph.order(), "nodes")
        print("Graph has", self.graph.size(), "edges")
        print("Graph has density: {}".format(nx.density(self.graph)))
        edge_type_count = {}
        for edge in self.graph.edges:
            relation = self.graph.edges[edge]['relation']
            if relation in edge_type_count:
                edge_type_count[relation] += 1
            else:
                edge_type_count[relation] = 1
        for edge_type in edge_type_count:
            print("Graph has", edge_type_count[edge_type], edge_type, "edges")

        subject_types = {}
        for edge in self.graph.edges:
            type = self.graph.edges[edge]["relation"]
            if type in subject_types:
                subject_types[type].add(edge[1])
            else:
                subject_types[type] = set()
                subject_types[type].add(edge[1])
        for type in subject_types:
            print("Graph has",
                  len(subject_types[type]),
                  type,
                  "subject nodes:",
                  subject_types[type])

    def save_graph(self, save_path=""):
        if save_path == "":
            save_path = self._save_path
        if not os.path.exists(save_path):
            mkdir(save_path)
        with open(os.path.join(save_path, "graph_data.pkl"), "wb") as fh:
            pickle.dump([self.graph, self.seeds], fh)
        with open(os.path.join(save_path, "misc.pkl"), "wb") as fh:
            pickle.dump([self.wn2name, self.wn2cn,
                         self.name2cn, self.name2wn], fh)

    def load_graph(self, save_path=""):
        if save_path == "":
            save_path = self._save_path

        with open(os.path.join(save_path, "graph_data.pkl"), "rb") as fh:
            self.graph, self.seeds = pickle.load(fh)

        self.wn2name = defaultdict(list)
        self.name2cn = {}
        self.wn2cn = defaultdict(list)
        for obj_id, wn_syn, cn_name in self.seeds:
            self.wn2name[wn_syn].append(obj_id)
            self.wn2cn[wn_syn].append(cn_name)
            self.name2cn[obj_id] = cn_name
            self.name2wn[obj_id] = wn_syn
        self.wn2name = dict(self.wn2name)
        self.wn2cn = dict(self.wn2cn)

    def prepare_embeddings(self, embedding_file):
        """
        This function prepares node embeddings for the graph

        :param embedding_file: location of the raw embedding file
        :return:
        """
        print("\n\nCreating node embeddings...")

        # For handling nodes that are not in the pretrained embeddings
        catch_all = {
            "numberbatch": {"disect": "dissect",
                            "spagetti": "spaghetti",
                            "kitche": "kitchen",
                            "pedastal": "pedestal",
                            "restrauent": "restaurant",
                            "refigerator": "refrigerator",
                            "steelmill": "steel_mill",
                            "pestil": "dried_fruit",
                            "mason_jar.n.01": "mason_jar"},
            "word2vec": {"can_opener": "opener",
                         "paint_roller": "roller",
                         "garlic_press": "crusher",
                         "measuring_cup": "cup",
                         "back_scratcher": "scratcher",
                         "backscratcher": "scratcher",
                         "pepper_mill": "grinder",
                         "dustcloth": "cloth",
                         "turn on": "turn",
                         "plug in": "plug",
                         "kitche": "kitchen",
                         "restrauent": "restaurant",
                         "refigerator": "refrigerator",
                         "pestil": "snack",
                         "change_integrity": "change",
                         "indispose": "agitate",
                         "compound_lever": "lever",
                         "piece_of_cloth": "cloth"},
            "glove": {"change_integrity": "change",
                      "indispose": "agitate",
                      "incise": "cut",
                      "rolling_pin": "roller",
                      "can_opener": "opener",
                      "paint_roller": "roller",
                      "garlic_press": "crusher",
                      "mixing_bowl": "bowl",
                      "coat_hanger": "hanger",
                      "salt_shaker": "shaker",
                      "saltshaker": "shaker",
                      "pepper_grinder": "grinder",
                      "hair_spray": "spray",
                      "watering_can": "container",
                      "measuring_cup": "cup",
                      "backscratcher": "scratcher",
                      "back_scratcher": "scratcher",
                      "saucepot": "pot",
                      "coffee_mug": "mug",
                      "beer_mug": "mug",
                      "pancake_turner": "spatula",
                      "pepper_mill": "grinder",
                      "cereal_bowl": "bowl",
                      "wooden_spoon": "spoon",
                      "frying_pan": "pan",
                      "scrub_brush": "brush",
                      "disect": "dissect",
                      "spagetti": "spaghetti",
                      "dustcloth": "cloth",
                      "kitche": "kitchen",
                      "pedastal": "pedestal",
                      "restrauent": "restaurant",
                      "resturant": "restaurant",
                      "refigerator": "refrigerator",
                      "pestil": "snack",
                      "toolbelt": "belt",
                      "compound_lever": "lever",
                      "piece_of_cloth": "cloth"},
            "fasttext": {"paint_roller": "roller",
                         "garlic_press": "crusher",
                         "pepper_grinder": "grinder",
                         "hair_spray": "spray",
                         "watering_can": "container",
                         "measuring_cup": "cup",
                         "coffee_mug": "mug",
                         "pancake_turner": "spatula",
                         "pepper_mill": "grinder",
                         "cereal_bowl": "bowl",
                         "kitche": "kitchen",
                         "pedastal": "pedestal",
                         "restrauent": "restaurant",
                         "refigerator": "refrigerator",
                         "pestil": "snack",
                         "scrub_brush": "brush",
                         "beer_mug": "mug",
                         "dustcloth": "cloth",
                         "change_integrity": "change",
                         "indispose": "agitate",
                         "compound_lever": "lever",
                         "piece_of_cloth": "cloth"}
        }

        embedding_model = ""
        if "glove" in embedding_file:
            embedding_model = "glove"
        elif "GoogleNews" in embedding_file:
            embedding_model = "word2vec"
        elif "subword" in embedding_file:
            embedding_model = "fasttext"
        elif "numberbatch" in embedding_file:
            embedding_model = "numberbatch"

        assert embedding_model in catch_all

        embedding_file = os.path.join(self._base_dir, embedding_file)

        def transform(compound_word):
            return [compound_word,
                    "_".join([w.capitalize() for w in compound_word.split("_")]),
                    "-".join([w for w in compound_word.split("_")])]

        node2vec = {}
        model = None
        node2vec_filename = os.path.join(
            self._save_path, "{}_node2vec.pkl".format(embedding_model))
        if os.path.exists(node2vec_filename):
            print("node2vec pickle file found: ", node2vec_filename)
        else:
            if os.path.exists(embedding_file):

                # glove has a slightly different format
                if embedding_model == "glove":
                    tmp_file = ".".join(
                        embedding_file.split(".")[:-1]) + "_tmp.txt"
                    glove2word2vec(embedding_file, tmp_file)
                    embedding_file = tmp_file

                # only native word2vec file needs binary flag to be true
                print("Loading pretrained embeddings from {} ...".format(embedding_file))
                model = KeyedVectors.load_word2vec_format(embedding_file, binary=(embedding_model == "word2vec"))
            else:
                raise Exception("pretrained embedding file not found")

            # retrieve embeddings for graph nodes
            no_match_nodes = []
            match_positions = []

            for node in tqdm(self.graph.nodes(), desc="Prepare node embeddings"):
                try_words = []
                node_type = self.graph.nodes[node]["type"]

                if node_type == "object_id":
                    try_words.extend(transform("_".join(node.split("_")[1:])))  # object name
                    # conceptnet name
                    try_words.extend(transform(self.name2cn[node]))
                    wordnet_synset = self.name2wn[node]
                    for lemma in wn.synset(wordnet_synset).lemma_names():
                        # wordnet synset name
                        try_words.extend(transform(lemma))

                    # remove id
                    node_name = "_".join(node.split("_")[1:])
                    if node_name in catch_all[embedding_model]:
                        try_words.append(catch_all[embedding_model][node_name])

                if node_type == "wordnet_synset" or node_type == "extracted_wordnet_synset":
                    for lemma in wn.synset(node).lemma_names():
                        try_words.extend(transform(lemma))  # wordnet synset name

                        if lemma in catch_all[embedding_model]:
                            try_words.append(catch_all[embedding_model][lemma])

                if node_type == "conceptnet_name" or node_type == "extracted_conceptnet_concept":
                    try_words.extend(transform(node))

                    if node in catch_all[embedding_model]:
                        try_words.append(catch_all[embedding_model][node])

                if node_type == "task":
                    try_words.extend(transform("_".join(node.split(" "))))

                if node in catch_all[embedding_model]:
                    try_words.append(catch_all[embedding_model][node])

                found_mapping = False
                for i, try_word in enumerate(try_words):
                    try:
                        node2vec[node] = model.get_vector(try_word)
                        match_positions.append(i + 1)
                        found_mapping = True
                    except KeyError:
                        pass
                    if found_mapping:
                        break

                if not found_mapping:
                    no_match_nodes.append([node, try_words])

            # print stats
            print("{}/{} nodes' embeddings found".format(len(match_positions),
                                                         len(self.graph.nodes)))
            print("Average position {}".format(sum(match_positions) * 1.0 / len(match_positions)))
            print("The following nodes have no matches:")
            for no_match in no_match_nodes:
                print(no_match)

            print("Saving synonym2vec to pickle file:", node2vec_filename)
            pickle.dump(node2vec, open(node2vec_filename, "wb"))

    def plot_object_distribution(self, save_path):
        """
        This function visualizes object distribution with a sunburst plot

        :param save_path: save directory
        :return:
        """

        import plotly.express as px
        import plotly.graph_objects as go

        # get hierarchies
        leaves = [v for v, d in self.graph.in_degree() if d == 0]
        root = [v for v, d in self.graph.out_degree() if d == 0]
        assert len(root) == 1
        all_paths = {}
        max_len = 0
        for leaf in leaves:
            paths = list(nx.all_simple_paths(self.graph, leaf, root[0]))

            for path in paths:
                path = path[1:]

                # remove wordnet post-fix
                path = [p[:-5] for p in path]
                if len(path) > max_len:
                    max_len = len(path)
                hash_path = tuple(path)
                if hash_path not in all_paths:
                    all_paths[hash_path] = 1
                else:
                    all_paths[hash_path] += 1

        organized_paths = []
        for path in all_paths:
            path = [""] * (max_len - len(path)) + \
                   list(path) + [all_paths[path]]
            organized_paths.append(path)

        df = pd.DataFrame(organized_paths)
        fig = px.sunburst(
            df, path=list(
                range(
                    max_len - 1, -1, -1)), values=max_len)

        new_labels = []
        new_parents = []
        new_values = []
        new_ids = []
        labels = fig['data'][0]['labels'].tolist()
        parents = fig['data'][0]['parents'].tolist()
        values = fig['data'][0]['values'].tolist()
        ids = fig['data'][0]['ids'].tolist()
        for label, parent, value, id in zip(labels, parents, values, ids):
            print(
                "{: <20s}{: <70s}{: <10s}{: <100s}".format(
                    label, parent, str(value), id))
            if "//" not in id and label:
                new_labels.append(label)
                new_parents.append(parent)
                new_values.append(value)
                new_ids.append(id)

        fig2 = go.Figure(go.Sunburst(
            labels=new_labels,
            parents=new_parents,
            values=new_values,
            ids=new_ids,
            domain={'x': [0.0, 1.0], 'y': [0.0, 1.0]},
            maxdepth=max_len,
            branchvalues="total",
        ))

        fig2.update_layout(
            margin=dict(t=0, l=0, r=0, b=0)
        )

        fig2.show()
        file = os.path.join(save_path, "object_sunburst.pdf")
        fig2.write_image(file, width=600, height=600, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG construction")
    parser.add_argument(
        '--base_dir',
        help='Location of knowledge_graph folder',
        default='',
        type=str)

    args = parser.parse_args()

    # To build a knowledge graph or prepare node embeddings for an already constructed graph, uncomment
    # a preset below

    ##########################################################################
    # Settings for creating different KGs
    ##########################################################################

    # # KG1: default graph with wordnet object hierarchy, conceptnet relations, object class-task edges
    # # Graph has 502 nodes
    # # Graph has 1707 edges
    # # Graph has density: 0.006787222368012978
    # kg = KnowledgeGraph(graph_name="kb2_all", base_dir=args.base_dir)
    # kg.populate_graph(object_definition_file="Scanned Objects.xlsx",
    #                   object_file="all_objects.txt",
    #                   add_wordnet_hierarchy=True,
    #                   add_wn_synset=True,
    #                   connect_tasks_and_obj_classes=True,
    #                   add_cn_name=False,
    #                   include_cn_phrase=False,
    #                   conceptnet_relations=["UsedFor", "CapableOf", "AtLocation", "HasProperty", "MadeOf"],
    #                   object_task_filename="task1_results.txt",
    #                   debug=True,
    #                   use_task1_grasps=True)
    # kg.save_graph()

    # # KG2: default graph abalation: No task edges (only wn + cn)
    # # Graph has 502 nodes
    # # Graph has 1014 edges
    # kg = KnowledgeGraph(graph_name="kb2_wn_cn", base_dir=args.base_dir)
    # kg.populate_graph(object_definition_file="Scanned Objects.xlsx",
    #                   object_file="all_objects.txt",
    #                   add_wordnet_hierarchy=True,
    #                   add_wn_synset=True,
    #                   connect_tasks_and_obj_classes=True,
    #                   add_cn_name=False,
    #                   include_cn_phrase=False,
    #                   conceptnet_relations=["UsedFor", "CapableOf", "AtLocation", "HasProperty", "MadeOf"],
    #                   object_task_filename="task1_results.txt",
    #                   remove_task_edges=True,
    #                   debug=True,
    #                   use_task1_grasps=True)
    # kg.save_graph()

    # # KG3: default graph abalation: No wn edges (only task + cn)
    # # Graph has 478 nodes
    # # Graph has 1601 edges
    # # Graph has density: 0.007021745041797146
    # kg = KnowledgeGraph(graph_name="kb2_task_cn", base_dir=args.base_dir)
    # kg.populate_graph(object_definition_file="Scanned Objects.xlsx",
    #                   object_file="all_objects.txt",
    #                   add_wordnet_hierarchy=False,
    #                   add_wn_synset=True,
    #                   connect_tasks_and_obj_classes=True,
    #                   add_cn_name=False,
    #                   include_cn_phrase=False,
    #                   conceptnet_relations=["UsedFor", "CapableOf", "AtLocation", "HasProperty", "MadeOf"],
    #                   object_task_filename="task1_results.txt",
    #                   debug=True,
    #                   use_task1_grasps=True)
    # kg.save_graph()

    # # KG4: default graph abalation: No wn or task edges (only cn)
    # # Graph has 478 nodes
    # # Graph has 908 edges
    # # Graph has density: 0.007021745041797146
    # kg = KnowledgeGraph(graph_name="kb2_cn", base_dir=args.base_dir)
    # kg.populate_graph(object_definition_file="Scanned Objects.xlsx",
    #                   object_file="all_objects.txt",
    #                   add_wordnet_hierarchy=False,
    #                   add_wn_synset=True,
    #                   connect_tasks_and_obj_classes=True,
    #                   add_cn_name=False,
    #                   include_cn_phrase=False,
    #                   conceptnet_relations=["UsedFor", "CapableOf", "AtLocation", "HasProperty", "MadeOf"],
    #                   object_task_filename="task1_results.txt",
    #                   remove_task_edges=True,
    #                   debug=True,
    #                   use_task1_grasps=True)
    # kg.save_graph()

    # # KG5: tasks + wn (no cn)
    # # Graph has 345 nodes
    # # Graph has 989 edges
    # # Graph has density: 0.006787222368012978
    # kg = KnowledgeGraph(graph_name="kb2_task_wn", base_dir=args.base_dir)
    # kg.populate_graph(object_definition_file="Scanned Objects.xlsx",
    #                   object_file="all_objects.txt",
    #                   add_wordnet_hierarchy=True,
    #                   add_wn_synset=True,
    #                   connect_tasks_and_obj_classes=True,
    #                   add_cn_name=False,
    #                   include_cn_phrase=False,
    #                   conceptnet_relations=[],
    #                   object_task_filename="task1_results.txt",
    #                   debug=True,
    #                   use_task1_grasps=True)
    # kg.save_graph()

    # # KG6: tasks(no cn or wn)
    # # Graph has 321 nodes
    # # Graph has 883 edges
    # # Graph has density: 0.006787222368012978
    # kg = KnowledgeGraph(graph_name="kb2_task", base_dir=args.base_dir)
    # kg.populate_graph(object_definition_file="Scanned Objects.xlsx",
    #                   object_file="all_objects.txt",
    #                   add_wordnet_hierarchy=False,
    #                   add_wn_synset=True,
    #                   connect_tasks_and_obj_classes=True,
    #                   add_cn_name=False,
    #                   include_cn_phrase=False,
    #                   conceptnet_relations=[],
    #                   object_task_filename="task1_results.txt",
    #                   debug=True,
    #                   use_task1_grasps=True)
    # kg.save_graph()

    # # KG7: only wn (no task or cn)
    # # Graph has 345 nodes
    # # Graph has 296 edges
    # # Graph has density: 0.006787222368012978
    # kg = KnowledgeGraph(graph_name="kb2_wn", base_dir=args.base_dir)
    # kg.populate_graph(object_definition_file="Scanned Objects.xlsx",
    #                   object_file="all_objects.txt",
    #                   add_wordnet_hierarchy=True,
    #                   add_wn_synset=True,
    #                   connect_tasks_and_obj_classes=True,
    #                   add_cn_name=False,
    #                   include_cn_phrase=False,
    #                   conceptnet_relations=[],
    #                   object_task_filename="task1_results.txt",
    #                   debug=True,
    #                   remove_task_edges=True,
    #                   use_task1_grasps=True)
    # kg.save_graph()

    # KG7: tasks + wn (no cn) and no instance nodes
    # Graph has 345 nodes
    # Graph has 989 edges
    # Graph has density: 0.006787222368012978
    # kg = KnowledgeGraph(graph_name="kb2_task_wn_noi", base_dir=args.base_dir)
    # kg.populate_graph(object_definition_file="Scanned Objects.xlsx",
    #                   object_file="all_objects.txt",
    #                   add_wordnet_hierarchy=True,
    #                   add_object_nodes=False,
    #                   add_wn_synset=True,
    #                   connect_tasks_and_obj_classes=True,
    #                   add_cn_name=False,
    #                   include_cn_phrase=False,
    #                   conceptnet_relations=[],
    #                   object_task_filename="task1_results.txt",
    #                   debug=True,
    #                   use_task1_grasps=True)
    # kg.save_graph()

    # KG8: tasks (no cn or wn) and no instance nodes
    # Graph has 131 nodes
    # Graph has 693 edges
    # Graph has density: 0.006787222368012978
    # kg = KnowledgeGraph(
    #     graph_name="kb2_task_noi_debug",
    #     base_dir=args.base_dir)
    # kg.populate_graph(object_definition_file="Scanned Objects.xlsx",
    #                   object_file="all_objects.txt",
    #                   add_wordnet_hierarchy=False,
    #                   add_object_nodes=False,
    #                   add_wn_synset=True,
    #                   connect_tasks_and_obj_classes=True,
    #                   add_cn_name=False,
    #                   include_cn_phrase=False,
    #                   conceptnet_relations=[],
    #                   object_task_filename="task1_results.txt",
    #                   debug=True,
    #                   use_task1_grasps=True,
    #                   sunburst=False)
    # kg.save_graph()

    # KG9: wn (no cn or task nodes) and no instance nodes
    # Graph has 155 nodes
    # Graph has 106 edges
    # Graph has density: 0.006787222368012978
    # kg = KnowledgeGraph(graph_name="kb2_wn_noi", base_dir=args.base_dir)
    # kg.populate_graph(object_definition_file="Scanned Objects.xlsx",
    #                   object_file="all_objects.txt",
    #                   add_wordnet_hierarchy=True,
    #                   add_object_nodes=False,
    #                   add_wn_synset=True,
    #                   connect_tasks_and_obj_classes=True,
    #                   add_cn_name=False,
    #                   include_cn_phrase=False,
    #                   conceptnet_relations=[],
    #                   object_task_filename="task1_results.txt",
    #                   debug=True,
    #                   remove_task_edges=True,
    #                   use_task1_grasps=True,
    #                   sunburst=False)
    # kg.save_graph()

    ##########################################################################
    # Settings for creating different node embeddings
    ##########################################################################
    # Download links:
    # conceptnet_numberbatch (19.08 en): https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
    # fasttext (wki-news-300d-1M-subword): https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
    # glove (Wikipedia+Gigaword 5 (6B)): http://nlp.stanford.edu/data/glove.6B.zip
    # word2vec (Google News (100B)): https://drive.google.com/uc?export=download&confirm=fdUw&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM

    # kg = KnowledgeGraph(graph_name="kbgrasp_complete", base_dir=args.base_dir)
    # kg.load_graph()
    # kg.prepare_embeddings(embedding_file="wiki-news-300d-1M-subword.vec")
    # kg.prepare_embeddings(embedding_file="GoogleNews-vectors-negative300.bin")
    # kg.prepare_embeddings(embedding_file="conceptnet/numberbatch-en-19.08.txt")
    # kg.prepare_embeddings(embedding_file="glove.6B.300d.txt")
    #
    # kg = KnowledgeGraph(graph_name="kbgrasp_minus_task_graph", base_dir=args.base_dir)
    # kg.load_graph()
    # kg.prepare_embeddings(embedding_file="wiki-news-300d-1M-subword.vec")
    # kg.prepare_embeddings(embedding_file="GoogleNews-vectors-negative300.bin")
    # kg.prepare_embeddings(embedding_file="conceptnet/numberbatch-en-19.08.txt")
    # kg.prepare_embeddings(embedding_file="glove.6B.300d.txt")
    #
    # kg = KnowledgeGraph(graph_name="kbgrasp_minus_wordnet", base_dir=args.base_dir)
    # kg.load_graph()
    # kg.prepare_embeddings(embedding_file="wiki-news-300d-1M-subword.vec")
    # kg.prepare_embeddings(embedding_file="GoogleNews-vectors-negative300.bin")
    # kg.prepare_embeddings(embedding_file="conceptnet/numberbatch-en-19.08.txt")
    # kg.prepare_embeddings(embedding_file="glove.6B.300d.txt")
    #
    # kg = KnowledgeGraph(graph_name="kbgrasp_minus_conceptnet", base_dir=args.base_dir)
    # kg.load_graph()
    # kg.prepare_embeddings(embedding_file="wiki-news-300d-1M-subword.vec")
    # kg.prepare_embeddings(embedding_file="GoogleNews-vectors-negative300.bin")
    # kg.prepare_embeddings(embedding_file="conceptnet/numberbatch-en-19.08.txt")
    # kg.prepare_embeddings(embedding_file="glove.6B.300d.txt")
    #
    # kg = KnowledgeGraph(graph_name="kbgrasp_task_object_instance", base_dir=args.base_dir)
    # kg.load_graph()
    # kg.prepare_embeddings(embedding_file="wiki-news-300d-1M-subword.vec")
    # kg.prepare_embeddings(embedding_file="GoogleNews-vectors-negative300.bin")
    # kg.prepare_embeddings(embedding_file="conceptnet/numberbatch-en-19.08.txt")
    # kg.prepare_embeddings(embedding_file="glove.6B.300d.txt")
