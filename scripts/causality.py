import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import causalnex
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.discretiser import Discretiser
from causalnex.structure import DAGRegressor
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.utils.network_utils import get_markov_blanket
from causalnex.network.sklearn import BayesianNetworkClassifier
from causalnex.discretiser.discretiser_strategy import (
    DecisionTreeSupervisedDiscretiserMethod,
)
from causalnex.network import BayesianNetwork
from causalnex.inference import InferenceEngine
from IPython.display import Image
import copy
import mlflow
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Causal:
    
    def __init__(self, filehandler):
        file_handler = logging.FileHandler(filehandler)
        formatter = logging.Formatter("time: %(asctime)s, function: %(funcName)s, module: %(name)s, message: %(message)s \n")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def generate_structure(self, data):
        '''
        Generate a StructureModel from the data
        This function applies the causalnex's NOTEARS algorithm to learn the structure
        Args:
            data: a pandas DataFrame that contains the the data to generate the Structure Model from
        Returns:
            A causalnex.structure.StructureModel generated from the data
        '''
        sm= from_pandas(data)
        return sm
    
    def ground_truth(self, data):

        sm = from_pandas(data)
        viz = plot_structure(
        sm,
        graph_attributes={"scale": "0.5"},
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK,
        prog='fdp',
        )
        return Image(viz.draw(format='png'))

    def visualize_structure(self, sm):
        '''
        This function visualizes a causalnex Structure Model
        Args:
            sm: causalnex.structure.StructureModel generated from the dataset
        '''
        viz = plot_structure(
            sm,
            graph_attributes={"scale": "0.5"},
            all_node_attributes=NODE_STYLE.WEAK,
            all_edge_attributes=EDGE_STYLE.WEAK,
            prog='fdp',
        )
        return Image(viz.draw(format='png'))
        
    def remove_weak_edges(self, sm, threshold):
        '''
        This function removes weak edges below a given threshold from a causalnex Structure Model. 
        Args:
            sm: causalnex.structure.StructureModel
                The StructureModel that you need to remove weak edges from
            threshold: float
                Edges below this threshold are removed
        Returns:
            a causalnex.structure.StructureModel with weak edges removed
        '''
        return sm.remove_edges_below_threshold(threshold)
    
    def remove_edges(self, sm, edges):
        '''
        This function removes the specified edges from a Structure Model
        Args:
            sm: causalnex.structure.StructureModel
                The StructureModel that you need to remove the edges from
            edges: list of tuples
                A list of tuples that define the edges to be removed. Each tuple contains two elements. The two elements in the tuple define the edge to be removed
              Returns:
                a causalnex.structure.StructureModel with the edges removed
        '''
        sm_new = sm
        for edge in edges:
            node1=edge[0]
            node2=edge[1]
            sm_new.remove_edge(node1, node2)
        return sm_new
      
    def add_edges(self, sm, edges):
        '''
        This function removes the specified edges from a Structure Model
        Args:
            sm: causalnex.structure.StructureModel
                The StructureModel that you need to remove the edges from
            edges: list of tuples
                A list of tuples that define the edges to be removed. Each tuple contains two elements. The two elements in the tuple define the edge to be removed
            Returns:
                a causalnex.structure.StructureModel with the edges added
        '''
        sm_new=sm
        for edge in edges:
            node1=edge[0]
            node2=edge[1]
            sm_new.add_edges(node1, node2)
        return sm_new
        
    def markov_blanket(self,sm):
        bn = BayesianNetwork(sm)
        blanket = get_markov_blanket(bn, 'diagnosis')
        edge_list = list(blanket.structure.edges)
        try: 
            edge_list = list(blanket.structure.edges)
            return edge_list
        except:
            print('Could not generate Markov blanket')
    

