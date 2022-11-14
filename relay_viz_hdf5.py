
import json
import numpy as np
from typing import Dict

from tvm.contrib.relay_viz.interface import (
    DefaultVizParser,
    Plotter,
    VizEdge,
    VizGraph,
    VizNode,
)

try:
    import h5py
    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    # add "from None" to silence
    # "During handling of the above exception, another exception occurred"
    raise ImportError(
        "The h5py package is required. "
        "Please install it first. For example, pip3 install h5py"
    ) from None

Hdf5VizParser = DefaultVizParser

class Hdf5Node:

    def __init__(self, viz_node: VizNode):
        self.name = viz_node.type_name + '_' + viz_node.identity
        self.type = viz_node.type_name
        self.params = self._detail_to_params(viz_node.detail)

    def _detail_to_params(self, detail: str) -> Dict:
        if len(detail) == 0:
            return {}

        ds = detail.split("\n")
        params = {}
        for p in ds:
            k, v = p.split(":")
            params[k] = v
        return params

class Hdf5Graph(VizGraph):
    """Hdf5 graph for relay IR.
    Parameters
    ----------
    name: str
        name of this graph.
    """

    def __init__(
        self,
        name: str
    ):
        self._name = name
        self._graph = {}
        self._id_to_hf_node = {}

    def node(self, viz_node: VizNode) -> None:
        """Add a node to the underlying graph."""

        if viz_node.identity not in self._graph:
            # Add the node into the graph.
            self._graph[viz_node.identity] = []

        node = Hdf5Node(viz_node)
        self._id_to_hf_node[viz_node.identity] = node

    def edge(self, viz_edge: VizEdge) -> None:
        """Add an edge to the underlying graph."""
        if viz_edge.end in self._graph:
            self._graph[viz_edge.end].append(viz_edge.start)
        else:
            self._graph[viz_edge.end] = [viz_edge.start]

    def get_layers(self):
        layers = []
        for id, in_ids in self._graph.items():
            layer = {}
            hf_node = self._id_to_hf_node[id]
            layer['name'] = hf_node.name
            layer['class_name'] = hf_node.type
            layer['inbound_nodes'] = []
            dtype_hint = ""
            for in_id in in_ids:
                in_hf_node = self._id_to_hf_node[in_id]
                item = [in_hf_node.name, 0, 0, {}]
                layer['inbound_nodes'].append(item)
                if 'dtype' in in_hf_node.params.keys():
                    dtype_hint = in_hf_node.params['dtype']

            layer['config'] = {'name': hf_node.params['name_hint'] if 'name_hint' in hf_node.params.keys() else hf_node.name}
            for k, v in hf_node.params.items():
                if 'out_dtype' in k and v and v == ' ':
                    layer['config']['dtype'] = dtype_hint
                    continue
                if 'name_hint' in k or not v or len(v) == 0:
                    continue
                layer['config'][k] = v

            layers.append(layer)
        return layers

class Hdf5Plotter(Plotter):
    """Hdf5 graph plotter"""

    def __init__(self):
        self._name_to_graph = {}

    def _save_attr_to_group(self, group, name, data):
        bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

        # Expecting this to never be true.
        if bad_attributes:
            raise RuntimeError('The following attributes cannot be saved to HDF5 '
                               'file because they are larger than %d bytes: %s' %
                               (HDF5_OBJECT_HEADER_LIMIT, ', '.join(bad_attributes)))

        data_npy = np.asarray(data)

        num_chunks = 1
        chunked_data = np.array_split(data_npy, num_chunks)

        # This will never loop forever thanks to the test above.
        while any(x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data):
            num_chunks += 1
            chunked_data = np.array_split(data_npy, num_chunks)

        if num_chunks > 1:
            for chunk_id, chunk_data in enumerate(chunked_data):
                group.attrs['%s%d' % (name, chunk_id)] = chunk_data
        else:
            group.attrs[name] = data

    def create_graph(self, name):
        self._name_to_graph[name] = Hdf5Graph(name)
        return self._name_to_graph[name]

    def render(self, filename: str = None):
        for name in self._name_to_graph:
            if filename is None:
                filename = name
            f = h5py.File(filename + '.h5', mode='w')
            g = f.create_group('model_weights')
            g.attrs['backend'] = 'tvm.relay'.encode('utf8')
            g.attrs['tvm_version'] = "0.11".encode('utf8')

            layers = self._name_to_graph[name].get_layers()
            mod_cfg = {"class_name": "Model", "config": {"name": "model", 'layers': layers}}
            f.attrs['model_config'] = json.dumps(mod_cfg).encode('utf8')
            self._save_attr_to_group(g, 'layer_names', [layer['name'].encode('utf8') for layer in layers])
            f.close()


