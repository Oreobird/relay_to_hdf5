
from tvm.relay.testing import mlp
from tvm.contrib import relay_viz
from relay_viz_hdf5 import Hdf5Plotter, Hdf5VizParser

mod, param = mlp.get_workload(batch_size=1, num_classes=10)
print("mod:{}\n\n".format(mod))

viz = relay_viz.RelayVisualizer(
    mod,
    relay_param=param,
    plotter=Hdf5Plotter(),
    parser=Hdf5VizParser()
    )

viz.render('mlp')

