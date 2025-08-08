import gdsfactory as gf
from kfactory import DPorts, kdb
from gdsfactory.pdk import get_layer
from gdsfactory.typings import (
        AngleInDegrees,
        LayerSpec,
        Position
    )
from kfactory.typings import MetaData

def add_port_with_info(
        self: gf.Component,
        name: str | None = None,
        center: Position | kdb.DPoint | None = None,
        width: float | None = None,
        orientation: AngleInDegrees = 0,
        layer: LayerSpec | None = None,
        port_type: str = "optical",
        info: dict[str, MetaData] | None = None,
    ):
    layer = get_layer(layer)
    x = float(center[0])
    y = float(center[1])
    trans = kdb.DCplxTrans(1, float(orientation), False, x, y)

    _port = DPorts(kcl=self.kcl, bases=self.ports.bases).create_port(
        name=name,
        width=width,
        layer=layer,
        port_type= port_type,
        dcplx_trans=trans,
        info=info
    )
    return _port

gf.Component.add_port_with_info = add_port_with_info