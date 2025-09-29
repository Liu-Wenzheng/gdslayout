import os
import yaml
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import gdsfactory as gf

from . import init_component
from .structure import Structure
from .layout import Ring_down_distance
from .utils import coupler

class Cluster():
    def __init__(self, name, config_folder, layer=(2, 0), safe_gap=10, config_edge_coupler="./config_edge_coupler.yaml"):
        self.name = name
        self.cluster_folder = f"./analysis/{name}/"
        os.makedirs(self.cluster_folder, exist_ok=True)
        
        self.config_folder = config_folder
        self.layer = layer
        with open(config_edge_coupler, 'r') as file:
            config_edge_coupler = yaml.safe_load(file)
            if 'in' not in config_edge_coupler and 'out' not in config_edge_coupler:
                self.edge_coupler_in, self.edge_offset_in, self.edge_coupler_efflen_in, self.edge_coupler_width_in, self.edge_coupler_port_length_in = coupler(config_edge_coupler, layer=self.layer)
                self.edge_coupler_out, self.edge_offset_out, self.edge_coupler_efflen_out, self.edge_coupler_width_out, self.edge_coupler_port_length_out = (self.edge_coupler_in, self.edge_offset_in, self.edge_coupler_efflen_in, self.edge_coupler_width_in, self.edge_coupler_port_length_in)
            else:
                self.edge_coupler_in, self.edge_offset_in, self.edge_coupler_efflen_in, self.edge_coupler_width_in, self.edge_coupler_port_length_in = coupler(config_edge_coupler, side='in', layer=self.layer)
                self.edge_coupler_out, self.edge_offset_out, self.edge_coupler_efflen_out, self.edge_coupler_width_out, self.edge_coupler_port_length_out = coupler(config_edge_coupler, side='out', layer=self.layer)

        self.safe_gap = safe_gap

    def load(self, config_id_list):
        self.num = len(config_id_list)

        if self.num >= 3:
            self.config_id_list = np.array(config_id_list)[::-1]
        else:
            self.config_id_list = np.array(config_id_list)
        
        self.device_xsize = np.zeros(self.num)
        self.component_ysize = np.zeros(self.num)
        self.δy_device_coupler = np.zeros(self.num)

        self.component = []
        self.device_path = []
        for i, config_id in enumerate(self.config_id_list):

            structure = Structure(config_id=config_id, config_folder=self.config_folder)
            component = structure.build()

            self.component.append(component)
            self.device_path.append(structure.device_path)
            self.component_ysize[i] = component.ysize
            self.device_xsize[i] = structure.device.xsize
            self.δy_device_coupler[i] = component.ymax - component.ports['in'].y - component.ports['in'].width/2


    def taper_connect(self, cluster, component_ref, port_name, taper, taper_length):
        port_info = component_ref.ports[port_name].info
        if port_info.get('etch'):
            etched_taper = gf.Component()
            u, b = port_info['etch_width_u'], port_info['etch_width_b']
            etch_box = gf.components.rectangle(size=(taper_length, u+b), centered=True, layer=self.layer)
            etched_taper.add_ref(etch_box).move([taper.center[0]-etch_box.center[0], taper.center[1]-etch_box.center[1]+(u-b)/2])
            final_taper = gf.boolean(A=etched_taper, B=taper, operation='A-B', layer=self.layer)
            final_taper.add_ports(taper.ports)

            taper_ref = cluster << final_taper
        else:
            taper_ref = cluster << taper
        return taper_ref


    def build_layout_1(self, x_border, y_border, length_tot, y_drift, tot_drift):
        if self.device_xsize[0] > length_tot-x_border[0]-x_border[1]:
            raise ValueError("Insufficient space to fit the cluster!")
        
        cluster = gf.Component()

        x_s = tot_drift[0] + (length_tot+x_border[0]-x_border[1])/2
        y_max = tot_drift[1]-(y_drift + y_border[0])

        # STRUCTURE
        component_ref = cluster.add_ref(self.component[0])
        component_ref.move(origin=component_ref.ports['device_center'].center, destination=(x_s, y_max-(component_ref.ymax - component_ref.ymin)/2))

        # TAPER IN
        taper_in_length = component_ref.ports['in'].center[0] - self.edge_coupler_efflen_in - tot_drift[0]
        taper_in = gf.components.taper(length=taper_in_length, width1=self.edge_coupler_width_in, width2=component_ref.ports['in'].width, layer=self.layer)
        taper_in_ref = self.taper_connect(cluster, component_ref, 'in', taper_in, taper_in_length)
        taper_in_ref.connect('o2', component_ref.ports['in'])

        # TAPER OUT
        taper_out_length = -component_ref.ports['out'].center[0] - self.edge_coupler_efflen_out + tot_drift[0] + length_tot
        taper_out = gf.components.taper(length=taper_out_length, width1=component_ref.ports['out'].width, width2=self.edge_coupler_width_out, layer=self.layer)
        taper_out_ref = self.taper_connect(cluster, component_ref, 'out', taper_out, taper_out_length)
        taper_out_ref.connect('o1', component_ref.ports['out'])

        y_len = cluster.ymax - cluster.ymin

        # L EDGE COUPLER
        if self.edge_coupler_in is not None:

            in_coupler = cluster << self.edge_coupler_in
            in_coupler.rotate(180)
            in_coupler.move(origin=(in_coupler.xmin, in_coupler.y),destination=(-self.edge_offset_in + tot_drift[0], taper_in_ref.ports['o1'].y))

            if not hasattr(self.edge_coupler_in, "ports") or "coupler" not in self.edge_coupler_in.ports:
                sig = gf.components.straight(length=self.edge_coupler_port_length_in, cross_section=gf.cross_section.strip(width=self.edge_coupler_width_in, layer=self.layer))
                sig_ref = cluster << sig
                sig_ref.connect('o1', in_coupler.ports['out'])
                sig_ref.movex(-self.edge_coupler_port_length_in)
            else:
                in_coupler.connect('coupler', taper_in_ref.ports['o1'])
            
            text_pos_in = np.array([in_coupler.x, in_coupler.ymax + 10])
        else:
            text_pos_in = np.array([taper_in_ref.ports['o1'].x, taper_in_ref.ymax + 10])

        if self.edge_coupler_out is not None:
            out_coupler = cluster << self.edge_coupler_out
            out_coupler.move(origin=(out_coupler.xmax, out_coupler.y), destination=(self.edge_offset_out + tot_drift[0] + length_tot, taper_out_ref.ports['o2'].y))

            if not hasattr(self.edge_coupler_out, "ports") or "coupler" not in self.edge_coupler_out.ports:
                sig2 = gf.components.straight(length=self.edge_coupler_port_length_out, cross_section=gf.cross_section.strip(width=self.edge_coupler_width_out, layer=self.layer))
                sig2_ref = cluster << sig2
                sig2_ref.connect('o1', out_coupler.ports['out'])
                sig2_ref.movex(self.edge_coupler_port_length_out)
            
            text_pos_out = np.array([out_coupler.x, out_coupler.ymax + 10])
        else:
            text_pos_out = np.array([taper_out_ref.ports['o2'].x, taper_out_ref.ymax + 10])

        self.text_pos = np.array([np.concatenate((text_pos_in, text_pos_out))])

        return cluster, y_len


    def build_layout_2(self, x_border, y_border, length_tot, y_drift, tot_drift):
        """if np.sum(self.device_xsize) > length_tot-x_border[0]-x_border[1]:
            raise ValueError("Insufficient space to fit the cluster!")"""
        
        sub_cluster = [gf.Component(), gf.Component()]

        x_s = tot_drift[0] + x_border[0] + self.device_xsize/2
        y_max = tot_drift[1]-(y_drift + y_border[0])
        y_flip = -self.safe_gap + y_max-np.max(self.component_ysize)/2
        ymax_arr = []
        self.text_pos = np.zeros((2, 4))

        for i in range(self.num):
            # STRUCTURE
            component_ref = sub_cluster[i].add_ref(self.component[i])
            component_ref.move(origin=(component_ref.ports['device_center'].x, component_ref.ports['in'].y), destination=(x_s[i], y_max))

            if i == 0:
                taper_in_length = component_ref.ports['in'].center[0] - self.edge_coupler_efflen_in - tot_drift[0]
                taper_out_length = -component_ref.ports['out'].center[0] - self.edge_coupler_efflen_out + tot_drift[0] + length_tot
            else:
                taper_in_length = component_ref.ports['in'].center[0] - self.edge_coupler_efflen_out - tot_drift[0]
                taper_out_length = -component_ref.ports['out'].center[0] - self.edge_coupler_efflen_in + tot_drift[0] + length_tot

            # TAPER IN
            taper_in = gf.components.taper(length=taper_in_length, width1=self.edge_coupler_width_in, width2=component_ref.ports['in'].width, layer=self.layer)
            taper_in_ref = self.taper_connect(sub_cluster[i], component_ref, 'in', taper_in, taper_in_length)
            taper_in_ref.connect('o2', component_ref.ports['in'])

            # TAPER OUT
            taper_out = gf.components.taper(length=taper_out_length, width1=component_ref.ports['out'].width, width2=self.edge_coupler_width_out, layer=self.layer)
            taper_out_ref = self.taper_connect(sub_cluster[i], component_ref, 'out', taper_out, taper_out_length)
            taper_out_ref.connect('o1', component_ref.ports['out'])

            ymax_arr.append(sub_cluster[i].ymax)
            
            # L EDGE COUPLER
            if self.edge_coupler_in is not None:
                in_coupler = sub_cluster[i] << self.edge_coupler_in
                if i == 0:
                    in_coupler.rotate(180)
                    in_coupler.move(origin=(in_coupler.xmin, in_coupler.y),destination=(-self.edge_offset_in + tot_drift[0], taper_in_ref.ports['o1'].y))
                else:
                    in_coupler.move(origin=(in_coupler.xmax, in_coupler.y), destination=(self.edge_offset_out + tot_drift[0] + length_tot, taper_out_ref.ports['o2'].y))

                if not hasattr(self.edge_coupler_in, "ports") or "coupler" not in self.edge_coupler_in.ports:
                    sig = gf.components.straight(length=self.edge_coupler_port_length_in, cross_section=gf.cross_section.strip(width=self.edge_coupler_width_in, layer=self.layer))
                    sig_ref = sub_cluster[i] << sig
                    sig_ref.connect('o1', in_coupler.ports['out'])
                    sig_ref.movex(-self.edge_coupler_port_length_in)
                else:
                    in_coupler.connect('coupler', taper_in_ref.ports['o1']) if i == 0 else in_coupler.connect('coupler', taper_out_ref.ports['o2'])
                
                text_pos_in = np.array([in_coupler.x, in_coupler.ymax + 10]) if i == 0 else np.array([2*tot_drift[0]+length_tot - in_coupler.x, in_coupler.ymax + 10 + 2*(y_flip - in_coupler.y )])
            else:
                text_pos_in = np.array([tot_drift[0], taper_in_ref.ymax + 10]) if i == 0 else np.array([tot_drift[0], taper_out_ref.ymax + 10 + 2*(y_flip - taper_out_ref.y )])

            # R EDGE COUPLER
            if self.edge_coupler_out is not None:
                out_coupler = sub_cluster[i] << self.edge_coupler_out
                if i == 0:
                    out_coupler.move(origin=(out_coupler.xmax, out_coupler.y), destination=(self.edge_offset_out + tot_drift[0] + length_tot, taper_out_ref.ports['o2'].y))
                else:
                    out_coupler.rotate(180)
                    out_coupler.move(origin=(out_coupler.xmin, out_coupler.y),destination=(-self.edge_offset_in + tot_drift[0], taper_in_ref.ports['o1'].y))

                if not hasattr(self.edge_coupler_out, "ports") or "coupler" not in self.edge_coupler_out.ports:
                    sig2 = gf.components.straight(length=self.edge_coupler_port_length_out, cross_section=gf.cross_section.strip(width=self.edge_coupler_width_out, layer=self.layer))
                    sig2_ref = sub_cluster[i] << sig2
                    sig2_ref.connect('o1', out_coupler.ports['out'])
                    sig2_ref.movex(self.edge_coupler_port_length_out)
                else:
                    out_coupler.connect('coupler', taper_out_ref.ports['o2']) if i == 0 else out_coupler.connect('coupler', taper_in_ref.ports['o1'])

                text_pos_out = np.array([out_coupler.x, out_coupler.ymax + 10]) if i == 0 else np.array([2*tot_drift[0]+length_tot - out_coupler.x, out_coupler.ymax + 10 + 2*(y_flip - out_coupler.y )])
                print("out_coupler: ",i, out_coupler.x)
            else:
                text_pos_out = np.array([tot_drift[0]+length_tot, taper_out_ref.ymax + 10]) if i == 0 else np.array([tot_drift[0]+length_tot, taper_in_ref.ymax + 10 + 2*(y_flip - taper_in_ref.y )])
            
            self.text_pos[i] = np.concatenate((text_pos_in, text_pos_out))
            
        sub_cluster[1] = sub_cluster[1].rotate(angle=180, center=(tot_drift[0] + (length_tot+x_border[0]-x_border[1])/2, y_flip))
        return (sub_cluster[0], sub_cluster[1]), np.max(self.component_ysize)+self.edge_coupler_width_in+2*self.safe_gap


    def build_layout_n(self, x_border, y_border, length_tot, y_drift, edge_coupler_distance, tot_drift, auto=(True, False), manual=([], [])):
        """if np.sum(self.device_xsize) > length_tot-x_border[0]-x_border[1]:
            raise ValueError("Insufficient space to fit the cluster!")"""
        
        cluster = gf.Component()
        
        length_eff = length_tot - x_border[0] - x_border[1]
        δl = (length_eff - np.sum(self.device_xsize))/(self.num-1)
        cavity_cumsum_stretch = np.concatenate(([0], np.cumsum(self.device_xsize)))

        x_s = tot_drift[0]+x_border[0]+np.array([cavity_cumsum_stretch[i]+self.device_xsize[i]/2+i*δl for i in range(self.num)])
        self.δy_cumsum = np.cumsum(np.maximum(0, self.δy_device_coupler[::-1] + self.safe_gap - edge_coupler_distance))
        y_max = np.array([tot_drift[1]-(y_drift + y_border[0]+i*edge_coupler_distance+self.δy_cumsum[i]) for i in range(self.num)])[::-1]
        component_shift = np.array([[x_s[i]-self.component[i].ports['device_center'].x, y_max[i]-self.component[i].ports['in'].y] for i in range(self.num)])

        if any(auto):
            component_gap = np.zeros((self.num-1, 2))
            
            # Use tqdm for progress bar without multiprocessing due to serialization issues
            for i in tqdm(range(self.num-1), desc="Calculating component gaps"):
                dist = Ring_down_distance(self.component[i], self.component[i+1], shift1=component_shift[i], shift2=component_shift[i+1], layer=self.layer)
                component_gap[i] = dist.calculate(type='exp', alpha=100)
            component_gap_ave = np.mean(component_gap, axis=0)
            if auto[0]:
                if component_gap_ave[0] < 0:
                    raise ValueError("Insufficient space to fit the cluster!")
                elif component_gap_ave[0] < self.safe_gap:
                    print(f"The average gap is {component_gap_ave[0]} um, less than the safe gap {self.safe_gap} um, which may lead to cross-talk.")

                component_shift_x = np.cumsum(component_gap_ave[0] - component_gap[:, 0])[:-1]
                component_shift[1:-1, 0] += component_shift_x
            if auto[1]:
                extra_shift_y = np.zeros(self.num-1)
                for i in range(self.num-1):
                    if component_gap[i, 1] < self.safe_gap and np.linalg.norm(component_gap[i]) < self.safe_gap:
                        extra_shift_y[i] = self.safe_gap - component_gap[i, 1]
                component_shift_y = np.cumsum(extra_shift_y[::-1])[::-1]
                component_shift[:-1, 1] -= component_shift_y

        elif any(manual):
            if manual[0]:
                if len(manual[0]) != self.num-2:
                    raise ValueError(f"Manual shift x should be a list of length {self.num-1}.")
                component_shift[1:-1, 0] += manual[0]
            if manual[1]:
                if len(manual[1]) != self.num:
                    raise ValueError(f"Manual shift y should be a list of length {self.num}.")
                component_shift[:, 1] += manual[1]

        self.text_pos = np.zeros((self.num, 4))
        ymax_arr = []
        ymin_arr = []
        for i in range(self.num):
            # STRUCTURE
            component_ref = cluster.add_ref(self.component[i])
            component_ref.move(component_shift[i])
            
            # TAPER IN
            taper_in_length = component_ref.ports['in'].center[0] - self.edge_coupler_efflen_in - tot_drift[0]
            taper_in = gf.components.taper(length=taper_in_length, width1=self.edge_coupler_width_in, width2=component_ref.ports['in'].width, layer=self.layer)
            taper_in_ref = self.taper_connect(cluster, component_ref, 'in', taper_in, taper_in_length)
            taper_in_ref.connect('o2', component_ref.ports['in'])

            # TAPER OUT
            taper_out_length = -component_ref.ports['out'].center[0] - self.edge_coupler_efflen_out + tot_drift[0] + length_tot
            taper_out = gf.components.taper(length=taper_out_length, width1=component_ref.ports['out'].width, width2=self.edge_coupler_width_out, layer=self.layer)
            taper_out_ref = self.taper_connect(cluster, component_ref, 'out', taper_out, taper_out_length)
            taper_out_ref.connect('o1', component_ref.ports['out'])

            ymax_arr.append(np.max([component_ref.ymax, taper_in_ref.ymax]))
            ymin_arr.append(np.min([component_ref.ymin, taper_out_ref.ymin]))

            # L EDGE COUPLER
            if self.edge_coupler_in is not None:

                in_coupler = cluster << self.edge_coupler_in
                in_coupler.rotate(180)
                in_coupler.move(origin=(in_coupler.xmin, in_coupler.y),destination=(-self.edge_offset_in + tot_drift[0], taper_in_ref.ports['o1'].y))

                if not hasattr(self.edge_coupler_in, "ports") or "coupler" not in self.edge_coupler_in.ports:
                    sig = gf.components.straight(length=self.edge_coupler_port_length_in, cross_section=gf.cross_section.strip(width=self.edge_coupler_width_in, layer=self.layer))
                    sig_ref = cluster << sig
                    sig_ref.connect('o1', in_coupler.ports['out'])
                    sig_ref.movex(-self.edge_coupler_port_length_in)
                else:
                    in_coupler.connect('coupler', taper_in_ref.ports['o1'])
                
                text_pos_in = np.array([in_coupler.x, in_coupler.ymax + 10])
            else:
                text_pos_in = np.array([taper_in_ref.ports['o1'].x, taper_in_ref.ymax + 10])

            # R EDGE COUPLER
            if self.edge_coupler_out is not None:
                out_coupler = cluster << self.edge_coupler_out
                out_coupler.move(origin=(out_coupler.xmax, out_coupler.y), destination=(self.edge_offset_out + tot_drift[0] + length_tot, taper_out_ref.ports['o2'].y))

                if not hasattr(self.edge_coupler_out, "ports") or "coupler" not in self.edge_coupler_out.ports:
                    sig2 = gf.components.straight(length=self.edge_coupler_port_length_out, cross_section=gf.cross_section.strip(width=self.edge_coupler_width_out, layer=self.layer))
                    sig2_ref = cluster << sig2
                    sig2_ref.connect('o1', out_coupler.ports['out'])
                    sig2_ref.movex(self.edge_coupler_port_length_out)
                
                text_pos_out = np.array([out_coupler.x, out_coupler.ymax + 10])
            else:
                text_pos_out = np.array([taper_out_ref.ports['o2'].x, taper_out_ref.ymax + 10])

            self.text_pos[i] = np.concatenate((text_pos_in, text_pos_out))

        return cluster, max(ymax_arr)-min(ymin_arr)


    def add_text(self, cluster, id, ref, size=20):
        text = self.text_pos[id]

        in_discrip_text_ref = gf.components.text(text=f'{self.config_id_list[id]}', size=size, justify='center', layer=self.layer)
        in_discrip_text_ref = cluster << in_discrip_text_ref
        # Position text directly
        in_discrip_text_ref.x = max(text[0], ref[0] + 2 * size)
        in_discrip_text_ref.ymin = text[1]

        out_discrip_text_ref = gf.components.text(text=f'{self.config_id_list[id]}', size=size, justify='center', layer=self.layer)
        out_discrip_text_ref = cluster << out_discrip_text_ref
        # Position text directly
        out_discrip_text_ref.x = min(text[2], ref[1] - 2 * size)
        out_discrip_text_ref.ymin = text[3]

        return cluster


    def build_layout(self, Chip, x_border, y_border, length_tot, y_drift, edge_coupler_distance, tot_drift, text=True, auto=(True, False), manual=([], [])):
        ref = [tot_drift[0], tot_drift[0]+length_tot]
        if self.num == 1:
            cluster, y_len = self.build_layout_1(x_border, y_border, length_tot, y_drift, tot_drift)
            if text:
                cluster = self.add_text(cluster, id=0, ref=ref)
            Chip << cluster

        elif self.num == 2:
            clusters, y_len = self.build_layout_2(x_border, y_border, length_tot, y_drift, tot_drift)
            for i, dev in enumerate(clusters):
                if text:
                    dev = self.add_text(dev, id=i, ref=ref)
                Chip << dev

        else:
            cluster, y_len = self.build_layout_n(x_border, y_border, length_tot, y_drift, edge_coupler_distance, tot_drift, auto=auto, manual=manual)
            if text:
                for i in range(self.num):
                    cluster = self.add_text(cluster, id=i, ref=ref)
            Chip << cluster

        y_len += y_border[0] + y_border[1]
        return Chip, y_len


    def _curvature(self):
        for p, id in zip(self.device_path, self.config_id_list):
            s, K = p.curvature()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(s, K)
            ax.set_title(f"Device {id}  length: {p.length()}um  FSR: {147000/p.length()}GHz")
            plt.savefig(self.cluster_folder + f'{id}_curvature.png')
            plt.close(fig)