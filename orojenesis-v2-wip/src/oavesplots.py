# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cairo
import math
import pickle
import pandas as pd
import io
import re
import socket
from collections import OrderedDict


def parse_oaves(pkl: str):
    df = pickle.load(open(pkl, "rb"))
    print(df.columns)
    cols = df.columns.tolist()
    cols[0] = "operational intensity"
    cols[1] = "footprint"
    cols[2] = "accesses"
    cols[-2] = "mapping"
    df.columns = cols
    return df


def plot_mapping(mapping: str, max_pixels=1000):
    belowdram_mapping = mapping[re.search('L\d+\[\w+\]', mapping).end():]

    # Names of tensor in outermost (first) L
    dram_tensors = list(re.findall('L\d+\[([ABD]+)\]', mapping)[0])
    # print(belowdram_mapping)
    dims = {}
    tiledims = {}
    for rank in ['M','N','K']:
        dims[rank] = 1
        tiledims[rank] = {}
        # print(rank)
        for count in re.findall(f"{rank}(\d+)", mapping):
            dims[rank] *= int(count)
        # Multiply every number seen past the first instance of that rank
        for matrix in ["A","B","D"]:
            tiledims[rank][matrix] = 1
            # print("".join(belowdram_mapping.split(matrix)[1:]))
            for count in re.findall(f"{rank}(\d+)", "".join(belowdram_mapping.split(matrix)[1:])):
                tiledims[rank][matrix] *= int(count)

    # Figure out how to draw arrows.  Keep track of last loop seen before variable is stored
    def find_rightmost_pattern(s, tensor, ranks_re='[MNK]'):
        later_index = s.find(tensor)
        matches = re.finditer(fr'{ranks_re}\d+', s)
        filtered_patterns = [(m.span()[0], m.group()) for m in matches if m.span()[0] < later_index]
        return filtered_patterns[-1] if filtered_patterns else (0,"")

    output = io.BytesIO()

    # Need to calculate aspect ratio before creating the output image so that I can set the correct height
    M = dims["M"]
    N = dims["N"]
    K = dims["K"]
    AK = tiledims["K"]["A"]
    AM = tiledims["M"]["A"]
    DM = tiledims["M"]["D"]
    DN = tiledims["N"]["D"]
    BN = tiledims["N"]["B"]
    BK = tiledims["K"]["B"]
    # Derive bytes (otherwise in elements)
    are = re.findall("Z(\d+)X", mapping)
    bre = re.findall("X(\d+)X", mapping)
    dre = re.findall("Y(\d+)X", mapping)
    if are and bre and dre:
        abytes = int(are[0])/8
        bbytes = int(bre[0])/8
        dbytes = int(dre[0])/8
    else:
        abytes = 1
        bbytes = 1 
        dbytes = 1

    # max dimension is 1000 pixels
    max_count_height = M+K
    max_count_width = N+K
    xms = int(max_count_width/20)
    yms = int(max_count_height/20)
    width_elements = max_count_width+3*xms
    height_elements = max_count_height+3*yms
    yoff = yms

    if max_count_width > max_count_height:
        width_pixels = max_pixels
        x_pixel_per_element = width_pixels/(width_elements)
        height_pixels = height_elements/width_elements*width_pixels
        y_pixel_per_element = height_pixels / height_elements
        ms = xms
    else:
        height_pixels = max_pixels
        y_pixel_per_element = height_pixels/(height_elements)
        width_pixels = width_elements/height_elements*height_pixels
        x_pixel_per_element = width_pixels / width_elements
        ms = yms
    # print(ms)
    #print(f"{x_pixel_per_element=},{y_pixel_per_element=},{width_pixels=},{height_pixels=},{max_count_width=},{max_count_height=}")
    def xscale(pixels):
        return x_pixel_per_element*pixels
    def yscale(pixels):
        return y_pixel_per_element*pixels


    surface = cairo.SVGSurface(output, width_pixels, height_pixels)
    # print(f"Surface {width_pixels}x{height_pixels}")
    ctx = cairo.Context(surface)
    ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(xscale(ms/2))
    ctx.set_line_width(xscale(ms/30))

    rgb_red = (0.770, 0.116, 0.0154)
    rgb_green = (0.263, 0.529, 0.216)
    rgb_black = (0, 0, 0)
    rgb_gray = (0.5, 0.5, 0.5)
    rgb_blue = (0.445, 0.651, 0.780)
    rgb_darkblue = (0.129, 0.357, 0.502)

    def draw_label(string, pos, theta = 0.0, face = 'Sans'):
        ctx.save()
        
        # build up an appropriate font
        #ctx.select_font_face(face , cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        #ctx.set_font_size(font_size)
        fascent, fdescent, fheight, fxadvance, fyadvance = ctx.font_extents()
        x_off, y_off, tw, th = ctx.text_extents(string)[:4]
        nx = -tw/2.0
        ny = fheight/2

        ctx.translate(pos[0], pos[1])
        ctx.rotate(theta)
        ctx.translate(nx, ny)
        ctx.move_to(0,0)
        ctx.show_text(string)

        ctx.restore()
                
    def draw_arrow(start_x, start_y, end_x, end_y, color, width, label=None):

        start_x_scaled = xscale(start_x)
        end_x_scaled = xscale(end_x)
        start_y_scaled = yscale(start_y)
        end_y_scaled = yscale(end_y)
        
        # Set the color
        ctx.set_source_rgb(*color)
        
        # Set the line width
        ctx.set_line_width(yscale(width))
        
        # Draw the line
        ctx.move_to(start_x_scaled, start_y_scaled)
        ctx.line_to(end_x_scaled, end_y_scaled)
        ctx.stroke()
        
        # Calculate the angle of the arrow
        angle = math.atan2(end_y_scaled - start_y_scaled, end_x_scaled - start_x_scaled)
        
        # Draw the arrowhead
        arrow_length = yscale(width)*5
        arrow_angle = math.pi / 6
        
        x1 = end_x_scaled - arrow_length * math.cos(angle - arrow_angle)
        y1 = end_y_scaled - arrow_length * math.sin(angle - arrow_angle)
        x2 = end_x_scaled - arrow_length * math.cos(angle + arrow_angle)
        y2 = end_y_scaled - arrow_length * math.sin(angle + arrow_angle)
        
        ctx.move_to(end_x_scaled,end_y_scaled)
        ctx.line_to(x1, y1)
        ctx.line_to(x2, y2)
        ctx.close_path()
        ctx.fill()

        if label:
            ctx.set_source_rgb(*rgb_black)
            draw_label(str(label), ((start_x_scaled+end_x_scaled)/2,(start_y_scaled+end_y_scaled)/2))

        

    def labeled_matrix(ll, ur, width, height, outline=False):
        #print(f"matrix at {ll=}:{xscale(ll)=} {ur=}:{yscale(ur)=}")
       
        ctx.rectangle(xscale(ll), yscale(ur), xscale(width), yscale(height))
        if outline:
            # Red outline
            if outline:
                ctx.set_source_rgb(*outline)
            ctx.stroke() 
            toff = -1/2
        else:
            # Blue fill
            ctx.set_source_rgb(*rgb_blue)
            ctx.fill() 
            toff = 1/2
            # Blue text
            ctx.set_source_rgb(*rgb_darkblue)
        
        draw_label(f"{width}", (xscale(ll+width/2), yscale(ur+toff*yms)))
        draw_label(f"{height}", (xscale(ll+toff*xms), yscale(ur+height/2)), theta=-math.pi/2)
        #ctx.show_text(f"{height}")
        
    # print(f"scale {pixel_per_element*(N+K)+ms} by {pixel_per_element*(M+K)+ms}")
    #ctx.scale(x_pixel_per_element*(N+K+3*ms), y_pixel_per_element*(M+K+3*ms))
    #ctx.scale(width_pixels,height_pixels)
    #print(f"{xms=},{yms=},{ms=}")


    # Draw A full matrix (x,y,width,height)
    labeled_matrix(xms, K+2*yms+yoff, K, M)
    # Draw D full matrix
    labeled_matrix(K+2*xms, K+2*yms+yoff, N, M)
    # Draw B full matrix
    labeled_matrix(K+2*xms, yms+yoff, N, K)
    # Draw A tile
    labeled_matrix(K-AK+xms, K+2*yms+yoff, AK, AM, outline=rgb_red if 'A' in dram_tensors else rgb_green)
    # Draw D tile
    labeled_matrix(K+2*xms, K+2*yms+yoff, DN, DM, outline=rgb_red if 'D' in dram_tensors else rgb_green)
    # Draw B tile
    labeled_matrix(K+2*xms, (K-BK)+yms+yoff, BN, BK, outline=rgb_red if 'B' in dram_tensors else rgb_green)


    def rank_tuples(tuples_list):
        sorted_elements = sorted(set([x for x in tuples_list]), reverse=True)
        rank_dict = {element: rank + 1 for rank, element in enumerate(sorted_elements)}
        ranked_tuples = [(rank_dict[x], x[1]) for x in tuples_list]
        return ranked_tuples

    Aarrow, Barrow, Darrow = rank_tuples([find_rightmost_pattern(belowdram_mapping, 'A', '[MK]'),
    find_rightmost_pattern(belowdram_mapping, 'B', '[NK]'),
    find_rightmost_pattern(belowdram_mapping, 'D', '[NM]')])

    # Draw A arrow
    if 'K' in Aarrow[1]:
        draw_arrow(K-AK+xms,K+2*yms+yoff+AM/2,xms,K+2*yms+yoff+AM/2,rgb_gray,yms/10,label=Aarrow[0])
    elif 'M' in Aarrow[1]:
        draw_arrow(K-AK+xms+AK/2,K+2*yms+yoff+AM,K-AK+xms+AK/2,K+2*yms+yoff+M,rgb_gray,yms/10,label=Aarrow[0])
    # Draw B arrow
    if 'K' in Barrow[1]:
        draw_arrow(K+2*xms+BN/2,(K-BK)+yms+yoff,K+2*xms+BN/2,yms+yoff,rgb_gray,yms/10,label=Barrow[0])
    elif 'N' in Barrow[1]:
        draw_arrow(K+2*xms+BN,(K-BK)+yms+yoff+BK/2,K+2*xms+N,(K-BK)+yms+yoff+BK/2,rgb_gray,yms/10,label=Barrow[0])
    # Draw B arrow
    if 'M' in Darrow[1]:
        draw_arrow(K+2*xms+DN/2,K+2*yms+yoff+DM,K+2*xms+DN/2,K+2*yms+yoff+M,rgb_gray,yms/10,label=Darrow[0])
    elif 'N' in Darrow[1]:
        draw_arrow(K+2*xms+DN,K+2*yms+yoff+DM/2,K+2*xms+N,K+2*yms+yoff+DM/2,rgb_gray,yms/10,label=Darrow[0])
    # print(f"{Aarrow=}, {Barrow=}, {Darrow=}")


    # Provide size information
    ctx.set_source_rgb(*rgb_black)
    ctx.set_font_size(yscale(yms/3))
    draw_label(f"DRAM: A = {K*M*abytes/10**6:.0f}MB B = {K*N*bbytes/10**6:.0f}MB D = {N*M*dbytes/10**6:.0f}MB - {(K*M*abytes+K*N*bbytes+N*M*dbytes)/10**6:.0f} MB total", (xscale(width_elements/2),yscale(yoff/2)))
    draw_label(f"L2: A = {AK*AM*abytes/10**3:.0f}KB, B = {BK*BN*bbytes/10**3:.0f}KB, D = {DN*DM*dbytes/10**3:.0f}KB - {(AK*AM*abytes+BK*BN*bbytes+DN*DM*dbytes)/10**3:.0f} KB total", (xscale(width_elements/2),yscale(yoff)))

    surface.finish()
    surface.flush()
    return output

import psutil
from typing import Optional

def get_next_free_port(
    start_port: int, 
    end_port: Optional[int] = None, 
    check_n_ports: Optional[int] = 100,
) -> int:

    used_ports = [nc.laddr.port for nc in psutil.net_connections()]

    if end_port is None:
        end_port = start_port + check_n_ports

    for port in range(start_port, end_port):
        if port not in used_ports:
            return port


def interactive_plot(fig):
    from dash import Dash, dcc, html, Input, Output, no_update
    import plotly.graph_objects as go
    import base64

    app = Dash()

    server = app.server

    # Layout of the app
    app.layout = html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure=fig
        ),
        html.Div(id='images-div')
    ])


    # Callback to update the image based on the clicked point
    @app.callback(
        Output('images-div', 'children'),
        Input('scatter-plot', 'clickData')
    )

    def display_image(clickData, max_pixels=500):

        if clickData is None:
            return []

        # demo only shows the first point, but other points may also be available
        #print(clickData)
        pt = clickData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]

        images = []

        # Support multiple mappings (each hover column)
        mappings = [x for x in list(pt["customdata"]) if type(x) == str and 'L' in x]
        for mapping in mappings:
            print(mapping)

            img_bytes = plot_mapping(mapping, max_pixels=max_pixels)
            # Convert the BytesIO object to a base64-encoded string
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode('ascii')
            src = 'data:image/svg+xml;base64,{}'.format(img_base64)
            images.append(html.Img(src=src, style={'display': 'inline', 'margin': '10px'}))
        
        return images

    sock = socket.socket()
    sock.bind(('', 0))
    portnum = sock.getsockname()[1]
    app.run(debug=True, port=get_next_free_port(8888), host=socket.gethostname())
