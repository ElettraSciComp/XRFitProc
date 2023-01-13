#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 09:49:09 2021

@author: matteo
"""

import numpy as np
import multiprocessing as mp
import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash_bootstrap_components as dbc
import FitTools as ft
# import time
import xraydb
import os
import pickle
from flask import redirect


class class_XRFitProc_webapp:        
    def __init__(self):
        
        # define app class attributes
        self.app = dash.Dash(name='XRFitProc',external_stylesheets = [dbc.themes.MATERIA])
        self.app.title = "XRFitProc"
                        
        # hardcoded initialisations
        self.n_lines = 100
        self.n_maps = 30
        self.PTE_indexing = ft.gen_PTE_indexing()
        self.tmp_fld, self.res_fld, self.input_fld = ft.organise_folder_structure()
        self.h5_path = os.path.join(self.input_fld,'xrf_flat_scan_aligned_2fit.h5')
        xrdb = xraydb.XrayDB()
        
        print(os.getcwd())
        
        self.temp_cmap = self.app.get_asset_url('temperature.cmap')
        self.fs = 24
        self.auc_res = 700
        
        
        # create temperature colormap
        temp_cmap = ft.reg_temperature_plotly(self.temp_cmap,'temperature',255)
        
        # initialise page elemnts and content
        content = [dcc.Store(id='xrf_lines_list',data=[])]
        content += [dcc.Store(id='xrf_lines_colors_list',data=[])]
        content += [dcc.Store(id='element_list',data=[])]
        content += [dcc.Store(id='loaded_data', data={})] 
        content += [dcc.Store(id='info_dict')]
        content += [dcc.Store(id='run_batch_dummy',data=False)]
        content += [dcc.Store(id='h5_path',data='')]
        
        content += [dcc.Store(id='batchfit_trigger', data=False)]     
        content += [dcc.Store(id='updateflags_trigger', data=False)]   
        content += [dcc.Store(id='batchfit_running', data=False)]   
        content += [dcc.Store(id='batchfit_finished', data=True)]        
        
        content += [dcc.Store(id='tot_pixels', data=0)]
        content += [dcc.Store(id='fit_sum_exec_time', data=0.0)]
        content += [dcc.Store(id='tmp_h5_path',data='')]
        content += [dcc.Store(id='tmp_h5_singleLine_path',data='')]
        content += [dcc.Store(id='alert_color',data={'avail_colors':np.asarray(["success", "primary", "warning", "danger", "info", "dark"]), 'current_alert_color':'success'})]
        
        # # Setup Fitting Tabs
        tab0_content = dbc.Card(
            dbc.CardBody([
                dcc.Location(id='url', refresh=True),
                dbc.Row([
                    dbc.Col(dbc.Button('Load H5 Dataset',color='primary',id='loadata_button'), width={'size':"3"}, align='center')])
                ],style={'left': '30px'}),
            className="mt-3", outline=False, style={'maxWidth':'50%'},
        )  
        
        
        # Tab 1
        tab1_content_left = dbc.Container([
            dbc.Row(dbc.Col(dbc.Card([dbc.CardHeader(html.H1("Sum Spectra", className="card-title"), style={'text-align':'center'}),dbc.CardBody(dbc.Row(dcc.Graph(id='sum_graph')))], className="mt-3", outline=False, style={'maxWidth':'100%'}), align='start', width={'size':'max'})),
            dbc.Row(dbc.Col(dbc.Card([dbc.CardHeader(html.H1("Fit Setup", className="card-title"), style={'text-align':'center'}), dbc.CardBody([                    
                    dbc.Row([dbc.Col('Beam Energy [eV]:', width={'size':'2'}, align='center'),
                             dbc.Col(dbc.Input(id='beam_en_val', type='number', value=1750, size="md", className="mb-3"), width={'size':'1'}, align='center'),
                             dbc.Col([], width={'size':'1'}, align='center'), # column space
                             dbc.Col([], width={'size':'1'}, align='center'), # column space
                             dbc.Col('Snip Width:', width={'size':'2'}, align='center'),
                             dbc.Col(dbc.Input(id='snip_width_val', type='number', value=20, size="md", className="mb-3"), width={'size':'1'}, align='center')]),
                                        
                    dbc.Row([dbc.Col('Fit boundaries:', width={'size':'2'}, align='center'),
                             dbc.Col(dbc.Input(id='fit_boundaries_LOW_val', type='number', min=0, value=180, size="md", className="mb-3"), width={'size':'1'}, align='center'),
                             dbc.Col(dbc.Input(id='fit_boundaries_HIGH_val', type='number', min=0, value=1200, size="md", className="mb-3"), width={'size':'1'}, align='center'),
                             
                             dbc.Col([], width={'size':'1'}, align='center'),
                             dbc.Col('Snip Iter:', width={'size':'2'}, align='center'),
                             dbc.Col(dbc.Input(id='snip_iter_val', type='number', value=10, size="md", className="mb-3"), width={'size':'1'}, align='center')]),
                                        
                    ])],
                className="mt-3", outline=False, style={'maxWidth':'100%'}
                ), align='start', width={'size':'max'})),
            
            html.Br(),
            dbc.Row(dbc.Col(dbc.Button('FIT',color='primary',id='fit_sum_button'), width={'size':'auto'}), align='center')], fluid=True)
                
        
        tab1_content_right = dbc.Container([
            dbc.Row(dbc.Col(dbc.Card([dbc.CardHeader(html.H1("XRF Lines", className="card-title"), style={'text-align':'center'}),
                    dbc.CardBody([
                        
                        dbc.Row([dbc.Col([dcc.Dropdown(id="XRFlines_element_dropdown",options=ft.gen_PTE_list(),value='',style={'color':'black'})], width={'size':'6'}, align='center'),
                                 dbc.Col(dbc.Button('Add',color='primary',id='add_xrf_button'), width={'size':"1"}, align='center')]),
                        
                        html.Br(),
                        
                        html.Div([dbc.Row([dbc.Col(dbc.Checklist(options=[dict()],value=[idx],id=f"checklist_xrfalert_{idx}", style={'fontWeight':'bold'}), width={'size':"2"}),
                                           dbc.Col(dbc.Alert("Empty alert", id=f'xrf_alert_{idx}',dismissable=True,is_open=True), width={'size':"10"})]) for idx in range(self.n_lines)], id = 'XRF_lines_checklist_div', style={'display':'none'})
                        
                        ])],
                    className="mt-3", outline=False, style={'maxWidth':'100%'},
                    ), align='start', width={'size':'max'}),),
            
            ],fluid=True)
        
        
        # Tab 2
        tab2_content = dbc.Container([dbc.Card(
            dbc.CardBody([dbc.Row([dbc.Col(dbc.Button('Run Batch Fit',id='batchfit_button',color='primary', disabled=False), width={'size':"auto"}, align='center'),
                                   dbc.Col([dcc.Interval(id='progress_interval', n_intervals = 0, interval = 1000, disabled = True), dbc.Progress(id='progress_bar', value=0, style={'height':'20px', 'display':'block'}, animated=True, striped=True)], width={'size':"3"}, align='center')
                                   ]),
                          
                          html.Div([dbc.Row([dbc.Col(dcc.Graph(id='fitMap_showcase_graph', style={'width':'6', 'height':'auto','display':'block'}), width={'size':"6"}, align='center'), dbc.Col(dcc.Graph(id='fit_showcase_graph'), width={'size':"6"}, style={"height": "100%"}, align='center')]),
                                    dbc.Row([dbc.Label("Elements", html_for="element_slider",style={'font-size': self.fs}),dbc.Col(dcc.Slider(min=0,max=self.n_maps-1,step=None,id='element_slider',value=0,marks={str(idx): str(idx) for idx in range(self.n_maps)}),width='5',style={'font-size': self.fs})]),
                                    html.Br(),
                                    dbc.Row([dbc.Label("Display Range", html_for="displayRange_LOW_val", style={'font-size': self.fs}),
                                             dbc.Col(dbc.Input(id='displayRange_LOW_val', type='number', value=0), width={'size':'auto'}, align='center'),
                                             dbc.Col(dbc.Input(id='displayRange_HIGH_val', type='number', value=1), width={'size':'auto'}, align='center')])], style={'display':'none'}, id='batchFit_div'),
                                    
                          dbc.Row([dbc.Col(dbc.Button('Save Elemental Map as HDF',id='hdf5_out_button',color='primary', disabled=False), width={'size':"auto"}, align='center'),
                                   dbc.Col(dbc.Button('Save XRF Lines Map as HDF',id='hdf5_singleLines_out_button',color='primary', disabled=False), width={'size':"auto"}, align='center'),
                                   dbc.Col(dbc.Button('Save Elemental Map as TIFF',id='tiff_out_button',color='primary', disabled=False), width={'size':"3"}, align='center')]),
                          
                          ],style={'left': '30px'}),
            className="mt-3", outline=False, style={'maxWidth':'100%'},
        )],fluid=True)
        
                            
        
        tabs = [dbc.Tabs(
            [
                dbc.Tab(tab0_content, label="Data Load", label_style={'font-size': self.fs}),
                dbc.Tab(dbc.Row([dbc.Col(tab1_content_left, align='start', width={'size':'7'}), dbc.Col(tab1_content_right, align='start', width={'size':'5'})]), label="Fit Parameters", label_style={'font-size': self.fs}, disabled=True, id='fit_tab'),           
                dbc.Tab(tab2_content, label="Batch Fitting", label_style={'font-size': self.fs}, disabled=True, id='batchfit_tab'),
            ]
        )]
             
        
        
        
        
        # finalise app layout
        self.app.layout = html.Div(tabs+content) # 'flex-direction':'column-reverse'
        
        
        ### CALLBACK FUNCTIONS           
        outputs = [Output('fit_tab', 'disabled'),Output('loaded_data','data'),Output('beam_en_val', 'value'),Output('fit_boundaries_LOW_val','value'),Output('fit_boundaries_HIGH_val','value'),Output('snip_width_val','value'),Output('snip_iter_val','value')]
        outputs += [Output('fit_boundaries_LOW_val','min'),Output('fit_boundaries_HIGH_val','min'),Output('fit_boundaries_LOW_val','max'),Output('fit_boundaries_HIGH_val','max')]
        @self.app.callback(outputs,Input('loadata_button', 'n_clicks'),prevent_initial_call=True)
        def load_data_button(n_clicks):   
            data = ft.load_HDF_aligned(self.h5_path,['n_SDD', 'channel_SUM', 'slope', 'offset', 'beam_en', 'im_shape','nonzeroxrf'])  
            n_channels = np.shape(data['xrfdata'])[1]
            xdata = np.array([data['offset']*1000 + data['slope']*1000*idx for idx in range(n_channels)],float)            
            data.update({'xrfdata':np.sum(data['xrfdata'],0), 'tot_pixels':np.shape(data['xrfdata'])[0], 'xdata':xdata})
            
            # clean tmp dir
            tmp_h5_list = os.listdir(self.tmp_fld)
            for fname in tmp_h5_list:
                os.remove(os.path.join(self.tmp_fld,fname))            
            
            return [False, data, data['beam_en'], 0, n_channels-1, 20 , 10, 0, 0, n_channels-1, n_channels-1]
        
        
        @self.app.callback(Output('run_batch_dummy','data'),[Input('hdf5_out_button','n_clicks'),Input('hdf5_singleLines_out_button','n_clicks'),Input('tiff_out_button','n_clicks')],[State('tmp_h5_path','data'),State('tmp_h5_singleLine_path','data'),State('element_list','data'),State('info_dict','data')],prevent_initial_call=True)
        def save_results(h5_button,h5_singleLines_button,tiff_button,tmp_h5_path,tmp_h5_singleLine_path,element_list,info_dict):
        
            ctx = dash.callback_context
            
            
            # start_time = time.time()
            
            if ctx.triggered[0]['prop_id'].split('.')[0]=='hdf5_out_button':
                auc_map = ft.load_tmp_h5_FITfile_ext(str(tmp_h5_path))
                ft.save_auc_elements_2_h5_allSDDs(self.res_fld + 'XRF_aligned_fits_elements.h5',list(info_dict['element_list']),auc_map,True)
                
            if ctx.triggered[0]['prop_id'].split('.')[0]=='hdf5_singleLines_out_button':
                auc_map = ft.load_tmp_h5_FITfile_ext(str(tmp_h5_singleLine_path))
                ft.save_auc_lines_2_h5_allSDDs(self.res_fld + 'XRF_aligned_fits_lines.h5',list(info_dict['element_list']),list(info_dict['lines_list_user']),auc_map,True)                
                
            if ctx.triggered[0]['prop_id'].split('.')[0]=='tiff_out_button':
                ft.save_auc_2_tiff_allSDDs(self.res_fld + 'XRF_aligned_fits_pymca_confront_ftol'+str(info_dict['ftol'])+'.tiff',list(info_dict['element_list']),auc_map,True)
                    
                    
            # print(f'save XRF results: {np.round(time.time() - start_time,2)} sec')

            return True
            
        
        inputs = [Input(f'xrf_alert_{idx}','is_open') for idx in range(self.n_lines)] + [Input(f'xrf_alert_{idx}','children') for idx in range(self.n_lines)] + [Input('add_xrf_button','n_clicks')]
        @self.app.callback([Output('xrf_lines_list','data'),Output('xrf_lines_colors_list','data'),Output('element_list','data'),Output('alert_color','data'),Output('XRF_lines_checklist_div','children'),Output('XRF_lines_checklist_div','style')],inputs,[State('alert_color','data'),State('xrf_lines_colors_list','data'),State('xrf_lines_list','data'),State("XRFlines_element_dropdown",'value'),State("beam_en_val",'value')],prevent_initial_call=True)
        def update_lines_list(*args):
                        
            beam_en = args[-1]
            el_val = args[-2]
            xrf_lines_list = args[-3]
            xrf_lines_colors_list = args[-4]
            alert_color_data = args[-5]
                                    
            ctx = dash.callback_context
            
            
            if ctx.triggered[0]['prop_id'].split('.')[0]=='add_xrf_button': # adding a new element with all its lines to xrf_lines_list
                
                avail_colors = np.asarray(alert_color_data['avail_colors'])
                xrf_lines_ordered = []
                xrf_lines_colors_ordered = []
                
                if el_val not in list(alert_color_data.keys()): # element is new in the evaluation and was not given a color yet 
                    curr_color = alert_color_data['current_alert_color']                    
                    curr_color_idx = np.argwhere(avail_colors==curr_color)[0,0]                    
                        
                    alert_color_data.update({'avail_colors':avail_colors, el_val:avail_colors[curr_color_idx]})
                    
                    # retrieve xrf lines information
                    xrf_lines = xraydb.xray_lines(el_val,excitation_energy=beam_en)
                    xrf_lines_strength = xrdb.xray_line_strengths(el_val,excitation_energy=beam_en)
                    xrf_lines_keys = list(xrf_lines.keys())
                    
                    for line in xrf_lines_keys:
                        label = el_val + ' - ' + line + ' - ' + str(np.round(xrf_lines_strength[line])) + ' cm2/g'# + ' - ' + str(xrf_lines[line][0]) + ' eV'
                        if label not in xrf_lines_list:
                            xrf_lines_list.append(label)
                            xrf_lines_colors_list.append(curr_color)
                            
                            
                    for key in list(self.PTE_indexing.keys()):
                        
                        tmp = []
                        
                        for line in xrf_lines_list:
                            el = line[:np.int64(line.find(' '))]
                            
                            if key==el:    
                                tmp.append(line)
                                
                        for line_key in list(xraydb.xray_lines(key,excitation_energy = beam_en).keys()):
                            for line in tmp:
                                
                                first_idx = np.int64(line.find('-'))
                                second_idx = np.int64(line.find('-', first_idx + 1))
                                l = line[first_idx+2:second_idx-1]
                                
                                if line_key==l:
                                    xrf_lines_ordered.append(line)
                                    
                    for line in xrf_lines_ordered:
                        el = line[:np.int64(line.find(' '))]
                        xrf_lines_colors_ordered.append(alert_color_data[el])
                        
                            
                    curr_color_idx += 1
                   
                    if curr_color_idx == len(avail_colors):
                        curr_color_idx = 0    
                            
                    alert_color_data.update({'current_alert_color':avail_colors[curr_color_idx]})
                    
                    
                else: # element is already present in xrf_lines_list and has been assigned a color
                    curr_color = str(alert_color_data[el_val])                    
                    curr_color_idx = np.argwhere(avail_colors==curr_color)[0,0]
                    
                    # retrieve xrf lines information
                    xrf_lines = xraydb.xray_lines(el_val,excitation_energy=beam_en)
                    xrf_lines_strength = xrdb.xray_line_strengths(el_val,excitation_energy=beam_en)
                    xrf_lines_keys = list(xrf_lines.keys())
                    
                    for line in xrf_lines_keys:
                        label = el_val + ' - ' + line + ' - ' + str(np.round(xrf_lines_strength[line])) + ' cm2/g'# + ' - ' + str(xrf_lines[line][0]) + ' eV'
                        if label not in xrf_lines_list:
                            xrf_lines_list.append(label)
                            xrf_lines_colors_list.append(curr_color)
                            
                            
                    for key in list(self.PTE_indexing.keys()):
                        
                        tmp = []
                        
                        for line in xrf_lines_list:
                            el = line[:np.int64(line.find(' '))]
                            
                            if key==el:    
                                tmp.append(line)
                                
                        for line_key in list(xraydb.xray_lines(key,excitation_energy = beam_en).keys()):
                            for line in tmp:
                                
                                first_idx = np.int64(line.find('-'))
                                second_idx = np.int64(line.find('-', first_idx + 1))
                                l = line[first_idx+2:second_idx-1]
                                
                                if line_key==l:
                                    xrf_lines_ordered.append(line)
                                    
                    for line in xrf_lines_ordered:
                        el = line[:np.int64(line.find(' '))]
                        xrf_lines_colors_ordered.append(alert_color_data[el])

                    
                
        
            else: # alert of an xrf line has been closed.            
                for idx in range(self.n_lines):
                    if args[idx] == False:
                        line = args[idx + self.n_lines]
                        
                        if line in xrf_lines_list:
                            xrf_lines_list.remove(line)
                            
                xrf_lines_ordered = xrf_lines_list.copy()
                xrf_lines_colors_ordered = []
                
                for line in xrf_lines_ordered:
                    el = line[:np.int64(line.find(' '))]                    
                    xrf_lines_colors_ordered.append(alert_color_data[el])

            
            div_children = [dbc.Row([dbc.Col(dbc.Checklist(options=[{"label":"", "value":idx}],value=[idx],id=f"checklist_xrfalert_{idx}", style={'fontWeight':'bold'}), width={'size':"2"}),
                                     dbc.Col(dbc.Alert(xrf_lines_ordered[idx], id=f'xrf_alert_{idx}',dismissable=True,is_open=True, color=xrf_lines_colors_ordered[idx]), width={'size':"10"})]) for idx in range(len(xrf_lines_ordered))] 
            
            div_children += [dbc.Row([dbc.Col([dbc.Checklist(options=[{"label":"", "value":idx, 'disabled':True}],value=[idx],id=f"checklist_xrfalert_{idx}", style={'fontWeight':'bold'})], width={'size':"2"}),
                                      dbc.Col(dbc.Alert("Empty alert", id=f'xrf_alert_{idx}',dismissable=True,is_open=False), width={'size':"10"})], style={'display':'none'}) for idx in range(len(xrf_lines_ordered),self.n_lines)]
                        
            element_list = []
            for idx in range(len(xrf_lines_ordered)):
                el = xrf_lines_ordered[idx][:np.int64(xrf_lines_ordered[idx].find(' '))]
                if el not in element_list:
                    element_list.append(el) 
                                                
            return [xrf_lines_ordered, xrf_lines_colors_ordered, element_list, alert_color_data, div_children, {'display':'inline-block'}] # dbc.ListGroup(ListGroup_children,flush=True,id='xrf_lines_ListGroup')
        
        
        states = [State('beam_en_val','value'),State('fit_boundaries_LOW_val','value'),State('fit_boundaries_HIGH_val','value'),State('snip_width_val','value'),State('snip_iter_val','value'),State('element_list','data'),State('xrf_lines_list','data'),State('loaded_data','data')]
        @self.app.callback([Output('sum_graph', 'figure'),Output('info_dict','data'),Output('batchfit_tab','disabled'),Output('tot_pixels','data'),Output('fit_sum_exec_time','data')],[Input('fit_sum_button', 'n_clicks'),Input('fit_tab', 'disabled')],states,prevent_initial_call=True) #[State('exp_info_div','style'),State('data_load_Col','style')]
        def update_fit_sum_graph(n_clicks_fit,n_clicks_load,beam_en,fb_low,fb_high,snip_width,snip_iter,element_list,xrf_lines_list,data):  

            # initialisations
            xrfdatasum = np.asarray(data['xrfdata'],dtype=np.int64)
            tot_pixels = np.int32(data['tot_pixels'])
            xdata = np.asarray(data['xdata'],float)
            delta_peak = 1
            nfev = 10**5
            ftol = 0.01
            sigma_bounds = (10,55)
            n_SDD = np.int64(data['n_SDD'])
            fit_bounds = (fb_low,fb_high)
            fit_bounds_en = (xdata[fit_bounds[0]],xdata[fit_bounds[1]])
            ctx = dash.callback_context
            slope = np.float64(data['slope'])
            offset = np.float64(data['offset'])
            
            if ctx.triggered[0]['prop_id'].split('.')[0]=='fit_sum_button':
                
                # start_time = time.time()   
                                            
                # retrieve and organize necessary fit information   
                lines_list, lines_list_user, lines_energy, lines_intensity, lines_strength = ft.sort_user_lines(element_list,xrf_lines_list,beam_en,fit_bounds_en,0,0)
                ref_lines, lin_coords_lines = ft.find_strongest_line_in_family(element_list, lines_list, beam_en)
                n_lines, peak_centers_ch, beam_en_ch = ft.calculate_fit_params(element_list,lines_list_user,lines_energy,beam_en,xdata[fit_bounds[0]:fit_bounds[1]])

                
                if n_clicks_fit==1:
                    # generate info_dict
                    info_dict = {'fit_bounds':fit_bounds, 'snip_width':snip_width, 'snip_iter':snip_iter, 'peak_centers_ch':peak_centers_ch, 'ftol':ftol, 'nfev':nfev,\
                                 'OPT_method':'L-BFGS-B','sigma_bounds':sigma_bounds, 'n_lines':n_lines, 'lines_intensity':lines_intensity, 'lin_coords_lines':lin_coords_lines, 'lines_energy':lines_energy, 'lines_energy_scattering':np.concatenate((lines_energy,np.array([beam_en])),axis=0),\
                                 'xdata':xdata, 'element_list':element_list, 'lines_list':lines_list, 'lines_list_user':lines_list_user, 'beam_en':beam_en, 'beam_en_ch':beam_en_ch, 'slope':slope, 'offset':offset, 'delta_peak':delta_peak, 'n_SDDs':n_SDD}
                        
                    _, _= ft.plotly_xfr_sum(xrfdatasum,info_dict)
                    
                    
                # generate info_dict
                info_dict = {'fit_bounds':fit_bounds, 'snip_width':snip_width, 'snip_iter':snip_iter, 'peak_centers_ch':peak_centers_ch, 'ftol':ftol, 'nfev':nfev,\
                             'OPT_method':'L-BFGS-B','sigma_bounds':sigma_bounds, 'n_lines':n_lines, 'lines_intensity':lines_intensity, 'lin_coords_lines':lin_coords_lines, 'lines_energy':lines_energy, 'lines_energy_scattering':np.concatenate((lines_energy,np.array([beam_en])),axis=0),\
                             'xdata':xdata, 'element_list':element_list, 'lines_list':lines_list, 'lines_list_user':lines_list_user, 'beam_en':beam_en, 'beam_en_ch':beam_en_ch, 'slope':slope, 'offset':offset, 'delta_peak':delta_peak, 'n_SDDs':n_SDD}
                
                print('checkpoint')
                with open('saved_dictionary.pkl', 'wb') as f:
                    pickle.dump(info_dict, f)
                    
                    
                fig, fit_sum_exec_time = ft.plotly_xfr_sum(xrfdatasum,info_dict)
                # print(f'fit_sum_exec_time = {fit_sum_exec_time}')
                
                disable_tab = False                
                # print(f'fit sum map: {np.round(time.time() - start_time,2)} sec')
                
            else: # first loading of data -> display graph
            
                # retrieve and organize necessary fit information   
                lines_list, lines_list_user, lines_energy, lines_intensity, lines_strength = ft.sort_user_lines([],[],beam_en,fit_bounds_en,None,None)
                ref_lines, lin_coords_lines = ft.find_strongest_line_in_family([], lines_list, beam_en)
                n_lines, peak_centers_ch, beam_en_ch = ft.calculate_fit_params([],lines_list_user,lines_energy,beam_en,xdata[fit_bounds[0]:fit_bounds[1]])
            
                # generate info_dict
                info_dict = {'fit_bounds':(0,np.shape(xdata)[0]), 'snip_width':snip_width, 'snip_iter':snip_iter, 'peak_centers_ch':peak_centers_ch, 'ftol':ftol, 'nfev':nfev,\
                             'OPT_method':'L-BFGS-B','sigma_bounds':(10,55), 'n_lines':n_lines, 'lines_intensity':lines_intensity, 'lin_coords_lines':lin_coords_lines, 'lines_energy':lines_energy, 'lines_energy_scattering':np.concatenate((lines_energy,np.array([beam_en])),axis=0),\
                             'xdata':xdata, 'element_list':element_list, 'lines_list':lines_list, 'lines_list_user':lines_list_user, 'beam_en':beam_en, 'beam_en_ch':beam_en_ch, 'slope':slope, 'offset':offset, 'delta_peak':delta_peak, 'n_SDDs':n_SDD}

            
                # start_time = time.time()
                fig = ft.plotly_sumdata_channels(xrfdatasum,xdata)
                fit_params, fits = ft.fit_wrapper_scattering_SDDsum(np.ones((n_SDD,np.shape(xrfdatasum)[0],1)),info_dict) # fake initialisation to compile numba njit prior to actual run
                disable_tab = True
                fit_sum_exec_time = 0.0
                # print(f'plot data: {np.round(time.time() - start_time,2)} sec')
            
            return [fig, info_dict, disable_tab, tot_pixels, fit_sum_exec_time]
            
        
        @self.app.callback([Output('fitMap_showcase_graph','figure'),Output('batchFit_div','style'),Output('element_slider','marks'),Output('displayRange_LOW_val','value'),Output('displayRange_HIGH_val','value')],[Input('tmp_h5_path','data'),Input('element_slider','value'),Input('displayRange_LOW_val','value'),Input('displayRange_HIGH_val','value')],[State('element_list','data'),State('info_dict','data')],prevent_initial_call=True)
        def populate_auc_graphs(tmp_h5_path,slider_val,dr_low,dr_high,element_list,info_dict):
            
            # start_time = time.time()  
            
            ctx = dash.callback_context
            
            # plot data + fit
            if slider_val <= len(element_list) - 1:
                el = element_list[slider_val]
                
            else:
                el = 'Scattering'   
                
            # load auc_map
            auc_map = ft.load_tmp_h5_FITfile(str(tmp_h5_path),slider_val)
            
            if ctx.triggered[0]['prop_id'].split('.')[0]=='displayRange_LOW_val' or ctx.triggered[0]['prop_id'].split('.')[0]=='displayRange_HIGH_val': 
                fig = px.imshow(auc_map, title='Element Viewed: ' + el, labels=dict(x="X", y="Y", color="Counts"),aspect="equal", zmin=dr_low, zmax=dr_high, color_continuous_scale=temp_cmap)
                dr_low_out = dr_low
                dr_high_out = dr_high
                
            else:
                fig = px.imshow(auc_map, title='Element Viewed: ' + el, labels=dict(x="X", y="Y", color="Counts"),aspect="equal", zmin=0, zmax=auc_map.max(), color_continuous_scale=temp_cmap)
                dr_low_out = 0
                dr_high_out = auc_map.max()
                
            fig.update_layout(font_size=self.fs, autosize=True)
            
            marks = {idx:element_list[idx] for idx in range(len(element_list))}
            marks.update({len(element_list):'Scattering'})
                                    
            # print(f'populate auc maps: {np.round(time.time() - start_time,2)} sec')
            
            return [fig, {'display':'block'}, marks, dr_low_out, dr_high_out]
        
        
        @self.app.callback(Output('fit_showcase_graph','figure'),Input('fitMap_showcase_graph','hoverData'),[State('tmp_h5_path','data'),State('info_dict','data'),State('loaded_data','data')],prevent_initial_call=True)
        def populate_single_pixel_fits(hoverData,tmp_h5_path,info_dict,loaded_data):
            
            # initialisations
            im_shape = loaded_data['im_shape']
            masked_idxs = np.asarray(loaded_data['nonzeroxrf'],dtype=np.int32)
            coord = np.array([hoverData['points'][0]['y'],hoverData['points'][0]['x']],int)

            try:
                lin_coord = np.argwhere(masked_idxs==np.ravel_multi_index(coord,im_shape))[0,0]
            
            except:
                fig = go.Figure()
                fig.update_yaxes(type="log")
                fig.update_layout(title=f"Fit of pixel {coord}",xaxis_title="Energy [eV]",yaxis_title="Counts", font_size=self.fs)
                
                return fig                
                
            lin_coords = [lin_coord]
            
            
            ydata = ft.load_HDF_aligned_singlePixel(self.h5_path,'channel_SUM',lin_coords)
            xdata = np.asarray(info_dict['xdata'],float) 
            fit_bounds = tuple(info_dict['fit_bounds'])
            fits, _ = ft.fit_pixel(ydata,xdata,info_dict)
            
            # plot data + fit
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xdata[fit_bounds[0]:fit_bounds[1]], y=ft.correct_decimal_spectrum(ydata[fit_bounds[0]:fit_bounds[1]]), mode='lines', showlegend=True, name='data'))
            fig.add_trace(go.Scatter(x=xdata[fit_bounds[0]:fit_bounds[1]], y=ft.correct_decimal_spectrum(fits[fit_bounds[0]:fit_bounds[1]]), mode='lines', showlegend=True, name='fit'))
            fig.update_yaxes(type="log")
            fig.update_layout(title=f"Fit of pixel {coord}",xaxis_title="Energy [eV]",yaxis_title="Counts", font_size=self.fs)
            
            return fig 
            
        
        @self.app.callback(Output('batchfit_trigger','data'),Input('batchfit_button','n_clicks'),State('batchfit_running','data'),prevent_initial_call=True)
        def batchfit_button(n_clicks,running_flag):
            
            if running_flag==False:
                return True
            
            else:
                return dash.no_update
            
            
        @self.app.callback([Output('batchfit_running','data'),Output('progress_interval','n_intervals'),Output('progress_interval','disabled')],Input('batchfit_finished','data'),Input('batchfit_button','n_clicks'),State('batchfit_running','data'),prevent_initial_call=True)
        def update_flags(finished_flag,trigger_flag,running_flag):
            
            ctx = dash.callback_context
            
            if ctx.triggered[0]['prop_id'].split('.')[0]=='batchfit_finished':
                return [False, 0, True]
            
            if ctx.triggered[0]['prop_id'].split('.')[0]=='batchfit_button':
                if running_flag:
                    return [dash.no_update, dash.no_update, dash.no_update]
                    
                else:
                    return [True, 0, False]            
            
        
        @self.app.callback([Output('tmp_h5_path','data'),Output('tmp_h5_singleLine_path','data'),Output('element_slider','max'),Output('batchfit_finished','data')],Input('batchfit_trigger','data'),[State('info_dict','data'),State('batchfit_running','data'),State('batchfit_finished','data')],prevent_initial_call=True)
        def run_batch_fitting(trigger,info_dict,running_flag,finished_flag):
            
            # save fit info to configuration_file
            txtpath = os.path.join(self.res_fld,'config.txt')
            ft.save_fit_info_2_txt(txtpath,info_dict)
                
            # start_time = time.time()  

            # initialisations
            data = ft.load_HDF_aligned(self.h5_path,['n_SDD', 'channel_SUM', 'slope', 'offset', 'im_shape', 'beam_en', 'nonzeroxrf'])
            xdata = np.asarray(info_dict['xdata'],float)
            lines_energy_scattering = np.asarray(info_dict['lines_energy_scattering'],float)
            fit_bounds = tuple(info_dict['fit_bounds'])
            xrfdata = data['xrfdata'][:,fit_bounds[0]:fit_bounds[1]]
            fit_params, fits = ft.fit_wrapper_scattering_SDDsum(xrfdata,info_dict)
            
            auc_map_flat = ft.fitparams2auc_flat_SDDsum(fit_params,lines_energy_scattering,xdata,2)
            auc_map = ft.fill_auc_mask(auc_map_flat,data['nonzeroxrf'],data['im_shape'])
            auc_map_el = ft.auc2element(auc_map,info_dict['element_list'],info_dict['lines_list_user'],1)
                        
            tmp_h5_path = ft.save_tmp_h5_FITfile(self.tmp_fld,{'auc_map':auc_map_el})
            tmp_h5_singleLine_path = ft.save_tmp_h5_FITfile(self.tmp_fld,{'auc_map':auc_map})
            
            # print(f'batch fitting: {np.round(time.time() - start_time,2)} sec')
                            
            return [tmp_h5_path, tmp_h5_singleLine_path, np.shape(auc_map_el)[2] - 1, True]
            
        
        @self.app.callback([Output('progress_bar','value'),Output('progress_bar','label')],Input('progress_interval','n_intervals'),[State('progress_interval','disabled'),State('fit_sum_exec_time','data'),State('tot_pixels','data'),State('xrf_lines_list','data'),State('progress_interval','interval'),State('progress_interval','max_intervals'),State('progress_bar','value')],prevent_initial_call=True)
        def update_progress(n_interval,disabled_interval,fit_sum_exec_time,tot_pixels,xrf_lines_list,interval,max_intervals,progress):
            
            tot_pixels = np.int64(tot_pixels)
            n_workers = mp.cpu_count()
                        
            if disabled_interval==False:        
                if tot_pixels > 0:
                    px_per_second = 1/fit_sum_exec_time * n_workers
                    tot_eta = tot_pixels/px_per_second # in seconds
                    
                    currently_elapsed_time = n_interval*interval/1000
                    currently_proc_vox = currently_elapsed_time*px_per_second
                    
                    remaining_eta_min, remaining_eta_sec = divmod(tot_eta - currently_elapsed_time,60)
                    remaining_eta_min = np.int32(remaining_eta_min)
                    remaining_eta_sec = np.int32(remaining_eta_sec)
                    progress = np.int32(np.round(currently_proc_vox/tot_pixels * 100))
                    
                    if progress >= 100:
                        progress = 100
                        remaining_eta_min = 0
                        remaining_eta_sec = 0                        
                    
                    return [progress, f"{progress}% - ETA {remaining_eta_min}:{remaining_eta_sec}"] # if progress >= 5 else ""
                
                else:
                    raise PreventUpdate()
                    
            else:
                return [100, "100%"]
