#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:45:36 2022

@author: matteo
"""

import DashAppTools
import argparse
import os

# define configuration parameters through argparse
parser = argparse.ArgumentParser(description='XRFitProc')
parser.add_argument('--app_host_url', type=str, required=False, default='0.0.0.0')
parser.add_argument('--port', type=int, required=False, default=8080)
parser.add_argument('--dash_debug', type=str, required=False, default=False)
parser.add_argument('--requests_url_base_pathname', type=str, required=False, default='/XRFitProc/')
args = parser.parse_args()

# call application class
appclass = DashAppTools.class_XRFitProc_webapp(args.requests_url_base_pathname)

if __name__ == '__main__':
    appclass.app.run(debug=eval(args.dash_debug), port=args.port, threaded=True)