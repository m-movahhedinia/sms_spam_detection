# -*- coding: utf-8 -*-
"""
Created on 

@author: mansour
"""

from logging import INFO, basicConfig, getLogger

from service.app import app

basicConfig(level=INFO)
log = getLogger()

if __name__ == "__main__":
    app.run(debug=True)
