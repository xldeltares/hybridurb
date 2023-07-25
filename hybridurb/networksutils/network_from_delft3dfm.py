"""
work_dir:
    geoms:        
    model:

get_geoms_from_model:
    input:  model root from user yml file
            catalogue.yml (from the data folder)
    output: geoms/*.geojson
            
setup_network:
    _config: yaml secifying which files are read as nodes and which as edges
    input: geoms/*.geojson
    output: model/graph.gpickle
    
simplify_network: get flow path
    _config: yaml secifying which files are read as nodes and which as edges
    input: geoms/*.geojson
    output: model/graph.gpickle
  
"""
