"""
work_dir:
    data:        
    model:

get_geoms_from_model:
    input:  model root from user yml file
            catalogue.yml (from the data folder)
    output: geoms/*.geojson
            
get_input_variables:
    _config: yaml secifying which input variables to read
    input: model root from user yml file
           model root/rr_to_flow.nc
    output: data/*_input.csv --> or nc
    
get_target_variables: get flow path
    _config: yaml secifying which target variables to read
    input: model root from user yml file
           model root/*_fou.nc
    output: data/*_target.csv --> or nc
  
"""
