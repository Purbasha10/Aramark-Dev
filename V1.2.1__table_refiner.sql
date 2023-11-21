create or replace procedure Unmapped_Table_Refiner()
    returns String
    language python
    runtime_version = 3.8
    packages =('snowflake-snowpark-python', 'pandas')
    handler = 'main'
    as '# The Snowpark package is required for Python Worksheets. 
# You can add more packages by selecting them using the Packages control and then importing them.

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd

def main(session: snowpark.Session): 
    # Your code goes here, inside the "main" handler.
    tableName = ''SANDBOX_DB.SANDBOX_STAGE.UNMAPPED_POS_HIERARCHY_STG''
    dataframe = session.table(tableName)
    df = dataframe.to_pandas()
    df = df[[''MASTER_ITEM_NAME_1'', ''MINOR_SOURCE_NAME'', ''MAJOR_SOURCE_NAME'']]
    df_processed = df[df.notnull().all(1)]
    session.write_pandas(df_processed, "UNMAPPED_DATA_REFINED",auto_create_table=True )
    # Return value will appear in the Results tab.
    return "Unmapped table processed and saved"';

CALL Unmapped_Table_Refiner();