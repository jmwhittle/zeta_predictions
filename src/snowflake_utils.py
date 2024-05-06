import snowflake.connector
import yaml
import json
import os
from snowflake.connector.pandas_tools import pd_writer
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL

### Snowflake utility functions
# define snowflake connection credentials
def get_snowflake_creds():
    try:
        client = botocore.session.get_session().create_client('secretsmanager')
        cache_config = SecretCacheConfig()
        cache = SecretCache( config = cache_config, client = client)
        str_creds = cache.get_secret_string('DSDS-snowflake-credentials')
        snow_creds = json.loads(str_creds)
        # print("AWS secrets manager credentials successfully loaded")
        return snow_creds
    except:
        with open(os.path.expanduser("~/pluralsight_db_credentials.yaml")) as fid:
            snow_creds = yaml.load(fid, Loader=yaml.FullLoader)
            # print("AWS scerets not found. Credentials loaded from yaml") 
            return snow_creds

        
# pull data from snowflake
def snow_panda(q, db=None, schema=None):
    
    with open(os.path.expanduser("~/pluralsight_db_credentials.yaml")) as fid:
        creds = yaml.load(fid, Loader=yaml.FullLoader)
    
    if ((db is None) | (schema is None)):
        ctx = snowflake.connector.connect(
            account = creds['account'],
            user = creds['username'],
            password = creds['password'],
            warehouse = 'transform_s'

            )
    else:    
        ctx = snowflake.connector.connect(
            account = creds['account'],
            user = creds['username'],
            password = creds['password'],
            database=db,
            schema=schema,
            warehouse = 'transform_s'
            )
        
    cur = ctx.cursor()

    cur.execute(q)
    df = cur.fetch_pandas_all()
    df.columns = [i.lower() for i in df.columns]
    
    return df


# write data from snowflake
def write_snow_panda(df, db, schema, table_name, **kwargs):
    
    with open(os.path.expanduser("~/pluralsight_db_credentials.yaml")) as fid:
        creds = yaml.load(fid, Loader=yaml.FullLoader)

    engine = create_engine(URL(
        account = creds['account'],
        user = creds['username'],
        password = creds['password'],
        database = db,
        schema = schema,
        warehouse = 'transform_s',
    ))    

    connection = engine.connect()
 
    df.columns = [i.upper() for i in df.columns]
    df.to_sql(table_name, con=engine, index=False, method=pd_writer, **kwargs)
    
    connection.close()
    engine.dispose()
    
    return


def execute_snow(q, db=None, schema=None):
    creds = get_snowflake_creds()    
    if ((db is None) | (schema is None)):
        ctx = snowflake.connector.connect(
            account = creds['account'],
            user = creds['username'],
            password = creds['password'],
            warehouse = 'transform_s')
    else:    
        ctx = snowflake.connector.connect(
            account = creds['account'],
            user = creds['username'],
            password = creds['password'],
            database=db,
            schema=schema,
            warehouse = 'transform_s'
            )        
    cur = ctx.cursor()
    cur.execute(q)
    return