# database.py
import psycopg2

def connect_to_db(host, dbname, user, password):
    """Establish a connection to the PostgreSQL database."""
    connection = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password
    )
    return connection

def get_open_tickets(connection):
    """Fetch incident descriptions and incident numbers from ays_intake_view table."""
    query = "SELECT incident_number, description FROM ays_intake_view WHERE state NOT IN ('Closed', 'Open')"
    df = pd.read_sql_query(query, connection)
    return df
