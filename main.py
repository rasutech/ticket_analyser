# main.py
from database import connect_to_db, get_open_tickets
from loader import load_ticket_analyzer
from elk import search_elk_logs
import pandas as pd

def write_analysis_output(connection, analysis_df, table_name="ticket_analysis_output"):
    """Write the final analysis dataframe into a PostgreSQL table."""
    analysis_df.to_sql(table_name, con=connection, if_exists='replace', index=False)

def ticket_analyzer(db_connection, elk_host, app_name):
    """Main function to orchestrate the analysis process for a specific application."""
    tickets_df = get_open_tickets(db_connection)
    analysis_data = []

    for _, row in tickets_df.iterrows():
        incident_number = row['incident_number']
        description = row['description']

        # Load the correct analyzer class based on the application
        analyzer = load_ticket_analyzer(app_name, incident_number, description, db_connection)
        analysis = analyzer.run_analysis()

        # Append the result for each incident
        analysis_data.append([incident_number, description, "\n".join(analysis)])

    # Create a pandas DataFrame and write the final analysis into the database
    analysis_df = pd.DataFrame(analysis_data, columns=['incident_number', 'description', 'analysis'])
    write_analysis_output(db_connection, analysis_df)

if __name__ == "__main__":
    db_connection = connect_to_db('host1', '
