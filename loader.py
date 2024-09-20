# loader.py
from app_analyzers.app1_analyzer import App1TicketAnalyzer
from app_analyzers.app2_analyzer import App2TicketAnalyzer

def load_ticket_analyzer(app_name, incident_number, description, db_connection):
    """Load the correct ticket analyzer class based on the application name"""
    if app_name == "App1":
        return App1TicketAnalyzer(incident_number, description, db_connection)
    elif app_name == "App2":
        return App2TicketAnalyzer(incident_number, description, db_connection)
    else:
        raise ValueError(f"No ticket analyzer available for app: {app_name}")
