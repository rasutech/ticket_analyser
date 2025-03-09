import psycopg2
import pandas as pd
import re
import os
import sys
from datetime import datetime

def connect_to_postgres(host, database, user, password, port=5432):
    """
    Establish a connection to the PostgreSQL database
    
    Args:
        host: Database host address
        database: Database name
        user: Database user
        password: Database password
        port: Database port (default 5432)
        
    Returns:
        Connection object or None if connection fails
    """
    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        print("Successfully connected to PostgreSQL database")
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

def extract_data_from_postgres(conn, query):
    """
    Execute a query and return the results as a pandas DataFrame
    
    Args:
        conn: PostgreSQL connection object
        query: SQL query to execute
        
    Returns:
        pandas DataFrame with query results
    """
    try:
        df = pd.read_sql_query(query, conn)
        print(f"Successfully fetched {len(df)} rows from database")
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

def extract_description_parts(description):
    """
    Parse the description field to extract actual description, build activity,
    and name-value pairs.
    
    Args:
        description: The description text to parse
        
    Returns:
        tuple: (actual_description, build_activity, name_value_dict)
    """
    if not isinstance(description, str) or not description.strip():
        return "", "", {}
    
    # Split the description into lines
    lines = description.strip().split('\n')
    
    # Extract actual description (first 1-2 lines)
    actual_description_lines = []
    current_line = 0
    
    # Get first line(s) as actual description (until we hit a line with name:value pattern)
    while current_line < len(lines) and current_line < 2:
        # Check if this line matches a name-value pair pattern
        if re.search(r'[A-Za-z\s]+:', lines[current_line]):
            # Skip this line if it's a name-value pair
            break
        actual_description_lines.append(lines[current_line])
        current_line += 1
    
    actual_description = '\n'.join(actual_description_lines).strip()
    
    # Look for Build/Pre-Build activity line
    build_activity = ""
    next_line_idx = current_line
    
    # Check if there's a build activity line between actual description and name-value pairs
    if next_line_idx < len(lines):
        if "Build" in lines[next_line_idx] or "Pre-Build" in lines[next_line_idx]:
            build_activity = lines[next_line_idx].strip()
            next_line_idx += 1
    
    # Extract name-value pairs
    name_value_dict = {}
    for i in range(next_line_idx, len(lines)):
        line = lines[i].strip()
        # Match pattern like "Network Activity:sometext," or "Network Activity: sometext,"
        match = re.match(r'([^:]+):\s*(.*?)(?:,\s*$|$)', line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            name_value_dict[key] = value
    
    return actual_description, build_activity, name_value_dict

def process_description_fields(df):
    """
    Process the description column to extract actual description, build activity,
    and name-value pairs as separate columns
    
    Args:
        df: DataFrame with a 'description' column
        
    Returns:
        DataFrame with extracted columns
    """
    # Create a copy of the input DataFrame
    result_df = df.copy()
    
    # Create columns for actual description and build activity
    result_df['actual_description'] = ""
    result_df['build_activity'] = ""
    
    # Dictionary to track all name-value keys found in the dataset
    all_keys = set()
    
    # First pass: extract description parts and collect all possible keys
    parsed_data = []
    for idx, row in df.iterrows():
        description = row.get('description', '')
        actual_desc, build_act, name_values = extract_description_parts(description)
        
        parsed_data.append({
            'idx': idx,
            'actual_description': actual_desc,
            'build_activity': build_act,
            'name_values': name_values
        })
        
        # Update the set of all keys
        all_keys.update(name_values.keys())
    
    # Sort the keys to ensure consistent column order
    all_keys = sorted(list(all_keys))
    
    # Initialize name-value columns with 'NA'
    for key in all_keys:
        column_name = f"{key.lower().replace(' ', '_')}"
        result_df[column_name] = "NA"
    
    # Second pass: populate the DataFrame with extracted values
    for item in parsed_data:
        idx = item['idx']
        result_df.at[idx, 'actual_description'] = item['actual_description']
        result_df.at[idx, 'build_activity'] = item['build_activity']
        
        # Add name-value pairs
        for key, value in item['name_values'].items():
            column_name = f"{key.lower().replace(' ', '_')}"
            result_df.at[idx, column_name] = value
    
    return result_df

def main():
    """
    Main function to extract and process data from PostgreSQL
    """
    # Database connection parameters
    # You should replace these with your actual database credentials
    host = input("Enter database host: ") or "localhost"
    database = input("Enter database name: ") or "postgres"
    user = input("Enter database user: ") or "postgres"
    password = input("Enter database password: ") or ""
    port = input("Enter database port (default 5432): ") or 5432
    
    try:
        port = int(port)
    except ValueError:
        print("Invalid port number, using default 5432")
        port = 5432
    
    # Connect to PostgreSQL
    conn = connect_to_postgres(host, database, user, password, port)
    if not conn:
        print("Failed to connect to database. Exiting.")
        sys.exit(1)
    
    try:
        # Query to fetch data from ays_intake table with time_to_resolve calculation
        # Adjust the query as needed, e.g., add WHERE clauses or limits
        query = """
        SELECT 
            incident_number, 
            short_description, 
            description, 
            created_on, 
            resolved_at, 
            resolution_code,
            -- Calculate time to resolve in hours
            CASE 
                WHEN resolved_at IS NOT NULL AND created_on IS NOT NULL 
                THEN EXTRACT(EPOCH FROM (resolved_at - created_on))/3600 
                ELSE NULL 
            END as time_to_resolve_hours,
            -- Calculate time to resolve in days
            CASE 
                WHEN resolved_at IS NOT NULL AND created_on IS NOT NULL 
                THEN EXTRACT(EPOCH FROM (resolved_at - created_on))/(3600*24) 
                ELSE NULL 
            END as time_to_resolve_days
        FROM 
            ays_intake
        """
        
        # Optional limit for testing
        limit = input("Enter row limit (leave blank for all rows): ")
        if limit:
            try:
                limit = int(limit)
                query += f" LIMIT {limit}"
            except ValueError:
                print("Invalid limit, fetching all rows")
        
        # Fetch data from PostgreSQL
        df = extract_data_from_postgres(conn, query)
        
        if df.empty:
            print("No data retrieved. Exiting.")
            conn.close()
            sys.exit(1)
        
        # Process the description field
        print("Processing description fields...")
        result_df = process_description_fields(df)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"ays_intake_processed_{timestamp}.csv"
        
        # Save the results to CSV
        result_df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
        
        # Optionally, also save as Excel file
        excel_output = input("Do you want to save as Excel file as well? (y/n): ")
        if excel_output.lower() == 'y':
            excel_file = f"ays_intake_processed_{timestamp}.xlsx"
            result_df.to_excel(excel_file, index=False)
            print(f"Processed data saved to {excel_file}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
    finally:
        # Close the database connection
        conn.close()
        print("Database connection closed")

if __name__ == "__main__":
    main()
