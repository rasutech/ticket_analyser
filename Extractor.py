import pandas as pd
import os
import re
import glob

def process_excel_file(input_filepath, output_filepath='system_gen.xlsx'):
    """
    Process an Excel file to extract specific groups and information from notes
    
    Args:
        input_filepath: Path to the input Excel file
        output_filepath: Path where the output Excel file will be saved
        
    Returns:
        str: Status message
    """
    try:
        # Read the Excel file
        print(f"Reading Excel file: {input_filepath}")
        df = pd.read_excel(input_filepath)
        
        # Ensure column names are lowercase for consistent access
        df.columns = [col.lower() for col in df.columns]
        
        # Filter only specified groups
        target_groups = ['group_a', 'group_b', 'group_c', 'group_d']
        filtered_df = df[df['group'].isin(target_groups)].copy()
        
        if filtered_df.empty:
            return "No matching groups found in the data."
        
        # Extract information from notes column
        filtered_df['Flow'] = filtered_df['notes'].apply(extract_flow)
        filtered_df['Issue'] = filtered_df['notes'].apply(lambda x: extract_between(x, 'Issue:', ['RCA:', 'RCA :', 'Resolution:', 'Resolution :']))
        filtered_df['Resolution'] = filtered_df['notes'].apply(extract_resolution)
        filtered_df['RCA'] = filtered_df['notes'].apply(extract_rca)
        
        # Save the result to a new Excel file
        filtered_df.to_excel(output_filepath, index=False)
        print(f"Successfully saved: {output_filepath}")
        
        return f"Processing complete! Output saved as {output_filepath}"
    
    except Exception as e:
        return f"Error processing Excel file: {str(e)}"

def extract_flow(notes):
    """Extract the Flow information from the first line of notes"""
    if not isinstance(notes, str):
        return 'NA'
    
    lines = notes.split('\n')
    if not lines or not lines[0].startswith('|Flow|'):
        return 'NA'
    
    flow_match = re.search(r'\|Flow\|(.*?)\|', lines[0])
    if flow_match:
        return flow_match.group(1).strip()
    return 'NA'

def extract_between(text, start_pattern, end_patterns):
    """
    Extract text between a start pattern and the earliest of several possible end patterns
    
    Args:
        text: The text to search in
        start_pattern: The pattern marking the start of the desired text
        end_patterns: List of patterns that could mark the end of the desired text
        
    Returns:
        str: Extracted text or 'NA' if not found
    """
    if not isinstance(text, str):
        return 'NA'
    
    start_idx = text.find(start_pattern)
    if start_idx == -1:
        return 'NA'
    
    content_start = start_idx + len(start_pattern)
    content_end = len(text)
    
    # Find the earliest occurrence of any end pattern
    for end_pattern in end_patterns:
        end_idx = text.find(end_pattern, content_start)
        if end_idx != -1 and end_idx < content_end:
            content_end = end_idx
    
    return text[content_start:content_end].strip()

def extract_resolution(notes):
    """Extract Resolution information with handling for space variations"""
    if not isinstance(notes, str):
        return 'NA'
    
    resolution = extract_between(
        notes, 
        'Resolution:', 
        ['RCA:', 'RCA :', 'Issue:', 'Issue :']
    )
    
    if resolution == 'NA':
        resolution = extract_between(
            notes, 
            'Resolution :', 
            ['RCA:', 'RCA :', 'Issue:', 'Issue :']
        )
    
    return resolution

def extract_rca(notes):
    """Extract RCA information with handling for space variations"""
    if not isinstance(notes, str):
        return 'NA'
    
    rca = extract_between(
        notes, 
        'RCA:', 
        ['Resolution:', 'Resolution :', 'Issue:', 'Issue :']
    )
    
    if rca == 'NA':
        rca = extract_between(
            notes, 
            'RCA :', 
            ['Resolution:', 'Resolution :', 'Issue:', 'Issue :']
        )
    
    return rca

def find_excel_files(directory='.'):
    """Find all Excel files in the specified directory"""
    excel_patterns = ['*.xlsx', '*.xls']
    excel_files = []
    
    for pattern in excel_patterns:
        excel_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    return excel_files

def main():
    """Main function to find and process Excel files"""
    # Find all Excel files in the current directory
    excel_files = find_excel_files()
    
    if not excel_files:
        print("No Excel files found. Please ensure your Excel file is in the current directory.")
        return
    
    print(f"Found {len(excel_files)} Excel file(s):")
    for i, file in enumerate(excel_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    # Use the first file by default, or let the user choose
    if len(excel_files) == 1:
        input_file = excel_files[0]
    else:
        try:
            choice = int(input(f"Enter the number of the file to process (1-{len(excel_files)}): "))
            input_file = excel_files[choice - 1]
        except (ValueError, IndexError):
            print("Invalid choice. Using the first file.")
            input_file = excel_files[0]
    
    # Process the selected file
    result = process_excel_file(input_file)
    print(result)

if __name__ == "__main__":
    main()
