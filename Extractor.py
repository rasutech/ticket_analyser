// Excel Processor for extracting groups and notes data
import * as XLSX from 'xlsx';

// Main function to process the Excel file
async function processExcelAndSave() {
  try {
    // List available files to find the Excel file
    const files = await window.fs.readdir('.');
    console.log('Available files:', files);
    
    // Filter for Excel files
    const excelFiles = files.filter(file => 
      file.endsWith('.xlsx') || file.endsWith('.xls')
    );
    
    if (excelFiles.length === 0) {
      console.error('No Excel files found. Please upload an Excel file.');
      return;
    }
    
    // Use the first Excel file found (you can modify this to use a specific filename)
    const inputFileName = excelFiles[0];
    console.log(`Processing file: ${inputFileName}`);
    
    // Read the file content
    const fileContent = await window.fs.readFile(inputFileName);
    
    // Process the Excel file
    const outputWorkbook = processExcelFile(fileContent);
    
    // Convert the workbook to a binary blob
    const outputData = XLSX.write(outputWorkbook, { bookType: 'xlsx', type: 'array' });
    
    // Save the processed file
    await window.fs.writeFile('system_gen.xlsx', new Uint8Array(outputData));
    console.log('Successfully saved system_gen.xlsx');
    
    return 'Processing complete! Output saved as system_gen.xlsx';
  } catch (error) {
    console.error('Error processing Excel file:', error);
    return `Error: ${error.message}`;
  }
}

// Function to process the Excel file content
function processExcelFile(fileContent) {
  // Read the workbook
  const workbook = XLSX.read(fileContent, {
    cellStyles: true,
    cellFormulas: true,
    cellDates: true,
    cellNF: true,
    sheetStubs: true
  });
  
  // Get the first sheet
  const firstSheetName = workbook.SheetNames[0];
  const worksheet = workbook.Sheets[firstSheetName];
  
  // Convert to array of arrays
  const data = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
  
  // Process the data
  const processedData = processData(data);
  
  // Create a new worksheet with the processed data
  const outputWs = XLSX.utils.aoa_to_sheet(processedData);
  const outputWb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(outputWb, outputWs, 'Processed');
  
  // Return the workbook for saving
  return outputWb;
}

// Function to process the data according to requirements
function processData(data) {
  // Extract headers and rows
  const headers = data[0];
  const rows = data.slice(1);
  
  // Find column indices based on headers
  let numberIdx = headers.findIndex(h => h?.toLowerCase() === 'number');
  let groupIdx = headers.findIndex(h => h?.toLowerCase() === 'group');
  let codeIdx = headers.findIndex(h => h?.toLowerCase() === 'code');
  let notesIdx = headers.findIndex(h => h?.toLowerCase() === 'notes');
  
  // If headers aren't found, use default positions
  numberIdx = numberIdx !== -1 ? numberIdx : 0;
  groupIdx = groupIdx !== -1 ? groupIdx : 1;
  codeIdx = codeIdx !== -1 ? codeIdx : 2;
  notesIdx = notesIdx !== -1 ? notesIdx : 3;
  
  // Create new headers for the output file
  const newHeaders = ['number', 'group', 'code', 'notes', 'Flow', 'Issue', 'Resolution', 'RCA'];
  
  // Function to extract text between two patterns
  function extractBetween(text, startPattern, endPatterns) {
    if (!text) return 'NA';
    
    const startIdx = text.indexOf(startPattern);
    if (startIdx === -1) return 'NA';
    
    const contentStart = startIdx + startPattern.length;
    let contentEnd = text.length;
    
    // Find the earliest occurrence of any end pattern
    for (const endPattern of endPatterns) {
      const endIdx = text.indexOf(endPattern, contentStart);
      if (endIdx !== -1 && endIdx < contentEnd) {
        contentEnd = endIdx;
      }
    }
    
    return text.substring(contentStart, contentEnd).trim();
  }
  
  // Process each row to extract the required information
  const processedRows = rows
    // Filter only the specified groups
    .filter(row => {
      const group = row[groupIdx];
      return ['group_a', 'group_b', 'group_c', 'group_d'].includes(group);
    })
    .map(row => {
      const number = row[numberIdx];
      const group = row[groupIdx];
      const code = row[codeIdx];
      const notes = row[notesIdx] || '';
      
      // Initialize new columns with default 'NA'
      let flow = 'NA';
      let issue = 'NA';
      let resolution = 'NA';
      let rca = 'NA';
      
      if (notes) {
        // Convert notes to string if it's not already
        const notesStr = String(notes);
        
        // Extract Flow from first line if it matches the pattern |Flow|sometext-sometext|
        const lines = notesStr.split('\n');
        if (lines[0] && lines[0].includes('|Flow|')) {
          const flowMatch = lines[0].match(/\|Flow\|(.*?)\|/);
          if (flowMatch && flowMatch[1]) {
            flow = flowMatch[1].trim();
          }
        }
        
        // Extract Issue - look for "Issue:" and extract until "RCA:" or "Resolution:"
        issue = extractBetween(
          notesStr, 
          'Issue:', 
          ['RCA:', 'RCA :', 'Resolution:', 'Resolution :']
        );
        
        // Extract Resolution
        resolution = extractBetween(
          notesStr, 
          'Resolution:', 
          ['RCA:', 'RCA :', 'Issue:', 'Issue :']
        );
        if (resolution === 'NA') {
          resolution = extractBetween(
            notesStr, 
            'Resolution :', 
            ['RCA:', 'RCA :', 'Issue:', 'Issue :']
          );
        }
        
        // Extract RCA
        rca = extractBetween(
          notesStr, 
          'RCA:', 
          ['Resolution:', 'Resolution :', 'Issue:', 'Issue :']
        );
        if (rca === 'NA') {
          rca = extractBetween(
            notesStr, 
            'RCA :', 
            ['Resolution:', 'Resolution :', 'Issue:', 'Issue :']
          );
        }
      }
      
      return [number, group, code, notes, flow, issue, resolution, rca];
    });
  
  // Combine headers and processed rows
  return [newHeaders, ...processedRows];
}

// Run the main function
processExcelAndSave().then(result => {
  console.log(result);
}).catch(error => {
  console.error('Error:', error);
});
