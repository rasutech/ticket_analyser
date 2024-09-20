# app_analyzers/app1_analyzer.py
import re
import pandas as pd
from base_analyzer import BaseTicketAnalyzer

class App1TicketAnalyzer(BaseTicketAnalyzer):
    def extract_order_numbers(self):
        """Step 2: Extract order numbers using regex"""
        self.order_numbers = re.findall(r'\d+', self.description)
        self.analysis.append(f"Extracted order numbers: {self.order_numbers}")

    def match_order_number(self):
        """Step 3: Match order numbers against t_raw_order"""
        for order_num in self.order_numbers:
            query = f"SELECT work_order_number, ORDER_SOURCE, ORDER_ACTION FROM t_raw_order WHERE work_order_number = {order_num}"
            df = pd.read_sql_query(query, self.db_connection)
            if not df.empty:
                self.order_number = order_num
                self.analysis.append(f"Matched order number: {self.order_number}")
                return
        self.analysis.append("t_raw_order has no matching order number")
        self.order_number = None

    def fetch_master_order_details(self):
        """Step 4: Fetch master order details from t_master_order"""
        query = f"SELECT * FROM t_master_order WHERE order_number = {self.order_number}"
        df = pd.read_sql_query(query, self.db_connection)
        if not df.empty:
            self.master_order = df.iloc[0]
            self.analysis.append(f"Fetched master order details: {self.master_order}")
        else:
            self.analysis.append("No matching master order found")
            self.master_order = None

    def fetch_transaction_steps(self):
        """Step 5: Fetch transaction steps from t_transaction_manager"""
        query = f"SELECT * FROM t_transaction_manager WHERE order_number = {self.order_number}"
        self.transaction_steps = pd.read_sql_query(query, self.db_connection)
        self.analysis.append(f"Fetched transaction steps: {len(self.transaction_steps)} steps found")

    def analyze_work_steps(self):
        """Step 6: Analyze work steps and determine if order is complete"""
        if self.master_order['STATUS'] == 'WO_COMPLETE' and all(self.transaction_steps['state'] == 'SUCCESS'):
            self.analysis.append("Order is marked Completed")
        else:
            failed_steps = self.transaction_steps[self.transaction_steps['state'] != 'SUCCESS']
            if not failed_steps.empty:
                self.analysis.append(f"Latest failed step: {failed_steps.iloc[-1]['work_step_name']}")
            else:
                self.analysis.append("No failed steps found")
