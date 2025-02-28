import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TicketAnalyzerv1:
    """
    A class for analyzing service desk tickets using NLP techniques.
    Performs ticket similarity analysis and root cause classification.
    No dependencies on Hugging Face or BERT.
    """
    
    def __init__(self, db_config, use_bigrams=True, max_features=512):
        """
        Initialize with database configuration.
        
        Args:
            db_config (dict): Database configuration parameters
            use_bigrams (bool): Whether to include bigrams in text vectorization
            max_features (int): Maximum number of features for TF-IDF vectorization
        """
        self.db_config = db_config
        self.use_bigrams = use_bigrams
        self.max_features = max_features
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def connect_to_db(self):
        """
        Establish connection to PostgreSQL database.
        
        Returns:
            connection: Database connection object
        """
        try:
            connection = psycopg2.connect(
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=self.db_config['port']
            )
            print("Database connection established successfully.")
            return connection
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None
    
    def fetch_tickets(self, app_id, start_date, end_date):
        """
        Fetch tickets from database based on app_id and date range.
        
        Args:
            app_id (str): Application ID to filter tickets
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pandas.DataFrame: DataFrame containing ticket data
        """
        try:
            # Establish database connection
            connection = self.connect_to_db()
            if connection is None:
                return None
            
            # Build the SQL query
            query = f"""
            SELECT 
                incident_number, app_id, state, impact, priority, urgency,
                created_on, description, short_description, assignment_group,
                calendar_stc, reopened_count, resolution_code, resolution_update
            FROM 
                incident_table
            WHERE 
                app_id = '{app_id}'
                AND created_on BETWEEN '{start_date}' AND '{end_date}'
            """
            
            # Load data into pandas DataFrame
            df = pd.read_sql(query, connection)
            connection.close()
            
            print(f"Successfully fetched {len(df)} tickets.")
            return df
        
        except Exception as e:
            print(f"Error fetching tickets: {e}")
            return None
    
    def preprocess_text(self, text):
        """
        Preprocess text data for NLP analysis.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, numbers and URLs
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def get_tfidf_embeddings(self, texts):
        """
        Generate TF-IDF embeddings for a list of texts.
        This is a BERT alternative using traditional NLP.
        
        Args:
            texts (list): List of preprocessed text strings
            
        Returns:
            numpy.ndarray: TF-IDF embeddings matrix
            TfidfVectorizer: The fitted vectorizer for future reference
        """
        # Set up TF-IDF vectorizer with appropriate parameters
        ngram_range = (1, 2) if self.use_bigrams else (1, 1)
        
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=ngram_range,
            min_df=2,          # Minimum document frequency
            max_df=0.85,       # Maximum document frequency
            sublinear_tf=True, # Apply sublinear TF scaling (1 + log(TF))
            norm='l2'          # L2 normalization for better cosine similarity
        )
        
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        print(f"Generated TF-IDF embeddings with {tfidf_matrix.shape[1]} features")
        
        # Return dense matrix for clustering algorithms that require it
        return tfidf_matrix.toarray(), vectorizer
    
    def perform_similarity_analysis(self, df):
        """
        Perform similarity analysis on ticket descriptions using TF-IDF embeddings.
        
        Args:
            df (pandas.DataFrame): DataFrame containing ticket data
            
        Returns:
            dict: Dictionary with resolution codes as keys and similarity matrices as values
            dict: Dictionary with resolution codes as keys and PCA data as values
        """
        # Preprocess all descriptions
        print("Preprocessing descriptions...")
        df['processed_description'] = df['description'].apply(self.preprocess_text)
        
        # Group by resolution code
        similarity_results = {}
        pca_results = {}
        
        # Get unique resolution codes
        resolution_codes = df['resolution_code'].unique()
        
        for code in resolution_codes:
            if pd.isna(code):
                continue
                
            print(f"\nAnalyzing tickets with resolution code: {code}")
            code_df = df[df['resolution_code'] == code].copy()
            
            if len(code_df) < 2:
                print(f"Skipping resolution code {code} - not enough samples")
                continue
                
            texts = code_df['processed_description'].tolist()
            
            # Get TF-IDF embeddings
            embeddings, _ = self.get_tfidf_embeddings(texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            similarity_results[code] = similarity_matrix
            
            # PCA or SVD for visualization
            if len(embeddings) > 2:
                # Use TruncatedSVD (better for sparse TF-IDF matrices) if many features
                if embeddings.shape[1] > 100:
                    svd = TruncatedSVD(n_components=2, random_state=42)
                    pca_data = svd.fit_transform(embeddings)
                else:
                    pca = PCA(n_components=2, random_state=42)
                    pca_data = pca.fit_transform(embeddings)
                    
                pca_results[code] = {
                    'pca_data': pca_data,
                    'indices': code_df.index.tolist(),
                    'incident_numbers': code_df['incident_number'].tolist()
                }
                
        return similarity_results, pca_results
    
    def classify_root_causes(self, df):
        """
        Classify tickets to identify root causes using clustering.
        
        Args:
            df (pandas.DataFrame): DataFrame containing ticket data
            
        Returns:
            pandas.DataFrame: DataFrame with added cluster labels
            dict: Dictionary with summary of root causes by group
        """
        # Make sure preprocessed descriptions are available
        if 'processed_description' not in df.columns:
            df['processed_description'] = df['description'].apply(self.preprocess_text)
        
        # Group by assignment_group and resolution_code
        groups = df.groupby(['assignment_group', 'resolution_code'])
        
        # Data structure to hold results
        root_cause_summary = {}
        cluster_labels_dict = {}
        
        for (assignment_group, resolution_code), group_df in groups:
            if pd.isna(assignment_group) or pd.isna(resolution_code):
                continue
                
            group_key = f"{assignment_group}_{resolution_code}"
            print(f"\nAnalyzing group: {group_key}")
            
            if len(group_df) < 3:
                print(f"Skipping group {group_key} - not enough samples for clustering")
                continue
                
            texts = group_df['processed_description'].tolist()
            
            # Get TF-IDF embeddings
            embeddings, vectorizer = self.get_tfidf_embeddings(texts)
            
            # Apply clustering - try both K-means and DBSCAN
            # First determine optimal K for K-means using elbow method
            distortions = []
            K_range = range(2, min(8, len(texts)))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(embeddings)
                distortions.append(kmeans.inertia_)
            
            # Simple elbow method - find the "elbow" in the distortion curve
            if len(K_range) > 2:
                deltas = np.diff(distortions)
                k_optimal = K_range[np.argmax(np.diff(deltas)) + 1]
            else:
                k_optimal = 2
                
            # Apply K-means with optimal k
            kmeans = KMeans(n_clusters=k_optimal, random_state=42)
            kmeans_labels = kmeans.fit_predict(embeddings)
            
            # Also try DBSCAN for comparison
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            dbscan_labels = dbscan.fit_predict(embeddings)
            
            # Choose clustering method based on number of clusters formed
            kmeans_clusters = len(set(kmeans_labels))
            dbscan_clusters = len(set([x for x in dbscan_labels if x >= 0]))
            
            if dbscan_clusters > 1:
                labels = dbscan_labels
                method = "DBSCAN"
            else:
                labels = kmeans_labels
                method = "K-means"
                
            # Store cluster labels for this group
            group_indices = group_df.index.tolist()
            for idx, label in zip(group_indices, labels):
                cluster_labels_dict[idx] = {
                    'cluster': int(label),
                    'method': method,
                    'group_key': group_key
                }
            
            # Extract important features for each cluster
            # Get top terms that characterize each cluster
            feature_names = vectorizer.get_feature_names_out()
            
            # Analyze root causes for each cluster
            root_causes = {}
            for cluster_id in set(labels):
                if method == "DBSCAN" and cluster_id < 0:
                    # Noise points in DBSCAN
                    cluster_name = "Outliers"
                else:
                    cluster_name = f"Cluster_{cluster_id}"
                
                # Get indices of samples in this cluster
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                
                if cluster_indices:
                    # Extract top terms for this cluster using TF-IDF scores
                    cluster_center = np.mean([embeddings[i] for i in cluster_indices], axis=0)
                    
                    # Get the top features for this cluster
                    top_indices = np.argsort(cluster_center)[-10:][::-1]
                    top_terms = [feature_names[i] for i in top_indices if cluster_center[i] > 0]
                    
                    # Extract sample tickets
                    sample_tickets = [group_df.iloc[i]['incident_number'] for i in cluster_indices[:3]]
                    
                    root_causes[cluster_name] = {
                        'count': len(cluster_indices),
                        'top_terms': top_terms[:5],  # Keep top 5 terms
                        'sample_tickets': sample_tickets
                    }
            
            # Store summary for this group
            root_cause_summary[group_key] = {
                'total_tickets': len(group_df),
                'clustering_method': method,
                'num_clusters': len(root_causes),
                'root_causes': root_causes
            }
        
        # Add cluster labels to the original dataframe
        df['cluster'] = pd.Series(dtype='int')
        df['cluster_method'] = pd.Series(dtype='str')
        df['group_key'] = pd.Series(dtype='str')
        
        for idx, cluster_info in cluster_labels_dict.items():
            df.loc[idx, 'cluster'] = cluster_info['cluster']
            df.loc[idx, 'cluster_method'] = cluster_info['method']
            df.loc[idx, 'group_key'] = cluster_info['group_key']
            
        return df, root_cause_summary
    
    def visualize_similarity(self, similarity_results, pca_results):
        """
        Visualize similarity matrices and PCA results.
        
        Args:
            similarity_results (dict): Dictionary with similarity matrices
            pca_results (dict): Dictionary with PCA results
        """
        # Create visualizations for each resolution code
        for code in similarity_results:
            # Plot similarity heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                similarity_results[code], 
                cmap='YlGnBu', 
                xticklabels=False, 
                yticklabels=False
            )
            plt.title(f'Ticket Similarity Matrix - Resolution Code: {code}')
            plt.tight_layout()
            plt.savefig(f'similarity_heatmap_{code}.png')
            plt.close()
            
            # Plot PCA scatter plot if available
            if code in pca_results:
                plt.figure(figsize=(10, 8))
                plt.scatter(
                    pca_results[code]['pca_data'][:, 0],
                    pca_results[code]['pca_data'][:, 1],
                    alpha=0.7
                )
                
                # Add ticket numbers as annotations for a few points
                for i, incident_number in enumerate(pca_results[code]['incident_numbers']):
                    if i % max(1, len(pca_results[code]['incident_numbers']) // 10) == 0:
                        plt.annotate(
                            incident_number,
                            (pca_results[code]['pca_data'][i, 0], pca_results[code]['pca_data'][i, 1]),
                            fontsize=8
                        )
                        
                plt.title(f'Ticket Similarity (2D PCA) - Resolution Code: {code}')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.tight_layout()
                plt.savefig(f'similarity_pca_{code}.png')
                plt.close()
    
    def visualize_root_causes(self, df, root_cause_summary):
        """
        Visualize root cause analysis results.
        
        Args:
            df (pandas.DataFrame): DataFrame with cluster labels
            root_cause_summary (dict): Summary of root causes by group
        """
        # 1. Create bar chart of ticket counts by assignment group
        plt.figure(figsize=(12, 6))
        group_counts = df['assignment_group'].value_counts().head(10)
        sns.barplot(x=group_counts.index, y=group_counts.values)
        plt.title('Top 10 Assignment Groups by Ticket Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('top_assignment_groups.png')
        plt.close()
        
        # 2. Create bar chart of ticket counts by resolution code
        plt.figure(figsize=(12, 6))
        code_counts = df['resolution_code'].value_counts().head(10)
        sns.barplot(x=code_counts.index, y=code_counts.values)
        plt.title('Top 10 Resolution Codes by Ticket Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('top_resolution_codes.png')
        plt.close()
        
        # 3. Visualize cluster distribution for top groups
        top_groups = sorted(
            root_cause_summary.items(),
            key=lambda x: x[1]['total_tickets'],
            reverse=True
        )[:5]
        
        for group_key, summary in top_groups:
            root_causes = summary['root_causes']
            cluster_names = list(root_causes.keys())
            cluster_counts = [info['count'] for info in root_causes.values()]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=cluster_names, y=cluster_counts)
            plt.title(f'Cluster Distribution for {group_key}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'cluster_distribution_{group_key}.png')
            plt.close()
            
            # 4. Create word cloud of top terms for each cluster
            try:
                from wordcloud import WordCloud
                
                for cluster_name, info in root_causes.items():
                    if not info['top_terms']:
                        continue
                        
                    # Create a frequency dictionary for the word cloud
                    word_freq = {term: 10 - i for i, term in enumerate(info['top_terms'])}
                    
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        max_words=100
                    ).generate_from_frequencies(word_freq)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'Top Terms for {group_key} - {cluster_name}')
                    plt.tight_layout()
                    plt.savefig(f'wordcloud_{group_key}_{cluster_name}.png')
                    plt.close()
                    
            except ImportError:
                print("WordCloud library not available. Skipping word cloud visualization.")
    
    def generate_report(self, df, root_cause_summary):
        """
        Generate a summary report of the analysis.
        
        Args:
            df (pandas.DataFrame): DataFrame with analysis results
            root_cause_summary (dict): Summary of root causes
            
        Returns:
            str: Summary report text
        """
        report = []
        report.append("# Ticket Analysis Summary Report")
        report.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n## Dataset Overview")
        report.append(f"- Total Tickets Analyzed: {len(df)}")
        report.append(f"- Date Range: {df['created_on'].min()} to {df['created_on'].max()}")
        report.append(f"- Unique Assignment Groups: {df['assignment_group'].nunique()}")
        report.append(f"- Unique Resolution Codes: {df['resolution_code'].nunique()}")
        
        report.append("\n## Top Assignment Groups")
        for group, count in df['assignment_group'].value_counts().head(5).items():
            report.append(f"- {group}: {count} tickets")
        
        report.append("\n## Top Resolution Codes")
        for code, count in df['resolution_code'].value_counts().head(5).items():
            report.append(f"- {code}: {count} tickets")
        
        report.append("\n## Root Cause Analysis")
        for group_key, summary in sorted(
            root_cause_summary.items(),
            key=lambda x: x[1]['total_tickets'],
            reverse=True
        )[:10]:
            report.append(f"\n### Group: {group_key}")
            report.append(f"- Total Tickets: {summary['total_tickets']}")
            report.append(f"- Clustering Method: {summary['clustering_method']}")
            report.append(f"- Number of Clusters: {summary['num_clusters']}")
            
            for cluster_name, info in summary['root_causes'].items():
                report.append(f"\n#### {cluster_name} ({info['count']} tickets)")
                report.append(f"- Key Terms: {', '.join(info['top_terms'])}")
                report.append(f"- Sample Tickets: {', '.join(info['sample_tickets'])}")
                
                # Add interpretation of the cluster
                report.append("- Likely Issue: " + self._interpret_cluster(info['top_terms']))
        
        return "\n".join(report)
    
    def _interpret_cluster(self, top_terms):
        """
        Generate a human-readable interpretation of what the cluster might represent
        based on its top terms.
        
        Args:
            top_terms (list): List of top terms characterizing the cluster
            
        Returns:
            str: Interpretation of the cluster
        """
        if not top_terms:
            return "Insufficient data to determine"
            
        # Join the top terms into a phrase
        phrase = " ".join(top_terms[:3])
        
        # Common patterns in IT tickets
        error_terms = ['error', 'fail', 'crash', 'bug', 'issue', 'exception', 'timeout']
        performance_terms = ['slow', 'performance', 'latency', 'delay', 'speed', 'response']
        access_terms = ['access', 'permission', 'login', 'credential', 'password', 'account']
        network_terms = ['network', 'connect', 'connection', 'disconnect', 'vpn', 'wifi']
        
        # Check for matching patterns
        for term in top_terms:
            if any(error in term for error in error_terms):
                return f"Application errors related to {phrase}"
            elif any(perf in term for perf in performance_terms):
                return f"Performance issues involving {phrase}"
            elif any(access in term for access in access_terms):
                return f"Access or authentication problems with {phrase}"
            elif any(net in term for net in network_terms):
                return f"Network connectivity issues affecting {phrase}"
                
        # Generic fallback
        return f"Issues related to {phrase}"
    
    def run_analysis(self, app_id, start_date, end_date):
        """
        Run the complete ticket analysis pipeline.
        
        Args:
            app_id (str): Application ID to filter tickets
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pandas.DataFrame: DataFrame with analysis results
            str: Summary report
        """
        # Step 1: Fetch ticket data
        print(f"Fetching tickets for app_id={app_id} from {start_date} to {end_date}...")
        df = self.fetch_tickets(app_id, start_date, end_date)
        if df is None or len(df) == 0:
            return None, "No tickets found matching the criteria."
        
        # Step 2: Perform similarity analysis
        print("\nPerforming ticket similarity analysis...")
        similarity_results, pca_results = self.perform_similarity_analysis(df)
        
        # Step 3: Classify root causes
        print("\nClassifying tickets to identify root causes...")
        df_with_clusters, root_cause_summary = self.classify_root_causes(df)
        
        # Step 4: Visualize results
        print("\nGenerating visualizations...")
        self.visualize_similarity(similarity_results, pca_results)
        self.visualize_root_causes(df_with_clusters, root_cause_summary)
        
        # Step 5: Generate report
        print("\nGenerating summary report...")
        report = self.generate_report(df_with_clusters, root_cause_summary)
        
        return df_with_clusters, report


# Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'tickets_db',
        'user': 'postgres',
        'password': 'your_password',
        'port': 5432
    }
    
    # Initialize the analyzer
    analyzer = TicketAnalyzer(
        db_config,
        use_bigrams=True,  # Include bigrams for better phrase detection
        max_features=512   # Dimensionality of the embeddings
    )
    
    # Run analysis
    app_id = "APP123"  # Replace with actual app_id
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    results_df, report = analyzer.run_analysis(app_id, start_date, end_date)
    
    # Save results to CSV
    if results_df is not None:
        results_df.to_csv(f"ticket_analysis_{app_id}_{start_date}_to_{end_date}.csv", index=False)
        
    # Save report to text file
    with open(f"ticket_analysis_report_{app_id}_{start_date}_to_{end_date}.md", "w") as f:
        f.write(report)
        
    print("\nAnalysis complete. Results saved to CSV and report files.")
