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
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
import pyLDAvis
import pyLDAvis.gensim_models
import re
import os
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from collections import Counter, defaultdict
import warnings
from wordcloud import WordCloud
import matplotlib.colors as mcolors
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TicketAnalyzer:
    """
    A class for analyzing service desk tickets using advanced NLP techniques.
    Performs ticket similarity analysis, root cause classification, and topic modeling.
    """
    
    def __init__(self, db_config, model_path=None, model_name="roberta-base"):
        """
        Initialize with database configuration and optional local model path.
        
        Args:
            db_config (dict): Database configuration parameters
            model_path (str, optional): Path to locally saved transformer model
            model_name (str): Name of the transformer model to use
                              (e.g., "bert-base-uncased", "roberta-base", "roberta-large")
        """
        self.db_config = db_config
        self.model_name = model_name
        
        # Initialize transformer model and tokenizer from local path if provided
        if model_path:
            print(f"Loading transformer model from local path: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
        else:
            # Try to download from Hugging Face (may fail behind VPN)
            try:
                print(f"Loading transformer model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
            except Exception as e:
                print(f"Error downloading transformer model: {e}")
                print("Please provide a local model path using the model_path parameter.")
                raise
        
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add IT/technical specific stop words
        self.tech_stop_words = {
            'please', 'help', 'issue', 'problem', 'error', 'ticket', 'request',
            'user', 'server', 'system', 'application', 'app', 'service', 'support',
            'hi', 'hello', 'thanks', 'thank', 'regards', 'dear', 'team',
            'incident', 'following', 'needed', 'need', 'requires', 'required',
            'get', 'getting', 'got', 'using', 'used', 'use', 'see', 'seen',
            'facing', 'faced', 'experiencing', 'experienced'
        }
        self.stop_words.update(self.tech_stop_words)
        
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
    
    def preprocess_for_topic_modeling(self, texts):
        """
        Preprocess a list of texts specifically for topic modeling.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of tokenized documents
            gensim.corpora.Dictionary: Dictionary mapping words to IDs
        """
        # Preprocess each text
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Tokenize
        tokenized_texts = [text.split() for text in processed_texts]
        
        # Create dictionary
        dictionary = Dictionary(tokenized_texts)
        
        # Filter out extremes (words that appear in less than 5 documents or more than 50% of documents)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        
        return tokenized_texts, dictionary
    
    def get_transformer_embeddings(self, texts, batch_size=16, max_length=512):
        """
        Generate embeddings for a list of texts using the transformer model.
        
        Args:
            texts (list): List of preprocessed text strings
            batch_size (int): Number of texts to process at once
            max_length (int): Maximum token length for the model
            
        Returns:
            numpy.ndarray: Transformer embeddings matrix
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and prepare inputs
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Generate embeddings (no gradient calculation needed)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # Use CLS token embedding (first token) as sentence embedding
            # For RoBERTa models, still use the first token which serves a similar purpose
            batch_embeddings = model_output.last_hidden_state[:, 0, :].numpy()
            embeddings.extend(batch_embeddings)
            
            print(f"Processed embeddings for {min(i+batch_size, len(texts))}/{len(texts)} texts")
            
        return np.vstack(embeddings)
    
    def perform_similarity_analysis(self, df):
        """
        Perform similarity analysis on ticket descriptions using transformer embeddings.
        
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
            
            # Get transformer embeddings
            embeddings = self.get_transformer_embeddings(texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            similarity_results[code] = similarity_matrix
            
            # PCA for visualization
            if len(embeddings) > 2:
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(embeddings)
                pca_results[code] = {
                    'pca_data': pca_data,
                    'indices': code_df.index.tolist(),
                    'incident_numbers': code_df['incident_number'].tolist()
                }
                
        return similarity_results, pca_results
    
    def perform_topic_modeling(self, df, min_tickets=100):
        """
        Perform topic modeling on ticket descriptions grouped by assignment group.
        Only analyzes groups with at least min_tickets.
        
        Args:
            df (pandas.DataFrame): DataFrame containing ticket data
            min_tickets (int): Minimum number of tickets required for topic modeling
            
        Returns:
            dict: Dictionary with assignment groups as keys and topic modeling results as values
        """
        # Preprocess descriptions if not already done
        if 'processed_description' not in df.columns:
            df['processed_description'] = df['description'].apply(self.preprocess_text)
        
        # Group by assignment_group
        topic_results = {}
        
        # Get assignment groups with enough tickets
        group_counts = df['assignment_group'].value_counts()
        eligible_groups = group_counts[group_counts >= min_tickets].index.tolist()
        
        print(f"\nPerforming topic modeling for {len(eligible_groups)} assignment groups with {min_tickets}+ tickets")
        
        for group in eligible_groups:
            print(f"\nAnalyzing topics for assignment group: {group}")
            group_df = df[df['assignment_group'] == group].copy()
            
            # Skip if no data after filtering
            if len(group_df) < min_tickets:
                continue
                
            # Get processed descriptions
            texts = group_df['processed_description'].tolist()
            
            # Preprocess for topic modeling
            tokenized_texts, dictionary = self.preprocess_for_topic_modeling(texts)
            
            # Create corpus (bag of words)
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            
            # Get optimal number of topics
            coherence_scores = []
            models = {}
            
            # Try different numbers of topics (range depends on dataset size)
            max_topics = min(20, len(texts) // 10)  # Upper bound based on dataset size
            topic_range = range(2, max_topics + 1, 2)  # Step by 2 for efficiency
            
            for num_topics in topic_range:
                print(f"  Training LDA model with {num_topics} topics...")
                lda_model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    random_state=42,
                    passes=10,
                    alpha='auto',
                    per_word_topics=True
                )
                
                models[num_topics] = lda_model
                
                # Calculate coherence score
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=tokenized_texts,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
                coherence_scores.append((num_topics, coherence_score))
                print(f"  Coherence score for {num_topics} topics: {coherence_score:.4f}")
            
            # Find optimal number of topics (highest coherence score)
            optimal_num_topics, best_coherence = max(coherence_scores, key=lambda x: x[1])
            print(f"Optimal number of topics for {group}: {optimal_num_topics} (coherence: {best_coherence:.4f})")
            
            # Get the best model
            best_model = models[optimal_num_topics]
            
            # Extract topics and their keywords
            topics = {}
            for topic_id in range(optimal_num_topics):
                # Get most significant words for this topic
                topic_words = best_model.show_topic(topic_id, topn=15)
                topics[topic_id] = {
                    'words': topic_words,
                    'name': self._generate_topic_name(topic_words)
                }
            
            # Assign topics to tickets
            topic_assignments = []
            for i, doc_bow in enumerate(corpus):
                # Get topic distribution for this document
                topic_dist = best_model.get_document_topics(doc_bow)
                # Find the dominant topic
                dominant_topic = max(topic_dist, key=lambda x: x[1]) if topic_dist else (0, 0)
                topic_assignments.append({
                    'incident_number': group_df.iloc[i]['incident_number'],
                    'topic_id': dominant_topic[0],
                    'topic_probability': dominant_topic[1]
                })
            
            # Create a DataFrame with topic assignments
            topic_df = pd.DataFrame(topic_assignments)
            
            # Generate LDAvis visualization data
            try:
                vis_data = pyLDAvis.gensim_models.prepare(
                    best_model, corpus, dictionary, sort_topics=False
                )
                # Save visualization data
                output_dir = f"ticket_analysis_visualizations/topics_{group.replace(' ', '_')}"
                os.makedirs(output_dir, exist_ok=True)
                pyLDAvis.save_html(vis_data, f"{output_dir}/ldavis.html")
            except Exception as e:
                print(f"Error generating LDAvis visualization: {e}")
                vis_data = None
            
            # Store results for this group
            topic_results[group] = {
                'model': best_model,
                'dictionary': dictionary,
                'corpus': corpus,
                'topics': topics,
                'optimal_num_topics': optimal_num_topics,
                'coherence_score': best_coherence,
                'topic_assignments': topic_df,
                'vis_data': vis_data
            }
            
            # Generate and save topic word clouds
            self._generate_topic_wordclouds(topics, group)
            
        return topic_results
            
    def _generate_topic_name(self, topic_words, max_words=3):
        """
        Generate a human-readable name for a topic based on its most significant words.
        
        Args:
            topic_words (list): List of (word, probability) tuples
            max_words (int): Maximum number of words to include in the name
            
        Returns:
            str: Human-readable topic name
        """
        # Extract words and their probabilities
        words = [word for word, _ in topic_words[:max_words]]
        
        # Common IT-related categories for pattern matching
        categories = {
            'login': ['login', 'password', 'credential', 'authentication', 'access'],
            'network': ['network', 'connection', 'internet', 'wifi', 'vpn', 'disconnect'],
            'hardware': ['hardware', 'device', 'printer', 'scanner', 'keyboard', 'mouse'],
            'software': ['software', 'install', 'update', 'upgrade', 'version'],
            'email': ['email', 'outlook', 'mail', 'message', 'inbox'],
            'database': ['database', 'sql', 'query', 'record', 'data'],
            'error': ['error', 'exception', 'crash', 'failure', 'failed'],
            'performance': ['slow', 'performance', 'speed', 'lag', 'latency'],
            'security': ['security', 'firewall', 'block', 'permission', 'deny']
        }
        
        # Check if any of the top words match a category
        for category, patterns in categories.items():
            if any(word in patterns for word in words):
                return f"{category.title()}: {' '.join(words)}"
        
        # If no category match, just use the top words
        return f"Topic: {' '.join(words)}"
    
    def _generate_topic_wordclouds(self, topics, group_name):
        """
        Generate word cloud visualizations for each topic.
        
        Args:
            topics (dict): Dictionary of topics with their words
            group_name (str): Name of the assignment group
        """
        # Create directory for word clouds
        safe_group_name = re.sub(r'[^\w\-_]', '_', str(group_name))
        output_dir = f"ticket_analysis_visualizations/topics_{safe_group_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Define a color map for the word clouds
        colors = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        
        # Generate a word cloud for each topic
        for topic_id, topic_info in topics.items():
            # Create a dictionary of word: weight
            word_weights = {word: weight for word, weight in topic_info['words']}
            
            # Generate the word cloud
            wc = WordCloud(
                background_color='white',
                max_words=100,
                width=800,
                height=400,
                contour_width=3,
                contour_color='steelblue',
                color_func=lambda *args, **kwargs: colors[topic_id % len(colors)]
            ).generate_from_frequencies(word_weights)
            
            # Plot and save
            plt.figure(figsize=(10, 6))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Topic {topic_id}: {topic_info['name']}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/topic_{topic_id}_wordcloud.png")
            plt.close()
    
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
            
            # Get transformer embeddings
            embeddings = self.get_transformer_embeddings(texts)
            
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
                cluster_texts = [texts[i] for i in cluster_indices]
                
                # Extract top terms for this cluster
                if cluster_texts:
                    # Use zero-shot classification for root cause identification if we have a large cluster
                    if len(cluster_texts) >= 10:
                        # Try to identify root cause using transformer-based approach
                        try:
                            combined_text = " ".join(cluster_texts[:5])  # Use a sample of texts
                            root_cause = self._identify_root_cause(combined_text)
                        except Exception as e:
                            print(f"Error in root cause identification: {e}")
                            root_cause = None
                    else:
                        root_cause = None
                    
                    # Traditional word frequency approach
                    combined_text = " ".join(cluster_texts)
                    words = combined_text.split()
                    word_counts = Counter(words)
                    top_terms = [word for word, count in word_counts.most_common(5)]
                    
                    # Create cluster info
                    cluster_info = {
                        'count': len(cluster_indices),
                        'top_terms': top_terms,
                        'sample_tickets': [group_df.iloc[i]['incident_number'] for i in cluster_indices[:3]]
                    }
                    
                    # Add root cause if available
                    if root_cause:
                        cluster_info['identified_root_cause'] = root_cause
                    
                    root_causes[cluster_name] = cluster_info
            
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
    
    def _identify_root_cause(self, text):
        """
        Use zero-shot classification to identify the root cause category.
        
        Args:
            text (str): Combined text from cluster
            
        Returns:
            str: Identified root cause category
        """
        try:
            # Create a zero-shot classification pipeline
            classifier = pipeline(
                "zero-shot-classification", 
                model=self.model_name
            )
            
            # Define possible root cause categories
            candidate_labels = [
                "authentication issue", 
                "network connectivity", 
                "software bug", 
                "hardware failure",
                "configuration error", 
                "permission issue", 
                "performance degradation",
                "data corruption", 
                "security incident", 
                "user error"
            ]
            
            # Classify the text
            result = classifier(text, candidate_labels)
            
            # Return the top category
            return result['labels'][0]
        except Exception as e:
            print(f"Error in zero-shot classification: {e}")
            return None
    
    def visualize_similarity(self, similarity_results, pca_results):
        """
        Visualize similarity matrices and PCA results.
        
        Args:
            similarity_results (dict): Dictionary with similarity matrices
            pca_results (dict): Dictionary with PCA results
        """
        # Create output directory if it doesn't exist
        output_dir = "ticket_analysis_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations for each resolution code
        for code in similarity_results:
            # Create code-specific directory for visualizations
            # Replace any characters that might be invalid in filenames
            safe_code = re.sub(r'[^\w\-_]', '_', str(code))
            code_dir = os.path.join(output_dir, safe_code)
            os.makedirs(code_dir, exist_ok=True)
            
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
            output_file = os.path.join(code_dir, 'similarity_heatmap.png')
            plt.savefig(output_file)
            print(f"Saved heatmap to {output_file}")
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
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.tight_layout()
                output_file = os.path.join(code_dir, 'similarity_pca.png')
                plt.savefig(output_file)
                print(f"Saved PCA plot to {output_file}")
                plt.close()
    
    def visualize_root_causes(self, df, root_cause_summary):
        """
        Visualize root cause analysis results.
        
        Args:
            df (pandas.DataFrame): DataFrame with cluster labels
            root_cause_summary (dict): Summary of root causes by group
        """
        # Create output directory if it doesn't exist
        output_dir = "ticket_analysis_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Create bar chart of ticket counts by assignment group
        plt.figure(figsize=(12, 6))
        group_counts = df['assignment_group'].value_counts().head(10)
        sns.barplot(x=group_counts.index, y=group_counts.values)
        plt.title('Top 10 Assignment Groups by Ticket Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'top_assignment_groups.png')
        plt.savefig(output_file)
        print(f"Saved assignment groups chart to {output_file}")
        plt.close()
        
        # 2. Create bar chart of ticket counts by resolution code
        plt.figure(figsize=(12, 6))
        code_counts = df['resolution_code'].value_counts().head(10)
        sns.barplot(x=code_counts.index, y=code_counts.values)
        plt.title('Top 10 Resolution Codes by Ticket Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'top_resolution_codes.png')
        plt.savefig(output_file)
        print(f"Saved resolution codes chart to {output_file}")
        plt.close()
        
        # 3. Visualize cluster distribution for top groups
        top_groups = sorted(
            root_cause_summary.items(),
            key=lambda x: x[1]['total_tickets'],
            reverse=True
        )[:5]
        
        for group_key, summary in top_groups:
            # Create a safe filename by replacing invalid characters
            safe_group_key = re.sub(r'[^\w\-_]', '_', str(group_key))
            group_dir = os.path.join(output_dir, safe_group_key)
            os.makedirs(group_dir, exist_ok=True)
            
            root_causes = summary['root_causes']
            cluster_names = list(root_causes.keys())
            cluster_counts = [info['count'] for info in root_causes.values()]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=cluster_names, y=cluster_counts)
            plt.title(f'Cluster Distribution for {group_key}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            output_file = os.path.join(group_dir, 'cluster_distribution.png')
            plt.savefig(output_file)
            print(f"Saved cluster distribution chart to {output_file}")
            plt.close()
    
    def visualize_topics(self, topic_results, df):
        """
        Visualize topic modeling results.
        
        Args:
            topic_results (dict): Dictionary with topic modeling results
            df (pandas.DataFrame): Original DataFrame with ticket data
        """
        # Create output directory
        output_dir = "ticket_analysis_visualizations/topics"
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot topic distribution by assignment group
        for group, results in topic_results.items():
            # Create a safe filename
            safe_group = re.sub(r'[^\w\-_]', '_', str(group))
            group_dir = os.path.join(output_dir, safe_group)
            os.makedirs(group_dir, exist_ok=True)
            
            # Get topic assignments
            topic_df = results['topic_assignments']
            
            # Count tickets per topic
            topic_counts = topic_df['topic_id'].value_counts().sort_index()
            
            # Get topic names
            topic_names = [f"T{topic_id}: {info['name']}" for topic_id, info in results['topics'].items()]
            
            # Create bar chart of tickets per topic
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(topic_counts)), topic_counts.values)
            
            # Add topic names as labels
            plt.xticks(
                range(len(topic_counts)),
                [topic_names[i] if i < len(topic_names) else f"T{i}" for i in topic_counts.index],
                rotation=45,
                ha='right'
            )
            
            plt.title(f'Topic Distribution for {group}')
            plt.xlabel('Topics')
            plt.ylabel('Number of Tickets')
            plt.tight_layout()
            
            # Save plot
            output_file = os.path.join(group_dir, 'topic_distribution.png')
            plt.savefig(output_file)
            print(f"Saved topic distribution chart to {output_file}")
            plt.close()
            
            # Generate and save word clouds for each topic (if not already generated)
            for topic_id, topic_info in results['topics'].items():
                wordcloud_file = os.path.join(group_dir, f'topic_{topic_id}_wordcloud.png')
                if not os.path.exists(wordcloud_file):
                    # Create a dictionary of word: weight
                    word_weights = {word: weight for word, weight in topic_info['words']}
                    
                    # Generate the word cloud
                    wc = WordCloud(
                        background_color='white',
                        max_words=100,
                        width=800,
                        height=400,
                        contour_width=3,
                        contour_color='steelblue'
                    ).generate_from_frequencies(word_weights)
                    
                    # Plot and save
                    plt.figure(figsize=(10, 6))
                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f"Topic {topic_id}: {topic_info['name']}")
                    plt.tight_layout()
                    plt.savefig(wordcloud_file)
                    plt.close()
                    print(f"Saved word cloud to {wordcloud_file}")
    
    def generate_report(self, df, root_cause_summary, topic_results=None):
        """
        Generate a summary report of the analysis.
        
        Args:
            df (pandas.DataFrame): DataFrame with analysis results
            root_cause_summary (dict): Summary of root causes
            topic_results (dict, optional): Results from topic modeling
            
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
        
        # Include topic modeling results if available
        if topic_results:
            report.append("\n## Topic Modeling Results")
            for group, results in topic_results.items():
                report.append(f"\n### Assignment Group: {group}")
                report.append(f"- Total Tickets: {results['topic_assignments'].shape[0]}")
                report.append(f"- Optimal Number of Topics: {results['optimal_num_topics']}")
                report.append(f"- Coherence Score: {results['coherence_score']:.4f}")
                
                report.append("\n#### Identified Topics:")
                for topic_id, topic_info in results['topics'].items():
                    # Get count of tickets in this topic
                    topic_count = results['topic_assignments'][results['topic_assignments']['topic_id'] == topic_id].shape[0]
                    percentage = 100 * topic_count / results['topic_assignments'].shape[0]
                    
                    report.append(f"\n##### Topic {topic_id}: {topic_info['name']}")
                    report.append(f"- Tickets: {topic_count} ({percentage:.1f}%)")
                    report.append(f"- Key Terms: {', '.join([word for word, _ in topic_info['words'][:7]])}")
                    
                    # Add example tickets
                    example_tickets = results['topic_assignments'][results['topic_assignments']['topic_id'] == topic_id]['incident_number'].head(3).tolist()
                    if example_tickets:
                        report.append(f"- Example Tickets: {', '.join(example_tickets)}")
                
                report.append(f"\n[LDA Visualization for {group}](ticket_analysis_visualizations/topics_{group.replace(' ', '_')}/ldavis.html)")
        
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
                if 'identified_root_cause' in info:
                    report.append(f"- Identified Root Cause: {info['identified_root_cause']}")
                report.append(f"- Key Terms: {', '.join(info['top_terms'])}")
                report.append(f"- Sample Tickets: {', '.join(info['sample_tickets'])}")
        
        return "\n".join(report)
    
    def run_analysis(self, app_id, start_date, end_date, perform_topic_modeling=True, min_tickets_for_topics=100):
        """
        Run the complete ticket analysis pipeline.
        
        Args:
            app_id (str): Application ID to filter tickets
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            perform_topic_modeling (bool): Whether to perform topic modeling
            min_tickets_for_topics (int): Minimum number of tickets required for topic modeling
            
        Returns:
            pandas.DataFrame: DataFrame with analysis results
            str: Summary report
            dict: Topic modeling results (if performed)
        """
        # Step 1: Fetch ticket data
        print(f"Fetching tickets for app_id={app_id} from {start_date} to {end_date}...")
        df = self.fetch_tickets(app_id, start_date, end_date)
        if df is None or len(df) == 0:
            return None, "No tickets found matching the criteria.", None
        
        # Step 2: Perform similarity analysis
        print("\nPerforming ticket similarity analysis...")
        similarity_results, pca_results = self.perform_similarity_analysis(df)
        
        # Step 3: Perform topic modeling (if requested)
        topic_results = None
        if perform_topic_modeling:
            print("\nPerforming topic modeling for large assignment groups...")
            topic_results = self.perform_topic_modeling(df, min_tickets=min_tickets_for_topics)
        
        # Step 4: Classify root causes
        print("\nClassifying tickets to identify root causes...")
        df_with_clusters, root_cause_summary = self.classify_root_causes(df)
        
        # Step 5: Visualize results
        print("\nGenerating visualizations...")
        self.visualize_similarity(similarity_results, pca_results)
        self.visualize_root_causes(df_with_clusters, root_cause_summary)
        
        if topic_results:
            self.visualize_topics(topic_results, df_with_clusters)
        
        # Step 6: Generate report
        print("\nGenerating summary report...")
        report = self.generate_report(df_with_clusters, root_cause_summary, topic_results)
        
        return df_with_clusters, report, topic_results


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
    
    # Path to locally saved transformer model
    local_model_path = "./transformer_model_cache"  # Update this to your local model path
    
    # Choose transformer model
    model_name = "roberta-base"  # More sophisticated model: roberta-base or roberta-large
    
    # Initialize the analyzer with local model path or model name
    analyzer = TicketAnalyzer(
        db_config, 
        model_path=local_model_path,  # Use this if you have a local model
        # model_name=model_name  # Use this if downloading from Hugging Face
    )
    
    # Run analysis
    app_id = "APP123"  # Replace with actual app_id
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    results_df, report, topic_results = analyzer.run_analysis(
        app_id, 
        start_date, 
        end_date,
        perform_topic_modeling=True,
        min_tickets_for_topics=100  # Only analyze groups with 100+ tickets
    )
    
    # Save results to CSV
    if results_df is not None:
        results_df.to_csv(f"ticket_analysis_{app_id}_{start_date}_to_{end_date}.csv", index=False)
        
    # Save report to markdown file
    with open(f"ticket_analysis_report_{app_id}_{start_date}_to_{end_date}.md", "w") as f:
        f.write(report)
        
    print("\nAnalysis complete. Results saved to CSV and report files.")
