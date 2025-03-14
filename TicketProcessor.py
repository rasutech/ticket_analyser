import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Union


class TicketProcessor:
    """
    A class to handle the complete ticket processing pipeline:
    1. Extract transaction IDs from ticket descriptions
    2. Enrich with data from master_order table
    3. Generate embeddings using BERT
    4. Determine optimal cluster count
    5. Perform K-means clustering
    """
    
    def __init__(
        self,
        tickets_db_uri: str,
        enrichment_db_uri: str,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the TicketProcessor with database connections and embedding model.
        
        Args:
            tickets_db_uri: SQLAlchemy connection string for the tickets database
            enrichment_db_uri: SQLAlchemy connection string for the enrichment database
            embedding_model: SentenceTransformer model to use for embeddings
        """
        self.tickets_engine = create_engine(tickets_db_uri)
        self.enrichment_engine = create_engine(enrichment_db_uri)
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Regular expression for finding transaction IDs (customize as needed)
        # This pattern looks for "order", "transaction", "id", "number" followed by digits
        # Customize this regex pattern based on your actual transaction ID format
        self.transaction_pattern = re.compile(
            r'(?:order|transaction|txn|id|number)[^\d]*(\d+)',
            re.IGNORECASE
        )
    
    def extract_tickets_data(self) -> pd.DataFrame:
        """
        Extract ticket data from the tickets database.
        
        Returns:
            DataFrame containing ticket information
        """
        query = """
        SELECT 
            incident_number,
            description,
            short_description,
            resolution_code,
            resolution_notes,
            assignment_group
        FROM TICKETS
        """
        return pd.read_sql(query, self.tickets_engine)
    
    def extract_transaction_ids(self, text: str) -> List[str]:
        """
        Extract transaction IDs from a text field.
        
        Args:
            text: Text to search for transaction IDs
            
        Returns:
            List of transaction IDs found in the text
        """
        if pd.isna(text) or not isinstance(text, str):
            return []
            
        matches = self.transaction_pattern.findall(text)
        return matches
    
    def enrich_with_transaction_data(self, tickets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract transaction IDs from ticket descriptions and enrich with data
        from the master_order table.
        
        Args:
            tickets_df: DataFrame containing ticket information
            
        Returns:
            Enriched DataFrame with transaction information
        """
        # Create a new DataFrame to store the enriched data
        enriched_df = tickets_df.copy()
        
        # Add columns for transaction IDs and enrichment data
        enriched_df['transaction_ids'] = None
        enriched_df['order_types'] = None
        enriched_df['order_flows'] = None
        enriched_df['order_activities'] = None
        
        # Process each ticket
        for idx, row in enriched_df.iterrows():
            # Extract transaction IDs from all text fields
            transaction_ids = []
            for field in ['description', 'short_description', 'resolution_notes']:
                if pd.notna(row[field]):
                    transaction_ids.extend(self.extract_transaction_ids(row[field]))
            
            # Remove duplicates while preserving order
            unique_ids = []
            for tid in transaction_ids:
                if tid not in unique_ids:
                    unique_ids.append(tid)
            
            if unique_ids:
                # Fetch enrichment data for each transaction ID
                order_types, order_flows, order_activities = [], [], []
                
                for tid in unique_ids:
                    query = f"""
                    SELECT order_type, order_flow, order_activity
                    FROM master_order
                    WHERE order_number = '{tid}'
                    """
                    try:
                        result = pd.read_sql(query, self.enrichment_engine)
                        if not result.empty:
                            order_types.append(result['order_type'].iloc[0])
                            order_flows.append(result['order_flow'].iloc[0])
                            order_activities.append(result['order_activity'].iloc[0])
                    except Exception as e:
                        print(f"Error fetching data for transaction ID {tid}: {e}")
                
                enriched_df.at[idx, 'transaction_ids'] = '|'.join(unique_ids)
                enriched_df.at[idx, 'order_types'] = '|'.join(order_types) if order_types else None
                enriched_df.at[idx, 'order_flows'] = '|'.join(order_flows) if order_flows else None
                enriched_df.at[idx, 'order_activities'] = '|'.join(order_activities) if order_activities else None
        
        return enriched_df
    
    def prepare_text_for_embedding(self, row: pd.Series) -> str:
        """
        Prepare a combined text representation from ticket data for embedding.
        
        Args:
            row: Series containing a single ticket record
            
        Returns:
            Combined text ready for embedding
        """
        components = []
        
        # Add ticket description fields
        for field in ['description', 'short_description']:
            if pd.notna(row[field]):
                components.append(str(row[field]))
        
        # Add resolution information
        if pd.notna(row['resolution_code']):
            components.append(f"Resolution Code: {row['resolution_code']}")
        
        if pd.notna(row['resolution_notes']):
            components.append(f"Resolution Notes: {row['resolution_notes']}")
        
        # Add enrichment data
        if pd.notna(row['order_types']):
            components.append(f"Order Types: {row['order_types']}")
        
        if pd.notna(row['order_flows']):
            components.append(f"Order Flows: {row['order_flows']}")
        
        if pd.notna(row['order_activities']):
            components.append(f"Order Activities: {row['order_activities']}")
        
        # Combine all components
        return " ".join(components)
    
    def generate_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate BERT embeddings for each ticket in the DataFrame.
        
        Args:
            df: DataFrame containing enriched ticket information
            
        Returns:
            Array of embeddings
        """
        # Prepare the text data for each ticket
        texts = df.apply(self.prepare_text_for_embedding, axis=1).tolist()
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        return embeddings
    
    def determine_optimal_clusters(
        self, 
        embeddings: np.ndarray,
        max_clusters: int = 20,
        method: str = 'both'
    ) -> Tuple[int, Optional[plt.Figure]]:
        """
        Determine the optimal number of clusters using the silhouette method,
        elbow method, or both.
        
        Args:
            embeddings: Array of embeddings
            max_clusters: Maximum number of clusters to consider
            method: Method to use ('silhouette', 'elbow', or 'both')
            
        Returns:
            Tuple containing the optimal number of clusters and a figure with the plots
        """
        # Standardize the embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        range_n_clusters = range(2, min(max_clusters + 1, len(embeddings) // 5))
        
        if method in ['silhouette', 'both']:
            silhouette_scores = []
            for n_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                labels = kmeans.fit_predict(scaled_embeddings)
                
                # Skip if there's only one cluster (silhouette score isn't defined)
                if len(set(labels)) <= 1:
                    silhouette_scores.append(0)
                    continue
                
                silhouette_avg = silhouette_score(scaled_embeddings, labels)
                silhouette_scores.append(silhouette_avg)
        
        if method in ['elbow', 'both']:
            wcss = []  # Within-Cluster Sum of Square
            for n_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                kmeans.fit(scaled_embeddings)
                wcss.append(kmeans.inertia_)
        
        # Create plots
        fig = None
        if method in ['silhouette', 'both'] or method in ['elbow', 'both']:
            fig, ax = plt.subplots(1, 2 if method == 'both' else 1, figsize=(15, 5) if method == 'both' else (8, 5))
            
            if method == 'both':
                # Silhouette plot
                ax[0].plot(list(range_n_clusters), silhouette_scores)
                ax[0].set_title('Silhouette Method')
                ax[0].set_xlabel('Number of clusters')
                ax[0].set_ylabel('Silhouette Score')
                ax[0].grid(True)
                
                # Elbow plot
                ax[1].plot(list(range_n_clusters), wcss)
                ax[1].set_title('Elbow Method')
                ax[1].set_xlabel('Number of clusters')
                ax[1].set_ylabel('WCSS')
                ax[1].grid(True)
            elif method == 'silhouette':
                ax.plot(list(range_n_clusters), silhouette_scores)
                ax.set_title('Silhouette Method')
                ax.set_xlabel('Number of clusters')
                ax.set_ylabel('Silhouette Score')
                ax.grid(True)
            else:  # method == 'elbow'
                ax.plot(list(range_n_clusters), wcss)
                ax.set_title('Elbow Method')
                ax.set_xlabel('Number of clusters')
                ax.set_ylabel('WCSS')
                ax.grid(True)
            
            plt.tight_layout()
        
        # Determine optimal number of clusters
        if method == 'silhouette':
            optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
        elif method == 'elbow':
            # Use the kneedle algorithm or a simple heuristic to find the elbow point
            # For simplicity, using a heuristic approach here
            wcss_diff = np.diff(wcss)
            wcss_diff_norm = wcss_diff / wcss[:-1]
            elbow_point = np.argmin(wcss_diff_norm) + 2  # +2 because range starts from 2
            optimal_clusters = elbow_point
        else:  # method == 'both'
            # Combine both methods (simple average)
            silhouette_optimal = range_n_clusters[np.argmax(silhouette_scores)]
            
            wcss_diff = np.diff(wcss)
            wcss_diff_norm = wcss_diff / wcss[:-1]
            elbow_point = np.argmin(wcss_diff_norm) + 2
            
            optimal_clusters = int((silhouette_optimal + elbow_point) / 2)
        
        return optimal_clusters, fig
    
    def perform_clustering(
        self, 
        embeddings: np.ndarray, 
        n_clusters: int
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform K-means clustering on the embeddings.
        
        Args:
            embeddings: Array of embeddings
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with cluster assignments and clustering results
        """
        # Standardize the embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_embeddings)
        
        # Calculate silhouette score (if more than one cluster)
        silhouette_avg = None
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(scaled_embeddings, cluster_labels)
        
        # Create results dictionary
        results = {
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_avg,
            'model': kmeans
        }
        
        return cluster_labels, results
    
    def analyze_clusters(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray
    ) -> Tuple[Dict, plt.Figure]:
        """
        Analyze the clusters to understand their characteristics.
        
        Args:
            df: DataFrame containing enriched ticket information
            cluster_labels: Array of cluster assignments
            
        Returns:
            Dictionary with cluster summaries and visualization figure
        """
        # Add cluster labels to the DataFrame
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Compute cluster sizes
        cluster_sizes = df_with_clusters['cluster'].value_counts().sort_index()
        
        # Create cluster summaries
        cluster_summaries = {}
        for cluster_id in range(len(cluster_sizes)):
            cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # Most common resolution codes
            if 'resolution_code' in cluster_df.columns:
                resolution_codes = cluster_df['resolution_code'].value_counts().head(3)
            else:
                resolution_codes = pd.Series()
            
            # Most common assignment groups
            if 'assignment_group' in cluster_df.columns:
                assignment_groups = cluster_df['assignment_group'].value_counts().head(3)
            else:
                assignment_groups = pd.Series()
            
            # Most common order types, flows, activities
            order_types = self._extract_most_common(cluster_df, 'order_types')
            order_flows = self._extract_most_common(cluster_df, 'order_flows')
            order_activities = self._extract_most_common(cluster_df, 'order_activities')
            
            # Sample incidents
            sample_incidents = cluster_df['incident_number'].sample(
                min(5, len(cluster_df))
            ).tolist()
            
            # Create summary
            cluster_summaries[cluster_id] = {
                'size': len(cluster_df),
                'percentage': len(cluster_df) / len(df_with_clusters) * 100,
                'top_resolution_codes': resolution_codes.to_dict(),
                'top_assignment_groups': assignment_groups.to_dict(),
                'top_order_types': order_types,
                'top_order_flows': order_flows,
                'top_order_activities': order_activities,
                'sample_incidents': sample_incidents
            }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot cluster sizes
        ax1.bar(cluster_sizes.index, cluster_sizes.values)
        ax1.set_title('Cluster Sizes')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Number of Tickets')
        ax1.set_xticks(cluster_sizes.index)
        
        # Plot cluster percentages
        percentages = (cluster_sizes / cluster_sizes.sum()) * 100
        ax2.bar(percentages.index, percentages.values)
        ax2.set_title('Cluster Percentages')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Percentage of Tickets')
        ax2.set_xticks(percentages.index)
        
        plt.tight_layout()
        
        return cluster_summaries, fig
    
    def _extract_most_common(self, df: pd.DataFrame, column: str, top_n: int = 3) -> Dict:
        """
        Extract most common values from a pipe-separated column.
        
        Args:
            df: DataFrame to analyze
            column: Column name to analyze
            top_n: Number of top values to return
            
        Returns:
            Dictionary of most common values and their counts
        """
        if column not in df.columns:
            return {}
            
        # Flatten the pipe-separated values
        all_values = []
        for value in df[column].dropna():
            all_values.extend(value.split('|'))
        
        # Count occurrences
        value_counts = pd.Series(all_values).value_counts().head(top_n)
        
        return value_counts.to_dict()
    
    def run_pipeline(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Run the complete pipeline:
        1. Extract ticket data
        2. Enrich with transaction data
        3. Generate embeddings
        4. Determine optimal clusters
        5. Perform clustering
        6. Analyze clusters
        
        Returns:
            Tuple containing:
            - DataFrame with enriched data and cluster assignments
            - Dictionary with clustering results
            - Dictionary with cluster summaries
        """
        # Extract ticket data
        print("Extracting ticket data...")
        tickets_df = self.extract_tickets_data()
        print(f"Extracted {len(tickets_df)} tickets.")
        
        # Enrich with transaction data
        print("Enriching with transaction data...")
        enriched_df = self.enrich_with_transaction_data(tickets_df)
        print("Enrichment complete.")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.generate_embeddings(enriched_df)
        print(f"Generated embeddings with shape {embeddings.shape}.")
        
        # Determine optimal number of clusters
        print("Determining optimal number of clusters...")
        n_clusters, cluster_fig = self.determine_optimal_clusters(
            embeddings, 
            max_clusters=20, 
            method='both'
        )
        print(f"Optimal number of clusters: {n_clusters}")
        
        # Perform clustering
        print("Performing clustering...")
        cluster_labels, clustering_results = self.perform_clustering(
            embeddings, 
            n_clusters
        )
        
        # Add cluster labels to the DataFrame
        enriched_df['cluster'] = cluster_labels
        
        # Analyze clusters
        print("Analyzing clusters...")
        cluster_summaries, analysis_fig = self.analyze_clusters(
            enriched_df, 
            cluster_labels
        )
        
        # Save results and figures
        cluster_fig.savefig('optimal_clusters.png')
        analysis_fig.savefig('cluster_analysis.png')
        
        clustering_results['figures'] = {
            'optimal_clusters': cluster_fig,
            'cluster_analysis': analysis_fig
        }
        
        print("Pipeline completed successfully.")
        
        return enriched_df, clustering_results, cluster_summaries


# Example usage
if __name__ == "__main__":
    # Database connection strings (replace with your actual credentials)
    tickets_db_uri = "postgresql://username:password@host:port/tickets_db"
    enrichment_db_uri = "postgresql://username:password@host:port/enrichment_db"
    
    # Initialize the processor
    processor = TicketProcessor(
        tickets_db_uri=tickets_db_uri,
        enrichment_db_uri=enrichment_db_uri,
        embedding_model="all-MiniLM-L6-v2"  # A good balance of quality and speed
    )
    
    # Run the pipeline
    enriched_df, clustering_results, cluster_summaries = processor.run_pipeline()
    
    # Save results
    enriched_df.to_csv('enriched_tickets_with_clusters.csv', index=False)
    
    # Display cluster summaries
    for cluster_id, summary in cluster_summaries.items():
        print(f"\nCluster {cluster_id} Summary:")
        print(f"Size: {summary['size']} tickets ({summary['percentage']:.2f}%)")
        
        print("\nTop Resolution Codes:")
        for code, count in summary['top_resolution_codes'].items():
            print(f"  - {code}: {count}")
        
        print("\nTop Assignment Groups:")
        for group, count in summary['top_assignment_groups'].items():
            print(f"  - {group}: {count}")
        
        print("\nTop Order Types:")
        for order_type, count in summary['top_order_types'].items():
            print(f"  - {order_type}: {count}")
        
        print("\nSample Incidents:")
        for incident in summary['sample_incidents']:
            print(f"  - {incident}")
        
        print("\n" + "-"*50)
