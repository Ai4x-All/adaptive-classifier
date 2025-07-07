import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import logging
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .models import Example, ModelConfig

logger = logging.getLogger(__name__)

class PrototypeMemory:
    """Memory system that maintains prototypes for each class, backed by PostgreSQL with pgvector."""

    def __init__(
        self,
        embedding_dim: int,
        config: Optional[ModelConfig] = None,
        db_config: Dict[str, Any] = None # New: Database connection details
    ):
        """Initialize the prototype memory system.

        Args:
            embedding_dim: Dimension of the embeddings
            config: Optional model configuration
            db_config: Dictionary containing PostgreSQL connection parameters (host, dbname, user, password, port)
        """
        self.embedding_dim = embedding_dim
        self.config = config or ModelConfig()

        if db_config is None:
            raise ValueError("db_config must be provided for PostgreSQL backend.")
        self.db_config = db_config

        # Initialize storage (these will now be reflections of database state)
        self.examples = defaultdict(list)  # label -> List[Example] - primarily used for pruning/aggregation in memory before persisting
        self.prototypes = {}  # label -> tensor - cache of current prototypes from DB
        self.strategic_prototypes = {}  # label -> strategic prototype tensor - still in-memory for simplicity

        self.conn = None # Database connection
        self._init_db()

        # Statistics (might need to query DB for precise counts now)
        self.updates_since_rebuild = 0 # Still useful for controlling _update_prototype frequency
        
        # Load existing prototypes from DB on initialization
        self._load_prototypes_from_db()


    def _init_db(self):
        """Initialize database connection and create tables if they don't exist."""
        try:
            self.conn = psycopg.connect(**self.db_config, row_factory=dict_row)
            register_vector(self.conn)
            with self.conn.cursor() as cur:
                # Enable pgvector extension if not already enabled
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create prototypes table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS prototypes (
                        label TEXT PRIMARY KEY,
                        embedding VECTOR({self.embedding_dim}),
                        num_examples INTEGER DEFAULT 0
                    );
                """)
                # Create examples table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS examples (
                        id SERIAL PRIMARY KEY,
                        label TEXT NOT NULL,
                        text TEXT,
                        embedding VECTOR({self.embedding_dim}),
                        added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            self.conn.commit()
            logger.info("PostgreSQL tables checked/created successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to or initialize database: {e}")
            raise

    def _load_prototypes_from_db(self):
        """Load all prototypes from the database into memory cache."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT label, embedding FROM prototypes;")
                for row in cur:
                    # pgvector returns numpy array, convert to torch.Tensor
                    self.prototypes[row['label']] = torch.from_numpy(row['embedding']).float()
            logger.info(f"Loaded {len(self.prototypes)} prototypes from database.")
        except Exception as e:
            logger.error(f"Failed to load prototypes from database: {e}")
            raise


    def add_example(self, example: Example, label: str):
        """Add a new example to memory and update prototype."""
        if example.embedding is None:
            raise ValueError("Example must have an embedding")
        if example.embedding.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Example embedding dimension {example.embedding.size(-1)} "
                f"does not match memory dimension {self.embedding_dim}"
            )
        
        embedding_np = example.embedding.cpu().numpy()

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO examples (label, text, embedding) VALUES (%s, %s, %s);",
                    (label, example.text, embedding_np)
                )
                self.conn.commit()
                logger.debug(f"Added example for label '{label}' to database.")

            self.examples[label].append(example)
            
            if self.config.max_examples_per_class > 0:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM examples WHERE label = %s;", (label,))
                    current_count = cur.fetchone()['count']

                if current_count > self.config.max_examples_per_class:
                    logger.debug(f"Pruning examples for label '{label}'. Current: {current_count}, Max: {self.config.max_examples_per_class}")
                    self._prune_examples(label)
            
            self._update_prototype(label)
            
            self.updates_since_rebuild += 1
            
            # **FIX STARTS HERE**
            # Call _rebuild_index (which is _load_prototypes_from_db) if update frequency is met
            if self.config.prototype_update_frequency > 0 and \
               self.updates_since_rebuild >= self.config.prototype_update_frequency:
                logger.info(f"Prototype update frequency met ({self.updates_since_rebuild}/{self.config.prototype_update_frequency}). Refreshing local prototype cache.")
                self._rebuild_index() # This will reset updates_since_rebuild to 0
            # **FIX ENDS HERE**

        except Exception as e:
            logger.error(f"Error adding example or updating prototype: {e}")
            self.conn.rollback()
            raise

    def get_nearest_prototypes(
            self,
            query_embedding: torch.Tensor,
            k: int = 5,
            min_similarity: Optional[float] = None
        ) -> List[Tuple[str, float]]:
        """Find the nearest prototype neighbors for a query using pgvector."""
        if not self.prototypes: # Check if there are any prototypes loaded (or in DB)
            logger.warning("No prototypes available to search.")
            return []

        # Ensure the query embedding is on CPU and convert to numpy for pgvector
        query_np = query_embedding.cpu().numpy()

        results = []
        try:
            with self.conn.cursor() as cur:
                # Using L2 distance operator (<->) provided by pgvector
                # We order by distance ascending and limit to k
                cur.execute(f"""
                    SELECT label, embedding <-> %s AS distance
                    FROM prototypes
                    ORDER BY distance
                    LIMIT %s;
                """, (query_np, k))
                
                rows = cur.fetchall()
                if not rows:
                    return []

                # Convert distances to similarities (similar to original FAISS logic)
                # Smaller distance means higher similarity
                distances = np.array([row['distance'] for row in rows])
                similarities = np.exp(-distances) # Using exp(-distance) for similarity

                # Collect results
                for row, similarity in zip(rows, similarities):
                    label = row['label']
                    score = float(similarity)
                    if min_similarity is None or score >= min_similarity:
                        results.append((label, score))
            
            # Normalize scores with softmax if results exist
            if results:
                # Only take the scores for softmax normalization
                scores_tensor = torch.tensor([score for _, score in results], dtype=torch.float32)
                normalized_scores = torch.nn.functional.softmax(scores_tensor, dim=0)
                
                results = [
                    (label, float(norm_score)) 
                    for (label, _), norm_score in zip(results, normalized_scores)
                ]
                # Sort again after softmax, just in case (though softmax preserves order if scores were already sorted)
                results.sort(key=lambda x: x[1], reverse=True)
            
            return results

        except Exception as e:
            logger.error(f"Error getting nearest prototypes from database: {e}")
            raise

    def _update_prototype(self, label: str):
        """Update the prototype for a given label in the database.
            This will recalculate the prototype based on current examples in DB.
        """
        try:
            with self.conn.cursor() as cur:
                # Get all embeddings for the given label from the examples table
                cur.execute("SELECT embedding FROM examples WHERE label = %s;", (label,))
                embeddings_np = [row['embedding'] for row in cur.fetchall()]

                if not embeddings_np:
                    # If no examples for this label, remove prototype
                    cur.execute("DELETE FROM prototypes WHERE label = %s;", (label,))
                    self.prototypes.pop(label, None) # Also remove from in-memory cache
                    self.conn.commit()
                    logger.debug(f"Removed prototype for label '{label}' as no examples exist.")
                    return

                # Convert numpy arrays back to torch tensors for mean calculation
                embeddings = torch.stack([torch.from_numpy(emb).float() for emb in embeddings_np])
                
                # Ensure calculations are done on CPU
                if embeddings.is_cuda:
                    embeddings = embeddings.cpu()

                prototype_tensor = torch.mean(embeddings, dim=0)
                prototype_np = prototype_tensor.numpy() # Convert back to numpy for pgvector

                # Update or insert the prototype in the prototypes table
                cur.execute(f"""
                    INSERT INTO prototypes (label, embedding, num_examples)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (label) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        num_examples = EXCLUDED.num_examples;
                """, (label, prototype_np, len(embeddings_np)))
                self.conn.commit()

                # Update in-memory cache
                self.prototypes[label] = prototype_tensor
                logger.debug(f"Updated prototype for label '{label}' in database and cache.")

        except Exception as e:
            logger.error(f"Error updating prototype for label '{label}': {e}")
            self.conn.rollback()
            raise

    def _rebuild_index(self):
        """With pgvector, the 'index' is implicitly managed by the database.
            This method can be repurposed to ensure our in-memory cache is synced
            with the database, or to perform VACUUM/ANALYZE for performance.
            The original FAISS _rebuild_index concept is not directly applicable.
        """
        logger.info("Rebuilding index for pgvector is implicitly handled by database. Refreshing local prototype cache.")
        self._load_prototypes_from_db() # Simply refresh local cache
        self.updates_since_rebuild = 0
        # self.just_rebuilt = False # This flag is less relevant now

    def _restore_from_save(self):
        """Restore index and mappings after loading from save.
            With PostgreSQL, this means re-establishing connection and
            re-loading prototypes from the database.
        """
        logger.info("Restoring from save: Re-establishing DB connection and loading prototypes.")
        if self.conn:
            self.conn.close() # Close existing connection if any
        self._init_db() # Re-initialize DB connection and tables
        self._load_prototypes_from_db() # Load prototypes into memory
        self.updates_since_rebuild = 0

    def _prune_examples(self, label: str):
        """Prune examples for a given label in the database to maintain memory bounds."""
        try:
            with self.conn.cursor() as cur:
                # Fetch all embeddings for the label from DB, ordered by added_at to prune oldest
                cur.execute("SELECT id, embedding FROM examples WHERE label = %s ORDER BY added_at ASC;", (label,))
                db_examples = cur.fetchall()

                if len(db_examples) <= self.config.max_examples_per_class:
                    logger.debug(f"No pruning needed for label '{label}'.")
                    return

                # Convert to torch tensors for distance calculation
                embeddings = torch.stack([torch.from_numpy(ex['embedding']).float() for ex in db_examples])
                
                # Ensure calculations are done on CPU
                if embeddings.is_cuda:
                    embeddings = embeddings.cpu()

                # Compute mean of embeddings (more stable than current prototype)
                mean_embedding = torch.mean(embeddings, dim=0)
                
                distances = []
                for ex_row in db_examples:
                    ex_embedding = torch.from_numpy(ex_row['embedding']).float()
                    # Ensure both tensors are on the same device for distance calculation
                    if ex_embedding.is_cuda and mean_embedding.is_cuda:
                            dist = torch.norm(ex_embedding - mean_embedding).item()
                    elif not ex_embedding.is_cuda and not mean_embedding.is_cuda:
                            dist = torch.norm(ex_embedding - mean_embedding).item()
                    else: # Mismatch, move to CPU
                            dist = torch.norm(ex_embedding.cpu() - mean_embedding.cpu()).item()
                    distances.append((dist, ex_row['id'])) # Store distance and original ID

                # Sort by distance and identify IDs to keep (closest to the mean)
                distances.sort(key=lambda x: x[0]) # Sort by distance
                ids_to_keep = {id for dist, id in distances[:self.config.max_examples_per_class]}
                
                # Identify IDs to remove
                all_ids = {ex_row['id'] for ex_row in db_examples}
                ids_to_remove = all_ids - ids_to_keep

                if ids_to_remove:
                    # Use sql.SQL and sql.Literal to safely construct the IN clause
                    # Convert set to list for consistent ordering (though not strictly necessary for IN)
                    ids_list_to_remove = list(ids_to_remove)
                    
                    if not ids_list_to_remove: # Double check if the list is empty
                        logger.debug(f"No examples to prune for label '{label}' after determining IDs.")
                        return

                    delete_query = sql.SQL("DELETE FROM examples WHERE id IN ({});").format(
                        sql.SQL(',').join(map(sql.Literal, ids_list_to_remove))
                    )
                    cur.execute(delete_query)
                    # **FIX ENDS HERE**
                    
                    self.conn.commit()
                    logger.info(f"Pruned {len(ids_to_remove)} examples for label '{label}'.")
                else:
                    logger.debug(f"No examples to prune for label '{label}'.")

        except Exception as e:
            logger.error(f"Error pruning examples for label '{label}': {e}")
            self.conn.rollback()
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics by querying the database."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(DISTINCT label) FROM prototypes;")
                num_classes = cur.fetchone()['count']

                cur.execute("SELECT label, COUNT(*) as count FROM examples GROUP BY label;")
                examples_per_class = {row['label']: row['count'] for row in cur.fetchall()}
                
                total_examples = sum(examples_per_class.values())

            return {
                'num_classes': num_classes,
                'examples_per_class': examples_per_class,
                'total_examples': total_examples,
                'prototype_dimensions': self.embedding_dim,
                'updates_since_rebuild': self.updates_since_rebuild # This counter remains local for update frequency
            }
        except Exception as e:
            logger.error(f"Error getting statistics from database: {e}")
            raise
    
    def clear(self):
        """Clear all memory by truncating database tables."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE prototypes RESTART IDENTITY;")
                cur.execute("TRUNCATE TABLE examples RESTART IDENTITY;")
            self.conn.commit()
            self.prototypes.clear() # Clear in-memory cache
            self.examples.clear() # Clear in-memory default dict
            self.strategic_prototypes.clear()
            self.updates_since_rebuild = 0
            logger.info("All memory cleared from database and cache.")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            self.conn.rollback()
            raise

    def compute_strategic_prototypes(self, cost_function, classifier_func):
        """Compute strategic prototypes for all classes.
        This remains in-memory as strategic computation might be dynamic and not persistable.
        """
        self.strategic_prototypes.clear() # Clear previous strategic prototypes

        # Iterate over labels that have examples in the database
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT DISTINCT label FROM examples;")
                labels_with_examples = [row['label'] for row in cur.fetchall()]

                for label in labels_with_examples:
                    cur.execute("SELECT embedding FROM examples WHERE label = %s;", (label,))
                    examples_embeddings_np = [row['embedding'] for row in cur.fetchall()]
                    
                    if not examples_embeddings_np:
                        continue # Skip if no examples found after query

                    strategic_embeddings = []
                    for embedding_np in examples_embeddings_np:
                        original_embedding = torch.from_numpy(embedding_np).float()
                        # Ensure original_embedding is on CPU before passing to compute_best_response
                        # unless cost_function expects GPU tensors. Assuming CPU for safety.
                        original_embedding_cpu = original_embedding.cpu() 

                        # Compute where this example would strategically move
                        strategic_embedding = cost_function.compute_best_response(
                            original_embedding_cpu, classifier_func
                        )
                        # Ensure the strategic_embedding is also on CPU before appending
                        strategic_embeddings.append(strategic_embedding.cpu())
                    
                    # Compute mean of strategic embeddings
                    if strategic_embeddings:
                        strategic_prototype = torch.stack(strategic_embeddings).mean(dim=0)
                        self.strategic_prototypes[label] = strategic_prototype
            logger.info("Strategic prototypes recomputed.")
        except Exception as e:
            logger.error(f"Error computing strategic prototypes: {e}")
            raise
    
    def get_strategic_prototypes(self, query_embedding: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """Get nearest strategic prototypes from the in-memory cache."""
        if not self.strategic_prototypes:
            logger.warning("No strategic prototypes available. Falling back to regular prototypes.")
            return self.get_nearest_prototypes(query_embedding, k)
        
        # Ensure query_embedding is on CPU for consistency with in-memory prototypes
        query_embedding_cpu = query_embedding.cpu()

        # Compute similarities to strategic prototypes
        similarities = []
        for label, prototype in self.strategic_prototypes.items():
            # Ensure prototype is on CPU as well
            prototype_cpu = prototype.cpu() 
            # Compute cosine similarity
            # unsqueeze(0) adds a batch dimension (1, embedding_dim) for batch-wise calculation
            sim = F.cosine_similarity(
                query_embedding_cpu.unsqueeze(0), 
                prototype_cpu.unsqueeze(0)
            ).item()
            similarities.append((label, sim))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")
