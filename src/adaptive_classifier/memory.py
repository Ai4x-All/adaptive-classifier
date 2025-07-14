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

# Assuming .models contains Example and ModelConfig classes
from .models import Example, ModelConfig

"""
基于 PostgreSQL 数据库的内存系统，每次实例化时都彻底删除并重新创建表，
以模拟 FAISS 的“每次实例化都是干净开始”的行为，且不对维度做区分。
"""

logger = logging.getLogger(__name__)

class PrototypeMemory:
    """Memory system that maintains prototypes for each class, backed by PostgreSQL with pgvector."""

    def __init__(
        self,
        embedding_dim: int,
        config: Optional[ModelConfig] = None,
        db_config: Dict[str, Any] = None # Database connection details
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
        self.examples = defaultdict(list)  
        self.prototypes = {}  # label -> tensor - cache of current prototypes from DB
        self.strategic_prototypes = {}  # label -> strategic prototype tensor - still in-memory for simplicity

        self.conn = None # Database connection

        # 【核心改动】: 表名是固定的，不再包含 embedding_dim
        self.prototypes_table_name = "prototypes"
        self.examples_table_name = "examples"

        self._init_db() # 调用初始化数据库的方法

        # Statistics
        self.updates_since_rebuild = 0 
        
        # Load existing prototypes from DB on initialization (should be empty after re-creation)
        self._load_prototypes_from_db()

    def _init_db(self):
        """Initialize database connection, and DROP/CREATE tables to ensure a clean slate."""
        try:
            self.conn = psycopg.connect(**self.db_config, row_factory=dict_row)
            register_vector(self.conn)
            with self.conn.cursor() as cur:
                # Enable pgvector extension if not already enabled
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                proto_table_ident = sql.Identifier(self.prototypes_table_name)
                examples_table_ident = sql.Identifier(self.examples_table_name)

                # 【核心改动】: 先删除现有表，无论其维度如何，以确保彻底清空
                # cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(examples_table_ident))
                # cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(proto_table_ident))

                # 【增强日志和错误处理】: 尝试删除现有表
                logger.info(f"尝试删除 PostgreSQL 表格 '{self.examples_table_name}' 和 '{self.prototypes_table_name}'...")
                try:
                    cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(examples_table_ident))
                    cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(proto_table_ident))
                    self.conn.commit() # 每次DROP后立即提交，释放潜在的锁
                    logger.info(f"成功删除 PostgreSQL 表格 '{self.examples_table_name}' 和 '{self.prototypes_table_name}'.")
                except psycopg.Error as db_err:
                    logger.error(f"删除 PostgreSQL 表格时发生数据库错误: {db_err}. 可能是由于锁或其他并发问题。")
                    self.conn.rollback() # 回滚以确保连接状态良好
                    raise # 重新抛出异常，因为无法删除表意味着无法进行后续操作
                except Exception as e:
                    logger.error(f"删除 PostgreSQL 表格时发生未知错误: {e}")
                    self.conn.rollback()
                    raise
                
                # 然后以当前 embedding_dim 重新创建表
                cur.execute(sql.SQL("""
                    CREATE TABLE {table_name} (
                        label TEXT PRIMARY KEY,
                        embedding VECTOR({embedding_dim}),
                        num_examples INTEGER DEFAULT 0
                    );
                """).format(
                    table_name=proto_table_ident,
                    embedding_dim=sql.Literal(self.embedding_dim)
                ))
                
                cur.execute(sql.SQL("""
                    CREATE TABLE {table_name} (
                        id SERIAL PRIMARY KEY,
                        label TEXT NOT NULL,
                        text TEXT,
                        embedding VECTOR({embedding_dim}),
                        added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """).format(
                    table_name=examples_table_ident,
                    embedding_dim=sql.Literal(self.embedding_dim)
                ))
                self.conn.commit()
                logger.info(f"PostgreSQL 表格 '{self.prototypes_table_name}' 和 '{self.examples_table_name}' 已删除并以维度 {self.embedding_dim} 重新创建。")
        except Exception as e:
            logger.error(f"连接或初始化数据库失败: {e}")
            self.conn.rollback()
            raise

    def _load_prototypes_from_db(self):
        """Load all prototypes from the database into memory cache."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT label, embedding FROM {};").format(
                    sql.Identifier(self.prototypes_table_name)
                ))
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
                cur.execute(sql.SQL(
                    "INSERT INTO {} (label, text, embedding) VALUES (%s, %s, %s);"
                ).format(sql.Identifier(self.examples_table_name)),
                    (label, example.text, embedding_np)
                )
                self.conn.commit()
                logger.debug(f"Added example for label '{label}' to database.")

            if self.config.max_examples_per_class > 0:
                with self.conn.cursor() as cur:
                    cur.execute(sql.SQL("SELECT COUNT(*) FROM {} WHERE label = %s;").format(
                        sql.Identifier(self.examples_table_name)), (label,))
                    current_count = cur.fetchone()['count']

                if current_count > self.config.max_examples_per_class:
                    logger.debug(f"Pruning examples for label '{label}'. Current: {current_count}, Max: {self.config.max_examples_per_class}")
                    self._prune_examples(label)
            
            self._update_prototype(label)
            
            self.updates_since_rebuild += 1
            
            if self.config.prototype_update_frequency > 0 and \
               self.updates_since_rebuild >= self.config.prototype_update_frequency:
                logger.info(f"Prototype update frequency met ({self.updates_since_rebuild}/{self.config.prototype_update_frequency}). Refreshing local prototype cache.")
                self._rebuild_index()

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
        if not self.prototypes: 
            logger.warning(f"No prototypes available to search.")
            return []

        query_np = query_embedding.cpu().numpy()

        results = []
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("""
                    SELECT label, embedding <-> %s AS distance
                    FROM {}
                    ORDER BY distance
                    LIMIT %s;
                """).format(sql.Identifier(self.prototypes_table_name)),
                (query_np, k))
                
                rows = cur.fetchall()
                if not rows:
                    return []

                distances = np.array([row['distance'] for row in rows])
                similarities = np.exp(-distances)

                for row, similarity in zip(rows, similarities):
                    label = row['label']
                    score = float(similarity)
                    if min_similarity is None or score >= min_similarity:
                        results.append((label, score))
            
            if results:
                scores_tensor = torch.tensor([score for _, score in results], dtype=torch.float32)
                normalized_scores = torch.nn.functional.softmax(scores_tensor, dim=0)
                
                results = [
                    (label, float(norm_score)) 
                    for (label, _), norm_score in zip(results, normalized_scores)
                ]
                results.sort(key=lambda x: x[1], reverse=True)
            
            return results

        except Exception as e:
            logger.error(f"Error getting nearest prototypes from database: {e}")
            raise

    def _update_prototype(self, label: str):
        """Update the prototype for a given label in the database."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT embedding FROM {} WHERE label = %s;").format(
                    sql.Identifier(self.examples_table_name)), (label,))
                embeddings_np = [row['embedding'] for row in cur.fetchall()]

                if not embeddings_np:
                    cur.execute(sql.SQL("DELETE FROM {} WHERE label = %s;").format(
                        sql.Identifier(self.prototypes_table_name)), (label,))
                    self.prototypes.pop(label, None)
                    self.conn.commit()
                    logger.debug(f"Removed prototype for label '{label}' as no examples exist.")
                    return

                embeddings = torch.stack([torch.from_numpy(emb).float() for emb in embeddings_np])
                
                if embeddings.is_cuda:
                    embeddings = embeddings.cpu()

                prototype_tensor = torch.mean(embeddings, dim=0)
                prototype_np = prototype_tensor.numpy()

                cur.execute(sql.SQL("""
                    INSERT INTO {} (label, embedding, num_examples)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (label) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        num_examples = EXCLUDED.num_examples;
                """).format(sql.Identifier(self.prototypes_table_name)),
                (label, prototype_np, len(embeddings_np)))
                self.conn.commit()

                self.prototypes[label] = prototype_tensor
                logger.debug(f"Updated prototype for label '{label}' in database and cache.")

        except Exception as e:
            logger.error(f"Error updating prototype for label '{label}': {e}")
            self.conn.rollback()
            raise

    def _rebuild_index(self):
        """With pgvector, the 'index' is implicitly managed by the database."""
        logger.info(f"Rebuilding index for pgvector is implicitly handled by database. Refreshing local prototype cache.")
        self._load_prototypes_from_db()
        self.updates_since_rebuild = 0

    def _restore_from_save(self):
        """Restore index and mappings after loading from save.
            With PostgreSQL, this means re-establishing connection and
            re-loading prototypes from the database.
        """
        logger.info(f"Restoring from save: Re-establishing DB connection and loading prototypes.")
        if self.conn:
            self.conn.close()
        self._init_db() # This will DROP and CREATE tables
        self._load_prototypes_from_db()
        self.updates_since_rebuild = 0

    def _prune_examples(self, label: str):
        """Prune examples for a given label in the database to maintain memory bounds."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT id, embedding FROM {} WHERE label = %s ORDER BY added_at ASC;").format(
                    sql.Identifier(self.examples_table_name)), (label,))
                db_examples = cur.fetchall()

                if len(db_examples) <= self.config.max_examples_per_class:
                    logger.debug(f"No pruning needed for label '{label}'.")
                    return

                embeddings = torch.stack([torch.from_numpy(ex['embedding']).float() for ex in db_examples])
                
                if embeddings.is_cuda:
                    embeddings = embeddings.cpu()

                mean_embedding = torch.mean(embeddings, dim=0)
                
                distances = []
                for ex_row in db_examples:
                    ex_embedding = torch.from_numpy(ex_row['embedding']).float()
                    if ex_embedding.is_cuda and mean_embedding.is_cuda:
                            dist = torch.norm(ex_embedding - mean_embedding).item()
                    elif not ex_embedding.is_cuda and not mean_embedding.is_cuda:
                            dist = torch.norm(ex_embedding - mean_embedding).item()
                    else:
                        dist = torch.norm(ex_embedding.cpu() - mean_embedding.cpu()).item()
                    distances.append((dist, ex_row['id']))

                distances.sort(key=lambda x: x[0])
                ids_to_keep = {id for dist, id in distances[:self.config.max_examples_per_class]}
                
                all_ids = {ex_row['id'] for ex_row in db_examples}
                ids_to_remove = all_ids - ids_to_keep

                if ids_to_remove:
                    ids_list_to_remove = list(ids_to_remove)
                    
                    if not ids_list_to_remove:
                        logger.debug(f"No examples to prune for label '{label}' after determining IDs.")
                        return

                    delete_query = sql.SQL("DELETE FROM {} WHERE id IN ({});").format(
                        sql.Identifier(self.examples_table_name),
                        sql.SQL(',').join(map(sql.Literal, ids_list_to_remove))
                    )
                    cur.execute(delete_query)
                    
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
                cur.execute(sql.SQL("SELECT COUNT(DISTINCT label) FROM {};").format(
                    sql.Identifier(self.prototypes_table_name)))
                num_classes = cur.fetchone()['count']

                cur.execute(sql.SQL("SELECT label, COUNT(*) as count FROM {} GROUP BY label;").format(
                    sql.Identifier(self.examples_table_name)))
                examples_per_class = {row['label']: row['count'] for row in cur.fetchall()}
                
                total_examples = sum(examples_per_class.values())

            return {
                'num_classes': num_classes,
                'examples_per_class': examples_per_class,
                'total_examples': total_examples,
                'prototype_dimensions': self.embedding_dim,
                'updates_since_rebuild': self.updates_since_rebuild 
            }
        except Exception as e:
            logger.error(f"Error getting statistics from database: {e}")
            raise
    
    def clear(self):
        """Clear all memory by truncating database tables."""
        # Note: This clear method is similar to the init's behavior in this simplified mode.
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY;").format(
                    sql.Identifier(self.prototypes_table_name)))
                cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY;").format(
                    sql.Identifier(self.examples_table_name)))
            self.conn.commit()
            self.prototypes.clear()
            self.strategic_prototypes.clear()
            self.updates_since_rebuild = 0
            logger.info(f"All memory cleared from database and cache.")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            self.conn.rollback()
            raise

    def compute_strategic_prototypes(self, cost_function, classifier_func):
        """Compute strategic prototypes for all classes."""
        self.strategic_prototypes.clear() 

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT DISTINCT label FROM {};").format(
                    sql.Identifier(self.examples_table_name)))
                labels_with_examples = [row['label'] for row in cur.fetchall()]

                for label in labels_with_examples:
                    cur.execute(sql.SQL("SELECT embedding FROM {} WHERE label = %s;").format(
                        sql.Identifier(self.examples_table_name)), (label,))
                    examples_embeddings_np = [row['embedding'] for row in cur.fetchall()]
                    
                    if not examples_embeddings_np:
                        continue 

                    strategic_embeddings = []
                    for embedding_np in examples_embeddings_np:
                        original_embedding = torch.from_numpy(embedding_np).float()
                        original_embedding_cpu = original_embedding.cpu() 

                        strategic_embedding = cost_function.compute_best_response(
                            original_embedding_cpu, classifier_func
                        )
                        strategic_embeddings.append(strategic_embedding.cpu())
                    
                    if strategic_embeddings:
                        strategic_prototype = torch.stack(strategic_embeddings).mean(dim=0)
                        self.strategic_prototypes[label] = strategic_prototype
            logger.info(f"Strategic prototypes recomputed.")
        except Exception as e:
            logger.error(f"Error computing strategic prototypes: {e}")
            raise
    
    def get_strategic_prototypes(self, query_embedding: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """Get nearest strategic prototypes from the in-memory cache."""
        if not self.strategic_prototypes:
            logger.warning(f"No strategic prototypes available. Falling back to regular prototypes.")
            return self.get_nearest_prototypes(query_embedding, k)
        
        query_embedding_cpu = query_embedding.cpu()

        similarities = []
        for label, prototype in self.strategic_prototypes.items():
            prototype_cpu = prototype.cpu() 
            sim = F.cosine_similarity(
                query_embedding_cpu.unsqueeze(0), 
                prototype_cpu.unsqueeze(0)
            ).item()
            similarities.append((label, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info(f"Database connection closed.")