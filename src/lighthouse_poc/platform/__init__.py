"""AI Marketplace data platform on Unity Catalog (ADR-012, docs/ai-marketplace-platform.md).

- naming:        catalog / schema / table names per environment (one prefix per developer)
- connection:    SQL warehouse engine from databricks-sdk unified auth
- migrations:    Alembic-managed schema changes (structure + data migrations) with restore points
- model_catalog: the governed model list (config/model_catalog.yaml) synced as a type-2 history
- knowledge:     chunking, gateway embeddings and Mosaic AI Vector Search (Delta Sync index)
- sql:           literal rendering for generated statements (trusted repo data only)
"""
