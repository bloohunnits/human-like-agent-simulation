# Environment

Environment variables that should be set to run the project

## Databases

- **POSTGRES_PASSWORD**  Password for the Postgres database
- **PG_DSN** connection to Postres database, e.g., `postgresql://postgres:$POSTGRES_PASSWORD@127.0.0.1:5432/memories`
- **QDRANT_URL**  Connection to the Qdrant vector database e.g., `http://localhost:6333`

### Testing

- **TEST_PG_DSN** If set use this as the PG_DSN for unit tests, othewise mock PG
- **TEST_QDRANT_URL** If set use this as the QDRANT_URL for unit tests, othewise mock Qdrant
