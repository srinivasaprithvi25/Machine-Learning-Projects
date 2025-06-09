import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()


def get_db_engine():
    """Create a SQLAlchemy engine using environment variables."""
    db_type = os.getenv("DB_TYPE")
    user = quote_plus(os.getenv("DB_USER", ""))
    password = quote_plus(os.getenv("DB_PASSWORD", ""))
    host = os.getenv("DB_HOST", "")
    port = os.getenv("DB_PORT", "")
    dbname = os.getenv("DB_NAME", "")

    if db_type == "postgresql":
        uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    elif db_type == "mysql":
        uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
    elif db_type == "mssql":
        driver = quote_plus("ODBC Driver 17 for SQL Server")
        uri = f"mssql+pyodbc://{user}:{password}@{host}/{dbname}?driver={driver}"
    elif db_type == "sqlite":
        uri = f"sqlite:///{os.getenv('DB_PATH')}"
    else:
        raise ValueError("Unsupported DB type")

    engine = create_engine(uri)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("\u2705 Connected to database")
        list_schemas_and_tables(engine)
    except Exception as e:
        print(f"\u274c Database connection failed: {e}")
        raise
    return engine


def list_schemas_and_tables(engine):
    """Print schemas and tables in the connected database."""
    try:
        with engine.connect() as conn:
            print("\n\U0001F4C2 Schemas and Tables in the Database:")
            result = conn.execute(text(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name;
                """
            ))
            rows = result.fetchall()
            current_schema = None
            for schema, table in rows:
                if schema != current_schema:
                    print(f"\n\U0001F4C1 Schema: {schema}")
                    current_schema = schema
                print(f"  \U0001F4C4 {table}")
    except Exception as e:
        print(f"\u274c Failed to list schemas/tables: {e}")

