from sqlalchemy import inspect
from db import engine

def extract_schema():
    inspector = inspect(engine)
    schema = []
    for table_name in inspector.get_table_names():
        table_info = {"table": table_name, "columns": [], "foreign_keys": []}
        for column in inspector.get_columns(table_name):
            table_info["columns"].append({
                "name": column["name"],
                "type": str(column["type"])
            })
        for fk in inspector.get_foreign_keys(table_name):
            table_info["foreign_keys"].append({
                "constrained_columns": fk["constrained_columns"],
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"]
            })
        schema.append(table_info)
    
    import json
    print(json.dumps(schema, indent=2))

if __name__ == "__main__":
    extract_schema()
