"""verl reward_fn stub for SQL (OPD uses no reward; interface only)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_sql

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    try:
        db = common_sql.DB.get(extra_info["db_path"])
        return 1.0 if common_sql.strict_check(solution_str, db, "stop")["valid"] else 0.0
    except Exception:
        return 0.0
