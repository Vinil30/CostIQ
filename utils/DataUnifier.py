import pandas as pd
import json
import os
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# TOOLS
# ============================================================================

@tool
def read_csv_preview(file_path: str, rows: int = 10) -> str:
    """Used to read CSV"""
    try:
        df = pd.read_csv(file_path, nrows=rows)

        info = {
            "file": file_path,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": df.head(5).to_dict(orient="records"),
            "null_counts": df.isnull().sum().to_dict(),
            "row_count_preview": len(df)
        }

        return json.dumps(info, indent=2, default=str)

    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"


@tool
def suggest_joins(files_info: str) -> str:
    """Suggested Joins"""
    try:
        files = json.loads(files_info)

        suggestions = []
        for f1, cols1 in files.items():
            for f2, cols2 in files.items():
                if f1 < f2:
                    common = list(set(cols1) & set(cols2))
                    if common:
                        suggestions.append(f"{f1} ↔ {f2} → {common}")

        return "\n".join(suggestions) if suggestions else "No common keys found."

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def merge_datasets(left_file: str, right_file: str, left_key: str, right_key: str, how: str = "left") -> str:
    """Merging operations"""
    try:
        left_df = pd.read_csv(left_file)
        right_df = pd.read_csv(right_file)

        merged = left_df.merge(right_df, left_on=left_key, right_on=right_key, how=how)

        output_path = f"merged_{os.path.basename(left_file).replace('.csv','')}_{os.path.basename(right_file)}"
        merged.to_csv(output_path, index=False)

        return json.dumps({
            "status": "success",
            "output_file": output_path,
            "rows": len(merged),
            "columns": list(merged.columns)
        })

    except Exception as e:
        return f"Error merging: {str(e)}"


@tool
def finalize_unified_dataset(df_path: str, selected_columns: str = None) -> str:
    """Unifying dataset"""
    try:
        df = pd.read_csv(df_path)

        if selected_columns:
            columns = json.loads(selected_columns)
            df = df[[c for c in columns if c in df.columns]]

        output_path = "unified_dataset.csv"
        df.to_csv(output_path, index=False)

        return json.dumps({
            "status": "success",
            "output_file": output_path,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records")
        })

    except Exception as e:
        return f"Error finalizing: {str(e)}"


# ============================================================================
# MAIN PIPELINE CLASS (NO AGENT)
# ============================================================================

class SchemaUnificationPipeline:

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(api_key=api_key, model=model, temperature=0)

    def unify(self, file_paths: List[str]) -> Dict[str, Any]:

        # Step 1: Read all previews
        previews = {}
        for path in file_paths:
            preview = read_csv_preview.invoke({"file_path": path})
            parsed = json.loads(preview)
            previews[path] = {
                "columns": parsed["columns"],
                "sample": parsed["sample_rows"][:2]
            }

        # Step 2: Ask LLM to plan the merge order and keys
        plan_prompt = f"""
You are a data engineer. You have these CSV files to merge into one unified dataset:

{json.dumps({path: info for path, info in previews.items()}, indent=2, default=str)}

Plan the exact merge sequence to produce one unified dataset with ALL important columns.
The goal is to retain: order financial data (price, freight_value, payment_value), 
product info (product_id, seller_id, product_category_name), 
review scores (review_score), customer info, and order timestamps.

Return ONLY a JSON array of merge steps, each step being:
{{
  "left": "path/to/left.csv",   // use exact file paths from above
  "right": "path/to/right.csv", // use exact file paths from above
  "left_key": "column_name",
  "right_key": "column_name",
  "how": "left"
}}

Rules:
- Use exact file paths as provided above
- The first step's "left" should be the file that is the central/fact table (orders)
- Each subsequent step's "left" should be the output of the previous merge (use the string "PREVIOUS")
- Only include steps where a valid join key exists
- Order matters: build up the dataset step by step

Example format:
[
  {{"left": "path/orders.csv", "right": "path/order_items.csv", "left_key": "order_id", "right_key": "order_id", "how": "left"}},
  {{"left": "PREVIOUS", "right": "path/payments.csv", "left_key": "order_id", "right_key": "order_id", "how": "left"}}
]
"""
        from langchain_core.messages import SystemMessage, HumanMessage

        response = self.llm.invoke([
            SystemMessage(content="You are a data engineer. Return only valid JSON arrays, no explanation, no markdown."),
            HumanMessage(content=plan_prompt)
        ])

        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        
        try:
            merge_plan = json.loads(raw)
        except Exception as e:
            print(f"⚠️ LLM merge plan parsing failed: {e}\nRaw response: {raw}")
            return {"output_file": None, "status": "error", "error": str(e)}

        print(f"\n📋 LLM Merge Plan ({len(merge_plan)} steps):")
        for i, step in enumerate(merge_plan):
            print(f"  {i+1}. {os.path.basename(step['left'])} + {os.path.basename(step['right'])} on '{step['left_key']}' = '{step['right_key']}'")

        # Step 3: Execute the plan
        current_file = None

        for i, step in enumerate(merge_plan):
            left = current_file if step["left"] == "PREVIOUS" else step["left"]

            if not left or not os.path.exists(left):
                print(f"⚠️ Step {i+1}: Left file not found: {left}")
                continue

            if not os.path.exists(step["right"]):
                print(f"⚠️ Step {i+1}: Right file not found: {step['right']}")
                continue

            print(f"\n➡️ Step {i+1}: Merging on '{step['left_key']}' ↔ '{step['right_key']}'")

            result = merge_datasets.invoke({
                "left_file": left,
                "right_file": step["right"],
                "left_key": step["left_key"],
                "right_key": step["right_key"],
                "how": step.get("how", "left")
            })

            try:
                result = json.loads(result)
                if result.get("status") == "success":
                    current_file = result["output_file"]
                    # Update previews so LLM knows what columns are available
                    previews[current_file] = {"columns": result["columns"]}
                    print(f"   ✅ {result['rows']} rows, {len(result['columns'])} columns")
                else:
                    print(f"   ❌ Merge failed: {result}")
            except Exception as e:
                print(f"   ❌ Result parsing failed: {e}")
                continue

        if not current_file:
            return {"output_file": None, "status": "error", "error": "All merge steps failed"}

        # Step 4: Ask LLM which columns to keep (drop duplicates/noise)
        final_cols_prompt = f"""
The merged dataset has these columns:
{previews.get(current_file, {}).get('columns', [])}

Select the most useful columns for business analysis, keeping:
- All financial columns (price, freight_value, payment_value, payment_type)
- All ID columns (order_id, customer_id, product_id, seller_id)  
- Date/timestamp columns
- Category columns (product_category_name)
- Review/quality columns (review_score)
- Customer location (customer_city, customer_state)
- Order status

Remove obvious duplicates (like _x/_y suffix columns, keep the cleaner one).
Return ONLY a JSON array of column names to keep, no explanation.
"""
        col_response = self.llm.invoke([
            SystemMessage(content="Return only a valid JSON array of column names."),
            HumanMessage(content=final_cols_prompt)
        ])

        raw_cols = col_response.content.strip().replace("```json", "").replace("```", "").strip()
        
        try:
            selected_columns = json.loads(raw_cols)
            # Validate columns exist
            df_check = pd.read_csv(current_file, nrows=1)
            selected_columns = [c for c in selected_columns if c in df_check.columns]
            print(f"\n✅ LLM selected {len(selected_columns)} columns to keep")
        except Exception as e:
            print(f"⚠️ Column selection failed: {e}. Keeping all columns.")
            selected_columns = None

        # Step 5: Finalize with run-specific path
        final = finalize_unified_dataset.invoke({
            "df_path": current_file,
            "selected_columns": json.dumps(selected_columns) if selected_columns else None
        })

        result = json.loads(final)
        print(f"\n🎯 Unified dataset: {result.get('rows')} rows × {len(result.get('columns', []))} columns")
        print(f"   Columns: {result.get('columns')}")
        
        return result
