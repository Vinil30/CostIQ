# business_analyst_agent.py

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


class BusinessAnalystAgent:
    """
    LLM-powered agent that dynamically identifies relevant columns from ANY dataset
    and extracts business metrics.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        verbose: bool = False,
        unified_dataset_path: str = "unified_dataset.csv"
    ):
        self.api_key = api_key
        self.model = model
        self.verbose = verbose
        self.unified_dataset_path = unified_dataset_path
        self.enriched_dataset_path = "enriched_dataset.csv"
        self.llm = ChatGroq(api_key=self.api_key, model=self.model, temperature=0)
    
    def _llm_detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Use LLM to identify relevant columns from ANY dataset."""
        
        # Prepare sample data for LLM
        columns_info = []
        for col in df.columns:
            sample_vals = df[col].dropna().head(3).tolist()
            # Convert non-serializable types
            sample_vals = [str(v) if not isinstance(v, (int, float, bool, type(None))) else v for v in sample_vals]
            columns_info.append({
                "name": col,
                "type": str(df[col].dtype),
                "sample_values": sample_vals
            })
        
        prompt = f"""
        You are a data analyst. Identify which columns in this ecommerce dataset are relevant for business analysis.
        
        Dataset columns:
        {json.dumps(columns_info, indent=2, default=str)}
        
        Identify and return ONLY these fields (use null if not found):
        - revenue_column: column containing monetary value (price, payment, amount)
        - cost_column: column containing cost (freight, shipping, delivery fee)
        - date_column: column with timestamp or date
        - order_id_column: column with unique order identifier
        - product_id_column: column with product identifier
        - seller_id_column: column with seller identifier
        - category_column: column with product category
        - review_column: column with rating/score
        - quantity_column: column with item quantity
        
        Return ONLY JSON with these keys. No explanation.
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a data analyst. Return only valid JSON."),
                HumanMessage(content=prompt)
            ])
            
            # Parse LLM response
            schema = json.loads(response.content)
            
            # Validate and clean
            valid_schema = {
                "revenue_col": schema.get("revenue_column"),
                "cost_col": schema.get("cost_column"),
                "date_col": schema.get("date_column"),
                "order_col": schema.get("order_id_column"),
                "product_col": schema.get("product_id_column"),
                "seller_col": schema.get("seller_id_column"),
                "category_col": schema.get("category_column"),
                "review_col": schema.get("review_column"),
                "qty_col": schema.get("quantity_column")
            }
            
            # Verify columns exist in dataframe
            for key, col in valid_schema.items():
                if col and col not in df.columns:
                    valid_schema[key] = None
            
            return valid_schema
            
        except Exception as e:
            if self.verbose:
                print(f"LLM schema detection failed: {e}")
            return self._fallback_schema(df)
    
    def _fallback_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback if LLM fails."""
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
        
        return {
            "revenue_col": numeric_cols[0] if len(numeric_cols) > 0 else None,
            "cost_col": None,
            "date_col": date_cols[0] if date_cols else None,
            "order_col": None,
            "product_col": None,
            "seller_col": None,
            "category_col": None,
            "review_col": None,
            "qty_col": None
        }
    
    def _calculate_kpis(self, df: pd.DataFrame, schema: Dict) -> Dict:
        """Calculate KPIs using detected columns."""
        
        kpis = {}
        rev = schema["revenue_col"]
        cost = schema["cost_col"]
        order = schema["order_col"]
        
        # Convert to numeric if needed
        if rev:
            df_rev = pd.to_numeric(df[rev], errors='coerce').fillna(0)
            kpis["total_revenue"] = float(df_rev.sum())
            kpis["avg_revenue"] = float(df_rev.mean())
        else:
            kpis["total_revenue"] = 0
            kpis["avg_revenue"] = 0
        
        # Cost
        if cost:
            df_cost = pd.to_numeric(df[cost], errors='coerce').fillna(0)
            kpis["total_cost"] = float(df_cost.sum())
        else:
            kpis["total_cost"] = 0
        
        # Profit
        kpis["total_profit"] = kpis["total_revenue"] - kpis["total_cost"]
        kpis["profit_margin"] = (kpis["total_profit"] / kpis["total_revenue"] * 100) if kpis["total_revenue"] > 0 else 0
        
        # Orders
        if order:
            kpis["total_orders"] = df[order].nunique()
            kpis["avg_order_value"] = kpis["total_revenue"] / kpis["total_orders"] if kpis["total_orders"] > 0 else 0
        else:
            kpis["total_orders"] = len(df)
            kpis["avg_order_value"] = kpis["total_revenue"] / kpis["total_orders"] if kpis["total_orders"] > 0 else 0
        
        # Quantity
        if schema["qty_col"]:
            df_qty = pd.to_numeric(df[schema["qty_col"]], errors='coerce').fillna(0)
            kpis["total_items"] = int(df_qty.sum())
        
        # Reviews
        if schema["review_col"]:
            df_review = pd.to_numeric(df[schema["review_col"]], errors='coerce')
            kpis["avg_review"] = float(df_review.mean()) if not df_review.isna().all() else None
            if kpis["avg_review"]:
                kpis["low_review_pct"] = float((df_review <= 2).mean() * 100)
        
        # Date range
        if schema["date_col"]:
            try:
                df_date = pd.to_datetime(df[schema["date_col"]], errors='coerce')
                kpis["date_start"] = df_date.min().strftime("%Y-%m-%d") if not pd.isna(df_date.min()) else None
                kpis["date_end"] = df_date.max().strftime("%Y-%m-%d") if not pd.isna(df_date.max()) else None
            except:
                kpis["date_start"] = None
                kpis["date_end"] = None
        
        return kpis
    
    def _calculate_trends(self, df: pd.DataFrame, schema: Dict) -> Dict:
        """Calculate trends if date and revenue exist."""
        
        if not schema["date_col"] or not schema["revenue_col"]:
            return {"trend": "insufficient_data"}
        
        try:
            df_copy = df.copy()
            df_copy[schema["date_col"]] = pd.to_datetime(df_copy[schema["date_col"]], errors='coerce')
            df_copy = df_copy.dropna(subset=[schema["date_col"], schema["revenue_col"]])
            
            if len(df_copy) == 0:
                return {"trend": "insufficient_data"}
            
            df_copy["date"] = df_copy[schema["date_col"]].dt.date
            df_rev = pd.to_numeric(df_copy[schema["revenue_col"]], errors='coerce').fillna(0)
            
            daily = df_copy.groupby("date")[schema["revenue_col"]].sum().reset_index()
            daily.columns = ["date", "revenue"]
            daily["revenue"] = pd.to_numeric(daily["revenue"], errors='coerce').fillna(0)
            
            if len(daily) >= 14:
                recent = daily["revenue"].tail(7).mean()
                previous = daily["revenue"].tail(14).head(7).mean()
                trend = "increasing" if recent > previous else "decreasing"
                trend_pct = ((recent - previous) / previous * 100) if previous > 0 else 0
            else:
                trend = "insufficient_data"
                trend_pct = 0
            
            return {
                "trend": trend,
                "trend_percentage": round(trend_pct, 2),
                "avg_daily_revenue": float(daily["revenue"].mean()),
                "peak_revenue": float(daily["revenue"].max()),
                "peak_date": str(daily.loc[daily["revenue"].idxmax(), "date"]) if len(daily) > 0 else None
            }
        except Exception as e:
            if self.verbose:
                print(f"Trend calculation error: {e}")
            return {"trend": "error", "error": str(e)}
    
    def _calculate_cost_leakage(self, df: pd.DataFrame, schema: Dict) -> Dict:
        """Identify cost leakages."""
        
        leakages = {"total_loss": 0, "issues": [], "recommendations": []}
        
        if not schema["revenue_col"]:
            return leakages
        
        try:
            df_temp = df.copy()
            
            # Convert to numeric
            df_temp[schema["revenue_col"]] = pd.to_numeric(df_temp[schema["revenue_col"]], errors='coerce').fillna(0)
            
            # High freight cost
            if schema["cost_col"]:
                df_temp[schema["cost_col"]] = pd.to_numeric(df_temp[schema["cost_col"]], errors='coerce').fillna(0)
                df_temp["freight_ratio"] = df_temp[schema["cost_col"]] / (df_temp[schema["revenue_col"]] + 1e-10)
                high_freight = df_temp[df_temp["freight_ratio"] > 0.3]
                
                if len(high_freight) > 0:
                    loss = high_freight[schema["cost_col"]].sum()
                    leakages["issues"].append({
                        "type": "excessive_freight",
                        "loss": round(float(loss), 2),
                        "affected": len(high_freight)
                    })
                    leakages["total_loss"] += loss
                    leakages["recommendations"].append("Negotiate freight rates for high-cost shipments")
            
            # Low reviews (returns proxy)
            if schema["review_col"]:
                df_temp[schema["review_col"]] = pd.to_numeric(df_temp[schema["review_col"]], errors='coerce')
                low_reviews = df_temp[df_temp[schema["review_col"]] <= 2]
                if len(low_reviews) > 0:
                    loss = low_reviews[schema["revenue_col"]].sum()
                    if schema["cost_col"]:
                        loss += low_reviews[schema["cost_col"]].sum()
                    leakages["issues"].append({
                        "type": "potential_returns",
                        "loss": round(float(loss), 2),
                        "affected": len(low_reviews)
                    })
                    leakages["total_loss"] += loss
                    leakages["recommendations"].append("Investigate low-rated products/sellers")
            
            # Low margin
            if schema["cost_col"]:
                df_temp["margin"] = (df_temp[schema["revenue_col"]] - df_temp[schema["cost_col"]]) / df_temp[schema["revenue_col"]] * 100
                low_margin = df_temp[df_temp["margin"] < 10]
                if len(low_margin) > 0:
                    loss = low_margin[schema["cost_col"]].sum()
                    leakages["issues"].append({
                        "type": "low_margin",
                        "loss": round(float(loss), 2),
                        "affected": len(low_margin)
                    })
                    leakages["total_loss"] += loss
                    leakages["recommendations"].append("Review pricing for low-margin products")
            
            leakages["total_loss"] = round(leakages["total_loss"], 2)
            
            if not leakages["recommendations"]:
                leakages["recommendations"].append("No significant cost leakages detected")
            
        except Exception as e:
            if self.verbose:
                print(f"Cost leakage error: {e}")
        
        return leakages
    
    def _generate_insights(self, kpis: Dict, trends: Dict, leakages: Dict) -> str:
        """Generate business insights using LLM."""
        
        prompt = f"""
        Based on this ecommerce data:
        - Revenue: ${kpis.get('total_revenue', 0):,.2f}
        - Profit Margin: {kpis.get('profit_margin', 0):.1f}%
        - Orders: {kpis.get('total_orders', 0):,}
        - Trend: {trends.get('trend', 'N/A')} ({trends.get('trend_percentage', 0)}%)
        - Cost Leakage: ${leakages.get('total_loss', 0):,.2f}
        
        Provide 3 short bullet points: 
        1. Current performance summary
        2. Biggest problem
        3. Actionable recommendation
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a business analyst. Be concise."),
                HumanMessage(content=prompt)
            ])
            return response.content
        except:
            return "Insights unavailable"
    
    def analyze(self) -> Dict[str, Any]:
        """Main method - run full analysis."""
        
        try:
            # Load data
            if not os.path.exists(self.unified_dataset_path):
                return {"status": "error", "error": "Dataset not found"}
            
            df = pd.read_csv(self.unified_dataset_path)
            
            if self.verbose:
                print(f"📂 Loaded: {len(df)} rows, {len(df.columns)} columns")
                print(f"Columns: {list(df.columns)}")
            
            # LLM detects schema
            schema = self._llm_detect_schema(df)
            
            if self.verbose:
                print(f"\n🔍 LLM Detected:")
                for key, val in schema.items():
                    if val:
                        print(f"   {key}: {val}")
            
            # Check if we have minimum required columns
            if not schema["revenue_col"]:
                return {
                    "status": "error",
                    "error": "No revenue column detected. Dataset may not contain monetary values.",
                    "detected_schema": schema
                }
            
            # Calculate metrics
            kpis = self._calculate_kpis(df, schema)
            trends = self._calculate_trends(df, schema)
            leakages = self._calculate_cost_leakage(df, schema)
            insights = self._generate_insights(kpis, trends, leakages)
            
            # Create enriched dataset
            df_enriched = df.copy()
            if schema["revenue_col"] and schema["cost_col"]:
                rev_numeric = pd.to_numeric(df_enriched[schema["revenue_col"]], errors='coerce').fillna(0)
                cost_numeric = pd.to_numeric(df_enriched[schema["cost_col"]], errors='coerce').fillna(0)
                df_enriched["profit"] = rev_numeric - cost_numeric
                df_enriched["margin_pct"] = (df_enriched["profit"] / (rev_numeric + 1e-10) * 100).round(2)
            if schema["review_col"]:
                review_numeric = pd.to_numeric(df_enriched[schema["review_col"]], errors='coerce')
                df_enriched["is_low_quality"] = review_numeric <= 2
            
            df_enriched.to_csv(self.enriched_dataset_path, index=False)
            
            return {
                "status": "success",
                "detected_schema": schema,
                "kpis": kpis,
                "trends": trends,
                "cost_leakages": leakages,
                "insights": insights,
                "enriched_dataset": self.enriched_dataset_path
            }
            
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def quick_analysis(self) -> Dict[str, Any]:
        """Quick analysis without LLM insights."""
        
        try:
            df = pd.read_csv(self.unified_dataset_path)
            schema = self._fallback_schema(df)
            kpis = self._calculate_kpis(df, schema)
            leakages = self._calculate_cost_leakage(df, schema)
            
            return {
                "status": "success",
                "kpis": kpis,
                "cost_leakages": leakages
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }