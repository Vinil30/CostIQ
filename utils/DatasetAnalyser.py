# business_analyst_agent.py

import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import hashlib


class BusinessAnalystAgent:
    """
    Universal LLM-powered agent that analyzes ANY dataset for business metrics.
    Works with pandas DataFrames directly, no file dependencies.
    """
    
    # Class-level cache for schema detection
    _schema_cache = {}
    
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        verbose: bool = False,
        use_cache: bool = True
    ):
        self.api_key = api_key
        self.model = model
        self.verbose = verbose
        self.use_cache = use_cache
        self.llm = ChatGroq(api_key=self.api_key, model=self.model, temperature=0)
    
    def _get_cache_key(self, df: pd.DataFrame, analysis_type: str) -> str:
        """Generate cache key from column names and analysis type."""
        # Use sorted column names as cache key (stable across runs)
        cols_str = ','.join(sorted(df.columns))
        return hashlib.md5(f"{cols_str}_{analysis_type}".encode()).hexdigest()
    
    def _llm_detect_schema(self, df: pd.DataFrame, analysis_type: str = "business") -> Dict[str, Any]:
        """Use LLM to identify relevant columns from ANY dataset with caching."""
        
        # Check cache first
        cache_key = self._get_cache_key(df, analysis_type)
        if self.use_cache and cache_key in self._schema_cache:
            if self.verbose:
                print(f"✅ Using cached schema for {cache_key[:8]}...")
            return self._schema_cache[cache_key]
        
        # OPTIMIZED: Send ONLY column names, not types (reduces tokens by 40%)
        columns_list = list(df.columns)
        
        # If too many columns, send first 30 + count
        if len(columns_list) > 50:
            columns_preview = columns_list[:30] + [f"... and {len(columns_list) - 30} more columns"]
        else:
            columns_preview = columns_list
        
        analysis_desc = {
            "business":   "revenue, cost, date, id, category, rating, quantity",
            "finance":    "revenue, expense, profit, asset, date, account",
            "operations": "time, quantity, status, resource, efficiency, error",
        }.get(analysis_type, "revenue, cost, date, id, category, rating, quantity")
        
        # MUCH SMALLER PROMPT - no data types, just column names
        prompt = f"""
        Map these columns to business metrics: {', '.join(columns_preview)}
        
        Expected metrics: {analysis_desc}
        
        Return ONLY JSON (no markdown):
        {{
            "primary_metric": null,
            "secondary_metric": null,
            "date_column": null,
            "id_column": null,
            "group_column": null,
            "quality_column": null,
            "quantity_column": null,
            "cost_column": null,
            "status_column": null
        }}
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="Return only valid JSON. No markdown."),
                HumanMessage(content=prompt)
            ])
            
            content = response.content.strip()
            for fence in ("```json", "```"):
                if content.startswith(fence):
                    content = content[len(fence):]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            schema = json.loads(content)
            
            # Validate columns exist
            valid_schema = {}
            for key, col in schema.items():
                if col and col in df.columns:
                    valid_schema[key] = col
                else:
                    valid_schema[key] = None
            
            # Smart fallbacks if LLM missed something
            if not valid_schema["primary_metric"]:
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
                if len(numeric_cols) > 0:
                    # Prefer columns with business keywords
                    metric_keywords = ['revenue', 'sales', 'price', 'amount', 'value', 'total', 'profit', 'cost']
                    for col in numeric_cols:
                        if any(kw in col.lower() for kw in metric_keywords):
                            valid_schema["primary_metric"] = col
                            break
                    if not valid_schema["primary_metric"]:
                        valid_schema["primary_metric"] = numeric_cols[0]
            
            if not valid_schema["date_column"]:
                date_keywords = ("date", "time", "timestamp", "day", "month", "year")
                date_cols = [c for c in df.columns if any(kw in c.lower() for kw in date_keywords)]
                valid_schema["date_column"] = date_cols[0] if date_cols else None
            
            # Cache the result
            if self.use_cache:
                self._schema_cache[cache_key] = valid_schema
            
            return valid_schema
        
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"JSON parsing error: {e}")
            return self._fallback_schema(df)
        except Exception as e:
            if self.verbose:
                print(f"LLM schema detection failed: {e}")
            return self._fallback_schema(df)
    
    def _fallback_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback if LLM fails - NO LLM CALL."""
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp', 'day', 'month', 'year'])]
        
        primary_metric = None
        if numeric_cols:
            metric_keywords = ['value', 'amount', 'price', 'revenue', 'cost', 'score', 'rating', 'quantity', 'count', 'sales', 'total']
            for col in numeric_cols:
                if any(keyword in col.lower() for keyword in metric_keywords):
                    primary_metric = col
                    break
            if not primary_metric:
                primary_metric = numeric_cols[0]
        
        return {
            "primary_metric": primary_metric,
            "secondary_metric": numeric_cols[1] if len(numeric_cols) > 1 else None,
            "date_column": date_cols[0] if date_cols else None,
            "id_column": None,
            "group_column": None,
            "quality_column": None,
            "quantity_column": None,
            "cost_column": None,
            "status_column": None
        }
    
    def _calculate_metrics(self, df: pd.DataFrame, schema: Dict) -> Dict[str, Any]:
        """Calculate universal metrics using detected columns."""
        metrics = {}
        metrics["row_count"] = len(df)
        metrics["column_count"] = len(df.columns)
        
        primary = schema["primary_metric"]
        if primary:
            df_primary = pd.to_numeric(df[primary], errors='coerce')
            metrics["primary_metric_name"] = primary
            metrics["primary_metric_sum"] = float(df_primary.sum()) if not df_primary.isna().all() else 0
            metrics["primary_metric_mean"] = float(df_primary.mean()) if not df_primary.isna().all() else 0
            metrics["primary_metric_median"] = float(df_primary.median()) if not df_primary.isna().all() else 0
            metrics["primary_metric_min"] = float(df_primary.min()) if not df_primary.isna().all() else 0
            metrics["primary_metric_max"] = float(df_primary.max()) if not df_primary.isna().all() else 0
            metrics["primary_metric_std"] = float(df_primary.std()) if not df_primary.isna().all() else 0
        
        secondary = schema["secondary_metric"]
        if secondary and secondary != primary:
            df_secondary = pd.to_numeric(df[secondary], errors='coerce')
            metrics["secondary_metric_name"] = secondary
            metrics["secondary_metric_sum"] = float(df_secondary.sum()) if not df_secondary.isna().all() else 0
            metrics["secondary_metric_mean"] = float(df_secondary.mean()) if not df_secondary.isna().all() else 0
        
        cost = schema["cost_column"]
        if cost and primary:
            df_cost = pd.to_numeric(df[cost], errors='coerce').fillna(0)
            df_primary = pd.to_numeric(df[primary], errors='coerce').fillna(0)
            metrics["total_cost"] = float(df_cost.sum())
            metrics["profit"] = float(df_primary.sum() - df_cost.sum())
            metrics["profit_margin"] = (metrics["profit"] / df_primary.sum() * 100) if df_primary.sum() > 0 else 0
            metrics["cost_leakage"] = float(df_cost.sum())
        
        if schema["id_column"]:
            metrics["unique_entities"] = df[schema["id_column"]].nunique()
        
        if schema["quality_column"]:
            df_quality = pd.to_numeric(df[schema["quality_column"]], errors='coerce')
            metrics["avg_quality"] = float(df_quality.mean()) if not df_quality.isna().all() else None
            if metrics["avg_quality"]:
                low_threshold = 2.5 if metrics["avg_quality"] > 3 else metrics["avg_quality"] * 0.6
                metrics["low_quality_percentage"] = float((df_quality <= low_threshold).mean() * 100)
        
        if schema["quantity_column"]:
            df_qty = pd.to_numeric(df[schema["quantity_column"]], errors='coerce').fillna(0)
            metrics["total_quantity"] = int(df_qty.sum())
            metrics["avg_quantity"] = float(df_qty.mean())
        
        if schema["date_column"]:
            try:
                df_date = pd.to_datetime(df[schema["date_column"]], errors='coerce')
                metrics["date_start"] = df_date.min().strftime("%Y-%m-%d") if not pd.isna(df_date.min()) else None
                metrics["date_end"] = df_date.max().strftime("%Y-%m-%d") if not pd.isna(df_date.max()) else None
                metrics["date_range_days"] = (df_date.max() - df_date.min()).days if not pd.isna(df_date.min()) else None
            except:
                metrics["date_start"] = None
                metrics["date_end"] = None
        
        return metrics
    
    def _calculate_trends(self, df: pd.DataFrame, schema: Dict) -> Dict[str, Any]:
        """Calculate trends if date and primary metric exist."""
        if not schema["date_column"] or not schema["primary_metric"]:
            return {"trend": "insufficient_data", "message": "Need date and primary metric columns for trend analysis"}
        
        try:
            df_copy = df.copy()
            df_copy[schema["date_column"]] = pd.to_datetime(df_copy[schema["date_column"]], errors='coerce')
            df_copy = df_copy.dropna(subset=[schema["date_column"], schema["primary_metric"]])
            
            if len(df_copy) == 0:
                return {"trend": "insufficient_data", "message": "No valid data points for trend analysis"}
            
            df_copy["date"] = df_copy[schema["date_column"]].dt.date
            daily = df_copy.groupby("date")[schema["primary_metric"]].sum().reset_index()
            daily.columns = ["date", "metric_value"]
            daily["metric_value"] = pd.to_numeric(daily["metric_value"], errors='coerce').fillna(0)
            daily = daily.sort_values("date")
            
            if len(daily) >= 2:
                if len(daily) >= 14:
                    recent = daily["metric_value"].tail(7).mean()
                    previous = daily["metric_value"].tail(14).head(7).mean()
                    trend = "increasing" if recent > previous else "decreasing" if recent < previous else "stable"
                    trend_pct = ((recent - previous) / previous * 100) if previous > 0 else 0
                else:
                    first_half = daily["metric_value"].head(len(daily)//2).mean()
                    second_half = daily["metric_value"].tail(len(daily)//2).mean()
                    trend = "increasing" if second_half > first_half else "decreasing" if second_half < first_half else "stable"
                    trend_pct = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                
                if len(daily) > 1:
                    first_val = daily["metric_value"].iloc[0]
                    last_val = daily["metric_value"].iloc[-1]
                    total_growth_pct = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
                else:
                    total_growth_pct = 0
                
                return {
                    "trend": trend,
                    "trend_percentage": round(trend_pct, 2),
                    "total_growth_percentage": round(total_growth_pct, 2),
                    "avg_daily_value": float(daily["metric_value"].mean()),
                    "peak_value": float(daily["metric_value"].max()),
                    "peak_date": str(daily.loc[daily["metric_value"].idxmax(), "date"]) if len(daily) > 0 else None,
                    "data_points": len(daily)
                }
            else:
                return {"trend": "insufficient_data", "message": f"Only {len(daily)} data points available"}
                
        except Exception as e:
            if self.verbose:
                print(f"Trend calculation error: {e}")
            return {"trend": "error", "error": str(e)}
    
    def _generate_insights(self, metrics: Dict, trends: Dict) -> str:
        """Generate business insights using LLM - OPTIMIZED prompt."""
        # Use concise format to reduce tokens
        prompt = f"""
        Data Summary:
        - Revenue: ${metrics.get('primary_metric_sum', 0):,.0f}
        - Margin: {metrics.get('profit_margin', 0):.0f}%
        - Cost: ${metrics.get('total_cost', 0):,.0f}
        - Trend: {trends.get('trend', 'N/A')} ({trends.get('trend_percentage', 0):.0f}%)
        
        3 bullet points:
        1. Key finding
        2. Critical issue
        3. Recommendation
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="Concise data analyst. 3 short bullet points."),
                HumanMessage(content=prompt)
            ])
            return response.content
        except Exception as e:
            if self.verbose:
                print(f"Insight generation error: {e}")
            return f"Analysis complete. Revenue: ${metrics.get('primary_metric_sum', 0):,.0f}"
    
    def _print_analysis_results(self, metrics: Dict, trends: Dict, insights: str, schema: Dict):
        """Print formatted analysis results to console."""
        # (Keep existing print function unchanged)
        print(f"\n{'='*60}")
        print(f"📊 BUSINESS ANALYSIS REPORT")
        print(f"{'='*60}")
        
        primary_name = metrics.get('primary_metric_name', 'Revenue').replace('_', ' ').title()
        primary_sum = metrics.get('primary_metric_sum', 0)
        print(f"\n💰 {primary_name}: ${primary_sum:,.2f}")
        
        if 'total_cost' in metrics:
            cost = metrics.get('total_cost', 0)
            print(f"💸 Total Cost: ${cost:,.2f}")
            if primary_sum > 0:
                print(f"📊 Cost % of Revenue: {(cost / primary_sum) * 100:.1f}%")
        
        if 'profit' in metrics:
            profit = metrics.get('profit', 0)
            margin = metrics.get('profit_margin', 0)
            print(f"💵 Profit: ${profit:,.2f}")
            print(f"📈 Profit Margin: {margin:.1f}%")
        
        print(f"\n📦 Total Records: {metrics.get('row_count', 0):,}")
        
        if 'unique_entities' in metrics:
            print(f"👥 Unique Entities: {metrics['unique_entities']:,}")
        
        if metrics.get('date_start') and metrics.get('date_end'):
            print(f"\n📅 PERIOD: {metrics['date_start']} to {metrics['date_end']}")
        
        if trends and trends.get('trend') not in ['insufficient_data', 'error']:
            print(f"\n📈 TREND: {trends.get('trend', 'N/A').upper()} ({trends.get('trend_percentage', 0)}%)")
        
        print(f"\n💡 INSIGHTS:\n{insights}")
        print(f"\n{'='*60}\n")
    
    def analyze(self, df: Union[pd.DataFrame, str], analysis_type: str = "business") -> Dict[str, Any]:
        """Main analysis method."""
        try:
            if isinstance(df, str):
                if not os.path.exists(df):
                    return {"status": "error", "error": f"File not found: {df}"}
                df = pd.read_csv(df)
            
            if not isinstance(df, pd.DataFrame):
                return {"status": "error", "error": "Input must be a pandas DataFrame or CSV file path"}
            
            if df.empty:
                return {"status": "error", "error": "DataFrame is empty"}
            
            if self.verbose:
                print(f"\n📂 Loading dataset: {len(df):,} rows, {len(df.columns)} columns")
            
            # LLM detects schema (with caching)
            schema = self._llm_detect_schema(df, analysis_type)
            
            if self.verbose:
                print(f"\n🔍 Detected Schema:")
                for key, val in schema.items():
                    if val:
                        print(f"   {key}: {val}")
            
            if not schema["primary_metric"]:
                if self.verbose:
                    print(f"\n⚠️ Warning: No primary metric column detected.")
                return {
                    "status": "warning",
                    "message": "No primary metric column detected.",
                    "detected_schema": schema,
                    "metrics": self._calculate_metrics(df, schema)
                }
            
            metrics = self._calculate_metrics(df, schema)
            trends = self._calculate_trends(df, schema)
            insights = self._generate_insights(metrics, trends)
            
            if self.verbose:
                self._print_analysis_results(metrics, trends, insights, schema)
            
            # Create enriched dataset
            enriched_df = df.copy()
            if schema["primary_metric"] and schema["cost_column"]:
                rev_numeric = pd.to_numeric(enriched_df[schema["primary_metric"]], errors='coerce').fillna(0)
                cost_numeric = pd.to_numeric(enriched_df[schema["cost_column"]], errors='coerce').fillna(0)
                enriched_df["calculated_profit"] = rev_numeric - cost_numeric
                enriched_df["calculated_margin_pct"] = (enriched_df["calculated_profit"] / (rev_numeric + 1e-10) * 100).round(2)
            
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "detected_schema": schema,
                "metrics": metrics,
                "trends": trends,
                "insights": insights,
                "enriched_dataframe": enriched_df if len(enriched_df) <= 1000 else None
            }
            
        except Exception as e:
            import traceback
            if self.verbose:
                print(f"\n❌ ERROR: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc() if self.verbose else None
            }