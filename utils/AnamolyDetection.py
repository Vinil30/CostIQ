# anomaly_detection_agent.py

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetectionAgent:
    """
    Detects specific anomalies at product, seller, category, and time levels.
    NO LLM SCHEMA DETECTION - uses provided schema or fallback patterns.
    """
    
    def __init__(
        self,
        api_key: str = None,  # Made optional since we don't use LLM anymore
        model: str = "openai/gpt-oss-120b",
        verbose: bool = False
    ):
        self.api_key = api_key
        self.model = model
        self.verbose = verbose
        # Only initialize LLM if api_key provided (for root cause insights)
        self.llm = None
        if api_key:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(api_key=self.api_key, model=self.model, temperature=0)
    
    def _detect_schema_fallback(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Smart fallback schema detection using column name patterns.
        NO LLM CALL - purely rule-based.
        """
        patterns = {
            "product_col": ["product_id", "product", "item_id", "item", "sku", "product_code"],
            "seller_col": ["seller_id", "seller", "vendor_id", "vendor", "supplier"],
            "category_col": ["category", "product_category", "cat", "product_cat"],
            "review_col": ["review_score", "rating", "score", "review", "star_rating"],
            "revenue_col": ["payment_value", "price", "revenue", "amount", "value", "total", "sales"],
            "cost_col": ["freight_value", "cost", "shipping", "freight", "shipping_cost"],
            "date_col": ["order_purchase_timestamp", "date", "timestamp", "created_at", "order_date", "purchase_date"]
        }
        
        schema = {}
        
        for key, pattern_list in patterns.items():
            found = None
            # Check for exact matches first
            for pattern in pattern_list:
                if pattern in df.columns:
                    found = pattern
                    break
            
            # If no exact match, check for partial matches (case insensitive)
            if not found:
                for col in df.columns:
                    col_lower = col.lower()
                    for pattern in pattern_list:
                        if pattern.lower() in col_lower:
                            found = col
                            break
                    if found:
                        break
            
            schema[key] = found
        
        # Additional validation for revenue_col (must be numeric)
        if schema["revenue_col"]:
            if not pd.api.types.is_numeric_dtype(df[schema["revenue_col"]]):
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    if any(kw in col.lower() for kw in ['price', 'value', 'amount', 'revenue', 'payment']):
                        schema["revenue_col"] = col
                        break
        
        return schema
    
    def _detect_product_anomalies(self, df: pd.DataFrame, schema: Dict) -> List[Dict]:
        """Find problematic products based on high return rate, low profit margin, high freight ratio."""
        anomalies = []
        product_col = schema.get("product_col")
        review_col = schema.get("review_col")
        revenue_col = schema.get("revenue_col")
        cost_col = schema.get("cost_col")
        
        if not product_col or not revenue_col:
            return anomalies
        
        # Group by product
        product_stats = df.groupby(product_col).agg({
            revenue_col: ['sum', 'mean', 'count'],
        }).reset_index()
        
        product_stats.columns = [product_col, 'total_revenue', 'avg_revenue', 'order_count']
        product_stats = product_stats[product_stats['order_count'] >= 5]
        
        if len(product_stats) == 0:
            return anomalies
        
        # Add review score if available
        if review_col and review_col in df.columns:
            review_stats = df.groupby(product_col)[review_col].mean().reset_index()
            review_stats.columns = [product_col, 'avg_review']
            product_stats = product_stats.merge(review_stats, on=product_col, how='left')
            product_stats['avg_review'] = product_stats['avg_review'].fillna(3)
            product_stats['return_rate_proxy'] = (5 - product_stats['avg_review']) / 4 * 100
        else:
            product_stats['return_rate_proxy'] = 0
            product_stats['avg_review'] = None
        
        # Add cost metrics
        if cost_col and cost_col in df.columns:
            df_cost = pd.to_numeric(df[cost_col], errors='coerce').fillna(0)
            cost_stats = df.groupby(product_col)[cost_col].sum().reset_index()
            cost_stats.columns = [product_col, 'total_cost']
            product_stats = product_stats.merge(cost_stats, on=product_col, how='left')
            product_stats['total_cost'] = product_stats['total_cost'].fillna(0)
            product_stats['profit'] = product_stats['total_revenue'] - product_stats['total_cost']
            product_stats['margin'] = (product_stats['profit'] / (product_stats['total_revenue'] + 1e-10) * 100).fillna(0)
            product_stats['freight_ratio'] = (product_stats['total_cost'] / (product_stats['total_revenue'] + 1e-10) * 100).fillna(0)
        else:
            product_stats['total_cost'] = 0
            product_stats['margin'] = 0
            product_stats['freight_ratio'] = 0
        
        # Define dynamic thresholds
        valid_products = product_stats[product_stats['total_revenue'] > 0]
        if len(valid_products) == 0:
            return anomalies
            
        avg_return_rate = valid_products['return_rate_proxy'].mean()
        avg_margin = valid_products['margin'].mean()
        avg_freight = valid_products['freight_ratio'].mean()
        
        # Find anomalies
        for _, row in product_stats.iterrows():
            if row['total_revenue'] <= 0:
                continue
                
            reasons = []
            impact = 0
            
            if review_col and row['return_rate_proxy'] > avg_return_rate * 1.5 and row['return_rate_proxy'] > 15:
                reasons.append(f"return_rate_{row['return_rate_proxy']:.0f}%_vs_avg_{avg_return_rate:.0f}%")
                impact += row['total_revenue'] * (min(row['return_rate_proxy'], 50) / 100)
            
            if cost_col and row['margin'] < 10 and row['margin'] < avg_margin - 15:
                reasons.append(f"margin_{row['margin']:.0f}%_vs_avg_{avg_margin:.0f}%")
                impact += row['total_cost'] * 0.5
            
            if cost_col and row['freight_ratio'] > avg_freight * 1.5 and row['freight_ratio'] > 25:
                reasons.append(f"freight_{row['freight_ratio']:.0f}%_vs_avg_{avg_freight:.0f}%")
                impact += row['total_cost'] * 0.3
            
            if reasons:
                anomalies.append({
                    "type": "product",
                    "id": str(row[product_col]),
                    "name": str(row.get('product_name', row[product_col]))[:100],
                    "reasons": reasons,
                    "estimated_loss": round(impact, 2),
                    "total_revenue": round(row['total_revenue'], 2),
                    "order_count": int(row['order_count']),
                    "metrics": {
                        "return_rate": round(row['return_rate_proxy'], 1) if review_col else None,
                        "margin": round(row['margin'], 1) if cost_col else None,
                        "freight_ratio": round(row['freight_ratio'], 1) if cost_col else None,
                        "avg_review": round(row['avg_review'], 1) if review_col and row['avg_review'] else None
                    }
                })
        
        return sorted(anomalies, key=lambda x: x['estimated_loss'], reverse=True)
    
    def _detect_seller_anomalies(self, df: pd.DataFrame, schema: Dict) -> List[Dict]:
        """Find problematic sellers based on low review scores and margins."""
        anomalies = []
        seller_col = schema.get("seller_col")
        product_col = schema.get("product_col")
        review_col = schema.get("review_col")
        revenue_col = schema.get("revenue_col")
        cost_col = schema.get("cost_col")
        
        if not seller_col or not revenue_col:
            return anomalies
        
        order_col = 'order_id' if 'order_id' in df.columns else None
        
        agg_dict = {revenue_col: ['sum', 'count']}
        if order_col:
            agg_dict[order_col] = 'nunique'
        
        seller_stats = df.groupby(seller_col).agg(agg_dict).reset_index()
        
        if order_col:
            seller_stats.columns = [seller_col, 'total_revenue', 'transaction_count', 'order_count']
        else:
            seller_stats.columns = [seller_col, 'total_revenue', 'order_count']
        
        seller_stats = seller_stats[seller_stats['order_count'] >= 3]
        
        if len(seller_stats) == 0:
            return anomalies
        
        if review_col and review_col in df.columns:
            review_stats = df.groupby(seller_col)[review_col].mean().reset_index()
            review_stats.columns = [seller_col, 'avg_review']
            seller_stats = seller_stats.merge(review_stats, on=seller_col, how='left')
            seller_stats['avg_review'] = seller_stats['avg_review'].fillna(3)
        
        if cost_col and cost_col in df.columns:
            cost_stats = df.groupby(seller_col)[cost_col].sum().reset_index()
            cost_stats.columns = [seller_col, 'total_cost']
            seller_stats = seller_stats.merge(cost_stats, on=seller_col, how='left')
            seller_stats['total_cost'] = seller_stats['total_cost'].fillna(0)
            seller_stats['profit'] = seller_stats['total_revenue'] - seller_stats['total_cost']
            seller_stats['margin'] = (seller_stats['profit'] / (seller_stats['total_revenue'] + 1e-10) * 100).fillna(0)
        
        if product_col and product_col in df.columns:
            product_count = df.groupby(seller_col)[product_col].nunique().reset_index()
            product_count.columns = [seller_col, 'unique_products']
            seller_stats = seller_stats.merge(product_count, on=seller_col, how='left')
            seller_stats['unique_products'] = seller_stats['unique_products'].fillna(0).astype(int)
        else:
            seller_stats['unique_products'] = 0
        
        avg_review = seller_stats['avg_review'].mean() if review_col else 5
        avg_margin = seller_stats['margin'].mean() if cost_col else 0
        
        for _, row in seller_stats.iterrows():
            reasons = []
            impact = 0
            
            if review_col and row['avg_review'] < 3 and row['avg_review'] < avg_review - 0.5:
                reasons.append(f"avg_review_{row['avg_review']:.1f}_vs_avg_{avg_review:.1f}")
                impact += row['total_revenue'] * 0.3
            
            if cost_col and row['margin'] < 5 and row['margin'] < avg_margin - 15:
                reasons.append(f"margin_{row['margin']:.0f}%_vs_avg_{avg_margin:.0f}%")
                impact += row['total_cost'] * 0.5
            
            if review_col and product_col and product_col in df.columns:
                seller_products = df[df[seller_col] == row[seller_col]]
                if len(seller_products) > 0 and review_col in seller_products.columns:
                    low_review_products = (seller_products[review_col] <= 2.5).sum()
                    if low_review_products > 3:
                        reasons.append(f"{low_review_products}_products_with_low_reviews")
                        impact += row['total_revenue'] * 0.2
            
            if reasons:
                anomalies.append({
                    "type": "seller",
                    "id": str(row[seller_col]),
                    "reasons": reasons,
                    "estimated_loss": round(impact, 2),
                    "total_revenue": round(row['total_revenue'], 2),
                    "order_count": int(row['order_count']),
                    "metrics": {
                        "avg_review": round(row['avg_review'], 1) if review_col and row.get('avg_review') else None,
                        "margin": round(row['margin'], 1) if cost_col else None,
                        "unique_products": int(row['unique_products'])
                    }
                })
        
        return sorted(anomalies, key=lambda x: x['estimated_loss'], reverse=True)
    
    def _detect_category_anomalies(self, df: pd.DataFrame, schema: Dict) -> List[Dict]:
        """Find problematic categories based on high return rates and low margins."""
        anomalies = []
        category_col = schema.get("category_col")
        review_col = schema.get("review_col")
        revenue_col = schema.get("revenue_col")
        cost_col = schema.get("cost_col")
        
        if not category_col or not revenue_col:
            return anomalies
        
        order_col = 'order_id' if 'order_id' in df.columns else None
        
        agg_dict = {revenue_col: ['sum', 'count']}
        if order_col:
            agg_dict[order_col] = 'nunique'
        
        cat_stats = df.groupby(category_col).agg(agg_dict).reset_index()
        
        if order_col:
            cat_stats.columns = [category_col, 'total_revenue', 'transaction_count', 'order_count']
        else:
            cat_stats.columns = [category_col, 'total_revenue', 'order_count']
        
        cat_stats = cat_stats[cat_stats['order_count'] >= 10]
        
        if len(cat_stats) == 0:
            return anomalies
        
        if review_col and review_col in df.columns:
            review_stats = df.groupby(category_col)[review_col].mean().reset_index()
            review_stats.columns = [category_col, 'avg_review']
            cat_stats = cat_stats.merge(review_stats, on=category_col, how='left')
            cat_stats['avg_review'] = cat_stats['avg_review'].fillna(3)
            cat_stats['return_rate'] = (5 - cat_stats['avg_review']) / 4 * 100
        else:
            cat_stats['return_rate'] = 0
        
        if cost_col and cost_col in df.columns:
            cost_stats = df.groupby(category_col)[cost_col].sum().reset_index()
            cost_stats.columns = [category_col, 'total_cost']
            cat_stats = cat_stats.merge(cost_stats, on=category_col, how='left')
            cat_stats['total_cost'] = cat_stats['total_cost'].fillna(0)
            cat_stats['profit'] = cat_stats['total_revenue'] - cat_stats['total_cost']
            cat_stats['margin'] = (cat_stats['profit'] / (cat_stats['total_revenue'] + 1e-10) * 100).fillna(0)
            cat_stats['freight_ratio'] = (cat_stats['total_cost'] / (cat_stats['total_revenue'] + 1e-10) * 100).fillna(0)
        else:
            cat_stats['margin'] = 0
            cat_stats['freight_ratio'] = 0
        
        valid_cats = cat_stats[cat_stats['total_revenue'] > 0]
        if len(valid_cats) == 0:
            return anomalies
            
        avg_return = valid_cats['return_rate'].mean() if review_col else 0
        avg_margin = valid_cats['margin'].mean() if cost_col else 0
        avg_freight = valid_cats['freight_ratio'].mean() if cost_col else 0
        
        for _, row in cat_stats.iterrows():
            if row['total_revenue'] <= 0:
                continue
                
            reasons = []
            impact = 0
            
            if review_col and row['return_rate'] > avg_return * 1.3 and row['return_rate'] > 10:
                reasons.append(f"return_rate_{row['return_rate']:.0f}%_vs_avg_{avg_return:.0f}%")
                impact += row['total_revenue'] * (min(row['return_rate'], 40) / 100)
            
            if cost_col and row['margin'] < 10 and row['margin'] < avg_margin - 10:
                reasons.append(f"margin_{row['margin']:.0f}%_vs_avg_{avg_margin:.0f}%")
                impact += row['total_cost'] * 0.3
            
            if cost_col and row['freight_ratio'] > avg_freight * 1.3 and row['freight_ratio'] > 20:
                reasons.append(f"freight_{row['freight_ratio']:.0f}%_vs_avg_{avg_freight:.0f}%")
                impact += row['total_cost'] * 0.2
            
            if reasons:
                anomalies.append({
                    "type": "category",
                    "id": str(row[category_col]),
                    "reasons": reasons,
                    "estimated_loss": round(impact, 2),
                    "total_revenue": round(row['total_revenue'], 2),
                    "order_count": int(row['order_count']),
                    "metrics": {
                        "return_rate": round(row['return_rate'], 1) if review_col else None,
                        "margin": round(row['margin'], 1) if cost_col else None,
                        "freight_ratio": round(row['freight_ratio'], 1) if cost_col else None,
                        "avg_review": round(row['avg_review'], 1) if review_col and row.get('avg_review') else None
                    }
                })
        
        return sorted(anomalies, key=lambda x: x['estimated_loss'], reverse=True)
    
    def _detect_time_anomalies(self, df: pd.DataFrame, schema: Dict) -> List[Dict]:
        """Find time-based anomalies like sudden drops or spikes."""
        anomalies = []
        date_col = schema.get("date_col")
        revenue_col = schema.get("revenue_col")
        
        if not date_col or not revenue_col:
            return anomalies
        
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy = df_copy.dropna(subset=[date_col])
            
            if len(df_copy) == 0:
                return anomalies
                
            df_copy['date'] = df_copy[date_col].dt.date
            df_copy[revenue_col] = pd.to_numeric(df_copy[revenue_col], errors='coerce').fillna(0)
            
            daily = df_copy.groupby('date')[revenue_col].sum().reset_index()
            daily.columns = ['date', 'revenue']
            daily = daily.sort_values('date')
            
            if len(daily) < 14:
                return anomalies
            
            daily['ma_7'] = daily['revenue'].rolling(window=7, min_periods=1).mean()
            daily['ma_14'] = daily['revenue'].rolling(window=14, min_periods=1).mean()
            daily['deviation'] = (daily['revenue'] - daily['ma_14']) / (daily['ma_14'] + 1e-10) * 100
            
            for _, row in daily.iterrows():
                if row['deviation'] < -25:
                    anomalies.append({
                        "type": "time",
                        "date": str(row['date']),
                        "issue": "significant_drop",
                        "deviation": round(row['deviation'], 1),
                        "revenue_loss": round(row['ma_14'] - row['revenue'], 2),
                        "actual_revenue": round(row['revenue'], 2),
                        "expected_revenue": round(row['ma_14'], 2)
                    })
                elif row['deviation'] > 40:
                    anomalies.append({
                        "type": "time",
                        "date": str(row['date']),
                        "issue": "significant_spike",
                        "deviation": round(row['deviation'], 1),
                        "revenue_gain": round(row['revenue'] - row['ma_14'], 2)
                    })
            
            return sorted(anomalies, key=lambda x: abs(x.get('deviation', 0)), reverse=True)[:10]
            
        except Exception as e:
            if self.verbose:
                print(f"Time anomaly detection error: {e}")
            return []
    
    def _generate_root_cause_insights(self, anomalies: Dict, schema: Dict) -> str:
        """Generate root cause analysis using LLM (only if LLM available)."""
        # Count total issues
        total_issues = (
            len(anomalies.get('products', [])) +
            len(anomalies.get('sellers', [])) +
            len(anomalies.get('categories', [])) +
            len(anomalies.get('time', []))
        )
        
        if total_issues == 0:
            return "No significant anomalies detected. Business appears to be operating within normal parameters."
        
        # If no LLM, return basic summary
        if not self.llm:
            return f"Found {total_issues} anomalies across products, sellers, categories, and time periods. Review the detailed anomalies for specific issues."
        
        # Prepare summary for LLM (limit to top items)
        summary = {
            "top_products": anomalies.get('products', [])[:5],
            "top_sellers": anomalies.get('sellers', [])[:5],
            "top_categories": anomalies.get('categories', [])[:5],
            "time_anomalies": anomalies.get('time', [])[:5]
        }
        
        # Optimized prompt - smaller and more concise
        prompt = f"""
        Anomalies: {total_issues} issues found.
        Top products: {len(summary['top_products'])}
        Top sellers: {len(summary['top_sellers'])}
        Top categories: {len(summary['top_categories'])}
        
        Provide:
        1. Root cause
        2. Recommendations
        3. Priority order
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="Root cause analyst. Be concise."),
                HumanMessage(content=prompt)
            ])
            return response.content
        except Exception as e:
            if self.verbose:
                print(f"Root cause insight generation error: {e}")
            return f"Root cause analysis: Found {total_issues} anomalies across products, sellers, categories, and time periods."
    
    def analyze(self, df: Union[pd.DataFrame, str, None] = None, schema: Dict = None) -> Dict[str, Any]:
        """
        Run complete anomaly detection.
        
        Args:
            df: pandas DataFrame OR path to CSV file
            schema: Pre-detected schema (from BusinessAnalystAgent) - REQUIRED for optimal performance
        
        Returns:
            Dict with anomalies at product, seller, category, and time levels
        """
        try:
            # Load data
            if df is None:
                return {"status": "error", "error": "No data provided"}
            elif isinstance(df, str):
                if not os.path.exists(df):
                    return {"status": "error", "error": f"File not found: {df}"}
                df = pd.read_csv(df)
                if self.verbose:
                    print(f"📂 Loaded from file: {len(df)} rows, {len(df.columns)} columns")
            elif isinstance(df, pd.DataFrame):
                if self.verbose:
                    print(f"📊 Analyzing dataframe: {len(df)} rows, {len(df.columns)} columns")
            else:
                return {"status": "error", "error": "Input must be a pandas DataFrame or file path"}
            
            if df.empty:
                return {"status": "error", "error": "DataFrame is empty"}
            
            # Use provided schema or fallback detection (NO LLM)
            if schema is None:
                if self.verbose:
                    print("\n🔍 Using fallback schema detection (no LLM)...")
                schema = self._detect_schema_fallback(df)
            else:
                if self.verbose:
                    print("\n🔍 Using provided schema from Business Analyst")
            
            if self.verbose:
                print(f"\n📋 Schema:")
                for key, val in schema.items():
                    if val:
                        print(f"   {key}: {val}")
            
            if not schema.get("revenue_col"):
                if self.verbose:
                    print("\n⚠️ Warning: No revenue column detected. Anomaly detection will be limited.")
            
            # Detect anomalies
            if self.verbose:
                print("\n🔎 Detecting anomalies...")
            product_anomalies = self._detect_product_anomalies(df, schema)
            seller_anomalies = self._detect_seller_anomalies(df, schema)
            category_anomalies = self._detect_category_anomalies(df, schema)
            time_anomalies = self._detect_time_anomalies(df, schema)
            
            # Calculate total loss
            total_loss = sum(p.get('estimated_loss', 0) for p in product_anomalies[:10])
            total_loss += sum(s.get('estimated_loss', 0) for s in seller_anomalies[:10])
            total_loss += sum(c.get('estimated_loss', 0) for c in category_anomalies[:10])
            
            # Generate insights
            anomalies_dict = {
                "products": product_anomalies,
                "sellers": seller_anomalies,
                "categories": category_anomalies,
                "time": time_anomalies
            }
            
            if self.verbose:
                print("\n🧠 Generating insights...")
            root_cause_insights = self._generate_root_cause_insights(anomalies_dict, schema)
            
            return {
                "status": "success",
                "detected_schema": schema,
                "anomalies": {
                    "products": product_anomalies[:20],
                    "sellers": seller_anomalies[:20],
                    "categories": category_anomalies[:20],
                    "time": time_anomalies
                },
                "summary": {
                    "total_anomaly_loss": round(total_loss, 2),
                    "problematic_products": len(product_anomalies),
                    "problematic_sellers": len(seller_anomalies),
                    "problematic_categories": len(category_anomalies),
                    "time_anomalies_count": len(time_anomalies)
                },
                "root_cause_insights": root_cause_insights
            }
            
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc() if self.verbose else None
            }
    
    def analyze_dataframe(self, df: pd.DataFrame, schema: Dict = None) -> Dict[str, Any]:
        """Convenience method to analyze a dataframe directly."""
        return self.analyze(df=df, schema=schema)
    
    def get_top_anomalies(self, df: Union[pd.DataFrame, str, None] = None, n: int = 10, schema: Dict = None) -> List[Dict]:
        """Get top N anomalies across all categories sorted by loss."""
        result = self.analyze(df=df, schema=schema)
        
        if result["status"] != "success":
            return []
        
        all_anomalies = []
        
        for product in result["anomalies"]["products"][:n]:
            all_anomalies.append({
                "rank": len(all_anomalies) + 1,
                "type": "Product",
                "name": product["id"],
                "issues": product["reasons"],
                "estimated_loss": product["estimated_loss"],
                "metrics": product["metrics"]
            })
        
        for seller in result["anomalies"]["sellers"][:n]:
            all_anomalies.append({
                "rank": len(all_anomalies) + 1,
                "type": "Seller",
                "name": seller["id"],
                "issues": seller["reasons"],
                "estimated_loss": seller["estimated_loss"],
                "metrics": seller["metrics"]
            })
        
        for category in result["anomalies"]["categories"][:n]:
            all_anomalies.append({
                "rank": len(all_anomalies) + 1,
                "type": "Category",
                "name": category["id"],
                "issues": category["reasons"],
                "estimated_loss": category["estimated_loss"],
                "metrics": category["metrics"]
            })
        
        return sorted(all_anomalies, key=lambda x: x["estimated_loss"], reverse=True)[:n]