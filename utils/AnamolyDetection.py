# anomaly_detection_agent.py

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetectionAgent:
    """
    Detects specific anomalies at product, seller, category, and time levels.
    Pinpoints exactly which entities are causing cost leakages.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "mixtral-8x7b-32768",
        verbose: bool = False,
        enriched_dataset_path: str = "enriched_dataset.csv"
    ):
        self.api_key = api_key
        self.model = model
        self.verbose = verbose
        self.enriched_dataset_path = enriched_dataset_path
        self.llm = ChatGroq(api_key=self.api_key, model=self.model, temperature=0)
    
    def _detect_product_anomalies(self, df: pd.DataFrame, schema: Dict) -> List[Dict]:
        """
        Find problematic products based on:
        - High return rate (low review scores)
        - Low profit margin
        - High freight ratio
        """
        
        anomalies = []
        product_col = schema.get("product_col")
        review_col = schema.get("review_col")
        revenue_col = schema.get("revenue_col")
        cost_col = schema.get("cost_col")
        
        if not product_col:
            return anomalies
        
        # Group by product
        product_stats = df.groupby(product_col).agg({
            revenue_col: ['sum', 'mean', 'count'],
        }).reset_index()
        
        product_stats.columns = [product_col, 'total_revenue', 'avg_revenue', 'order_count']
        product_stats = product_stats[product_stats['order_count'] >= 10]  # Min sample size
        
        # Add review score if available
        if review_col:
            review_stats = df.groupby(product_col)[review_col].mean().reset_index()
            review_stats.columns = [product_col, 'avg_review']
            product_stats = product_stats.merge(review_stats, on=product_col, how='left')
            product_stats['return_rate_proxy'] = (5 - product_stats['avg_review']) / 4 * 100
        else:
            product_stats['return_rate_proxy'] = 0
        
        # Add cost metrics
        if cost_col:
            cost_stats = df.groupby(product_col)[cost_col].sum().reset_index()
            cost_stats.columns = [product_col, 'total_cost']
            product_stats = product_stats.merge(cost_stats, on=product_col, how='left')
            product_stats['profit'] = product_stats['total_revenue'] - product_stats['total_cost']
            product_stats['margin'] = (product_stats['profit'] / product_stats['total_revenue'] * 100).fillna(0)
            
            # Freight ratio
            freight_ratio = df.groupby(product_col).apply(
                lambda x: (x[cost_col].sum() / x[revenue_col].sum() * 100) if x[revenue_col].sum() > 0 else 0
            ).reset_index()
            freight_ratio.columns = [product_col, 'freight_ratio']
            product_stats = product_stats.merge(freight_ratio, on=product_col, how='left')
        
        # Define thresholds
        avg_return_rate = product_stats['return_rate_proxy'].mean()
        avg_margin = product_stats['margin'].mean() if cost_col else 0
        avg_freight = product_stats['freight_ratio'].mean() if cost_col else 0
        
        # Find anomalies
        for _, row in product_stats.iterrows():
            reasons = []
            impact = 0
            
            # High return rate
            if row['return_rate_proxy'] > avg_return_rate * 2 and row['return_rate_proxy'] > 20:
                reasons.append(f"return_rate_{row['return_rate_proxy']:.0f}%_vs_avg_{avg_return_rate:.0f}%")
                impact += row['total_revenue'] * (row['return_rate_proxy'] / 100)
            
            # Low margin
            if cost_col and row['margin'] < 10 and row['margin'] < avg_margin - 20:
                reasons.append(f"margin_{row['margin']:.0f}%_vs_avg_{avg_margin:.0f}%")
                impact += row['total_cost'] * 0.5
            
            # High freight
            if cost_col and row['freight_ratio'] > avg_freight * 2 and row['freight_ratio'] > 30:
                reasons.append(f"freight_{row['freight_ratio']:.0f}%_vs_avg_{avg_freight:.0f}%")
                impact += row['total_cost'] * 0.3
            
            if reasons:
                anomalies.append({
                    "type": "product",
                    "id": row[product_col],
                    "name": row.get('product_name', row[product_col]),
                    "reasons": reasons,
                    "estimated_loss": round(impact, 2),
                    "total_revenue": round(row['total_revenue'], 2),
                    "order_count": int(row['order_count']),
                    "metrics": {
                        "return_rate": round(row['return_rate_proxy'], 1),
                        "margin": round(row['margin'], 1) if cost_col else None,
                        "freight_ratio": round(row['freight_ratio'], 1) if cost_col else None,
                        "avg_review": round(row['avg_review'], 1) if review_col else None
                    }
                })
        
        return sorted(anomalies, key=lambda x: x['estimated_loss'], reverse=True)
    
    def _detect_seller_anomalies(self, df: pd.DataFrame, schema: Dict) -> List[Dict]:
        """
        Find problematic sellers based on:
        - Low review scores
        - High return rates
        - Low profit margin
        """
        
        anomalies = []
        seller_col = schema.get("seller_col")
        product_col = schema.get("product_col")
        review_col = schema.get("review_col")
        revenue_col = schema.get("revenue_col")
        cost_col = schema.get("cost_col")
        
        if not seller_col:
            return anomalies
        
        # Group by seller
        seller_stats = df.groupby(seller_col).agg({
            revenue_col: ['sum', 'count'],
            'order_id': 'nunique'
        }).reset_index()
        
        seller_stats.columns = [seller_col, 'total_revenue', 'transaction_count', 'order_count']
        seller_stats = seller_stats[seller_stats['order_count'] >= 5]
        
        # Add review scores
        if review_col:
            review_stats = df.groupby(seller_col)[review_col].mean().reset_index()
            review_stats.columns = [seller_col, 'avg_review']
            seller_stats = seller_stats.merge(review_stats, on=seller_col, how='left')
            seller_stats['bad_review_rate'] = (seller_stats['avg_review'] <= 2.5).astype(int)
        
        # Add cost metrics
        if cost_col:
            cost_stats = df.groupby(seller_col)[cost_col].sum().reset_index()
            cost_stats.columns = [seller_col, 'total_cost']
            seller_stats = seller_stats.merge(cost_stats, on=seller_col, how='left')
            seller_stats['profit'] = seller_stats['total_revenue'] - seller_stats['total_cost']
            seller_stats['margin'] = (seller_stats['profit'] / seller_stats['total_revenue'] * 100).fillna(0)
        
        # Add product count
        product_count = df.groupby(seller_col)[product_col].nunique().reset_index()
        product_count.columns = [seller_col, 'unique_products']
        seller_stats = seller_stats.merge(product_count, on=seller_col, how='left')
        
        # Define thresholds
        avg_review = seller_stats['avg_review'].mean() if review_col else 5
        avg_margin = seller_stats['margin'].mean() if cost_col else 0
        
        # Find anomalies
        for _, row in seller_stats.iterrows():
            reasons = []
            impact = 0
            
            # Low review scores
            if review_col and row['avg_review'] < 3 and row['avg_review'] < avg_review - 1:
                reasons.append(f"avg_review_{row['avg_review']:.1f}_vs_avg_{avg_review:.1f}")
                impact += row['total_revenue'] * 0.3
            
            # Low margin
            if cost_col and row['margin'] < 5 and row['margin'] < avg_margin - 20:
                reasons.append(f"margin_{row['margin']:.0f}%_vs_avg_{avg_margin:.0f}%")
                impact += row['total_cost'] * 0.5
            
            # High number of products with issues
            if review_col:
                seller_products = df[df[seller_col] == row[seller_col]]
                if len(seller_products) > 0 and review_col in seller_products.columns:
                    low_review_products = (seller_products[review_col] <= 2.5).sum()
                    if low_review_products > 3:
                        reasons.append(f"{low_review_products}_products_with_low_reviews")
                        impact += row['total_revenue'] * 0.2
            
            if reasons:
                anomalies.append({
                    "type": "seller",
                    "id": row[seller_col],
                    "reasons": reasons,
                    "estimated_loss": round(impact, 2),
                    "total_revenue": round(row['total_revenue'], 2),
                    "order_count": int(row['order_count']),
                    "metrics": {
                        "avg_review": round(row['avg_review'], 1) if review_col else None,
                        "margin": round(row['margin'], 1) if cost_col else None,
                        "unique_products": int(row['unique_products'])
                    }
                })
        
        return sorted(anomalies, key=lambda x: x['estimated_loss'], reverse=True)
    
    def _detect_category_anomalies(self, df: pd.DataFrame, schema: Dict) -> List[Dict]:
        """
        Find problematic categories based on:
        - High return rates
        - Low margins
        - High freight costs
        """
        
        anomalies = []
        category_col = schema.get("category_col")
        review_col = schema.get("review_col")
        revenue_col = schema.get("revenue_col")
        cost_col = schema.get("cost_col")
        
        if not category_col:
            return anomalies
        
        # Group by category
        cat_stats = df.groupby(category_col).agg({
            revenue_col: ['sum', 'count'],
            'order_id': 'nunique'
        }).reset_index()
        
        cat_stats.columns = [category_col, 'total_revenue', 'transaction_count', 'order_count']
        cat_stats = cat_stats[cat_stats['order_count'] >= 20]
        
        # Add review scores
        if review_col:
            review_stats = df.groupby(category_col)[review_col].mean().reset_index()
            review_stats.columns = [category_col, 'avg_review']
            cat_stats = cat_stats.merge(review_stats, on=category_col, how='left')
            cat_stats['return_rate'] = (5 - cat_stats['avg_review']) / 4 * 100
        
        # Add cost metrics
        if cost_col:
            cost_stats = df.groupby(category_col)[cost_col].sum().reset_index()
            cost_stats.columns = [category_col, 'total_cost']
            cat_stats = cat_stats.merge(cost_stats, on=category_col, how='left')
            cat_stats['profit'] = cat_stats['total_revenue'] - cat_stats['total_cost']
            cat_stats['margin'] = (cat_stats['profit'] / cat_stats['total_revenue'] * 100).fillna(0)
            
            freight_ratio = df.groupby(category_col).apply(
                lambda x: (x[cost_col].sum() / x[revenue_col].sum() * 100) if x[revenue_col].sum() > 0 else 0
            ).reset_index()
            freight_ratio.columns = [category_col, 'freight_ratio']
            cat_stats = cat_stats.merge(freight_ratio, on=category_col, how='left')
        
        # Define thresholds
        avg_return = cat_stats['return_rate'].mean() if review_col else 0
        avg_margin = cat_stats['margin'].mean() if cost_col else 0
        avg_freight = cat_stats['freight_ratio'].mean() if cost_col else 0
        
        # Find anomalies
        for _, row in cat_stats.iterrows():
            reasons = []
            impact = 0
            
            # High return rate
            if review_col and row['return_rate'] > avg_return * 1.5 and row['return_rate'] > 15:
                reasons.append(f"return_rate_{row['return_rate']:.0f}%_vs_avg_{avg_return:.0f}%")
                impact += row['total_revenue'] * (row['return_rate'] / 100)
            
            # Low margin
            if cost_col and row['margin'] < 15 and row['margin'] < avg_margin - 10:
                reasons.append(f"margin_{row['margin']:.0f}%_vs_avg_{avg_margin:.0f}%")
                impact += row['total_cost'] * 0.3
            
            # High freight
            if cost_col and row['freight_ratio'] > avg_freight * 1.5 and row['freight_ratio'] > 25:
                reasons.append(f"freight_{row['freight_ratio']:.0f}%_vs_avg_{avg_freight:.0f}%")
                impact += row['total_cost'] * 0.2
            
            if reasons:
                anomalies.append({
                    "type": "category",
                    "id": row[category_col],
                    "reasons": reasons,
                    "estimated_loss": round(impact, 2),
                    "total_revenue": round(row['total_revenue'], 2),
                    "order_count": int(row['order_count']),
                    "metrics": {
                        "return_rate": round(row['return_rate'], 1) if review_col else None,
                        "margin": round(row['margin'], 1) if cost_col else None,
                        "freight_ratio": round(row['freight_ratio'], 1) if cost_col else None,
                        "avg_review": round(row['avg_review'], 1) if review_col else None
                    }
                })
        
        return sorted(anomalies, key=lambda x: x['estimated_loss'], reverse=True)
    
    def _detect_time_anomalies(self, df: pd.DataFrame, schema: Dict) -> List[Dict]:
        """
        Find time-based anomalies like sudden drops or spikes.
        """
        
        anomalies = []
        date_col = schema.get("date_col")
        revenue_col = schema.get("revenue_col")
        
        if not date_col or not revenue_col:
            return anomalies
        
        # Prepare time series
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy['date'] = df_copy[date_col].dt.date
        
        daily = df_copy.groupby('date')[revenue_col].sum().reset_index()
        daily.columns = ['date', 'revenue']
        daily['revenue'] = pd.to_numeric(daily['revenue'], errors='coerce').fillna(0)
        
        if len(daily) < 30:
            return anomalies
        
        # Calculate moving averages
        daily['ma_7'] = daily['revenue'].rolling(window=7, min_periods=1).mean()
        daily['ma_30'] = daily['revenue'].rolling(window=30, min_periods=1).mean()
        daily['deviation'] = (daily['revenue'] - daily['ma_30']) / daily['ma_30'] * 100
        
        # Find significant drops/spikes
        for _, row in daily.iterrows():
            if row['deviation'] < -30:  # 30% drop
                anomalies.append({
                    "type": "time",
                    "date": str(row['date']),
                    "issue": "significant_drop",
                    "deviation": round(row['deviation'], 1),
                    "revenue_loss": round(row['ma_30'] - row['revenue'], 2),
                    "actual_revenue": round(row['revenue'], 2),
                    "expected_revenue": round(row['ma_30'], 2)
                })
            elif row['deviation'] > 50:  # 50% spike
                anomalies.append({
                    "type": "time",
                    "date": str(row['date']),
                    "issue": "significant_spike",
                    "deviation": round(row['deviation'], 1),
                    "revenue_gain": round(row['revenue'] - row['ma_30'], 2)
                })
        
        return sorted(anomalies, key=lambda x: abs(x.get('deviation', 0)), reverse=True)[:10]
    
    def _generate_root_cause_insights(self, anomalies: Dict, schema: Dict) -> str:
        """
        Use LLM to generate root cause analysis and recommendations.
        """
        
        # Prepare summary for LLM
        summary = {
            "top_products": anomalies['products'][:5],
            "top_sellers": anomalies['sellers'][:5],
            "top_categories": anomalies['categories'][:5],
            "time_anomalies": anomalies['time'][:5]
        }
        
        prompt = f"""
        Based on the anomaly detection results below, provide root cause analysis and recommendations.
        
        TOP PROBLEMATIC PRODUCTS:
        {json.dumps(summary['top_products'], indent=2, default=str)}
        
        TOP PROBLEMATIC SELLERS:
        {json.dumps(summary['top_sellers'], indent=2, default=str)}
        
        TOP PROBLEMATIC CATEGORIES:
        {json.dumps(summary['top_categories'], indent=2, default=str)}
        
        TIME ANOMALIES:
        {json.dumps(summary['time_anomalies'], indent=2, default=str)}
        
        Provide:
        1. Root Cause Analysis (what's causing these issues)
        2. Actionable Recommendations (what to do about each)
        3. Priority Order (which issues to fix first)
        
        Be specific and use the actual product/seller names.
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a root cause analyst. Provide specific, actionable insights."),
                HumanMessage(content=prompt)
            ])
            return response.content
        except Exception as e:
            return f"Root cause analysis unavailable: {str(e)}"
    
    def analyze(self, custom_thresholds: Dict = None) -> Dict[str, Any]:
        """
        Run complete anomaly detection.
        
        Returns:
            Dict with anomalies at product, seller, category, and time levels
        """
        
        try:
            # Load enriched dataset
            if not os.path.exists(self.enriched_dataset_path):
                return {"status": "error", "error": "Enriched dataset not found"}
            
            df = pd.read_csv(self.enriched_dataset_path)
            
            if self.verbose:
                print(f"📂 Loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Detect schema from enriched dataset
            schema = self._detect_schema(df)
            
            if self.verbose:
                print(f"\n🔍 Detected columns:")
                print(f"   Product: {schema.get('product_col')}")
                print(f"   Seller: {schema.get('seller_col')}")
                print(f"   Category: {schema.get('category_col')}")
                print(f"   Review: {schema.get('review_col')}")
            
            # Detect anomalies at all levels
            if self.verbose:
                print("\n🔎 Detecting product anomalies...")
            product_anomalies = self._detect_product_anomalies(df, schema)
            
            if self.verbose:
                print("🔎 Detecting seller anomalies...")
            seller_anomalies = self._detect_seller_anomalies(df, schema)
            
            if self.verbose:
                print("🔎 Detecting category anomalies...")
            category_anomalies = self._detect_category_anomalies(df, schema)
            
            if self.verbose:
                print("🔎 Detecting time anomalies...")
            time_anomalies = self._detect_time_anomalies(df, schema)
            
            # Calculate total loss from anomalies
            total_loss = sum(p['estimated_loss'] for p in product_anomalies[:10])
            total_loss += sum(s['estimated_loss'] for s in seller_anomalies[:10])
            total_loss += sum(c['estimated_loss'] for c in category_anomalies[:10])
            
            # Generate root cause insights
            anomalies_dict = {
                "products": product_anomalies,
                "sellers": seller_anomalies,
                "categories": category_anomalies,
                "time": time_anomalies
            }
            
            if self.verbose:
                print("\n🧠 Generating root cause insights...")
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
                "traceback": traceback.format_exc()
            }
    
    def _detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect schema from enriched dataset."""
        
        return {
            "product_col": "product_id" if "product_id" in df.columns else None,
            "seller_col": "seller_id" if "seller_id" in df.columns else None,
            "category_col": "product_category_name" if "product_category_name" in df.columns else None,
            "review_col": "review_score" if "review_score" in df.columns else None,
            "revenue_col": "payment_value" if "payment_value" in df.columns else "price" if "price" in df.columns else None,
            "cost_col": "freight_value" if "freight_value" in df.columns else None,
            "date_col": "order_purchase_timestamp" if "order_purchase_timestamp" in df.columns else None
        }
    
    def get_top_anomalies(self, n: int = 10) -> List[Dict]:
        """
        Get top N anomalies across all categories sorted by loss.
        """
        
        result = self.analyze()
        
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