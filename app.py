import os
import uuid
import json
from datetime import datetime, timedelta
from functools import wraps
from langchain_core.messages import ToolMessage, AIMessage
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from bson.json_util import dumps

from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────────────
# App & Config
# ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "1234321")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 50 MB total
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ──────────────────────────────────────────────
# MongoDB
# ──────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client     = MongoClient(MONGO_URI)
db         = client["langgraph_app"]

users_col    = db["users"]
analyses_col = db["analyses"]

users_col.create_index("email", unique=True)
analyses_col.create_index([("user_id", 1), ("created_at", DESCENDING)])

ALLOWED_EXTENSIONS = {"csv"}

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("signin"))
        return f(*args, **kwargs)
    return decorated


def serialize_doc(doc):
    """Convert MongoDB doc to JSON-safe dict."""
    if doc is None:
        return None
    doc = dict(doc)
    doc["_id"] = str(doc["_id"])
    if "user_id" in doc:
        doc["user_id"] = str(doc["user_id"])
    return doc

def serialize_doc(doc):
    """Convert MongoDB doc to JSON-safe dict."""
    if doc is None:
        return None
    doc = dict(doc)
    doc["_id"] = str(doc["_id"])
    if "user_id" in doc:
        doc["user_id"] = str(doc["user_id"])
    # Convert all datetime fields to ISO strings
    for key, value in doc.items():
        if isinstance(value, datetime):
            doc[key] = value.isoformat()
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, datetime):
                    value[k] = v.isoformat()
    return doc
def get_last_3_analyses(user_id):
    """Fetch the last 3 analysis documents for a user."""
    cursor = analyses_col.find(
        {"user_id": str(user_id)},
        sort=[("created_at", DESCENDING)],
        limit=3
    )
    return [serialize_doc(d) for d in cursor]


def get_last_5_analyses(user_id):
    """Fetch the last 5 analysis documents for a user."""
    cursor = analyses_col.find(
        {"user_id": str(user_id)},
        sort=[("created_at", DESCENDING)],
        limit=5
    )
    return [serialize_doc(d) for d in cursor]


def run_langgraph_pipeline(file_paths, api_key, prev_actions=None,
                            prev_bas=None, prev_anomalies=None, action_taken=None):
    """
    Invoke the LangGraph compiled app and return the final state.
    Accepts lists of previous actions, BA outputs, and anomaly outputs.
    """
    try:
        from utils.Graph import app as langgraph_app   # import here to keep Flask startup fast

        initial_state = {
            "api_key":               api_key,
            "file_paths":            file_paths,
            "unified_dataset_path":  None,
            "ba_output":             {},
            "anomaly_output":        {},
            "previous_actions":      prev_actions,
            "previous_bas":          prev_bas,
            "previous_anomalies":    prev_anomalies,
            "action_taken":          action_taken,
            "messages":              [],
        }
        final_state = langgraph_app.invoke(initial_state)

        # Extract manager / tool outputs from messages
        action_output     = None
        validation_output = None
        for msg in final_state.get("messages", []):
            if isinstance(msg, ToolMessage):
                try:
                    parsed = json.loads(msg.content)
                    if not isinstance(parsed, dict):
                        continue
                    status = parsed.get("status", "")
                    if status == "action_suggested":
                        action_output = parsed.get("action")   # the actual action dict
                    elif status == "validation_complete":
                        validation_output = parsed.get("validation_result")
                except Exception:
                    pass
        
        return {
        "success": True,
        "ba_output": final_state.get("ba_output", {}),
        "anomaly_output": final_state.get("anomaly_output", {}),
        "action_output": action_output,
        "validation_output": validation_output,
        "unified_dataset_path": final_state.get("unified_dataset_path"),
    }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ──────────────────────────────────────────────
# Error Handlers
# ──────────────────────────────────────────────

@app.errorhandler(413)
def request_too_large(e):
    flash("Upload too large. Total size of all files must be under 50 MB.", "danger")
    return redirect(url_for("dashboard"))


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Auth ──────────────────────────────────────

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        # ── Step 1: Account fields ──
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # ── Step 2: Business fields ──
        business_name         = request.form.get("business_name", "").strip()
        industry              = request.form.get("industry", "").strip()
        sales_channel         = request.form.get("sales_channel", "").strip()
        order_volume          = request.form.get("order_volume", "").strip()
        monthly_revenue_label = request.form.get("monthly_revenue_label", "").strip()

        # ── Step 3: Context fields ──
        challenges     = request.form.getlist("challenges")
        current_method = request.form.get("current_method", "").strip()
        focus_notes    = request.form.get("focus_notes", "").strip()

        # ── Validation ──
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
            return render_template("signup.html")

        if users_col.find_one({"email": email}):
            flash("An account with that email already exists.", "danger")
            return render_template("signup.html")

        # ── Build business context string for the BA agent ──
        business_context_parts = []

        if business_name:
            business_context_parts.append(f"Business: {business_name}")
        if industry:
            business_context_parts.append(f"Industry: {industry}")
        if sales_channel:
            business_context_parts.append(f"Sales Channel: {sales_channel}")
        if order_volume:
            business_context_parts.append(f"Monthly Order Volume: {order_volume}")
        if monthly_revenue_label:
            business_context_parts.append(f"Monthly Revenue: {monthly_revenue_label}")
        if challenges:
            challenge_labels = {
                "high_returns":   "High Return Rates",
                "freight_costs":  "Freight / Shipping Costs",
                "dead_inventory": "Dead / Slow Inventory",
                "ad_spend":       "Inefficient Ad Spend",
                "seller_quality": "Seller / Supplier Quality",
                "margin_erosion": "Margin Erosion",
                "pricing":        "Pricing Issues",
                "refunds":        "Refund / Chargeback Volume",
            }
            selected = [challenge_labels.get(c, c) for c in challenges]
            business_context_parts.append(f"Known Cost Challenges: {', '.join(selected)}")
        if current_method:
            business_context_parts.append(f"Current Analysis Method: {current_method}")
        if focus_notes:
            business_context_parts.append(f"Owner Notes: {focus_notes}")

        business_context = "\n".join(business_context_parts)

        # ── Insert user document ──
        user_doc = {
            "name":            name,
            "email":           email,
            "password_hash":   generate_password_hash(password),
            "created_at":      datetime.utcnow(),
            "total_runs":      0,

            "business_profile": {
                "business_name":   business_name,
                "industry":        industry,
                "sales_channel":   sales_channel,
                "order_volume":    order_volume,
                "monthly_revenue": monthly_revenue_label,
                "cost_challenges": challenges,
                "current_method":  current_method,
                "focus_notes":     focus_notes,
            },

            "business_context": business_context,
        }

        result = users_col.insert_one(user_doc)
        session.permanent = True
        session["user_id"]   = str(result.inserted_id)
        session["user_name"] = name
        flash("Account created! Welcome aboard.", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = users_col.find_one({"email": email})
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "danger")
            return render_template("signin.html")

        session.permanent = True
        session["user_id"]   = str(user["_id"])
        session["user_name"] = user["name"]
        return redirect(url_for("dashboard"))

    return render_template("signin.html")


@app.route("/signout")
def signout():
    session.clear()
    flash("You've been signed out.", "warning")
    return redirect(url_for("signin"))


# ── Dashboard ─────────────────────────────────
def normalize_list(val):
    if val is None:
        return []
    if isinstance(val, dict):
        return list(val.values())
    if isinstance(val, list):
        return val
    return [val]


@app.route("/dashboard")
@login_required
def dashboard():
    user = users_col.find_one({"_id": ObjectId(session["user_id"])})
    last_5_analyses = get_last_5_analyses(session["user_id"])
    latest = last_5_analyses[0] if last_5_analyses else None

    # Normalize all list fields in every analysis so templates can safely slice
    for analysis in last_5_analyses:
        # Only normalize anomalies list — leave ba_output sub-dicts alone
        ao = analysis.get("anomaly_output") or {}
        anoms = ao.get("anomalies")
        if isinstance(anoms, dict):
            # keep as dict — template accesses .products, .sellers, .time directly
            pass
        elif isinstance(anoms, list):
            pass  # already a list, fine

    latest = last_5_analyses[0] if last_5_analyses else None

    prev_action_json = ""
    if latest and latest.get("action_output"):
        prev_action_json = json.dumps(latest["action_output"], indent=2)

    return render_template(
        "dashboard.html",
        user=user,
        last_5_analyses=last_5_analyses,
        latest=latest,
        prev_action_json=prev_action_json,
    )

@app.route("/run-analysis", methods=["POST"])
@login_required
def run_analysis():
    api_key = os.environ.get("GROQ_API_KEY")

    files = request.files.getlist("csv_files")
    if not files or all(f.filename == "" for f in files):
        flash("Please upload at least one CSV file.", "danger")
        return redirect(url_for("dashboard"))

    # Save uploaded files
    run_id     = str(uuid.uuid4())
    run_folder = os.path.join(app.config["UPLOAD_FOLDER"], run_id)
    os.makedirs(run_folder, exist_ok=True)

    saved_paths = []
    file_names  = []
    for f in files:
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            dest  = os.path.join(run_folder, fname)
            f.save(dest)
            saved_paths.append(dest)
            file_names.append(fname)

    if not saved_paths:
        flash("No valid CSV files found.", "danger")
        return redirect(url_for("dashboard"))

    # Parse optional action_taken flag
    action_taken = None
    at_val = request.form.get("action_taken", "")
    if at_val == "true":
        action_taken = True
    elif at_val == "false":
        action_taken = False

    # Get last 3 analyses for context
    last_3_analyses = get_last_3_analyses(session["user_id"])

    prev_actions   = []
    prev_bas       = []
    prev_anomalies = []

    for analysis in last_3_analyses:
        if analysis.get("action_output"):
            prev_actions.append(analysis["action_output"])
        if analysis.get("ba_output"):
            prev_bas.append(analysis["ba_output"])
        if analysis.get("anomaly_output"):
            prev_anomalies.append(analysis["anomaly_output"])

    # ── Run the pipeline ──
    result = run_langgraph_pipeline(
        file_paths     = saved_paths,
        api_key        = api_key,
        prev_actions   = prev_actions   if prev_actions   else None,
        prev_bas       = prev_bas       if prev_bas       else None,
        prev_anomalies = prev_anomalies if prev_anomalies else None,
        action_taken   = action_taken,
    )

    # ── Persist result ──
    analysis_doc = {
        "run_id":               run_id,
        "user_id":              session["user_id"],
        "session_id":           session.sid if hasattr(session, "sid") else run_id,
        "created_at":           datetime.utcnow(),
        "file_names":           file_names,
        "file_paths":           saved_paths,

        "success":              result.get("success", False),
        "error":                result.get("error"),
        "ba_output":            result.get("ba_output", {}),
        "anomaly_output":       result.get("anomaly_output", {}),
        "action_output":        result.get("action_output"),
        "validation_output":    result.get("validation_output"),
        "unified_dataset_path": result.get("unified_dataset_path"),

        "context": {
            "previous_actions":   prev_actions,
            "previous_bas":       prev_bas,
            "previous_anomalies": prev_anomalies,
            "action_taken":       action_taken,
        },

        "api_key_prefix": api_key[:8] + "…" if api_key else None,
    }
    analyses_col.insert_one(analysis_doc)

    users_col.update_one(
        {"_id": ObjectId(session["user_id"])},
        {"$inc": {"total_runs": 1}, "$set": {"last_run_at": datetime.utcnow()}}
    )

    if result.get("success"):
        flash("Pipeline completed successfully.", "success")
    else:
        flash(f"Pipeline error: {result.get('error', 'unknown')}", "danger")

    return redirect(url_for("dashboard"))


# ── History ───────────────────────────────────

@app.route("/history")
@login_required
def history():
    cursor = analyses_col.find(
        {"user_id": session["user_id"]},
        sort=[("created_at", DESCENDING)],
        limit=50
    )
    analyses = [serialize_doc(d) for d in cursor]
    return render_template("history.html", analyses=analyses)


# ── API endpoints (JSON) ───────────────────────

@app.route("/api/history")
@login_required
def api_history():
    """Return last N analyses as JSON. Query param: n (default 5)."""
    n = min(int(request.args.get("n", 5)), 50)
    cursor = analyses_col.find(
        {"user_id": session["user_id"]},
        sort=[("created_at", DESCENDING)],
        limit=n
    )
    return jsonify([serialize_doc(d) for d in cursor])


@app.route("/api/run/<run_id>")
@login_required
def api_run_detail(run_id):
    """Return full detail of a single run."""
    doc = analyses_col.find_one({
        "run_id":  run_id,
        "user_id": session["user_id"]
    })
    if not doc:
        return jsonify({"error": "Not found"}), 404
    return jsonify(serialize_doc(doc))


@app.route("/api/me")
@login_required
def api_me():
    user = users_col.find_one({"_id": ObjectId(session["user_id"])})
    safe = {
        "name":        user["name"],
        "email":       user["email"],
        "total_runs":  user.get("total_runs", 0),
        "last_run_at": str(user.get("last_run_at", "")),
    }
    return jsonify(safe)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)