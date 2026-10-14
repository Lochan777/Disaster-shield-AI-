import os
import re
import math
import sqlite3
import csv
import io
import time
import threading
from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response

try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    class DecisionTreeClassifier:
        def __init__(self, random_state=None):
            self.threshold = 11.0
        def fit(self, X, y):
            max_med = max(x[0] for x, label in zip(X, y) if label == 'Medium')
            min_high = min(x[0] for x, label in zip(X, y) if label == 'High')
            self.threshold = (max_med + min_high) / 2.0
            return self
        def predict(self, X):
            return ['High' if x[0] > self.threshold else 'Medium' for x in X]

app = Flask(__name__)

# --- 1. AI RISK MODEL ---
X_train = [[2, 1], [5, 1], [8, 2], [10, 2], [12, 3], [15, 3], [20, 3], [1, 1], [4, 2]]
y_train = ['Medium', 'Medium', 'Medium', 'Medium', 'High', 'High', 'High', 'Medium', 'Medium']
ml_model = DecisionTreeClassifier(random_state=42)
ml_model.fit(X_train, y_train)

def encode_urgency(urgency_str):
    if 'Medical' in urgency_str: return 2
    elif 'Water entering' in urgency_str: return 3
    return 1

# --- 2. KARNATAKA DISTRICT & HOBLI MICRO-ZONING ---
HOBLI_MAP = {
    'jayapura': ('Jayapura', 'Mysuru'), 'varuna': ('Varuna', 'Mysuru'), 'hunsur': ('Hunsur', 'Mysuru'),
    'madikeri': ('Madikeri', 'Kodagu'), 'virajpet': ('Virajpet', 'Kodagu'), 'somwarpet': ('Somwarpet', 'Kodagu'),
    'yelahanka': ('Yelahanka', 'Bengaluru Urban'), 'kengeri': ('Kengeri', 'Bengaluru Urban'),
    'mudigere': ('Mudigere', 'Chikkamagaluru'), 'sringeri': ('Sringeri', 'Chikkamagaluru'),
    'athani': ('Athani', 'Belagavi'), 'gokak': ('Gokak', 'Belagavi'), 'bhadravati': ('Bhadravati', 'Shivamogga')
}

DISTRICT_MAP = {
    'mysuru': 'Mysuru', 'mysore': 'Mysuru', 'saraswathipuram': 'Mysuru', 'vijayanagar': 'Mysuru',
    'bengaluru': 'Bengaluru Urban', 'bangalore': 'Bengaluru Urban', 'mg road': 'Bengaluru Urban',
    'chikkamagaluru': 'Chikkamagaluru', 'chikmagalur': 'Chikkamagaluru', 'davangere': 'Davangere',
    'kodagu': 'Kodagu', 'coorg': 'Kodagu', 'madikeri': 'Kodagu', 'mangaluru': 'Dakshina Kannada',
    'mangalore': 'Dakshina Kannada', 'udupi': 'Udupi', 'hassan': 'Hassan', 'shivamogga': 'Shivamogga',
    'shimoga': 'Shivamogga', 'mandya': 'Mandya', 'belagavi': 'Belagavi', 'belgaum': 'Belagavi',
    'hubballi': 'Dharwad', 'dharwad': 'Dharwad', 'kalaburagi': 'Kalaburagi', 'gulbarga': 'Kalaburagi'
}

DISTRICT_COLORS = {
    'Mysuru': '#2563eb', 'Bengaluru Urban': '#7c3aed', 'Chikkamagaluru': '#059669', 'Davangere': '#d97706',
    'Kodagu': '#dc2626', 'Dakshina Kannada': '#0284c7', 'Udupi': '#0d9488', 'Hassan': '#78716c',
    'Shivamogga': '#16a34a', 'Mandya': '#ca8a04', 'Belagavi': '#9333ea', 'Dharwad': '#c2410c', 'Kalaburagi': '#475569'
}

def detect_location_hierarchy(text):
    text_lower = text.lower()
    for h_key, (hobli, dist) in HOBLI_MAP.items():
        if h_key in text_lower: return hobli, dist
    for d_key, dist in DISTRICT_MAP.items():
        if d_key in text_lower: return 'General Area', dist
    return 'General Area', 'Mysuru'

def analyze_news_nlp(news_text):
    text_lower = news_text.lower()
    numbers = re.findall(r'\b(\d+)\b', news_text)
    people = max(int(n) for n in numbers if int(n) < 10000) if numbers else (25 if any(w in text_lower for w in ['hundreds', 'many', 'dozens', 'massive']) else 10)

    if any(w in text_lower for w in ['injured', 'hospital', 'medical', 'ambulance', 'critical', 'hurt']): urgency = 'Medical assistance reported'
    elif any(w in text_lower for w in ['submerged', 'stranded', 'entering homes', 'trapped', 'rooftop', 'inundated']): urgency = 'Water entering homes'
    else: urgency = 'Flooding affecting access'

    loc_match = re.search(r'\b(?:in|at|near|across|from)\s+([a-zA-Z\s,]+?)(?=\s+(?:where|with|after|as|leaving|affecting|water|people|homes|houses|residents|stranded|trapped|injured|submerged|\d+)|\.|$)', news_text, re.IGNORECASE)
    location = loc_match.group(1).strip(' ,.-').title() if loc_match and len(loc_match.group(1).strip(' ,.-')) >= 2 else "Unknown"
    if location == "Unknown":
        found = [p.title() for k, p in DISTRICT_MAP.items() if k in text_lower]
        if found: location = found[0]
    return location, people, urgency

def generate_ai_rescue_plan(inc):
    people = int(inc['people'])
    urgency = inc['urgency']
    severity = inc['severity']
    boats = max(1, math.ceil(people / 8))
    jackets = people + (boats * 2)
    medics = max(2, math.ceil(people / 5)) if 'Medical' in urgency or severity == 'High' else 1
    ambulances = max(1, math.ceil(people / 6)) if 'Medical' in urgency else (1 if severity == 'High' else 0)

    if 'Medical' in urgency:
        team, advice, eta = "Alpha Medical Trauma Unit", f"Priority medical extraction. Deploy {ambulances} ambulances and {medics} paramedics.", "8 mins"
    elif 'Water entering' in urgency or severity == 'High':
        team, advice, eta = "NDRF Heavy Flood Rescue Boat Squadron", f"Deploy {boats} motorized rescue boats with {jackets} life vests.", "10 mins"
    else:
        team, advice, eta = "Bravo Civil Defense Unit", f"Deploy drainage units and route safety guidance for {people} citizens.", "14 mins"
    return {'team': team, 'boats': boats, 'jackets': jackets, 'medics': medics, 'ambulances': ambulances, 'advice': advice, 'eta': eta}

def calculate_predictive_forecast(rainfall_mm, weather_alert, base_risk):
    multiplier = 1.0
    if rainfall_mm > 100 or 'Warning' in weather_alert or 'Alert' in weather_alert:
        multiplier = 1.35
        trend_label = "📈 High Escalation Risk (+35% 24h)"
        badge_class = "high"
    elif rainfall_mm > 70:
        multiplier = 1.20
        trend_label = "📊 Moderate Escalation (+20% 24h)"
        badge_class = "medium"
    else:
        multiplier = 1.05
        trend_label = "📉 Stable / Normal Trend"
        badge_class = "resolved"
    return round(base_risk * multiplier, 1), trend_label, badge_class

def simulate_lstm_river_forecast(current_level_m, inflow_rate, rainfall):
    predicted_6h_crest = round(current_level_m + (inflow_rate * 0.05) + (rainfall * 0.015), 2)
    delta = predicted_6h_crest - current_level_m
    if delta > 1.2 or predicted_6h_crest > 85.0:
        status, badge_style = "🔴 Danger: Rapid Surge / Breach Imminent", "background: #ef4444; color: white;"
    elif delta > 0.5:
        status, badge_style = "🟡 Warning: Water Level Rising", "background: #f59e0b; color: white;"
    else:
        status, badge_style = "🟢 Stable Basin Flow", "background: #10b981; color: white;"
    return predicted_6h_crest, status, badge_style

def init_db():
    conn = sqlite3.connect('disaster_shield.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT, location TEXT NOT NULL, hobli TEXT NOT NULL DEFAULT 'General Area',
        district TEXT NOT NULL DEFAULT 'Mysuru', people INTEGER NOT NULL, urgency TEXT NOT NULL, details TEXT NOT NULL,
        severity TEXT DEFAULT 'Medium', risk_score REAL DEFAULT 0.0, status TEXT DEFAULT 'Pending', source TEXT DEFAULT 'Command Center'
    )''')
    cursor.execute("PRAGMA table_info(incidents)")
    cols = [col[1] for col in cursor.fetchall()]
    if 'district' not in cols: cursor.execute("ALTER TABLE incidents ADD COLUMN district TEXT DEFAULT 'Mysuru'")
    if 'source' not in cols: cursor.execute("ALTER TABLE incidents ADD COLUMN source TEXT DEFAULT 'Command Center'")
    if 'hobli' not in cols: cursor.execute("ALTER TABLE incidents ADD COLUMN hobli TEXT DEFAULT 'General Area'")
    if 'dispatched_at' not in cols: cursor.execute("ALTER TABLE incidents ADD COLUMN dispatched_at REAL DEFAULT 0.0")
    if 'created_at' not in cols: cursor.execute("ALTER TABLE incidents ADD COLUMN created_at REAL DEFAULT 0.0")

    cursor.execute('''CREATE TABLE IF NOT EXISTS districts (
        district_name TEXT PRIMARY KEY, boats INTEGER NOT NULL, life_jackets INTEGER NOT NULL,
        medics INTEGER NOT NULL, ambulances INTEGER NOT NULL, rainfall_mm INTEGER NOT NULL, weather_alert TEXT NOT NULL
    )''')
    if cursor.execute('SELECT COUNT(*) FROM districts').fetchone()[0] == 0:
        karnataka_districts = [
            ('Mysuru', 15, 60, 12, 6, 85, 'Heavy Rain Flood Watch'), ('Bengaluru Urban', 25, 100, 20, 12, 110, 'Urban Waterlogging Warning'),
            ('Chikkamagaluru', 12, 50, 10, 5, 130, 'Landslide & Flood Red Alert'), ('Davangere', 10, 40, 8, 4, 45, 'Normal Monitoring'),
            ('Kodagu', 18, 70, 14, 7, 145, 'Cloudburst & Flash Flood Warning'), ('Dakshina Kannada', 20, 80, 16, 8, 125, 'Coastal Storm Surge Alert'),
            ('Udupi', 15, 60, 12, 6, 95, 'High Tide Flood Watch'), ('Hassan', 10, 45, 9, 4, 60, 'Moderate Rain Advisory'),
            ('Shivamogga', 14, 55, 11, 5, 90, 'River Overflow Warning'), ('Mandya', 10, 40, 8, 4, 50, 'Normal Monitoring'),
            ('Belagavi', 16, 65, 13, 6, 88, 'Catchment Basin Alert'), ('Dharwad', 12, 50, 10, 5, 40, 'Normal Monitoring'), ('Kalaburagi', 10, 40, 8, 4, 30, 'Clear')
        ]
        cursor.executemany('INSERT OR IGNORE INTO districts VALUES (?, ?, ?, ?, ?, ?, ?)', karnataka_districts)

    cursor.execute('''CREATE TABLE IF NOT EXISTS river_gauges (
        basin_name TEXT PRIMARY KEY, district TEXT NOT NULL, current_level_m REAL NOT NULL, 
        inflow_cusecs INTEGER NOT NULL, danger_mark_m REAL NOT NULL
    )''')
    if cursor.execute('SELECT COUNT(*) FROM river_gauges').fetchone()[0] == 0:
        basins = [
            ('Cauvery Basin (Krishnarajasagara Dam)', 'Mysuru', 82.4, 35000, 84.0),
            ('Hemavathi Basin (Gorur Dam)', 'Hassan', 78.1, 18000, 79.5),
            ('Tunga Basin (Gajanur Dam)', 'Shivamogga', 568.5, 42000, 570.0),
            ('Kabini Basin (H.D. Kote)', 'Mysuru', 227.2, 28000, 228.3),
            ('Netravathi Basin', 'Dakshina Kannada', 24.5, 55000, 26.0)
        ]
        cursor.executemany('INSERT OR IGNORE INTO river_gauges VALUES (?, ?, ?, ?, ?)', basins)

    conn.commit(); conn.close()

# --- AI AUTO-DISPATCH THREAD ---
def ai_auto_deployer():
    while True:
        time.sleep(4)
        try:
            conn = sqlite3.connect('disaster_shield.db', timeout=10)
            conn.row_factory = sqlite3.Row
            pending = conn.execute("SELECT * FROM incidents WHERE status = 'Pending'").fetchall()
            now = time.time()
            for inc in pending:
                if inc['created_at'] > 0 and (now - inc['created_at']) >= 60:
                    plan = generate_ai_rescue_plan(inc)
                    dist = inc['district']
                    d_stock = conn.execute("SELECT * FROM districts WHERE district_name = ?", (dist,)).fetchone()
                    if d_stock and d_stock['boats'] >= plan['boats'] and d_stock['ambulances'] >= plan['ambulances']:
                        conn.execute("UPDATE districts SET boats = boats - ?, life_jackets = life_jackets - ?, medics = medics - ?, ambulances = ambulances - ? WHERE district_name = ?", (plan['boats'], plan['jackets'], plan['medics'], plan['ambulances'], dist))
                        conn.execute("UPDATE incidents SET status = ?, dispatched_at = ? WHERE id = ?", (f"Dispatched: {plan['team']} [AI Auto-Pilot]", now, inc['id']))
            conn.commit()
            conn.close()
        except Exception:
            pass

threading.Thread(target=ai_auto_deployer, daemon=True).start()

# --- COMMAND CENTER HTML ---
SENDER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DisasterShield AI - Command Center</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        :root { --primary: #0f766e; --bg: #f8fafc; --card: #ffffff; --text: #1e293b; --high: #ef4444; --medium: #f59e0b; --ai: #4f46e5; --disp: #0284c7; }
        body { font-family: 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
        .container { max-width: 1250px; margin: 0 auto; }
        .top-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; flex-wrap: wrap; gap: 10px; }
        h1 { font-size: 24px; margin: 0; }
        .nav-group { display: flex; gap: 8px; flex-wrap: wrap; }
        .nav-btn { background: #4f46e5; color: white; padding: 9px 14px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 12px; }
        .nav-sos { background: #ef4444; color: white; padding: 9px 14px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 12px; }
        .nav-rescue-sep { background: #0284c7; color: white; padding: 9px 14px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 12px; }
        .nav-export { background: #059669; color: white; padding: 9px 14px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 12px; }
        .banner { background: #e6fffa; border-left: 4px solid var(--primary); padding: 10px 14px; margin-bottom: 16px; font-size: 13px; border-radius: 4px; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px; }
        .stat-card { background: var(--card); padding: 14px; border-radius: 8px; border: 1px solid #e2e8f0; }
        .stat-card h3 { margin: 0 0 5px 0; font-size: 11px; color: #64748b; text-transform: uppercase; }
        .stat-card .num { font-size: 24px; font-weight: bold; }
        .panel { background: var(--card); padding: 18px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 18px; }
        .ai-panel { background: linear-gradient(to bottom right, #ffffff, #eef2ff); border: 1px solid #c7d2fe; padding: 18px; border-radius: 8px; margin-bottom: 18px; }
        #map { height: 460px; width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
        .main-grid { display: grid; grid-template-columns: 1.7fr 1.3fr; gap: 18px; }
        .incident-item { padding: 12px 0; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: flex-start; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; color: white; margin-top: 4px; }
        .high { background: var(--high); } .medium { background: var(--medium); } .resolved { background: #64748b; } .dispatched { background: var(--disp); }
        .district-badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: bold; color: white; margin-right: 5px; }
        .hobli-badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: bold; color: white; background-color: #475569; margin-right: 5px; }
        .rec-box { background: #f1f5f9; border-left: 3px solid var(--primary); padding: 6px 10px; margin-top: 6px; font-size: 11px; }
        .actions { display: flex; flex-direction: column; gap: 5px; }
        .btn-action { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; text-decoration: none; text-align: center; }
        .btn-res { background: #d1fae5; color: #065f46; } .btn-del { background: #fee2e2; color: #991b1b; }
        .form-group { margin-bottom: 10px; }
        label { display: block; margin-bottom: 4px; font-weight: 600; font-size: 12px; }
        input, select, textarea { width: 100%; padding: 7px; border: 1px solid #cbd5e1; border-radius: 4px; box-sizing: border-box; }
        textarea { height: 58px; }
        .btn { background: var(--primary); color: white; border: none; padding: 9px; width: 100%; border-radius: 4px; font-weight: bold; cursor: pointer; }
        .btn-ai { background: var(--ai); color: white; border: none; padding: 9px; width: 100%; border-radius: 4px; font-weight: bold; cursor: pointer; }
        .chip { display: inline-block; background: #e0e7ff; color: #3730a3; padding: 4px 8px; border-radius: 4px; font-size: 11px; margin: 2px; cursor: pointer; border: 1px solid #c7d2fe; }
        .district-table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 8px; }
        .district-table th, .district-table td { border: 1px solid #e2e8f0; padding: 5px 8px; text-align: left; }
        .district-table th { background: #f1f5f9; color: #475569; }
    </style>
</head>
<body>
    <div class="container">
        <div class="top-bar">
            <div>
                <h1>DisasterShield AI - Command Center</h1>
                <p style="color: #64748b; margin: 2px 0 0 0; font-size: 13px;">Karnataka State Zoom Locked & Tracking Active + Cinematic Auto-Zoom</p>
            </div>
            <div class="nav-group">
                <a href="/sos" target="_blank" class="nav-sos">Citizen SOS Portal &nearr;</a>
                <a href="/rescue" target="_blank" class="nav-rescue-sep">Rescue Console (Separate Link) &rarr;</a>
                <a href="/export_report" class="nav-export">Download CSV Report &darr;</a>
            </div>
        </div>

        <div class="banner"><strong>CINEMATIC AUTO-ZOOM ACTIVE:</strong> When a rescue vehicle is dispatched, the map automatically flies and zooms directly to the active route.</div>

        <div class="stats">
            <div class="stat-card"><h3>Total Incidents</h3><div class="num">{{ incidents|length }}</div></div>
            <div class="stat-card"><h3>High Priority</h3><div class="num">{{ incidents | selectattr('severity', 'equalto', 'High') | list | length }}</div></div>
            <div class="stat-card"><h3>Teams Dispatched</h3><div class="num">{{ incidents | rejectattr('status', 'equalto', 'Pending') | rejectattr('status', 'equalto', 'Resolved') | list | length }}</div></div>
            <div class="stat-card"><h3>Resolved</h3><div class="num">{{ incidents | selectattr('status', 'equalto', 'Resolved') | list | length }}</div></div>
        </div>

        <div class="panel" style="border-left: 4px solid var(--ai);">
            <h2 style="margin-top:0; font-size:17px; color: #312e81;">🌊 LSTM Time-Series River &amp; Dam Telemetry (6h Crest Forecasts)</h2>
            <div style="overflow-x: auto;">
                <table class="district-table">
                    <tr><th>River Basin / Dam</th><th>District</th><th>Current Level (m)</th><th>Inflow (cusecs)</th><th>Danger Mark (m)</th><th>LSTM 6h Predicted Crest</th><th>Basin Trend Status</th></tr>
                    {% for r in river_gauges %}
                    <tr>
                        <td><strong>{{ r.basin_name }}</strong></td>
                        <td>{{ r.district }}</td>
                        <td>{{ r.current_level_m }} m</td>
                        <td>{{ r.inflow_cusecs }} cusecs</td>
                        <td>{{ r.danger_mark_m }} m</td>
                        <td><strong style="color: #0284c7;">{{ r.pred_crest }} m</strong></td>
                        <td><span style="padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 10px; {{ r.badge_style }}">{{ r.status_label }}</span></td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>

        <div class="panel">
            <h2 style="margin-top:0; font-size:17px;">Live Karnataka District Base Stations &amp; Weather Alert Table</h2>
            <div style="overflow-x: auto;">
                <table class="district-table">
                    <tr><th>District Hub</th><th>District Color Code</th><th>Rainfall (mm/hr)</th><th>Weather Alert Status</th><th>Available Boats</th><th>Life Jackets</th><th>Medics</th><th>Ambulances</th></tr>
                    {% for d in districts %}
                    <tr>
                        <td><strong>{{ d.district_name }}</strong></td>
                        <td><span style="display:inline-block; width:14px; height:14px; background: {{ district_colors.get(d.district_name, '#4f46e5') }}; border-radius:50%; vertical-align:middle; margin-right:5px;"></span> <code style="font-size:11px;">{{ district_colors.get(d.district_name, '#4f46e5') }}</code></td>
                        <td>{{ d.rainfall_mm }} mm</td>
                        <td><span style="color: {% if 'Warning' in d.weather_alert or 'Alert' in d.weather_alert %}#ef4444{% else %}#059669{% endif %}; font-weight:bold;">{{ d.weather_alert }}</span></td>
                        <td>{{ d.boats }}</td><td>{{ d.life_jackets }}</td><td>{{ d.medics }}</td><td>{{ d.ambulances }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>

        <div class="panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h2 style="margin:0; font-size:16px;">🛰️ Tactical Karnataka State Map (Zoom Locked & Tracked)</h2>
                <span style="font-size: 11px; color: #64748b; background: #f1f5f9; padding: 4px 8px; border-radius: 4px;">Zero-Delay ESRI World Imagery Active</span>
            </div>
            <div id="map"></div>
        </div>

        <div class="main-grid">
            <div class="panel">
                <h2 style="margin-top:0; font-size:17px;">Incident Log & Predictive Risk Forecasts</h2>
                {% if incidents %}
                    {% for inc in incidents %}
                    <div class="incident-item">
                        <div style="width: 76%;">
                            <span class="district-badge" style="background: {{ district_colors.get(inc.district, '#4f46e5') }};">{{ inc.district }}</span>
                            <span class="hobli-badge">{{ inc.hobli }}</span>
                            <strong>{{ inc.location }}</strong>
                            <p style="margin: 3px 0; font-size: 12px; color: #64748b;">People: {{ inc.people }} | Urgency: {{ inc.urgency }}</p>
                            <p style="margin: 3px 0; font-size: 12px;">{{ inc.details }}</p>
                            <span class="badge {% if inc.severity == 'High' %}high{% else %}medium{% endif %}">AI: {{ inc.severity | upper }}</span>
                            <span class="badge {{ inc.pred_class }}">24h FORECAST: {{ inc.pred_score }} [{{ inc.pred_trend }}]</span>
                            <span class="badge {% if inc.status == 'Resolved' %}resolved{% elif 'Dispatched' in inc.status %}dispatched{% else %}medium{% endif %}">STATUS: {{ inc.status | upper }}</span>
                            <div class="rec-box">
                                {% if inc.status == 'Resolved' %} Mission Complete. (Hidden from map view).
                                {% elif 'Dispatched' in inc.status %} {{ inc.status }} (GPS En Route)
                                {% else %} Pending manual dispatch OR 60s AI Auto-Pilot Countdown.
                                {% endif %}
                            </div>
                        </div>
                        <div class="actions">
                            {% if inc.status != 'Resolved' %}<a href="/resolve/{{ inc.id }}" class="btn-action btn-res">Resolve</a>{% endif %}
                            <a href="/delete/{{ inc.id }}" class="btn-action btn-del">Delete</a>
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <p style="color: #64748b; font-size: 13px;">No active incidents reported.</p>
                {% endif %}
            </div>

            <div>
                <div class="ai-panel">
                    <h2 style="margin-top: 0; color: #312e81; font-size: 17px;">AI Breaking News Scanner (NLP)</h2>
                    <form id="newsForm">
                        <div class="form-group"><textarea id="news_text" placeholder="Paste breaking news headline..." required></textarea></div>
                        <div style="margin-bottom: 10px;">
                            <span class="chip" onclick="document.getElementById('news_text').value='TV9 Breaking: Flash floods in Jayapura, Mysuru leaving 25 people stranded in submerged houses.'">Mysuru Flood (25)</span>
                            <span class="chip" onclick="document.getElementById('news_text').value='Alert: Landslide near Madikeri, Kodagu where 30 residents are trapped and need urgent help.'">Kodagu Landslide (30)</span>
                        </div>
                        <button type="submit" class="btn-ai">Analyze News & Route</button>
                    </form>
                </div>

                <div class="panel">
                    <h2 style="margin-top: 0; font-size: 17px;">Manual Incident Report</h2>
                    <form id="incidentForm">
                        <div class="form-group"><label>Location / Area (Include Hobli if known)</label><input type="text" id="manual_loc" placeholder="e.g. Jayapura, Mysuru" required></div>
                        <div class="form-group"><label>People Affected</label><input type="number" id="manual_ppl" value="1" min="1" required></div>
                        <div class="form-group"><label>Urgency Level</label>
                            <select id="manual_urg">
                                <option value="Flooding affecting access">Flooding affecting access</option>
                                <option value="Medical assistance reported">Medical assistance reported</option>
                                <option value="Water entering homes">Water entering homes</option>
                            </select>
                        </div>
                        <div class="form-group"><label>Incident Details</label><textarea id="manual_det" placeholder="Describe current situation..."></textarea></div>
                        <button type="submit" class="btn">Submit Manual Flood Report</button>
                    </form>
                </div>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const districtColors = {{ district_colors | tojson }};
        const incidents = {{ incidents | tojson }};
        const serverTimeOnLoad = {{ server_time }};
        const localTimeOnLoad = Date.now() / 1000;

        const LOC_COORDS = {
            'mysuru': [12.3051, 76.6413], 'mysore': [12.3051, 76.6413], 'jayapura': [12.2000, 76.5500], 'varuna': [12.3300, 76.7500], 'hunsur': [12.3114, 76.2923],
            'bengaluru': [12.9716, 77.5946], 'bangalore': [12.9716, 77.5946], 'yelahanka': [13.1007, 77.5963], 'kengeri': [12.9143, 77.4848],
            'kodagu': [12.4244, 75.7382], 'madikeri': [12.4244, 75.7382], 'virajpet': [12.2000, 75.8000], 'somwarpet': [12.6000, 75.8600],
            'chikkamagaluru': [13.3153, 75.7754], 'chikmagalur': [13.3153, 75.7754], 'mudigere': [13.1400, 75.6300], 'sringeri': [13.4190, 75.2570],
            'davangere': [14.4644, 75.9218], 'shivamogga': [13.9299, 75.5681], 'shimoga': [13.9299, 75.5681], 'bhadravati': [13.8400, 75.7000],
            'dakshina kannada': [12.9141, 74.8560], 'mangaluru': [12.9141, 74.8560], 'mangalore': [12.9141, 74.8560],
            'udupi': [13.3409, 74.7421], 'hassan': [13.0068, 76.1004], 'mandya': [12.5229, 76.8951],
            'belagavi': [15.8497, 74.4977], 'belgaum': [15.8497, 74.4977], 'athani': [16.7200, 75.0600], 'gokak': [16.1600, 74.8300],
            'dharwad': [15.4589, 75.0078], 'hubballi': [15.3647, 75.1240], 'kalaburagi': [17.3297, 76.8343], 'gulbarga': [17.3297, 76.8343]
        };

        function getCoords(locStr, distStr) {
            let key = locStr.toLowerCase().trim();
            for (let k in LOC_COORDS) {
                if (key.includes(k)) return LOC_COORDS[k];
            }
            let dKey = distStr.toLowerCase().trim();
            if (LOC_COORDS[dKey]) return LOC_COORDS[dKey];
            return [12.3051, 76.6413];
        }

        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 19, attribution: 'Esri Satellite' });
        
        const karnatakaBounds = L.latLngBounds([11.4, 74.0], [18.6, 78.6]);
        const map = L.map('map', { 
            center: [14.5204, 75.7223], 
            zoom: 7, 
            minZoom: 7, 
            maxZoom: 14,
            maxBounds: karnatakaBounds,
            maxBoundsViscosity: 1.0,
            layers: [satelliteLayer] 
        });

        async function fetchRoadRoute(startLat, startLon, endLat, endLon) {
            try {
                let url = `https://router.project-osrm.org/route/v1/driving/${startLon},${startLat};${endLon},${endLat}?overview=full&geometries=geojson`;
                let res = await fetch(url);
                let json = await res.json();
                if (json.code === 'Ok' && json.routes && json.routes.length > 0) {
                    return json.routes[0].geometry.coordinates.map(c => [c[1], c[0]]);
                }
            } catch(e) {}
            return [[startLat, startLon], [endLat, endLon]];
        }

        incidents.forEach(async (inc, index) => {
            if (inc.status === 'Resolved') return;

            let coords = getCoords(inc.location, inc.district);
            let lat = coords[0], lon = coords[1];
            let dColor = districtColors[inc.district] || '#4f46e5';

            L.circle([lat, lon], { color: dColor, fillColor: dColor, fillOpacity: 0.15, radius: 600 }).addTo(map);

            if (inc.urgency.includes("Water") || inc.urgency.includes("Flooding") || inc.severity === 'High') {
                let radBase = inc.severity === 'High' ? 450 : 250;
                L.circle([lat, lon], { color: '#00f6ff', fillColor: '#00d4ff', fillOpacity: 0.55, weight: 1, radius: radBase }).addTo(map);
            }

            const districtPin = L.divIcon({ className: 'custom-pin', html: `<div style="background-color: ${dColor}; width:16px; height:16px; border-radius:50%; border:3px solid white; box-shadow:0 0 8px rgba(0,0,0,0.5);"></div>`, iconSize: [22, 22] });
            L.marker([lat, lon], { icon: districtPin }).addTo(map);

            if (inc.status === 'Pending') {
                let baseLat = lat - 0.25, baseLon = lon - 0.25;
                L.polyline([[baseLat, baseLon], [lat, lon]], { color: '#ef4444', weight: 4, dashArray: '4, 4', opacity: 0.9 }).addTo(map);
                
                let redIcon = L.divIcon({
                    className: 'red-countdown-icon',
                    html: `<div style="background:#ef4444; color:white; font-size:11px; font-weight:bold; padding:4px 8px; border-radius:12px; border:2px solid white; box-shadow:0 2px 6px rgba(0,0,0,0.6); white-space:nowrap;" id="cnt_${inc.id}">⏳ 60s Auto-Pilot</div>`,
                    iconSize: [110, 30], iconAnchor: [55, 15]
                });
                L.marker([lat, lon], { icon: redIcon, zIndexOffset: 2000 }).addTo(map);

                setInterval(() => {
                    let now = (Date.now() / 1000) - localTimeOnLoad + serverTimeOnLoad;
                    let created = inc.created_at || now;
                    let left = Math.max(0, Math.ceil(60 - (now - created)));
                    let el = document.getElementById(`cnt_${inc.id}`);
                    if (el) { el.innerHTML = `⏳ Auto-Pilot: <b>${left}s</b>`; }
                }, 1000);
            }

            if (inc.status.includes('Dispatched')) {
                let vEmoji = inc.status.includes('Medical') ? '🚑' : (inc.status.includes('Boat') ? '🚤' : '🚒');
                let routeColor = inc.status.includes('Medical') ? '#ef4444' : (inc.status.includes('Boat') ? '#0284c7' : '#f59e0b');

                const vIcon = L.divIcon({
                    className: 'vehicle-icon',
                    html: `<div style="background: #0f172a; color: white; width: 42px; height: 42px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 3px solid ${routeColor}; box-shadow: 0 4px 14px rgba(0,0,0,0.5); font-size: 20px;">${vEmoji}</div>`,
                    iconSize: [42, 42], iconAnchor: [21, 21]
                });
                
                let startLat = lat - 0.25, startLon = lon - 0.25;
                let roadPoints = await fetchRoadRoute(startLat, startLon, lat, lon);
                
                let vMarker = L.marker(roadPoints[0], { icon: vIcon, zIndexOffset: 1000 }).addTo(map);
                let routeLine = L.polyline(roadPoints, { color: routeColor, weight: 5, opacity: 0.85 }).addTo(map);

                if (index === 0 || inc.status.includes('Dispatched')) {
                    map.flyTo([lat, lon], 10, { duration: 1.5 });
                }

                let totalPoints = roadPoints.length;
                let animTimer = setInterval(() => {
                    let now = (Date.now() / 1000) - localTimeOnLoad + serverTimeOnLoad;
                    let elapsed = now - (inc.dispatched_at || serverTimeOnLoad);
                    let driveTime = 30;

                    if (elapsed < driveTime) {
                        let progress = elapsed / driveTime;
                        let idx = Math.min(Math.floor(progress * totalPoints), totalPoints - 1);
                        vMarker.setLatLng(roadPoints[idx]);
                    } else {
                        clearInterval(animTimer);
                        vMarker.setLatLng([lat, lon]);
                        routeLine.setStyle({color: '#64748b', opacity: 0.3});
                    }
                }, 100);
            }
        });

        document.getElementById("newsForm").addEventListener("submit", function(e) {
            e.preventDefault();
            fetch('/analyze_news', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ news_text: document.getElementById("news_text").value.trim() }) })
            .then(r => r.json()).then(d => { if (d.success) window.location.href = '/'; });
        });

        document.getElementById("incidentForm").addEventListener("submit", function(e) {
            e.preventDefault();
            fetch('/', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ location: document.getElementById("manual_loc").value.trim(), people: Number(document.getElementById("manual_ppl").value), urgency: document.getElementById("manual_urg").value, details: document.getElementById("manual_det").value.trim() })
            }).then(r => r.json()).then(d => { if (d.success) window.location.href = '/'; });
        });
    </script>
</body>
</html>"""

# --- RESCUE RECEIVER HTML ---
RECEIVER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DisasterShield AI - Rescue Team Receiver Console</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        :root { --primary: #0f766e; --accent: #4f46e5; --bg: #f1f5f9; --card: #ffffff; --text: #0f172a; --muted: #64748b; --high: #ef4444; --med: #f59e0b; --blue: #0284c7; --green: #10b981; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 22px; }
        .container { max-width: 1250px; margin: 0 auto; }
        .header-bar { background: linear-gradient(135deg, #0f172a, #1e293b); color: white; padding: 18px 22px; border-radius: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
        .live-dot { width: 12px; height: 12px; background: #10b981; border-radius: 50%; display: inline-block; margin-right: 10px; box-shadow: 0 0 8px #10b981; }
        h1 { font-size: 22px; margin: 0; display: flex; align-items: center; }
        .btn-auto { background: linear-gradient(135deg, #4f46e5, #0284c7); color: white; border: none; padding: 10px 16px; border-radius: 8px; font-weight: bold; font-size: 13px; cursor: pointer; }
        .btn-back { background: #0f766e; color: white; padding: 9px 14px; border-radius: 8px; text-decoration: none; font-size: 13px; font-weight: 600; }
        .toast { background: #ecfdf5; border-left: 5px solid var(--green); color: #065f46; padding: 12px 16px; border-radius: 8px; margin-bottom: 18px; font-size: 14px; font-weight: 600; }
        .toast-warn { background: #fff1f2; border-left: 5px solid #ef4444; color: #991b1b; padding: 12px 16px; border-radius: 8px; margin-bottom: 18px; font-size: 14px; font-weight: 600; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px; }
        .stat-card { background: var(--card); padding: 16px; border-radius: 10px; border: 1px solid #e2e8f0; border-top: 4px solid var(--primary); }
        .stat-card h3 { margin: 0 0 6px 0; font-size: 11px; color: var(--muted); text-transform: uppercase; }
        .stat-card .num { font-size: 26px; font-weight: 800; }
        .panel { background: var(--card); padding: 18px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 18px; }
        #map { height: 380px; width: 100%; border-radius: 8px; }
        .mission-card { background: var(--card); border: 1px solid #e2e8f0; border-left: 6px solid var(--med); border-radius: 10px; padding: 18px; margin-bottom: 14px; display: grid; grid-template-columns: 1.7fr 1.3fr; gap: 20px; }
        .mission-card.high-p { border-left-color: var(--high); } .mission-card.disp-p { border-left-color: var(--blue); } .mission-card.res-p { border-left-color: #94a3b8; opacity: 0.85; }
        .badge { display: inline-block; padding: 4px 9px; border-radius: 20px; font-size: 10px; font-weight: bold; color: white; margin-left: 5px; }
        .bg-high { background: var(--high); } .bg-med { background: var(--med); } .bg-disp { background: var(--blue); } .bg-res { background: #64748b; }
        .district-badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: white; margin-right: 6px; }
        .hobli-badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: white; background-color: #475569; margin-right: 6px; }
        .ai-box { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 8px; padding: 12px; margin-top: 10px; }
        .equip-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 8px; }
        .equip-pill { background: white; border: 1px solid #e0e7ff; border-radius: 6px; padding: 6px; text-align: center; font-size: 11px; }
        .equip-pill strong { display: block; font-size: 15px; color: var(--accent); }
        .ctrl-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between; }
        select { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid #cbd5e1; margin: 6px 0 10px 0; font-size: 12px; }
        .btn-send { background: var(--primary); color: white; border: none; padding: 10px; width: 100%; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 12px; }
        .btn-done { background: #d1fae5; color: #065f46; text-decoration: none; padding: 8px; border-radius: 6px; font-weight: bold; font-size: 12px; text-align: center; display: block; margin-top: 8px; }
        .btn-gps { background: #e0f2fe; color: #0369a1; text-decoration: none; padding: 8px; border-radius: 6px; font-weight: bold; font-size: 12px; text-align: center; display: block; margin-top: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-bar">
            <div>
                <h1><span class="live-dot"></span>DisasterShield AI - Rescue Team Receiver Console</h1>
                <p style="color: #94a3b8; margin: 4px 0 0 22px; font-size: 13px;">Separate Rescue Portal &bull; Equipment Allocation &bull; Karnataka State Zoom</p>
            </div>
            <div style="display: flex; gap: 10px; align-items: center;">
                <form action="/ai_auto_dispatch" method="POST" style="margin:0;">
                    <button type="submit" class="btn-auto">Manual Auto-Dispatch</button>
                </form>
                <a href="/" class="btn-back">&larr; Command Center (Separate Link)</a>
            </div>
        </div>

        {% if dispatched_msg %}<div class="toast">{{ dispatched_msg }}</div>{% endif %}
        {% if error_msg %}<div class="toast-warn">⚠️ {{ error_msg }}</div>{% endif %}

        <div class="stats">
            <div class="stat-card"><h3>Incoming Queries</h3><div class="num">{{ incidents|length }}</div></div>
            <div class="stat-card"><h3>Awaiting Dispatch</h3><div class="num" style="color:#f59e0b;">{{ incidents | selectattr('status', 'equalto', 'Pending') | list | length }}</div></div>
            <div class="stat-card"><h3>Teams Responding</h3><div class="num" style="color:#0284c7;">{{ incidents | rejectattr('status', 'equalto', 'Pending') | rejectattr('status', 'equalto', 'Resolved') | list | length }}</div></div>
            <div class="stat-card"><h3>Missions Completed</h3><div class="num" style="color:#10b981;">{{ incidents | selectattr('status', 'equalto', 'Resolved') | list | length }}</div></div>
        </div>

        <div class="panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h2 style="margin:0; font-size:16px;">🛰️ Live District Tactical Map (Karnataka State Locked)</h2>
            </div>
            <div id="map"></div>
        </div>

        <div class="panel">
            <h2 style="margin-top:0; font-size:17px;">Incoming District Emergency Queries &amp; Equipment Ledger</h2>
            {% if incidents %}
                {% for inc in incidents %}
                <div class="mission-card {% if inc.status == 'Resolved' %}res-p{% elif 'Dispatched' in inc.status %}disp-p{% elif inc.severity == 'High' %}high-p{% endif %}">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span class="district-badge" style="background: {{ district_colors.get(inc.district, '#4f46e5') }};">{{ inc.district }}</span>
                                <span class="hobli-badge">{{ inc.hobli }}</span>
                                <strong style="font-size:18px;">{{ inc.location }}</strong>
                            </div>
                            <div>
                                <span class="badge {% if inc.severity == 'High' %}bg-high{% else %}bg-med{% endif %}">AI: {{ inc.severity | upper }}</span>
                                <span class="badge {{ inc.pred_class }}">24h FORECAST: {{ inc.pred_score }}</span>
                                <span class="badge {% if inc.status == 'Resolved' %}bg-res{% elif 'Dispatched' in inc.status %}bg-disp{% else %}bg-med{% endif %}">{{ inc.status | upper }}</span>
                            </div>
                        </div>
                        <p style="font-size:13px; color:var(--muted); margin:6px 0;">
                            <strong>People:</strong> {{ inc.people }} &nbsp;|&nbsp; <strong>Urgency:</strong> {{ inc.urgency }} &nbsp;|&nbsp; <strong>ETA:</strong> {{ inc.ai_plan.eta }}
                        </p>
                        <p style="font-size:12px; background:#f8fafc; padding:8px; border-radius:6px; margin:6px 0;">{{ inc.details }}</p>

                        <div class="ai-box">
                            <div style="font-size:11px; font-weight:bold; color:#3730a3; text-transform:uppercase;">AI Resource Allocation &amp; Equipment Required</div>
                            <div style="font-size:12px; margin-top:4px;">{{ inc.ai_plan.advice }}</div>
                            <div class="equip-grid">
                                <div class="equip-pill"><strong>{{ inc.ai_plan.boats }}</strong>Boats Required</div>
                                <div class="equip-pill"><strong>{{ inc.ai_plan.medics }}</strong>Medics / Kits</div>
                                <div class="equip-pill"><strong>{{ inc.ai_plan.jackets }}</strong>Life Vests</div>
                                <div class="equip-pill"><strong>{{ inc.ai_plan.ambulances }}</strong>Ambulances</div>
                            </div>
                        </div>
                    </div>

                    <div class="ctrl-box">
                        <div>
                            <div style="font-size:12px; font-weight:bold; color:var(--primary);">DISTRICT BASE: {{ inc.district }}</div>
                            <div style="font-size:11px; color:var(--muted); margin:4px 0;">Recommended: <strong style="color:var(--text);">{{ inc.ai_plan.team }}</strong></div>
                            {% if inc.status != 'Resolved' %}
                            <form action="/dispatch/{{ inc.id }}" method="POST">
                                <select name="team_name">
                                    <option value="{{ inc.ai_plan.team }}">[AI Pick] {{ inc.ai_plan.team }}</option>
                                    <option value="NDRF Heavy Flood Rescue Boat Squadron">NDRF Heavy Flood Rescue Boat Squadron</option>
                                    <option value="Alpha Medical Trauma Unit">Alpha Medical Trauma Unit</option>
                                    <option value="Bravo Civil Defense Unit">Bravo Civil Defense Unit</option>
                                </select>
                                <button type="submit" class="btn-send">{% if 'Dispatched' in inc.status %}Update Dispatch &amp; Stock{% else %}Dispatch Instantly{% endif %}</button>
                            </form>
                            {% else %}
                            <div style="background:#ecfdf5; color:#065f46; padding:10px; border-radius:6px; font-size:12px; text-align:center; margin-top:8px;"><strong>Mission Completed</strong></div>
                            {% endif %}
                        </div>
                        <div>
                            {% if inc.status != 'Resolved' %}
                            <a href="/resolve/{{ inc.id }}?from=rescue" class="btn-done">Mark Mission Complete (Restock)</a>
                            {% endif %}
                            <a href="https://www.google.com/maps/search/?api=1&query={{ inc.location }}" target="_blank" class="btn-gps">Open GPS Route (Google Maps)</a>
                        </div>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <p style="color:var(--muted); font-size:13px;">Waiting for incoming queries from Command Center...</p>
            {% endif %}
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const districtColors = {{ district_colors | tojson }};
        const incidents = {{ incidents | tojson }};
        const serverTimeOnLoad = {{ server_time }};
        const localTimeOnLoad = Date.now() / 1000;

        const LOC_COORDS = {
            'mysuru': [12.3051, 76.6413], 'mysore': [12.3051, 76.6413], 'jayapura': [12.2000, 76.5500], 'varuna': [12.3300, 76.7500], 'hunsur': [12.3114, 76.2923],
            'bengaluru': [12.9716, 77.5946], 'bangalore': [12.9716, 77.5946], 'yelahanka': [13.1007, 77.5963], 'kengeri': [12.9143, 77.4848],
            'kodagu': [12.4244, 75.7382], 'madikeri': [12.4244, 75.7382], 'virajpet': [12.2000, 75.8000], 'somwarpet': [12.6000, 75.8600],
            'chikkamagaluru': [13.3153, 75.7754], 'chikmagalur': [13.3153, 75.7754], 'mudigere': [13.1400, 75.6300], 'sringeri': [13.4190, 75.2570],
            'davangere': [14.4644, 75.9218], 'shivamogga': [13.9299, 75.5681], 'shimoga': [13.9299, 75.5681], 'bhadravati': [13.8400, 75.7000],
            'dakshina kannada': [12.9141, 74.8560], 'mangaluru': [12.9141, 74.8560], 'mangalore': [12.9141, 74.8560],
            'udupi': [13.3409, 74.7421], 'hassan': [13.0068, 76.1004], 'mandya': [12.5229, 76.8951],
            'belagavi': [15.8497, 74.4977], 'belgaum': [15.8497, 74.4977], 'athani': [16.7200, 75.0600], 'gokak': [16.1600, 74.8300],
            'dharwad': [15.4589, 75.0078], 'hubballi': [15.3647, 75.1240], 'kalaburagi': [17.3297, 76.8343], 'gulbarga': [17.3297, 76.8343]
        };

        function getCoords(locStr, distStr) {
            let key = locStr.toLowerCase().trim();
            for (let k in LOC_COORDS) {
                if (key.includes(k)) return LOC_COORDS[k];
            }
            let dKey = distStr.toLowerCase().trim();
            if (LOC_COORDS[dKey]) return LOC_COORDS[dKey];
            return [12.3051, 76.6413];
        }

        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 19, attribution: 'Esri Satellite' });
        
        const karnatakaBounds = L.latLngBounds([11.4, 74.0], [18.6, 78.6]);
        const map = L.map('map', { 
            center: [14.5204, 75.7223], 
            zoom: 7, 
            minZoom: 7, 
            maxZoom: 14,
            maxBounds: karnatakaBounds,
            maxBoundsViscosity: 1.0,
            layers: [satelliteLayer] 
        });

        async function fetchRoadRoute(startLat, startLon, endLat, endLon) {
            try {
                let url = `https://router.project-osrm.org/route/v1/driving/${startLon},${startLat};${endLon},${endLat}?overview=full&geometries=geojson`;
                let res = await fetch(url);
                let json = await res.json();
                if (json.code === 'Ok' && json.routes && json.routes.length > 0) {
                    return json.routes[0].geometry.coordinates.map(c => [c[1], c[0]]);
                }
            } catch(e) {}
            return [[startLat, startLon], [endLat, endLon]];
        }

        incidents.forEach(async (inc, index) => {
            if (inc.status === 'Resolved') return;

            let coords = getCoords(inc.location, inc.district);
            let lat = coords[0], lon = coords[1];
            let dColor = districtColors[inc.district] || '#4f46e5';

            L.circle([lat, lon], { color: dColor, fillColor: dColor, fillOpacity: 0.15, radius: 600 }).addTo(map);

            if (inc.urgency.includes("Water") || inc.urgency.includes("Flooding") || inc.severity === 'High') {
                let radBase = inc.severity === 'High' ? 450 : 250;
                L.circle([lat, lon], { color: '#00f6ff', fillColor: '#00d4ff', fillOpacity: 0.55, weight: 1, radius: radBase }).addTo(map);
            }

            const districtPin = L.divIcon({ className: 'custom-pin', html: `<div style="background-color: ${dColor}; width:16px; height:16px; border-radius:50%; border:3px solid white; box-shadow:0 0 8px rgba(0,0,0,0.5);"></div>`, iconSize: [22, 22] });
            L.marker([lat, lon], { icon: districtPin }).addTo(map);

            if (inc.status === 'Pending') {
                let baseLat = lat - 0.25, baseLon = lon - 0.25;
                L.polyline([[baseLat, baseLon], [lat, lon]], { color: '#ef4444', weight: 4, dashArray: '4, 4', opacity: 0.9 }).addTo(map);
                
                let redIcon = L.divIcon({
                    className: 'red-countdown-icon',
                    html: `<div style="background:#ef4444; color:white; font-size:11px; font-weight:bold; padding:4px 8px; border-radius:12px; border:2px solid white; box-shadow:0 2px 6px rgba(0,0,0,0.6); white-space:nowrap;" id="resc_cnt_${inc.id}">⏳ 60s Auto-Pilot</div>`,
                    iconSize: [110, 30], iconAnchor: [55, 15]
                });
                L.marker([lat, lon], { icon: redIcon, zIndexOffset: 2000 }).addTo(map);

                setInterval(() => {
                    let now = (Date.now() / 1000) - localTimeOnLoad + serverTimeOnLoad;
                    let created = inc.created_at || now;
                    let left = Math.max(0, Math.ceil(60 - (now - created)));
                    let el = document.getElementById(`resc_cnt_${inc.id}`);
                    if (el) { el.innerHTML = `⏳ Auto-Pilot: <b>${left}s</b>`; }
                }, 1000);
            }

            if (inc.status.includes('Dispatched')) {
                let vEmoji = inc.status.includes('Medical') ? '🚑' : (inc.status.includes('Boat') ? '🚤' : '🚒');
                let routeColor = inc.status.includes('Medical') ? '#ef4444' : (inc.status.includes('Boat') ? '#0284c7' : '#f59e0b');

                const vIcon = L.divIcon({
                    className: 'vehicle-icon',
                    html: `<div style="background: #0f172a; color: white; width: 42px; height: 42px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 3px solid ${routeColor}; box-shadow: 0 4px 14px rgba(0,0,0,0.5); font-size: 20px;">${vEmoji}</div>`,
                    iconSize: [42, 42], iconAnchor: [21, 21]
                });
                
                let startLat = lat - 0.25, startLon = lon - 0.25;
                let roadPoints = await fetchRoadRoute(startLat, startLon, lat, lon);
                
                let vMarker = L.marker(roadPoints[0], { icon: vIcon, zIndexOffset: 1000 }).addTo(map);
                let routeLine = L.polyline(roadPoints, { color: routeColor, weight: 5, opacity: 0.85 }).addTo(map);

                if (index === 0 || inc.status.includes('Dispatched')) {
                    map.flyTo([lat, lon], 10, { duration: 1.5 });
                }

                let totalPoints = roadPoints.length;
                let animTimer = setInterval(() => {
                    let now = (Date.now() / 1000) - localTimeOnLoad + serverTimeOnLoad;
                    let elapsed = now - (inc.dispatched_at || serverTimeOnLoad);
                    let driveTime = 30;

                    if (elapsed < driveTime) {
                        let progress = elapsed / driveTime;
                        let idx = Math.min(Math.floor(progress * totalPoints), totalPoints - 1);
                        vMarker.setLatLng(roadPoints[idx]);
                    } else {
                        clearInterval(animTimer);
                        vMarker.setLatLng([lat, lon]);
                        routeLine.setStyle({color: '#64748b', opacity: 0.3});
                    }
                }, 100);
            }
        });

        document.getElementById("newsForm").addEventListener("submit", function(e) {
            e.preventDefault();
            fetch('/analyze_news', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ news_text: document.getElementById("news_text").value.trim() }) })
            .then(r => r.json()).then(d => { if (d.success) window.location.href = '/'; });
        });

        document.getElementById("incidentForm").addEventListener("submit", function(e) {
            e.preventDefault();
            fetch('/', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ location: document.getElementById("manual_loc").value.trim(), people: Number(document.getElementById("manual_ppl").value), urgency: document.getElementById("manual_urg").value, details: document.getElementById("manual_det").value.trim() })
            }).then(r => r.json()).then(d => { if (d.success) window.location.href = '/'; });
        });
    </script>
</body>
</html>"""

# --- SOS HTML ---
SOS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DisasterShield AI - Citizen SOS Portal</title>
    <style>
        body { font-family: sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin:0; }
        .sos-box { background: #1e293b; border: 2px solid #ef4444; border-radius: 12px; padding: 30px; width: 400px; text-align: center; }
        input, select { width: 100%; padding: 10px; margin-bottom: 15px; background: #0f172a; border: 1px solid #475569; color: white; border-radius: 6px; box-sizing: border-box; }
        .btn-sos { background: #ef4444; color: white; padding: 14px; width: 100%; border: none; border-radius: 8px; font-weight: bold; font-size: 16px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="sos-box">
        <h1 style="color:#ef4444; margin-top:0;">🚨 EMERGENCY SOS</h1>
        <form action="/sos" method="POST">
            <input type="hidden" id="location" name="location" value="Mysuru">
            <input type="text" name="details" placeholder="Your Name / Contact" required>
            <input type="number" name="people" value="2" required>
            <select name="urgency"><option value="Water entering homes">Water entering homes</option><option value="Medical assistance reported">Medical Emergency</option></select>
            <p id="gps" style="font-size:12px; color:#38bdf8;">📍 Acquiring GPS...</p>
            <button type="submit" class="btn-sos">Broadcast SOS to Command</button>
        </form>
    </div>
    <script>
        navigator.geolocation.getCurrentPosition(
            p => { document.getElementById("location").value = "GPS: " + p.coords.latitude.toFixed(4) + "," + p.coords.longitude.toFixed(4); document.getElementById("gps").innerHTML = "📍 GPS Locked"; document.getElementById("gps").style.color = "#10b981"; },
            e => { document.getElementById("gps").innerHTML = "⚠️ GPS denied. Defaulting to base."; }
        );
    </script>
</body>
</html>"""

def setup_environment():
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f: f.write(SENDER_HTML)
    with open('templates/rescue.html', 'w', encoding='utf-8') as f: f.write(RECEIVER_HTML)
    with open('templates/sos.html', 'w', encoding='utf-8') as f: f.write(SOS_HTML)
    init_db()

setup_environment()

def enrich_incidents_with_forecast(incidents_list):
    conn = sqlite3.connect('disaster_shield.db')
    conn.row_factory = sqlite3.Row
    for inc in incidents_list:
        dist_row = conn.execute("SELECT rainfall_mm, weather_alert FROM districts WHERE district_name = ?", (inc['district'],)).fetchone()
        rain = dist_row['rainfall_mm'] if dist_row else 50
        alert = dist_row['weather_alert'] if dist_row else 'Normal'
        
        p_score, p_trend, p_class = calculate_predictive_forecast(rain, alert, inc['risk_score'])
        inc['pred_score'] = p_score
        inc['pred_trend'] = p_trend
        inc['pred_class'] = p_class
    conn.close()
    return incidents_list

@app.route('/api/state')
def api_state():
    conn = sqlite3.connect('disaster_shield.db')
    rows = conn.execute('SELECT id, status, risk_score FROM incidents ORDER BY id DESC').fetchall()
    conn.close()
    return jsonify({'signature': "|".join([f"{r[0]}:{r[1]}:{r[2]}" for r in rows])})

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    conn = sqlite3.connect('disaster_shield.db'); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        loc = data.get('location', '').strip()
        ppl = int(data.get('people', 1))
        urg = data.get('urgency', 'Flooding affecting access')
        det = data.get('details', '').strip()

        hobli, dist = detect_location_hierarchy(loc + " " + det)
        rain = (cursor.execute("SELECT rainfall_mm FROM districts WHERE district_name = ?", (dist,)).fetchone() or [50])[0]
        u_code = encode_urgency(urg)
        sev = str(ml_model.predict([[ppl, u_code]])[0])
        score = (ppl * 1.5) + ((3.0 if u_code==3 else (2.0 if u_code==2 else 1.0)) * 10) + (25.0 if sev == 'High' else 0.0) + (15.0 if rain > 100 else 0.0)
        sev = 'High' if rain > 100 else sev

        cursor.execute('''INSERT INTO incidents (location, hobli, district, people, urgency, details, severity, risk_score, status, source, created_at) VALUES (?,?,?,?,?,?,?,?,?,'Command Center',?)''', (loc, hobli, dist, ppl, urg, det, sev, score, 'Pending', time.time()))
        conn.commit(); conn.close(); return jsonify({'success': True})

    incs = [dict(r) for r in cursor.execute('SELECT * FROM incidents ORDER BY risk_score DESC, id DESC').fetchall()]
    incs = enrich_incidents_with_forecast(incs)
    dsts = [dict(r) for r in cursor.execute('SELECT * FROM districts ORDER BY district_name ASC').fetchall()]
    
    gauges_raw = [dict(r) for r in cursor.execute('SELECT * FROM river_gauges').fetchall()]
    for g in gauges_raw:
        crest, status, style = simulate_lstm_river_forecast(g['current_level_m'], g['inflow_cusecs'], 85)
        g['pred_crest'] = crest
        g['status_label'] = status
        g['badge_style'] = style

    conn.close()
    return render_template('index.html', incidents=incs, districts=dsts, river_gauges=gauges_raw, district_colors=DISTRICT_COLORS, server_time=time.time())

@app.route('/rescue')
def rescue_dashboard():
    conn = sqlite3.connect('disaster_shield.db'); conn.row_factory = sqlite3.Row
    raw = [dict(r) for r in conn.execute('SELECT * FROM incidents ORDER BY risk_score DESC, id DESC').fetchall()]; conn.close()
    raw = enrich_incidents_with_forecast(raw)
    for inc in raw: inc['ai_plan'] = generate_ai_rescue_plan(inc)
    return render_template('rescue.html', incidents=raw, district_colors=DISTRICT_COLORS, server_time=time.time())

@app.route('/sos', methods=['GET', 'POST'])
def citizen_sos():
    if request.method == 'POST':
        loc, ppl, urg = request.form.get('location', 'Mysuru').strip(), int(request.form.get('people', 2)), request.form.get('urgency', 'Water entering homes')
        det = "[Citizen SOS Portal] " + request.form.get('details', '').strip()
        hobli, dist = detect_location_hierarchy(loc + " " + det)
        conn = sqlite3.connect('disaster_shield.db')
        conn.execute('''INSERT INTO incidents (location, hobli, district, people, urgency, details, severity, risk_score, status, source, created_at) VALUES (?,?,?,?,?,?,'High',?,?,'Citizen SOS',?)''', (loc, hobli, dist, ppl, urg, det, (ppl*1.5)+50.0, 'Pending', time.time()))
        conn.commit(); conn.close(); return redirect(url_for('dashboard'))
    return render_template('sos.html')

@app.route('/analyze_news', methods=['POST'])
def analyze_news():
    txt = (request.get_json(silent=True) or request.form).get('news_text', '').strip()
    if not txt: return jsonify({'success': False})
    loc, ppl, urg = analyze_news_nlp(txt)
    hobli, dist = detect_location_hierarchy(txt)
    if loc == "Unknown": loc = hobli if hobli != 'General Area' else dist
    u_code = encode_urgency(urg)
    sev = str(ml_model.predict([[ppl, u_code]])[0])
    score = (ppl * 1.5) + ((3.0 if u_code==3 else (2.0 if u_code==2 else 1.0)) * 10) + (25.0 if sev == 'High' else 0.0)
    conn = sqlite3.connect('disaster_shield.db')
    conn.execute('''INSERT INTO incidents (location, hobli, district, people, urgency, details, severity, risk_score, status, source, created_at) VALUES (?,?,?,?,?,?,?,?,?,'NLP News Scanner',?)''', (loc, hobli, dist, ppl, urg, f"[AI News Feed] {txt}", sev, score, 'Pending', time.time()))
    conn.commit(); conn.close()
    return jsonify({'success': True, 'summary': f"Location: {loc} | Priority: {sev.upper()}"})

@app.route('/dispatch/<int:id>', methods=['POST'])
def dispatch_team(id):
    team = request.form.get('team_name', 'AI Tactical Rescue Unit')
    conn = sqlite3.connect('disaster_shield.db'); conn.row_factory = sqlite3.Row
    inc = conn.execute("SELECT * FROM incidents WHERE id = ?", (id,)).fetchone()
    if inc:
        plan, dist = generate_ai_rescue_plan(inc), inc['district']
        conn.execute("UPDATE districts SET boats = boats - ?, life_jackets = life_jackets - ?, medics = medics - ?, ambulances = ambulances - ? WHERE district_name = ?", (plan['boats'], plan['jackets'], plan['medics'], plan['ambulances'], dist))
        conn.execute("UPDATE incidents SET status = ?, dispatched_at = ? WHERE id = ?", (f"Dispatched: {team}", time.time(), id))
        conn.commit()
    conn.close(); return redirect(url_for('rescue_dashboard'))

@app.route('/ai_auto_dispatch', methods=['POST'])
def ai_auto_dispatch():
    conn = sqlite3.connect('disaster_shield.db', timeout=10)
    conn.row_factory = sqlite3.Row
    pending = conn.execute("SELECT * FROM incidents WHERE status = 'Pending'").fetchall()
    now = time.time()
    for inc in pending:
        plan, dist = generate_ai_rescue_plan(inc), inc['district']
        d_stock = conn.execute("SELECT * FROM districts WHERE district_name = ?", (dist,)).fetchone()
        if d_stock and d_stock['boats'] >= plan['boats'] and d_stock['ambulances'] >= plan['ambulances']:
            conn.execute("UPDATE districts SET boats = boats - ?, life_jackets = life_jackets - ?, medics = medics - ?, ambulances = ambulances - ? WHERE district_name = ?", (plan['boats'], plan['jackets'], plan['medics'], plan['ambulances'], dist))
            conn.execute("UPDATE incidents SET status = ?, dispatched_at = ? WHERE id = ?", (f"Dispatched: {plan['team']}", now, inc['id']))
    conn.commit()
    conn.close()
    return redirect(url_for('rescue_dashboard'))

@app.route('/resolve/<int:id>')
def resolve_incident(id):
    conn = sqlite3.connect('disaster_shield.db'); conn.row_factory = sqlite3.Row
    inc_row = conn.execute("SELECT * FROM incidents WHERE id = ?", (id,)).fetchone()
    if inc_row and inc_row['status'] != 'Resolved':
        plan, dist = generate_ai_rescue_plan(inc_row), inc_row['district']
        conn.execute("UPDATE districts SET boats = boats + ?, life_jackets = life_jackets + ?, medics = medics + ?, ambulances = ambulances + ? WHERE district_name = ?", (plan['boats'], plan['jackets'], plan['medics'], plan['ambulances'], dist))
        conn.execute("UPDATE incidents SET status = 'Resolved' WHERE id = ?", (id,))
        conn.commit()
    conn.close()
    if request.args.get('from') == 'rescue': return redirect(url_for('rescue_dashboard'))
    return redirect(url_for('dashboard'))

@app.route('/delete/<int:id>')
def delete_incident(id):
    c = sqlite3.connect('disaster_shield.db'); c.execute('DELETE FROM incidents WHERE id = ?', (id,)); c.commit(); c.close(); return redirect(url_for('dashboard'))

@app.route('/export_report')
def export_report():
    c = sqlite3.connect('disaster_shield.db')
    data = c.execute('SELECT id, location, hobli, district, people, urgency, severity, risk_score, status, source FROM incidents').fetchall(); c.close()
    si = io.StringIO(); cw = csv.writer(si)
    cw.writerow(['ID', 'Location', 'Hobli/Taluk', 'District Hub', 'People', 'Urgency', 'Severity', 'Score', 'Status', 'Source'])
    for r in data: cw.writerow(r)
    out = make_response(si.getvalue()); out.headers["Content-Disposition"] = "attachment; filename=Report.csv"; out.headers["Content-type"] = "text/csv"
    return out

if __name__ == '__main__':
    def show_links():
        print("\n" + "="*70)
        print(" 🚀 DISASTERSHIELD AI (FULLY LOCKED & SYNCHRONIZED) READY:")
        print(" 👉 1. COMMAND CENTER (Sender)     : http://127.0.0.1:5000")
        print(" 👉 2. RESCUE CONSOLE (Receiver)   : http://127.0.0.1:5000/rescue")
        print(" 👉 3. CITIZEN SOS EMERGENCY PORTAL : http://127.0.0.1:5000/sos")
        print("="*70 + "\n")
    threading.Timer(1.2, show_links).start()
    app.run(debug=True)