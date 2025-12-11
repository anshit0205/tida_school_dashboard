# TIDA Sports Academy Dashboard

A comprehensive analytics dashboard for sports academy management, tracking revenue, student enrollments, renewals, and multi-sport memberships.

## Features
- 📊 Real-time revenue tracking and MRR analysis
- 👥 Student enrollment and retention analytics
- 🔄 Renewal management with risk assessment
- 🎯 Multi-sport vs single-sport comparison
- 🏫 School and academy performance metrics
- 📍 Location-based analytics
- 📱 Digital marketing attribution
- 📦 Package duration analysis

## Tech Stack
- **Frontend:** Streamlit
- **Database:** MySQL (Railway)
- **Visualization:** Plotly
- **Deployment:** Streamlit Cloud

---

## 📋 Prerequisites

- Python 3.8+
- MySQL Workbench (for data import)
- Railway account (free tier available)
- Streamlit Cloud account (free)
- Your academy data in CSV/Excel format

---

## 🚀 Setup Instructions

### Step 1: Set Up Railway MySQL Database

1. **Create Railway Account**
   - Go to [Railway.app](https://railway.app)
   - Sign up with GitHub (recommended)

2. **Create New Project**
   - Click "New Project"
   - Select "Provision MySQL"
   - Railway will automatically create a MySQL database

3. **Get Database Credentials**
   - Click on your MySQL service
   - Go to **Variables** tab
   - Note down these values:
     - `MYSQLHOST` (e.g., `caboose.proxy.rlwy.net`)
     - `MYSQLPORT` (e.g., `58323`)
     - `MYSQLUSER` (e.g., `root`)
     - `MYSQLPASSWORD` (long random string)
     - `MYSQLDATABASE` (e.g., `railway`)

4. **Enable Public Access**
   - Go to **Settings** → **Networking**
   - Ensure **TCP Proxy** is enabled (should be by default)
   - This allows external connections from Streamlit Cloud

---

### Step 2: Import Your Data Using MySQL Workbench

1. **Install MySQL Workbench**
   - Download from [MySQL Official Site](https://dev.mysql.com/downloads/workbench/)
   - Install and open

2. **Connect to Railway Database**
   - Click **+** icon next to "MySQL Connections"
   - Enter connection details from Railway:
     - **Connection Name:** Railway MySQL
     - **Hostname:** (your MYSQLHOST)
     - **Port:** (your MYSQLPORT)
     - **Username:** (your MYSQLUSER)
     - **Password:** Click "Store in Vault" → enter MYSQLPASSWORD
   - Click **Test Connection** → should succeed
   - Click **OK**

3. **Import Your Data**
   - Double-click your connection to open it
   - Right-click on your database (e.g., `railway`) → **Table Data Import Wizard**
   - Select your CSV/Excel file
   - Follow the wizard to create table (recommended name: `data_final_slim`)
   - Verify import by running: `SELECT * FROM data_final_slim LIMIT 10;`

4. **Create Streamlit User for Security** (Recommended)
   - Open a SQL query tab
   - Run these commands:
```sql
   CREATE USER 'streamlit_user'@'%' IDENTIFIED BY 'YourSecurePassword123!';
   GRANT ALL PRIVILEGES ON railway.* TO 'streamlit_user'@'%';
   FLUSH PRIVILEGES;
```
   - Note down the username (`streamlit_user`) and password for later

---

### Step 3: Set Up Local Development (Optional)

1. **Clone Repository**
```bash
   git clone <your-repo-url>
   cd tida-sports-dashboard
```

2. **Install Dependencies**
```bash
   pip install -r requirements.txt
```

3. **Create Secrets File**
   - Create `.streamlit/secrets.toml` in project root:
```toml
   [database]
   DB_HOST = "your-railway-host.proxy.rlwy.net"
   DB_PORT = 58323
   DB_USER = "streamlit_user"
   DB_PASS = "YourSecurePassword123!"
   DB_NAME = "railway"
   TABLE_NAME = "data_final_slim"
```

4. **Run Locally**
```bash
   streamlit run app.py
```

---

### Step 4: Deploy to Streamlit Cloud

1. **Push Code to GitHub**
```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **New app**
   - Select your repository, branch (`main`), and file (`app.py`)

3. **Configure Secrets**
   - Before deploying, click **Advanced settings**
   - In the **Secrets** section, paste:
```toml
   [database]
   DB_HOST = "your-railway-host.proxy.rlwy.net"
   DB_PORT = 58323
   DB_USER = "streamlit_user"
   DB_PASS = "YourSecurePassword123!"
   DB_NAME = "railway"
   TABLE_NAME = "data_final_slim"
```
   - Click **Deploy**

4. **Verify Connection**
   - Wait for deployment to complete
   - Check if "✅ Loaded X records from MySQL database" appears
   - If connection fails, verify Railway credentials match exactly

---

## 🔧 Troubleshooting

### Connection Issues

**Error: "Access denied for user"**
- Verify credentials in Streamlit Cloud secrets match Railway exactly
- Ensure you created the `streamlit_user` with proper permissions
- Check that TCP Proxy is enabled in Railway

**Error: "No data loaded"**
- Verify table name matches exactly (`TABLE_NAME` in secrets)
- Check Railway database has data: `SELECT COUNT(*) FROM data_final_slim;`
- Ensure Railway service is running (not paused)

**Slow Dashboard Loading**
- Railway free tier may have performance limits
- Consider upgrading Railway plan for production use
- Check data volume (large datasets may need optimization)

## 🔄 Data Updates

### How Updates Work
- Dashboard caches data for **5 minutes** for performance
- Click **"🔄 Force Refresh Now"** button to get latest data immediately
- Cache automatically expires every 5 minutes
- After cache expires, next page load fetches fresh data

### Updating Data in Railway
1. Connect to Railway MySQL via MySQL Workbench
2. Update/insert/delete records as needed
3. Changes appear in dashboard within 5 minutes (or click refresh button)

### For Real-Time Updates
If you need more frequent updates:
- Reduce cache time: Change `ttl=300` to `ttl=60` (1 minute) in code
- Enable auto-refresh: Install `streamlit-autorefresh` for automatic page refresh
- Display screens: Set up auto-refresh every 2-5 minutes

---

## 📊 Dashboard Features Guide

- **Filters (Sidebar):** Time period, sports, membership type, schools, locations
- **KPIs:** Revenue, MRR, customers, retention metrics
- **Sports Analysis:** Enrollment trends, revenue by sport
- **Renewal Pipeline:** Critical/high-risk customers needing follow-up
- **Tabs:** Revenue trends, location analysis, school analytics, academy performance
- **Export:** Download filtered data as CSV

---

## 🔐 Security Notes

- Never commit `secrets.toml` to GitHub (add to `.gitignore`)
- Use strong passwords for database users
- Railway free tier is suitable for development/small teams
- Consider Railway Pro for production with sensitive data
- Regularly update dependencies for security patches

---

## 📝 Data Requirements

Your CSV should include these columns:
- `wc_order_for_name` (student name)
- `sport` (sport name or multi-sport)
- `school`, `place`, `academy_name` (location info)
- `product_net_revenue` (revenue amount)
- `start_date`, `next_payment_date` (dates)
- `billing_period`, `billing_interval` (subscription info)
- `phone`, `email` (contact info)
- Optional: `coupon_amount`, `wc_order_attribution_utm_source`

---

## 🆘 Support

- **Railway Issues:** Check [Railway Docs](https://docs.railway.app)
- **Streamlit Issues:** Check [Streamlit Docs](https://docs.streamlit.io)
- **Dashboard Issues:** Open GitHub issue in this repository

---


**Thank You**