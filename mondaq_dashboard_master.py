
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
import io
from pandas import ExcelWriter

# --- PAGE CONFIG ---
st.set_page_config(page_title='Mondaq Master Dashboard', layout='wide')

# --- LOAD DATA ---
@st.cache_data
def load_data():
    reader_df = pd.read_csv("Reader-MondaqAnalytics.csv")
    article_df = pd.read_csv("Article-MondaqAnalytics.csv")
    author_df = pd.read_csv("Author-MondaqAnalytics.csv")

    reader_df.columns = reader_df.columns.str.strip()
    article_df.columns = article_df.columns.str.strip()
    author_df.columns = author_df.columns.str.strip()

    reader_df['Last Access Date'] = pd.to_datetime(reader_df['Last Access Date'], errors='coerce')
    article_df['Date'] = pd.to_datetime(article_df['Date'], errors='coerce')

    merged_df = article_df.merge(author_df, on='Author Id', how='left', suffixes=('_article', '_author'))
    merged_df = merged_df.rename(columns={'Reads_article': 'Article Reads', 'Author Name_article': 'Author Name'})

    return reader_df, merged_df

reader_df, merged_df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("📌 Filter Data")
start_date, end_date = st.sidebar.date_input("Select Date Range", [merged_df['Date'].min(), merged_df['Date'].max()])
country_filter = st.sidebar.multiselect("Filter by Country", options=reader_df['Country'].unique(), default=reader_df['Country'].unique())
industry_filter = st.sidebar.multiselect("Filter by Industry", options=reader_df['Industry'].dropna().unique(), default=reader_df['Industry'].dropna().unique())

filtered_articles = merged_df[(merged_df['Date'] >= pd.to_datetime(start_date)) & 
                              (merged_df['Date'] <= pd.to_datetime(end_date))]
filtered_readers = reader_df[(reader_df['Country'].isin(country_filter)) & 
                             (reader_df['Industry'].isin(industry_filter))]

# --- KPIs ---
st.title("📊 Mondaq Master Analytics Dashboard")
st.markdown("Advanced insights, predictions, and report generation for reader and article engagement.")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Readers", len(filtered_readers))
col2.metric("Total Reads", int(filtered_articles['Article Reads'].sum()))
col3.metric("Top Author", filtered_articles.groupby('Author Name')['Article Reads'].sum().idxmax())
col4.metric("Top Article", filtered_articles.loc[filtered_articles['Article Reads'].idxmax(), 'Title'])

# --- TABS ---
tabs = st.tabs([
    "📈 Reader Insights", 
    "📰 Article Insights", 
    "✍️ Author Insights", 
    "🏢 Company Deep Dive", 
    "🔍 Search Tools", 
    "⏰ Engagement Timing", 
    "📤 Report Generator", 
    "📉 Predictive Insights"
])

# --- Reader Insights ---
with tabs[0]:
    st.subheader("Top Countries by Readers")
    st.bar_chart(filtered_readers['Country'].value_counts().nlargest(10))

    st.subheader("Top Industries")
    st.bar_chart(filtered_readers['Industry'].value_counts().nlargest(10))

    st.subheader("Top Positions")
    st.bar_chart(filtered_readers['Position'].value_counts().nlargest(10))

# --- Article Insights ---
with tabs[1]:
    st.subheader("Top Articles by Reads")
    top_articles = filtered_articles.nlargest(10, 'Article Reads')[['Title', 'Author Name', 'Article Reads']]
    st.dataframe(top_articles)

    st.subheader("Reads Over Time")
    reads_over_time = filtered_articles.groupby(filtered_articles['Date'].dt.to_period('M'))['Article Reads'].sum().reset_index()
    reads_over_time['Date'] = reads_over_time['Date'].dt.to_timestamp()
    fig = px.line(reads_over_time, x='Date', y='Article Reads', markers=True)
    st.plotly_chart(fig, use_container_width=True)

# --- Author Insights ---
with tabs[2]:
    st.subheader("Top Authors by Total Reads")
    top_authors = filtered_articles.groupby('Author Name')['Article Reads'].sum().nlargest(10).reset_index()
    fig2 = px.bar(top_authors, x='Article Reads', y='Author Name', orientation='h')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Author Table with Key Stats")
    author_summary = filtered_articles.groupby('Author Name').agg({
        'Article Id': 'count',
        'Article Reads': 'sum',
        'Historic Reads': 'sum',
        'Profile Views': 'sum'
    }).rename(columns={'Article Id': 'Articles'}).sort_values('Article Reads', ascending=False).reset_index()
    st.dataframe(author_summary)

    st.download_button("📥 Download Author Summary as CSV", data=author_summary.to_csv(index=False), file_name="author_summary.csv", mime="text/csv")

# --- Company Deep Dive ---
with tabs[3]:
    st.subheader("🔎 Company Deep Dive")
    selected_company = st.selectbox("Select a Company", sorted(reader_df['Company Name'].unique()))
    company_data = reader_df[reader_df['Company Name'] == selected_company]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Readers", len(company_data))
    col2.metric("Total Reads", int(company_data['Reads'].sum()))
    col3.metric("Avg Reads per Person", round(company_data['Reads'].mean(), 2))

    st.markdown("#### Top Positions in This Company")
    st.bar_chart(company_data['Position'].value_counts().nlargest(5))

    st.markdown("#### Readers Table")
    st.dataframe(company_data[['Full Name', 'Email', 'Position', 'Reads']])
    st.download_button("📥 Download Company Report as CSV", data=company_data.to_csv(index=False), file_name=f"{selected_company}_readers.csv", mime="text/csv")

# --- Search Tools ---
with tabs[4]:
    st.subheader("🔎 Search Articles by Title or Keyword")
    article_search = st.text_input("Enter article title or keyword").lower()
    if article_search:
        results = merged_df[merged_df['Title'].str.lower().str.contains(article_search)]
        st.write(f"Found {len(results)} article(s):")
        st.dataframe(results[['Title', 'Author Name', 'Article Reads', 'Date']])

    st.subheader("🔍 Search Readers by Name, Email, or Company")
    reader_search = st.text_input("Enter reader's name, email, or company").lower()
    if reader_search:
        results = reader_df[
            reader_df['Full Name'].str.lower().str.contains(reader_search) |
            reader_df['Email'].str.lower().str.contains(reader_search) |
            reader_df['Company Name'].str.lower().str.contains(reader_search)
        ]
        st.write(f"Found {len(results)} reader(s):")
        st.dataframe(results[['Full Name', 'Email', 'Company Name', 'Position', 'Reads']])

# --- Engagement Timing ---
with tabs[5]:
    st.subheader("⏰ Engagement Heatmap (Day of Week vs Hour of Day)")
    valid_times = reader_df.dropna(subset=['Last Access Date']).copy()
    valid_times['Day'] = valid_times['Last Access Date'].dt.day_name()
    valid_times['Hour'] = valid_times['Last Access Date'].dt.hour
    heatmap_data = valid_times.pivot_table(index='Day', columns='Hour', values='Reads', aggfunc='sum', fill_value=0)
    heatmap_data = heatmap_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='Blues', ax=ax)
    ax.set_title('Reader Engagement by Day and Hour')
    st.pyplot(fig)

# --- Report Generator ---
with tabs[6]:
    st.subheader("📤 Generate and Download Report")
    top_articles = filtered_articles.groupby('Title').agg({
        'Article Reads': 'sum',
        'Date': 'min',
        'Author Name': 'first'
    }).sort_values('Article Reads', ascending=False).reset_index().head(10)
    top_authors = filtered_articles.groupby('Author Name')['Article Reads'].sum().reset_index().sort_values('Article Reads', ascending=False).head(10)
    country_breakdown = filtered_readers['Country'].value_counts().reset_index()
    country_breakdown.columns = ['Country', 'Reader Count']
    report_data = {
        "Summary Metrics": {
            "Total Reads": [int(filtered_articles['Article Reads'].sum())],
            "Unique Articles": [filtered_articles['Article Id'].nunique()],
            "Unique Readers": [filtered_readers['User Id'].nunique()]
        },
        "Top Articles": top_articles,
        "Top Authors": top_authors,
        "Reader Country Breakdown": country_breakdown
    }
    excel_buffer = io.BytesIO()
    with ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        pd.DataFrame(report_data["Summary Metrics"]).to_excel(writer, sheet_name='Summary', index=False)
        report_data["Top Articles"].to_excel(writer, sheet_name='Top Articles', index=False)
        report_data["Top Authors"].to_excel(writer, sheet_name='Top Authors', index=False)
        report_data["Reader Country Breakdown"].to_excel(writer, sheet_name='Country Breakdown', index=False)
    st.download_button(
        label="📊 Download Excel Report",
        data=excel_buffer,
        file_name="Mondaq_Analytics_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Predictive Insights ---
with tabs[7]:
    st.subheader("📉 Forecast Total Article Reads")
    agg_df = merged_df.groupby(merged_df['Date'].dt.to_period('M'))['Article Reads'].sum().reset_index()
    agg_df['Date'] = agg_df['Date'].dt.to_timestamp()
    ts_df = agg_df.rename(columns={'Date': 'ds', 'Article Reads': 'y'})
    if len(ts_df) >= 6:
        m = Prophet()
        m.fit(ts_df)
        future = m.make_future_dataframe(periods=3, freq='M')
        forecast = m.predict(future)
        fig = px.line(forecast, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Predicted Reads'}, title="Predicted Total Article Reads")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data to generate a forecast.")
