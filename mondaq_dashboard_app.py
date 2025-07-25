
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Mondaq Analytics Dashboard', layout='wide')
st.title("📊 Mondaq Analytics Dashboard")
st.markdown("Get insights into readership, author performance, and article trends.")

@st.cache_data
def load_data():
    reader_df = pd.read_csv("Reader-MondaqAnalytics.csv")
    article_df = pd.read_csv("Article-MondaqAnalytics.csv")
    author_df = pd.read_csv("Author-MondaqAnalytics.csv")

    reader_df.columns = reader_df.columns.str.strip()
    article_df.columns = article_df.columns.str.strip()
    author_df.columns = author_df.columns.str.strip()

    merged_df = article_df.merge(author_df, on='Author Id', how='left', suffixes=('_article', '_author'))
    merged_df = merged_df.rename(columns={'Reads_article': 'Article Reads', 'Author Name_article': 'Author Name'})
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])

    return reader_df, merged_df

reader_df, merged_df = load_data()

tab1, tab2, tab3 = st.tabs(["📈 Reader Insights", "📰 Article Insights", "✍️ Author Insights"])

with tab1:
    st.subheader("Top Countries by Readers")
    top_countries = reader_df['Country'].value_counts().nlargest(10)
    st.bar_chart(top_countries)

    st.subheader("Top Industries")
    top_industries = reader_df['Industry'].value_counts().dropna().nlargest(10)
    st.bar_chart(top_industries)

    st.subheader("Top Positions")
    top_positions = reader_df['Position'].value_counts().dropna().nlargest(10)
    st.bar_chart(top_positions)

with tab2:
    st.subheader("Top Articles by Reads")
    top_articles = merged_df.nlargest(10, 'Article Reads')[['Title', 'Author Name', 'Article Reads']]
    st.dataframe(top_articles)

    st.subheader("Reads Over Time")
    reads_over_time = merged_df.groupby(merged_df['Date'].dt.to_period('M'))['Article Reads'].sum().reset_index()
    reads_over_time['Date'] = reads_over_time['Date'].dt.to_timestamp()
    fig = px.line(reads_over_time, x='Date', y='Article Reads', markers=True)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Top Authors by Total Reads")
    top_authors = merged_df.groupby('Author Name')['Article Reads'].sum().nlargest(10).reset_index()
    fig2 = px.bar(top_authors, x='Article Reads', y='Author Name', orientation='h')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Video-Tagged Articles by Author")
    video_df = merged_df[merged_df['Mondaq Tags'] == 'Video']
    video_authors = video_df['Author Name'].value_counts().reset_index()
    video_authors.columns = ['Author Name', 'Video Count']
    fig3 = px.bar(video_authors, x='Video Count', y='Author Name', orientation='h')
    st.plotly_chart(fig3, use_container_width=True)
