import streamlit as st
import anthropic
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import tempfile
import os

st.set_page_config(
    page_title="CSV Report Generator",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Automated Data Report Generator")
st.caption("Upload a CSV file and get a professional PDF report with AI insights.")

@st.cache_resource
def get_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

def generate_charts(df, tmpdir):
    chart_paths = []

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) == 0:
        return chart_paths

    # Chart 1: Bar chart of first numeric column
    fig, ax = plt.subplots(figsize=(8, 4))
    col = numeric_cols[0]
    if len(df) > 20:
        df[col].head(20).plot(kind='bar', ax=ax, color='#4F81BD')
        ax.set_title(f'{col} (first 20 rows)')
    else:
        df[col].plot(kind='bar', ax=ax, color='#4F81BD')
        ax.set_title(f'{col}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    plt.tight_layout()
    path1 = os.path.join(tmpdir, 'chart1.png')
    plt.savefig(path1, dpi=100, bbox_inches='tight')
    plt.close()
    chart_paths.append(path1)

    # Chart 2: If 2+ numeric cols, scatter plot
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6, color='#C0504D')
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
        plt.tight_layout()
        path2 = os.path.join(tmpdir, 'chart2.png')
        plt.savefig(path2, dpi=100, bbox_inches='tight')
        plt.close()
        chart_paths.append(path2)

    return chart_paths

def get_ai_insights(df, filename):
    client = get_client()

    stats = df.describe().to_string()
    columns = ", ".join(df.columns.tolist())
    sample = df.head(5).to_string()
    shape = f"{df.shape[0]} rows, {df.shape[1]} columns"

    prompt = f"""You are a data analyst. Analyze this dataset and provide a professional report.

Dataset: {filename}
Shape: {shape}
Columns: {columns}

Sample data (first 5 rows):
{sample}

Statistical summary:
{stats}

Please provide:
1. EXECUTIVE SUMMARY (2-3 sentences about what this data represents)
2. KEY FINDINGS (3-5 bullet points of the most important insights)
3. DATA QUALITY NOTES (any issues, missing values, or anomalies)
4. RECOMMENDATIONS (2-3 actionable recommendations based on the data)

Keep it professional and concise. Use plain text, no markdown symbols."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def create_pdf(df, insights, chart_paths, filename):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    elements = []

    # Title style
    title_style = ParagraphStyle('CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#1F3864'),
        spaceAfter=6)

    heading_style = ParagraphStyle('CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#4F81BD'),
        spaceBefore=16,
        spaceAfter=6,
        borderPad=4)

    body_style = ParagraphStyle('CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=4)

    # Title
    elements.append(Paragraph(f"Data Analysis Report", title_style))
    elements.append(Paragraph(f"File: {filename}", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    # Dataset overview table
    elements.append(Paragraph("Dataset Overview", heading_style))
    overview_data = [
        ['Metric', 'Value'],
        ['Total Rows', str(df.shape[0])],
        ['Total Columns', str(df.shape[1])],
        ['Numeric Columns', str(len(df.select_dtypes(include='number').columns))],
        ['Missing Values', str(df.isnull().sum().sum())],
        ['Columns', ', '.join(df.columns.tolist()[:6]) + ('...' if len(df.columns) > 6 else '')]
    ]
    table = Table(overview_data, colWidths=[2.5*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3864')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#EBF3FB')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.2*inch))

    # AI Insights
    elements.append(Paragraph("AI Analysis & Insights", heading_style))
    for line in insights.split('\n'):
        if line.strip():
            elements.append(Paragraph(line.strip(), body_style))
    elements.append(Spacer(1, 0.2*inch))

    # Charts
    if chart_paths:
        elements.append(Paragraph("Visual Analysis", heading_style))
        for path in chart_paths:
            img = Image(path, width=6*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.1*inch))

    # Statistics table
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        elements.append(Paragraph("Statistical Summary", heading_style))
        stats = numeric_df.describe().round(2)
        stat_data = [['Stat'] + list(stats.columns[:5])]
        for idx, row in stats.iterrows():
            stat_data.append([idx] + [str(v) for v in row.values[:5]])

        col_width = 6.5 * inch / (len(stat_data[0]))
        stat_table = Table(stat_data, colWidths=[col_width] * len(stat_data[0]))
        stat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#EBF3FB')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(stat_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Main UI
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success(f"✅ Loaded: {uploaded_file.name} — {df.shape[0]} rows, {df.shape[1]} columns")

    with st.expander("Preview Data"):
        st.dataframe(df.head(10))

    if st.button("🚀 Generate PDF Report", type="primary"):
        with st.spinner("Analyzing data and generating report..."):

            # Get AI insights
            with st.status("Getting AI insights..."):
                insights = get_ai_insights(df, uploaded_file.name)
                st.write("✅ AI analysis complete")

            # Generate charts
            with st.status("Creating charts..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    chart_paths = generate_charts(df, tmpdir)
                    st.write(f"✅ {len(chart_paths)} charts created")

                    # Build PDF
                    with st.status("Building PDF..."):
                        pdf_buffer = create_pdf(df, insights, chart_paths, uploaded_file.name)
                        st.write("✅ PDF ready")

        st.success("🎉 Report generated!")

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"report_{uploaded_file.name.replace('.csv', '')}.pdf",
            mime="application/pdf"
        )

        with st.expander("Preview AI Insights"):
            st.write(insights)
else:
    st.info("👆 Upload a CSV file to get started.")
    st.markdown("""
    **What this tool does:**
    - Analyzes your CSV data automatically
    - Generates professional charts and visualizations
    - Provides AI-powered insights and recommendations
    - Exports everything as a polished PDF report
    """)