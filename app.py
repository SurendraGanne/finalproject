import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -- Load assets --
clf = joblib.load('risk_classifier.joblib')
scaler = joblib.load('feature_scaler.joblib')
feature_names = joblib.load('feature_names.joblib')
<<<<<<< HEAD
reference_df = pd.read_csv("output/csv/combined_ehr_cost_dataset.csv")  # Your summary dataset
=======
reference_df = pd.read_csv("combined_ehr_cost_dataset.csv")  # Your summary dataset
>>>>>>> 171ce994f3dcff5543d94ce285cd6b553346dc19

st.set_page_config(page_title="Patient Risk Prediction App", layout="wide")
st.title('🏥 Patient Risk Prediction App')
st.markdown("> **Enter patient information to estimate healthcare risk level (Low / Medium / High).**")

# ---- Helper function to get data-driven ranges ----
def get_field_stats(df, field):
    return {
        "min": int(df[field].min()),
        "max": int(df[field].max()),
        "median": int(df[field].median()),
        "q1": int(df[field].quantile(0.25)),
        "q3": int(df[field].quantile(0.75))
    }

fields = [
    'Age', 'Income', 'Number_of_Encounters', 'Avg_Encounter_Cost',
    'Number_of_Procedure', 'Avg_Procedure_Cost', 'Number_of_Medication',
    'Avg_Medication_Cost', 'Number_of_Immunization', 'Avg_Immunization_Cost',
    'Number_of_Conditions'
]
field_stats = {field: get_field_stats(reference_df, field) for field in fields}

# ---- Sidebar with dynamic inputs ----
with st.sidebar:
    st.header("Patient Information")
    age = st.slider(
        'Age',
        min_value=field_stats['Age']["min"], max_value=field_stats['Age']["max"], value=field_stats['Age']["median"],
        help=f"Min: {field_stats['Age']['min']}, Median: {field_stats['Age']['median']}, Max: {field_stats['Age']['max']}"
    )
    income = st.number_input(
        'Income', min_value=field_stats['Income']["min"], max_value=field_stats['Income']["max"], value=field_stats['Income']["median"],
        help=f"Min: {field_stats['Income']['min']}, Median: {field_stats['Income']['median']}, Max: {field_stats['Income']['max']}"
    )
    number_of_encounters = st.slider(
        'Number of Encounters', min_value=field_stats['Number_of_Encounters']["min"], max_value=field_stats['Number_of_Encounters']["max"],
        value=field_stats['Number_of_Encounters']["median"],
        help=f"Min: {field_stats['Number_of_Encounters']['min']}, Median: {field_stats['Number_of_Encounters']['median']}, Max: {field_stats['Number_of_Encounters']['max']}"
    )
    avg_encounter_cost = st.slider(
        'Avg Encounter Cost', min_value=float(field_stats['Avg_Encounter_Cost']["min"]), max_value=float(field_stats['Avg_Encounter_Cost']["max"]),
        value=float(field_stats['Avg_Encounter_Cost']["median"]),
        help=f"Min: {field_stats['Avg_Encounter_Cost']['min']}, Median: {field_stats['Avg_Encounter_Cost']['median']}, Max: {field_stats['Avg_Encounter_Cost']['max']}"
    )
    number_of_procedure = st.slider(
        'Number of Procedures', min_value=field_stats['Number_of_Procedure']["min"], max_value=field_stats['Number_of_Procedure']["max"],
        value=field_stats['Number_of_Procedure']["median"],
        help=f"Min: {field_stats['Number_of_Procedure']['min']}, Median: {field_stats['Number_of_Procedure']['median']}, Max: {field_stats['Number_of_Procedure']['max']}"
    )
    avg_procedure_cost = st.slider(
        'Avg Procedure Cost', min_value=float(field_stats['Avg_Procedure_Cost']["min"]), max_value=float(field_stats['Avg_Procedure_Cost']["max"]),
        value=float(field_stats['Avg_Procedure_Cost']["median"]),
        help=f"Min: {field_stats['Avg_Procedure_Cost']['min']}, Median: {field_stats['Avg_Procedure_Cost']['median']}, Max: {field_stats['Avg_Procedure_Cost']['max']}"
    )
    number_of_medication = st.slider(
        'Number of Medications', min_value=field_stats['Number_of_Medication']["min"], max_value=field_stats['Number_of_Medication']["max"],
        value=field_stats['Number_of_Medication']["median"],
        help=f"Min: {field_stats['Number_of_Medication']['min']}, Median: {field_stats['Number_of_Medication']['median']}, Max: {field_stats['Number_of_Medication']['max']}"
    )
    avg_medication_cost = st.slider(
        'Avg Medication Cost', min_value=float(field_stats['Avg_Medication_Cost']["min"]), max_value=float(field_stats['Avg_Medication_Cost']["max"]),
        value=float(field_stats['Avg_Medication_Cost']["median"]),
        help=f"Min: {field_stats['Avg_Medication_Cost']['min']}, Median: {field_stats['Avg_Medication_Cost']['median']}, Max: {field_stats['Avg_Medication_Cost']['max']}"
    )
    number_of_immunization = st.slider(
        'Number of Immunizations', min_value=field_stats['Number_of_Immunization']["min"], max_value=field_stats['Number_of_Immunization']["max"],
        value=field_stats['Number_of_Immunization']["median"],
        help=f"Min: {field_stats['Number_of_Immunization']['min']}, Median: {field_stats['Number_of_Immunization']['median']}, Max: {field_stats['Number_of_Immunization']['max']}"
    )
    avg_immunization_cost = st.slider(
        'Avg Immunization Cost', min_value=float(field_stats['Avg_Immunization_Cost']["min"]), max_value=float(field_stats['Avg_Immunization_Cost']["max"]),
        value=float(field_stats['Avg_Immunization_Cost']["median"]),
        help=f"Min: {field_stats['Avg_Immunization_Cost']['min']}, Median: {field_stats['Avg_Immunization_Cost']['median']}, Max: {field_stats['Avg_Immunization_Cost']['max']}"
    )
    number_of_conditions = st.slider(
        'Number of Conditions', min_value=field_stats['Number_of_Conditions']["min"], max_value=field_stats['Number_of_Conditions']["max"],
        value=field_stats['Number_of_Conditions']["median"],
        help=f"Min: {field_stats['Number_of_Conditions']['min']}, Median: {field_stats['Number_of_Conditions']['median']}, Max: {field_stats['Number_of_Conditions']['max']}"
    )
    gender = st.selectbox('Gender', ['M', 'F'])
    race = st.selectbox('Race', ['white', 'black', 'asian', 'other'])
    region = st.selectbox('Region', ['Massachusetts'])  # Update as needed
    last_encounter_type = st.selectbox(
        'Last Encounter Type',
        ['wellness', 'ambulatory', 'emergency']
    )

# One-hot encoded input
feature_dict = {
    'Age': age,
    'Income': income,
    'Number_of_Encounters': number_of_encounters,
    'Avg_Encounter_Cost': avg_encounter_cost,
    'Number_of_Procedure': number_of_procedure,
    'Avg_Procedure_Cost': avg_procedure_cost,
    'Number_of_Medication': number_of_medication,
    'Avg_Medication_Cost': avg_medication_cost,
    'Number_of_Immunization': number_of_immunization,
    'Avg_Immunization_Cost': avg_immunization_cost,
    'Number_of_Conditions': number_of_conditions,
}
for g in ['M', 'F']:
    feature_dict[f'Gender_{g}'] = 1 if gender == g else 0
for r in ['white', 'black', 'asian', 'other']:
    feature_dict[f'Race_{r}'] = 1 if race == r else 0
for reg in ['Massachusetts']:
    feature_dict[f'Region_{reg}'] = 1 if region == reg else 0
for typ in ['wellness', 'ambulatory', 'emergency']:
    feature_dict[f'Last_Encounter_Type_{typ}'] = 1 if last_encounter_type == typ else 0

input_df = pd.DataFrame([feature_dict])

# Add any missing columns (from training) as 0, reorder columns
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# -- Scale numeric columns --
input_df[fields] = scaler.transform(input_df[fields])

# ---- PREDICTION AND INSIGHTS ----
if st.button('Predict Risk Level'):
    pred = clf.predict(input_df)
    probs = clf.predict_proba(input_df)[0]
    pred_risk = pred[0]

    # ----- 1. Summary Card -----
    st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:12px;margin-bottom:16px;">
            <h2 style="color:#0078ff;">Predicted Risk Level: <b>{pred_risk}</b></h2>
            <p>
                <b>High:</b> {probs[list(clf.classes_).index('High')]:.0%} |
                <b>Medium:</b> {probs[list(clf.classes_).index('Medium')]:.0%} |
                <b>Low:</b> {probs[list(clf.classes_).index('Low')]:.0%}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ----- 2. Personalized Recommendations -----
    if pred_risk == "High":
        st.warning("⚠️ **High Risk:** Please consult your healthcare provider for a personalized care plan. Consider a full check-up and review of chronic conditions.")
    elif pred_risk == "Medium":
        st.info("🟠 **Medium Risk:** Stay proactive with regular check-ups and discuss risk reduction with your doctor.")
    else:
        st.success("🟢 **Low Risk:** Maintain your current healthy lifestyle and schedule routine wellness visits.")

    # ----- 3. Population Comparison -----
    try:
        cond_percentile = (reference_df['Number_of_Conditions'] < number_of_conditions).mean()
        st.markdown(f"**Population Benchmark:** Your number of conditions is higher than **{cond_percentile:.0%}** of patients in our population.")
    except Exception:
        st.info("Population data unavailable for benchmarking.")

    # ----- 4. Dynamic Health Tips -----
    if number_of_medication > reference_df['Number_of_Medication'].quantile(0.8):
        st.info("💊 You have a higher number of medications than most patients. Consider a medication review for interactions or simplification.")
    if number_of_procedure > reference_df['Number_of_Procedure'].quantile(0.8):
        st.info("🦾 High procedure count! Review with your healthcare provider for optimization.")

    # ----- 5. Feature Importances -----
    importances = clf.feature_importances_
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True).tail(15)
    fig = px.bar(
        feature_importances, y='Feature', x='Importance', orientation='h',
        title="Top 15 Feature Importances",
        color='Importance', color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----- 6. Prediction probability breakdown -----
    prob_df = pd.DataFrame({
        'Risk Level': clf.classes_,
        'Probability': np.round(probs, 3)
    })
    fig2 = px.pie(
        prob_df, names='Risk Level', values='Probability', hole=0.4,
        title='Prediction Probability Breakdown',
        color='Risk Level', color_discrete_map={'High':'red','Medium':'orange','Low':'green'}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ----- 7. Input Data -----
    st.markdown("### Summary of Input Data")
    st.write("A summary of the patient’s input data along with helpful interpretations:")

    interpreted_data = []
    for field in fields:
        val = feature_dict[field]
        stats = field_stats[field]
        if val < stats['q1']:
            interpretation = "Below normal range"
        elif val > stats['q3']:
            interpretation = "Above normal range"
        else:
            interpretation = "Within typical range"

        interpreted_data.append({
            "Variable": field.replace("_", " "),
            "Entered Value": val,
            "Typical Range": f"{stats['q1']} - {stats['q3']}",
            "Interpretation": interpretation
        })

    interpreted_df = pd.DataFrame(interpreted_data)
    st.dataframe(interpreted_df)


    # ----- 8. Variable Ranges Table -----
    stats_df = pd.DataFrame(field_stats).T
    with st.expander(" Input variable ranges from data"):
        st.dataframe(stats_df)

    # ----- 9. (Optional) Download report -----
    csv = input_df.to_csv(index=False)
    st.download_button("Download Input Data as CSV", data=csv, file_name='input_data.csv', mime='text/csv')

st.markdown("---")
st.markdown(
    "| Powered by Streamlit and scikit-learn</sub>",
    unsafe_allow_html=True
)
