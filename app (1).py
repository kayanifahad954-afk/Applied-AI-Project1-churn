import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Churn Risk Advisor', page_icon='📡',
                   layout='wide')

RAW_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
            'TotalCharges']
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
PAYMENTS = ['Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)']

# Palette
RED, AMBER, GREEN = '#e4572e', '#f5a524', '#2e9e5b'
TEAL, BLUE, PURPLE = '#00838f', '#2f6fb0', '#7b4fc9'
BAND_COLOR = {'HIGH': RED, 'WATCH': AMBER, 'LOW': GREEN}
SENIOR_AGE = 65     # the model only knows SeniorCitizen (0/1), not exact age


# ---------------- Styling ----------------
CSS = """
<style>
.hero {
    background: linear-gradient(120deg, #00838f 0%, #2f6fb0 50%, #7b4fc9 100%);
    padding: 1.4rem 1.8rem; border-radius: 18px; color: white;
    margin-bottom: 1.2rem; box-shadow: 0 6px 18px rgba(47,111,176,.35);
}
.hero h1 { margin: 0; font-size: 2rem; color: white; }
.hero p  { margin: .3rem 0 0 0; opacity: .92; font-size: .95rem; }

.card {
    background: #ffffff; color: #1f2d3d; border-radius: 16px;
    padding: 1rem 1.2rem; border-left: 8px solid var(--c);
    box-shadow: 0 3px 12px rgba(0,0,0,.12); height: 100%;
}
.card .label { font-size: .8rem; text-transform: uppercase;
               letter-spacing: .06em; color: #5b6b7c; }
.card .value { font-size: 2rem; font-weight: 800; color: var(--c);
               line-height: 1.2; }
.card .sub   { font-size: .8rem; color: #5b6b7c; }

.gauge-wrap { margin: 1.1rem 0 .4rem 0; }
.gauge { position: relative; height: 18px; border-radius: 10px;
         background: linear-gradient(90deg, #2e9e5b, #f5a524 45%, #e4572e); }
.gauge .pin { position: absolute; top: -5px; width: 28px; height: 28px;
              margin-left: -14px; border-radius: 50%; background: white;
              border: 5px solid #1f2d3d; box-shadow: 0 2px 6px rgba(0,0,0,.4); }
.gauge .thr { position: absolute; top: -8px; width: 3px; height: 34px;
              background: #1f2d3d; opacity: .75; }
.gauge-legend { display: flex; justify-content: space-between;
                font-size: .75rem; opacity: .8; margin-top: .6rem; }

.section-title { font-size: 1.25rem; font-weight: 700; margin: 1.2rem 0 .4rem;
                 padding-left: .6rem; border-left: 6px solid var(--c); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(0,131,143,.16),
                                         rgba(123,79,201,.16));
}
[data-testid="stSidebar"] h2 { color: #00838f; }

button[data-baseweb="tab"] { font-weight: 700; font-size: 1rem; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2f6fb0; border-bottom: 3px solid #7b4fc9;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def card(label, value, sub, color):
    return (f'<div class="card" style="--c:{color}">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def section(title, color):
    st.markdown(f'<div class="section-title" style="--c:{color}">{title}</div>',
                unsafe_allow_html=True)


def _mix(c1, c2, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def heat_style(col):
    """Green -> amber -> red cell colours for probabilities (no matplotlib)."""
    green, amber, red = (46, 158, 91), (245, 165, 36), (228, 87, 46)
    styles = []
    for v in col:
        v = min(max(float(v), 0.0), 1.0)
        rgb = (_mix(green, amber, v / 0.5) if v < 0.5
               else _mix(amber, red, (v - 0.5) / 0.5))
        text = '#ffffff' if v >= 0.6 else '#1f2d3d'
        styles.append(f'background-color: rgb{rgb}; color: {text}; '
                      f'font-weight: 600')
    return styles


def age_group(age):
    if age < 30:
        return 'Young adult'
    if age < 50:
        return 'Adult'
    if age < SENIOR_AGE:
        return 'Mature'
    return 'Senior'


# ---------------- Model ----------------
@st.cache_resource            # load once per server, not on every click
def load_artifacts():
    model = joblib.load('churn_model.joblib')
    with open('model_meta.json') as f:
        meta = json.load(f)
    return model, meta


def prepare_input(raw, columns):
    X = pd.get_dummies(raw[RAW_COLS])            # no drop_first at serving
    return X.reindex(columns=columns, fill_value=0).astype(float)


model, meta = load_artifacts()
COLS = meta['feature_columns']


def score(raw):
    return model.predict_proba(prepare_input(raw, COLS))[:, 1]


# ---------------- Sidebar: one customer ----------------
with st.sidebar:
    st.header('🧑 Customer profile')

    # Age preference
    age = st.slider('Age (years)', 18, 90, 35)
    senior = age >= SENIOR_AGE
    st.caption(f'Age group: **{age_group(age)}**. The model only uses a '
               f'Senior citizen flag (age {SENIOR_AGE}+).')

    tenure = st.slider('Tenure (months)', 0, 72, 4)
    contract = st.selectbox('Contract',
                            ['Month-to-month', 'One year', 'Two year'])
    monthly = st.number_input('Monthly charges (USD)', 18.0, 120.0, 85.0,
                              step=0.5)
    internet = st.selectbox('Internet service', ['Fiber optic', 'DSL', 'No'])
    services = []
    if internet != 'No':
        services = st.multiselect('Add-on services', SERVICES,
                                  default=['StreamingTV'])
    phone = st.radio('Phone service', ['Yes', 'No'], horizontal=True)
    multi = 'No phone service'
    if phone == 'Yes':
        multi = st.radio('Multiple lines', ['No', 'Yes'], horizontal=True)
    payment = st.selectbox('Payment method', PAYMENTS)
    paperless = st.radio('Paperless billing', ['Yes', 'No'], horizontal=True)
    with st.expander('Demographics'):
        gender = st.radio('Gender', ['Female', 'Male'], horizontal=True)
        partner = st.checkbox('Has partner')
        dependents = st.checkbox('Has dependents')
    st.divider()
    threshold = st.slider('🎯 Contact threshold', 0.05, 0.95,
                          float(meta['threshold']), 0.05)

row = {'gender': gender, 'SeniorCitizen': int(senior),
       'Partner': 'Yes' if partner else 'No',
       'Dependents': 'Yes' if dependents else 'No', 'tenure': tenure,
       'PhoneService': phone, 'MultipleLines': multi,
       'InternetService': internet, 'Contract': contract,
       'PaperlessBilling': paperless, 'PaymentMethod': payment,
       'MonthlyCharges': monthly,
       'TotalCharges': tenure * monthly}           # derived, not typed
for s in SERVICES:
    row[s] = ('No internet service' if internet == 'No'
              else 'Yes' if s in services else 'No')
customer = pd.DataFrame([row])

# ---------------- Main area ----------------
st.markdown(
    f'<div class="hero"><h1>📡 Customer Churn Risk Advisor</h1>'
    f'<p>{meta["model_name"]} &nbsp;|&nbsp; '
    f'CV AUC {meta["cv_auc"]:.3f} &plusmn; {meta["cv_auc_std"]:.3f} '
    f'&nbsp;|&nbsp; Decision support only</p></div>',
    unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['👤 One customer', '📂 Batch scoring', 'ℹ️ About'])

with tab1:
    p = float(score(customer)[0])
    band = ('HIGH' if p >= threshold else
            'WATCH' if p >= threshold / 2 else 'LOW')
    color = BAND_COLOR[band]
    action = {'HIGH': 'Contact now', 'WATCH': 'Monitor',
              'LOW': 'No action'}[band]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card('Churn probability', f'{p:.0%}',
                     f'threshold {threshold:.0%}', color),
                unsafe_allow_html=True)
    c2.markdown(card('Risk band', band, 'based on your threshold', color),
                unsafe_allow_html=True)
    c3.markdown(card('Action', action, 'recommended next step', BLUE),
                unsafe_allow_html=True)
    c4.markdown(card('Age group', age_group(age),
                     f'{age} years' + (' (senior flag on)' if senior else ''),
                     PURPLE),
                unsafe_allow_html=True)

    st.markdown(
        f'<div class="gauge-wrap"><div class="gauge">'
        f'<div class="thr" style="left:{threshold * 100:.1f}%"></div>'
        f'<div class="pin" style="left:{p * 100:.1f}%"></div></div>'
        f'<div class="gauge-legend"><span>0% low risk</span>'
        f'<span>▍ contact threshold</span><span>100% high risk</span></div>'
        f'</div>', unsafe_allow_html=True)

    section('🔀 What would change the risk?', TEAL)
    changes = [('Contract', c) for c in ['One year', 'Two year']
               if c != contract]
    if payment == 'Electronic check':
        changes.append(('PaymentMethod', 'Credit card (automatic)'))
    if internet != 'No' and 'TechSupport' not in services:
        changes.append(('TechSupport', 'Yes'))
    rows = []
    for feature, value in changes:
        alt = customer.copy()
        alt[feature] = value
        q = float(score(alt)[0])
        rows.append({'Change': f'{feature}: {value}',
                     'New probability': round(q, 3),
                     'Difference': round(q - p, 3)})
    if rows:
        table = pd.DataFrame(rows)

        def color_diff(col):
            return [f'color: {GREEN}; font-weight: 700' if v < 0 else
                    f'color: {RED}; font-weight: 700' if v > 0 else ''
                    for v in col]

        st.dataframe(
            table.style.apply(color_diff, subset=['Difference'])
                       .format({'New probability': '{:.3f}',
                                'Difference': '{:+.3f}'}),
            hide_index=True, use_container_width=True)
    st.caption('Green = lower risk, red = higher risk. Associations learned '
               'from data, not guaranteed effects.')

    final = model[-1] if hasattr(model, 'steps') else model
    if hasattr(model, 'steps') and hasattr(final, 'coef_'):
        section('🔍 Why this score? (Logistic Regression)', PURPLE)
        with st.expander('Show the eight biggest contributions', expanded=True):
            z = model[:-1].transform(prepare_input(customer, COLS))[0]
            contrib = pd.Series(final.coef_[0] * z, index=COLS)
            top = contrib.reindex(
                contrib.abs().sort_values(ascending=False).index).head(8)
            chart = pd.DataFrame({
                'Pushes toward churn': top.clip(lower=0),
                'Pushes away from churn': top.clip(upper=0)})
            st.bar_chart(chart, color=[RED, GREEN])
            st.caption('Red bars push toward churn, green bars push away.')

with tab2:
    section('📂 Score a customer file', BLUE)
    st.write('Upload a CSV with the original Telco columns.')
    file = st.file_uploader('Customer file', type='csv')
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing:
            st.error(f'Missing columns: {missing}')
            st.stop()
        data['TotalCharges'] = pd.to_numeric(
            data['TotalCharges'], errors='coerce').fillna(0)
        data['p_churn'] = score(data).round(3)
        data['contact'] = np.where(data['p_churn'] >= threshold, 'Yes', '')

        # Age preference for the batch view
        age_pref = st.radio('Age preference', ['All customers',
                                               f'Seniors ({SENIOR_AGE}+) only',
                                               'Non-seniors only'],
                            horizontal=True)
        if age_pref.startswith('Seniors'):
            view = data[data['SeniorCitizen'] == 1]
        elif age_pref.startswith('Non-seniors'):
            view = data[data['SeniorCitizen'] == 0]
        else:
            view = data

        n_hi = int((view['contact'] == 'Yes').sum())
        avg = view['p_churn'].mean() if len(view) else 0.0
        b1, b2, b3 = st.columns(3)
        b1.markdown(card('Customers shown', f'{len(view)}',
                         f'of {len(data)} uploaded', BLUE),
                    unsafe_allow_html=True)
        b2.markdown(card('Above threshold', f'{n_hi}',
                         f'threshold {threshold:.0%}', RED),
                    unsafe_allow_html=True)
        b3.markdown(card('Average risk', f'{avg:.0%}',
                         'mean churn probability', AMBER),
                    unsafe_allow_html=True)

        st.write('')
        ranked = view.sort_values('p_churn', ascending=False).head(50)
        st.dataframe(
            ranked.style.apply(heat_style, subset=['p_churn']),
            use_container_width=True)
        st.download_button('⬇️ Download scored CSV', data.to_csv(index=False),
                           'scored_customers.csv', 'text/csv')

with tab3:
    section('ℹ️ Model details', PURPLE)
    st.json({k: v for k, v in meta.items() if k != 'feature_columns'})
    st.markdown('**Limitations:** trained on one US telecom dataset; '
                'not validated for other markets. See the model card.')
    st.markdown(f'**About age:** the training data has no exact age column, '
                f'only a Senior citizen flag. The age slider converts age '
                f'{SENIOR_AGE}+ to that flag, so ages below and above the '
                f'cutoff score the same.')
