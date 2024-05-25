import streamlit as st
import numpy as np
import pandas as pd
from collections import OrderedDict

def get_user_input_features():
    """
    사용자로부터 입력 값을 받아오는 함수

    :return: 입력 값 딕셔너리
    """
    AT = st.sidebar.slider('Ambient Temperature (AT)', -10.0, 50.0, 25.0)
    V = st.sidebar.slider('Exhaust Vacuum (V)', 25.0, 81.0, 60.0)
    AP = st.sidebar.slider('Ambient Pressure (AP)', 990.0, 1030.0, 1013.0)
    RH = st.sidebar.slider('Relative Humidity (RH)', 10.0, 100.0, 75.0)

    data = {
        'AT': AT,
        'V': V,
        'AP': AP,
        'RH': RH
    }

    features = pd.DataFrame(data, index=[0])
    return features

def get_raw_input_features():
    """
    사용자로부터 원본 입력 값을 받아오는 함수

    :return: 입력 값 딕셔너리
    """
    data = {
        'AT': st.sidebar.slider('Ambient Temperature (AT)', -10.0, 50.0, 25.0),
        'V': st.sidebar.slider('Exhaust Vacuum (V)', 25.0, 81.0, 60.0),
        'AP': st.sidebar.slider('Ambient Pressure (AP)', 990.0, 1030.0, 1013.0),
        'RH': st.sidebar.slider('Relative Humidity (RH)', 10.0, 100.0, 75.0)
    }

    return [data]

def draw_shap_plot(base_value, shap_values, data):
    """
    SHAP 값을 시각화하는 함수

    :param base_value: 기본 값
    :param shap_values: SHAP 값
    :param data: 입력 데이터
    """
    import matplotlib.pyplot as plt
    import shap

    explainer = shap.Explainer(shap_values, data)
    shap_values = explainer(data)

    fig, ax = plt.subplots()
    shap.force_plot(base_value, shap_values.values[0], data.iloc[0], matplotlib=True, ax=ax)
    st.pyplot(fig)

def streamlit_main():
    """
    streamlit main 함수

    :return: None
    """
    st.title('CCPP Power Output Predictor')

    # sidebar input 값 선택 UI 생성
    st.sidebar.header('User Menu')
    user_input_data = get_user_input_features()

    st.sidebar.header('Raw Input Features')
    raw_input_data = get_raw_input_features()

    submit = st.sidebar.button('Get predictions')
    if submit:
        # 임의의 예측 결과 및 SHAP 값을 생성
        results = {'prediction': 123.45}  # 임의의 예측 값
        shap_results = {'base_value': 0.5, 'shap_values': [0.1, -0.2, 0.05, 0.1]}  # 임의의 SHAP 값

        # 예측 결과 표시
        st.subheader('Results')
        prediction = results["prediction"]
        st.write("Prediction: ", round(prediction, 2))

        # expander 형식으로 model input 표시
        st.subheader('Input Features')
        features_selected = ['AT', 'V', 'AP', 'RH']

        model_input_expander = st.expander('Model Input')
        model_input_expander.write('Input Features: ')
        model_input_expander.text(", ".join(list(raw_input_data[0].keys())))
        model_input_expander.json(raw_input_data[0])
        model_input_expander.write('Selected Features: ')
        model_input_expander.text(", ".join(features_selected))
        selected_features_values = OrderedDict((k, raw_input_data[0][k]) for k in features_selected)
        model_input_expander.json(selected_features_values)

        # SHAP force plot 표시
        st.subheader('Interpretation Plot')
        draw_shap_plot(shap_results['base_value'], shap_results['shap_values'], pd.DataFrame(raw_input_data)[features_selected])

        # expander 형식으로 SHAP detail 값 표시
        shap_detail_expander = st.expander('Shap Detail')
        for key, item in zip(features_selected, shap_results['shap_values']):
            shap_detail_expander.text('%s: %s' % (key, item))

if __name__ == '__main__':
    streamlit_main()
