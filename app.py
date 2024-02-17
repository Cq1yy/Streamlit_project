import streamlit as st
from PIL import Image
from eda import open_data, make_correlation, draw_distribution_graphs,target_addiction, target_quantity_column,plot_graph,get_numeric_features,get_categorical_features



def process_main_page():
    show_main_page()
    gistagrams()
    correlation()
    addiction()
    numeric_features()
    categorical_features()
    st.write("### Дополнительные исследования выполнены в файле .ipynb, число пропусков и дубликов равно нулю.")

def show_main_page():
    image = Image.open('data/money.jpeg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Bank",
        page_icon=image,

    )

    st.write(
        """
        # Реакция клиентов на рассылки услуг от банка
        Определяем, какие люди могут быть заинтересованы в услугах банка.
        """
    )

    st.image(image)
    st.write("## С чем мы будем работать ?")
    df = open_data()
    st.table(df.head(5))
    st.write(
        """
        ####  Где:
          - AGREEMENT_RK — уникальный идентификатор объекта в выборке;
          - TARGET — целевая переменная: отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было);
          - AGE — возраст клиента;
          - GEN_TITLE - профессия клиента;
          - SOCSTATUS_WORK_FL — социальный статус клиента относительно работы (1 — работает, 0 — не работает);
          - SOCSTATUS_PENS_FL — социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер);
          - GENDER — пол клиента (1 — мужчина, 0 — женщина);
          - CHILD_TOTAL — количество детей клиента;
          - DEPENDANTS — количество иждивенцев клиента;
          - PERSONAL_INCOME — личный доход клиента (в рублях);
          - LOAN_NUM_TOTAL — количество ссуд клиента;
          - LOAN_NUM_CLOSED — количество погашенных ссуд клиента.
        """
    )

def gistagrams():
    df = open_data()

    st.title("Взглянем на распределение некоторых числовых признаков")

    st.write("## График распределения зарплаты")
    fig = draw_distribution_graphs(df,"PERSONAL_INCOME",0, 80005, bins = 40)
    st.pyplot(fig)

    st.write("## График распределения возраста клиентов")
    fig = draw_distribution_graphs(df,"AGE", 20, 70, bins=45)
    st.pyplot(fig)
    st.write("### Исходя из нарисованных данных мы можем понять, что основной контингент клиентов находятся в диапазоне от 20 до 30 которые имеют зарплату примерно от 15 до 20 тыщ. рублей.")

def correlation():
    st.title("Какие выводы мы можем сделать из матрицы корреляция ?")

    st.write("## Матрица корреляций числовых признаков")

    df = open_data()
    fig = make_correlation(df)
    st.pyplot(fig)

    st.write("### Матрица корреляций дает нам понять, что есть категории, сильно коррелирующие друг с другом. ")

def addiction():
    st.title("В каком возрасте чаще всего берут кредиты ?")

    st.write("## График зависимости целевой переменной от возраста клиента")
    df = open_data()
    fig = target_addiction(df, "AGE", "TARGET")
    st.pyplot(fig)

    st.write("## График зависимости целевой переменной от дохода клиента")
    df_new = target_quantity_column(df, "PERSONAL_INCOME", "PERSONAL_INCOME_NEW")
    fig = plot_graph(df_new,"PERSONAL_INCOME_NEW",0, 100000)
    st.pyplot(fig)

    st.write("### Мы узнали, что чем моложе, и чем меньше получает человек, тем больше он склонен к оформлению кредита. ")

def numeric_features():
    st.title("Посмотрим на некоторые характеристики категориальных и числовых признаков")
    st.write("## Числовые характеристики числовых столбцов")

    df = open_data()
    features = get_numeric_features(df)
    st.table(features)

def categorical_features():
    st.write("## Числовая характеристика категориального столбца")
    df = open_data()
    col=['GEN_TITLE']
    features = get_categorical_features(df, col)
    st.table(features)


if __name__ == "__main__":
    process_main_page()
