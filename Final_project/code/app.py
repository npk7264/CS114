import streamlit as st
import pandas as pd
import joblib
import re


# Import hàm từ phoBERT.ipynb
# import sys
# sys.path.insert(0, '../code')
from preprocessing_module import remove_stopwords,word_token,abbreviate,remove_duplicate_characters,remove_unnecessary_charactor,to_lower_case,convert_unicode, remove_html, feature_extraction


def comment_preprocess(comment):
    # Xoá HTML tag
    comment = remove_html(comment)

    # Chuẩn hoá dấu câu
    comment = convert_unicode(comment)

    # Đưa về dạng viết thường
    comment =  to_lower_case(comment)
    
    # Xoá các ký tự không cần thiết
    comment = remove_unnecessary_charactor(comment)

    # Chuẩn hoá những từ lặp âm tiết
    comment = remove_duplicate_characters(comment)
    
    # Chuẩn hoá viết tắt
    comment = abbreviate(comment)

    # Tách từ tiếng Việt
    comment = word_token(comment)

    # xử lý stopword
    comment = remove_stopwords(comment)

    return comment

# Title
st.header("Phân tích cảm xúc bình luận")

text_input = st.text_input(
    "Enter the URL to the comment on foody 👇",
    placeholder="URL"
)

if text_input:
    cmt = comment_preprocess(text_input)
    # st.text(cmt)
    feature = feature_extraction(cmt)
    svm = joblib.load("./Final_project/code/clf_svm.pkl")
    predict = svm.predict(feature)
    if predict == -1:
        st.text("😠 Tiêu cực")
    elif predict == 0: 
        st.text("😐 Trung lập")
    elif predict == 1:
        st.text("😋 Tích cực")




