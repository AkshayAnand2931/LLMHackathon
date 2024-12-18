import streamlit as st
from agentic import final_response

st.title("Pharmaceutical Assistant")
input_container = st.container()

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'something' not in st.session_state:
    st.session_state.something = ''

def display_messages_chatbot():

    for message in st.session_state.messages:

        if message["role"] == "user":
            input_container.markdown(f'<div style="background-color:#DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;">👤 <strong>You: </strong>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            input_container.markdown(f'<div style="background-color:#E4E6EB; padding: 10px; border-radius: 10px; margin-bottom: 10px;">🤖 <strong>Bot: </strong>{message["content"]}</div>', unsafe_allow_html=True)


def submit():
    st.session_state.something = st.session_state.user_input
    st.session_state.user_input = ""


if __name__ == "__main__":

    st.text_input("👤 You : ", key="user_input", on_change=submit)
    user_input = st.session_state.something

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = final_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": response})

        display_messages_chatbot()