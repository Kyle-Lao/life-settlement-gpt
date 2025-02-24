import streamlit as st
import openai
import os
from retrieve_texts import extract_states_from_query, load_statutes_for_states, generate_gpt_response

# Get API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["general"]["OPENAI_API_KEY"]

client = openai.OpenAI(api_key=OPENAI_API_KEY)


st.title("📜 Life Settlements GPT")
st.write("Ask me anything about life settlement laws!")

# User input
question = st.text_input("Enter your legal question:")

if st.button("Get Answer"):
    if question:
        # Extract relevant states from query
        states = extract_states_from_query(question)

        if not states:
            st.warning("❌ Could not detect a valid state. Please specify a state in your question.")
        else:
            # Load statutes for detected states
            statute_texts = load_statutes_for_states(states)

            if not statute_texts:
                st.warning(f"❌ No statutes found for: {', '.join(states)}.")
            else:
                # Generate answer using your function
                response = generate_gpt_response(question, statute_texts)
                st.write(response)
    else:
        st.warning("Please enter a question before clicking the button.")
