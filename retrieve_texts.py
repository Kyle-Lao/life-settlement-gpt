import os
import re
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# Get OpenAI API Key from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Pass API key to OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Ensure embeddings also use the API key
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# 🔹 State Abbreviation Mapping
STATE_ABBREVIATIONS = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR"
}

# 🔹 Ensure multi-word states are checked first
SPECIAL_CASES = {
    "west virginia": "WV", "new york": "NY", "new mexico": "NM",
    "new hampshire": "NH", "north carolina": "NC", "south carolina": "SC",
    "north dakota": "ND", "south dakota": "SD"
}

def extract_states_from_query(query):
    """Extracts state abbreviations from a given user query."""
    query_lower = query.lower()
    found_states = set()  # ✅ Use a set to avoid duplicates

    # ✅ Check **special cases first** (Prevents false matches)
    for full_name, abbreviation in SPECIAL_CASES.items():
        if re.search(rf"\b{full_name}\b", query_lower):
            found_states.add(abbreviation)

    # ✅ Check **full state names**
    for full_name, abbreviation in STATE_ABBREVIATIONS.items():
        if re.search(rf"\b{full_name}\b", query_lower):
            found_states.add(abbreviation)

    # ✅ Check for **state abbreviations (CA, TX, FL)** 
    for abbr in STATE_ABBREVIATIONS.values():
        if re.search(rf"\b{abbr}\b", query):  # ✅ Match uppercase directly
            found_states.add(abbr)  # ✅ Add original uppercase abbreviation

    return list(found_states)  # ✅ Return all detected states



# 🔹 Loads statute files for ALL detected states
def load_statutes_for_states(states):
    statute_folder = "texts"
    statute_texts = {}

    for state_abbr in states:
        filename_patterns = [
            f"{state_abbr} Life Settlements Statutes.txt",
            f"{state_abbr} Life Settlement Statutes.txt",
            f"{state_abbr} State Life Settlements Statutes.txt"
        ]

        # ✅ Check **ALL** possible filenames and load them
        for filename in filename_patterns:
            file_path = os.path.join(statute_folder, filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    statute_texts[state_abbr] = (file.read(), filename)
                    break  # ✅ Prevents unnecessary iterations for that state

    return statute_texts  # ✅ Should return multiple states if found



def find_relevant_statutes(query, statute_text):
    sections = statute_text.split("\n\n")  # Split by double newlines
    embeddings = OpenAIEmbeddings()  
    db = FAISS.from_texts(sections, embeddings)
    
    # ✅ Filter by Exact Keyword Match
    keyword_filtered = [s for s in sections if all(word in s.lower() for word in query.lower().split())]
    
    if keyword_filtered:
        relevant_sections = db.similarity_search(query, k=3)  # Retrieve top 3 matches
        return "\n\n".join([s.page_content for s in relevant_sections])
    
    return ""  # Return empty if nothing relevant is found



import re

def reinterpret_followup_query(user_query, last_query):
    """
    Reinterprets follow-up queries by integrating newly mentioned states into the correct position within the last query.
    Ensures sentence structure remains natural and avoids duplicate or misplaced insertions.
    """
    # Extract states from both queries
    user_states = extract_states_from_query(user_query)
    last_states = extract_states_from_query(last_query)

    # If the user already provided specific states, return the query as-is
    if user_states:
        return user_query  

    # If no new states are mentioned, use the last known states
    if last_states:
        # Remove filler words like "what about", "how about", etc.
        cleaned_query = re.sub(r"\b(what about|how about|and in|in)\b", "", user_query, flags=re.IGNORECASE).strip()

        # Identify the sentence structure of the last query
        query_without_states = re.sub(r"\b(" + "|".join(map(re.escape, last_states)) + r")\b", "{}", last_query, flags=re.IGNORECASE)

        # Replace the placeholders with the new states
        reinterpreted_query = query_without_states.format(*user_states)

        return reinterpreted_query.strip()

    return user_query  # Default to original query if no inference can be made



def generate_gpt_response(query, statute_texts):
    responses = []

    for state_abbr, (statute_text, statute_file) in statute_texts.items():
        relevant_statutes = find_relevant_statutes(query, statute_text)

        prompt = f"""
        You are an expert in life settlement laws. Answer the question using only the provided statute.

        **User Question:** {query}

        **Relevant Law from {statute_file} ({state_abbr}):**
        {relevant_statutes}

        **Instructions:**
        - Answer concisely, summarizing the relevant law in 1-2 sentences.
        - **DO NOT add extra commentary or disclaimers**.
        - **DO NOT make assumptions**—answer **only** using the statute text.
        - **Always include the statute section, chapter, and number** in this format:

          ✅ **Answer:**  
          [Concise answer, directly referencing the most important parts of the law]

          📌 **Relevant Statutes:**  
          "[Exact statute text]"  
          (**[Statute Chapter]**, **Statute Section**)

          📌 **Source:** {statute_file}
        """

        response = openai.ChatCompletion.create(
    api_key=OPENAI_API_KEY,  # Ensure API key is included
    model="gpt-4-turbo",
    messages=[{"role": "system", "content": prompt}],
    temperature=0
)


        responses.append(f"📌 **State: {state_abbr}**\n{response.choices[0].message.content.strip()}")

    return "\n\n".join(responses)









def main():
    last_query = None  # Store the last user query
    last_states = []  # Store the last states

    while True:
        user_query = input("\nEnter your life settlements legal question (or type 'exit' to quit): ").strip()

        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        # **Handle Follow-Up Queries**
        if user_query.lower().startswith("what about"):
            if last_query and last_states:
                new_state = user_query.replace("what about", "").strip()  # Extract new state
                user_query = re.sub(r"(in|between|for|,)?\s*" + "|".join(last_states), f"in {new_state}", last_query, flags=re.IGNORECASE)
                
            else:
                print("❌ No previous question found to reference. Please ask a full question.")
                continue

        # Extract states
        states = extract_states_from_query(user_query)
        if not states:
            print("❌ Could not detect a valid state. Please specify a state in your question.")
            continue

        print(f"\n🔎 Searching for Life Settlement Statutes specific to: {', '.join(states)}...")

        # Load statutes for detected states
        statute_texts = load_statutes_for_states(states)
        if not statute_texts:
            print(f"❌ No statutes found for {', '.join(states)}.")
            continue

        # Generate response
        response = generate_gpt_response(user_query, statute_texts)
        print(f"\n{response}\n")

        # **Save the last valid query & states**
        last_query = user_query
        last_states = states






# Run the program
if __name__ == "__main__":
    main()
