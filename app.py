import os
import openai
import streamlit as st
from streamlit_chat import message
import pandas as pd

from utils import format_doc, format_prompt, doc_prompt_cfg, get_embedding, search_docs
from utils_sepaq import PARCS


# Setting page title and header
st.set_page_config(page_title="SepaqGPT", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>SépaqGPT - l'assistant pour vos randos! 🤖⛰️</h1>", unsafe_allow_html=True)

configs = {
    "path_embeddings": "./df_embeddings.csv",
    "model_name_embeddings": "text-embedding-ada-002"
}

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0
if 'docs_sent' not in st.session_state:
    st.session_state['docs_sent'] = []
if 'df_rando' not in st.session_state:
    st.session_state['df_rando'] = []
if 'parcs' not in st.session_state:
    st.session_state['parcs'] = ["jacques_cartier"]

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation

parc = st.sidebar.multiselect(
    'Par quel parc êtes-vous intéressé?',
    (PARCS.keys()),
    default="Jacques-Cartier")
st.sidebar.markdown("***")
api_key = st.sidebar.text_input('Entrer votre clé API', '')
st.sidebar.markdown("***")  
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Cout total de la conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Effacer la conversation", key="clear")


# Map model names to OpenAI model IDs
model = "gpt-3.5-turbo"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    st.session_state["docs_sent"] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(prompt, configs):

    df_embeddings = pd.read_csv(configs["path_embeddings"])
    df_embeddings = df_embeddings[df_embeddings.parc.isin(st.session_state["parcs"])]

    prompt_embedding = get_embedding(prompt)
    similar_rando = search_docs(df_embeddings, prompt_embedding)

    # for parc in st.session_state["parcs"]:
        # df_rando = pd.read_csv(f"./data/randos/{parc}.csv")

    for doc_sent in st.session_state["docs_sent"]:
        for i, message in enumerate(st.session_state['messages']):
            st.session_state['messages'][i]['content'] = message['content'].replace(doc_sent, '')

    docs = format_doc(similar_rando, max_tokens=3000)
    # save it in a list of remove it from context later
    st.session_state["docs_sent"].append(docs)

    doc_prompt = format_prompt(docs, doc_prompt_cfg)
    prompt = doc_prompt + prompt

    st.session_state['messages'].append({"role": "user", "content": prompt})

    completion = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages'],
        temperature=0.5
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:

    st.session_state["parcs"] = [PARCS[i] for i in parc if i in PARCS]
    if os.getenv("OPENAI_API_KEY") is not None:
        st.session_state["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    else:
        st.session_state["openai_api_key"] = api_key

    openai.api_key = api_key

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Votre question:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input, configs)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        cost = total_tokens * 0.002 / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {model}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")