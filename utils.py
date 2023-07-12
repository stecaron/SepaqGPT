import openai
from openai.embeddings_utils import cosine_similarity

doc_prompt_cfg = {
        "max_tokens": 3500,
        "text_before_documents": (
            "Vous etes un assistant qui soumet des recommendations sur les randonnées offertes part la Sépaq, un organisme mettant de l'avant les lieux de nature au Québec. Si la question concerne d'autres activités ou les hébergements, veuillez répondre que vous êtes seulement disposé à répondre aux questions concernant les randonnées, pour tout autre question veuillez téléphoner à 1-800-sepaq."
            "Voici la liste de randonnées"
            "<DOCUMENTS> "
        ),
        "text_before_prompt": (
            "<\DOCUMENTS>\n"
            "Réponds à la question suivante:\n"  
        )
      }


def format_prompt(documents_str, config):
        """
        Prepare the system prompt with prompt engineering.

        Joins the text before and after documents with
        """
        formatter = "{text_before_documents}\n{documents}\n{text_before_prompt}"
        system_prompt = formatter.format(
            text_before_documents=config["text_before_documents"], documents=documents_str, text_before_prompt=config["text_before_prompt"]
        )
        token_count = len(system_prompt.split())
        if token_count > config["max_tokens"]:
            raise ValueError(f"System prompt tokens > {config['max_tokens']}")
        return system_prompt


def format_doc(matched_documents, max_tokens):
        """Format our matched documents to plaintext.
        We also make sure they fit in the alloted max_tokens space.
        """
        documents_str = ""
        total_tokens = 0

        num_total_docs = len(matched_documents)
        num_preserved_docs = 0

        for _, row in matched_documents.iterrows():
            doc = ""
            for col in row.index:
                doc = doc + f"{col}: {row[col]}\n"

            num_preserved_docs += 1
            token_count = len(doc.split())
            if total_tokens + token_count <= max_tokens:
                documents_str += f"<DOCUMENT>{doc}<\\DOCUMENT>"
                total_tokens += token_count
            else:
                remaining_tokens = max_tokens - total_tokens
                truncated_doc = " ".join(doc.split()[:remaining_tokens])
                documents_str += f"<DOCUMENT>{truncated_doc}<\\DOCUMENT>"
                break

        if num_preserved_docs < (num_total_docs):
            matched_documents = matched_documents.iloc[:num_preserved_docs]

        return documents_str


def search_docs(doc_df, q_embedding, max_rando=15):
    doc_df.reset_index(drop=True, inplace=True)
    emb_cols = [col for col in doc_df if col.startswith('emb_')]
    test_emb = doc_df[emb_cols].apply(lambda x: cosine_similarity(x, q_embedding), axis=1)
    res = doc_df.iloc[test_emb.sort_values(ascending=False).index][0:(max_rando-1)][["parc", "text"]]
    return res


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']