from edullm.llm.rag import rag_pipeline
import streamlit as st
from edullm.llm.rag.rag_pipeline import RAGPipeline


@st.cache_resource(show_spinner=True)
def pipeline():
    return RAGPipeline()


def main():
    rag_pipeline = pipeline()

    st.header("TutorLLM: Your learning assistant")

    # Initialize the chat message history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about your documents"}
        ]

    # initial message and append messages to session state
    query = st.chat_input(placeholder="Ask your question here")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

    # display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # generate a new response from the LLM if the last message in the queue is from the user
    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Processing"):
                response = rag_pipeline._query_chat_engine(query)
                st.write(response.response)
                # message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
