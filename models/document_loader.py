from typing import List
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def web_loader(urls: List[str], vector_store, company_name: str, classes=None, metadata=None):
    for url in urls:
        loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=classes
            ) if classes else None
        ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        for split in all_splits:
            split.metadata["source_url"] = url
            split.metadata["company"] = company_name

        if metadata:
            for split in all_splits:
                for key, value in metadata.items():
                    split.metadata[key] = value

        indices = vector_store.add_documents(documents=all_splits)

    return vector_store
