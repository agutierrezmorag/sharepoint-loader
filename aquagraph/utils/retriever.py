import json
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document

load_dotenv()


class CustomAzureAISearchRetriever(AzureAISearchRetriever):
    """Custom retriever for Azure AI Search with metadata parsing capabilities.

    Extends the base AzureAISearchRetriever to add custom metadata parsing
    and document retrieval functionality. Handles both synchronous and
    asynchronous document retrieval operations.
    """

    def _parse_metadata(self, raw_metadata: Dict) -> Dict:
        """Parse nested metadata JSON string into dictionary.

        Args:
            raw_metadata (Dict): Raw metadata dictionary potentially containing
                               nested JSON string under 'metadata' key

        Returns:
            Dict: Parsed metadata with extracted title, source and page number.
                 Returns original metadata if parsing fails.
        """
        if isinstance(raw_metadata.get("metadata"), str):
            try:
                parsed_metadata = json.loads(raw_metadata["metadata"])
                return {
                    "title": parsed_metadata.get("title", ""),
                    "source": parsed_metadata.get("source", ""),
                    "page": parsed_metadata.get("page", ""),
                }
            except json.JSONDecodeError:
                return raw_metadata
        return raw_metadata

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents for a given query synchronously.

        Args:
            query (str): Search query to find relevant documents
            run_manager (CallbackManagerForRetrieverRun): Callback manager for the retrieval operation

        Returns:
            List[Document]: List of Document objects containing content and parsed metadata
        """
        search_results = self._search(query)

        documents = []
        for result in search_results:
            content = result.pop(self.content_key)
            parsed_metadata = self._parse_metadata(result)
            documents.append(Document(page_content=content, metadata=parsed_metadata))
        return documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents for a given query asynchronously.

        Args:
            query (str): Search query to find relevant documents
            run_manager (AsyncCallbackManagerForRetrieverRun): Async callback manager for the retrieval operation

        Returns:
            List[Document]: List of Document objects containing content and parsed metadata
        """
        search_results = await self._asearch(query)

        documents = []
        for result in search_results:
            content = result.pop(self.content_key)
            parsed_metadata = self._parse_metadata(result)
            documents.append(Document(page_content=content, metadata=parsed_metadata))
        return documents
